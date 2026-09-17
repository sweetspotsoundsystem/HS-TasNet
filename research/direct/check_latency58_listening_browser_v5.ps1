param(
    [Parameter(Mandatory=$true)][string]$PageUrl,
    [Parameter(Mandatory=$true)][string]$ReceiptPath,
    [int]$DebugPort = 9234
)

# Exercise decoding and controls in an isolated, muted browser. This script
# records no ratings, preferences or human listening responses.
$ErrorActionPreference = "Stop"
if (([Uri]$PageUrl).Host -ne "127.0.0.1" -or (Test-Path -LiteralPath $ReceiptPath)) {
    throw "Use a local page and a fresh receipt"
}
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path -LiteralPath $chrome)) { throw "Chrome is unavailable" }
$profile = Join-Path $env:TEMP ("latency58-player-check-" + [Guid]::NewGuid().ToString("N"))
$endpoint = "http://127.0.0.1:$DebugPort"
$occupied = $false
try { Invoke-RestMethod "$endpoint/json/version" -TimeoutSec 2 | Out-Null; $occupied = $true } catch {}
if ($occupied) { throw "Debug port is already occupied" }
New-Item -ItemType Directory -Path $profile | Out-Null
$socket = $null
$script:NextId = 0
$began = [Diagnostics.Stopwatch]::StartNew()
$deadline = [Threading.CancellationTokenSource]::new(90000)
$report = $null

function Invoke-Cdp([string]$Method, [hashtable]$Params = @{}) {
    $id = ++$script:NextId
    $payload = @{ id=$id; method=$Method; params=$Params } | ConvertTo-Json -Compress -Depth 20
    $bytes = [Text.Encoding]::UTF8.GetBytes($payload)
    $socket.SendAsync([ArraySegment[byte]]::new($bytes),
        [System.Net.WebSockets.WebSocketMessageType]::Text, $true,
        $deadline.Token).GetAwaiter().GetResult() | Out-Null
    while ($true) {
        $stream = [IO.MemoryStream]::new()
        $buffer = New-Object byte[] 65536
        do {
            $received = $socket.ReceiveAsync([ArraySegment[byte]]::new($buffer),
                $deadline.Token).GetAwaiter().GetResult()
            if ($received.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) {
                throw "Browser connection closed"
            }
            $stream.Write($buffer, 0, $received.Count)
        } until ($received.EndOfMessage)
        $message = [Text.Encoding]::UTF8.GetString($stream.ToArray()) | ConvertFrom-Json
        $stream.Dispose()
        if ($message.id -eq $id) {
            if ($null -ne $message.error) { throw ($message.error | ConvertTo-Json -Compress) }
            return $message.result
        }
    }
}

try {
    $arguments = @("--headless=new", "--disable-gpu", "--no-first-run",
        "--no-default-browser-check", "--mute-audio", "--disk-cache-size=1000000",
        "--media-cache-size=1000000", "--autoplay-policy=no-user-gesture-required",
        "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=$DebugPort",
        "--user-data-dir=`"$profile`"", "about:blank")
    $process = Start-Process -FilePath $chrome -ArgumentList $arguments -PassThru
    $target = $null
    while ($began.Elapsed.TotalSeconds -lt 20 -and $null -eq $target) {
        try {
            $targets = Invoke-RestMethod "$endpoint/json/list" -TimeoutSec 2
            $target = $targets |
                Where-Object { $_.type -eq "page" -and $_.url -eq "about:blank" } |
                Select-Object -First 1
        } catch {}
        if ($null -eq $target) { Start-Sleep -Milliseconds 100 }
    }
    if ($null -eq $target) { throw "Isolated browser did not start" }
    $socket = [System.Net.WebSockets.ClientWebSocket]::new()
    $socket.ConnectAsync([Uri]$target.webSocketDebuggerUrl,
        $deadline.Token).GetAwaiter().GetResult() | Out-Null
    $version = Invoke-Cdp "Browser.getVersion"
    Invoke-Cdp "Page.enable" | Out-Null
    Invoke-Cdp "Page.navigate" @{url=$PageUrl} | Out-Null
    Start-Sleep -Milliseconds 500
    $expression = @'
(async () => {
  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
  function require(value, message) { if (!value) throw new Error(message); }
  async function until(test, label) {
    const deadline = performance.now() + 10000;
    while (!test() && performance.now() < deadline) await wait(25);
    require(test(), 'Timeout: ' + label);
  }
  await until(() => document.getElementById('sources')?.children.length >= 5, 'page controls');
  const data = JSON.parse(document.getElementById('listening-data').textContent);
  const audio = document.getElementById('player');
  const stem = document.getElementById('stem');
  const button = id => document.querySelector('[data-source="' + id + '"]');
  const ready = () => audio.readyState >= 2 && !audio.error && Number.isFinite(audio.duration);
  const estimates = data.sources.filter(x => x.kind === 'estimate');
  require([3, 4].includes(estimates.length) && data.sources.length === estimates.length + 2 && stem.options.length === 4,
    'Unexpected player inventory');
  require(estimates[0].id === 'working' && estimates.at(-1).id === 'candidate' && estimates.some(x => x.id === 'parent'),
    'Unexpected comparison sources');
  const expectedFiles = estimates.length * 4 + 5;
  require(data.human_listening_verdict === null, 'Player contains a listening verdict');
  const files = [];
  for (const source of data.sources) {
    for (const name of source.kind === 'mixture' ? ['bass'] : ['drums', 'bass', 'vocals', 'other']) {
      audio.pause();
      stem.value = name;
      stem.dispatchEvent(new Event('change'));
      button(source.id).click();
      await until(ready, source.id + '/' + name);
      require(Math.abs(audio.duration - 15) < 1e-6, 'Wrong clip duration');
      require(button(source.id).getAttribute('aria-pressed') === 'true', 'Source state differs');
      require(audio.currentSrc === document.getElementById('open-wav').href, 'WAV link differs');
      audio.currentTime = 3;
      await until(() => !audio.seeking && Math.abs(audio.currentTime - 3) < 0.01, 'nonzero seek');
      await audio.play();
      await until(() => audio.currentTime > 3.05, 'playback advances');
      audio.pause();
      files.push({source:source.id, stem:source.kind === 'mixture' ? null : name,
        url:audio.currentSrc, duration:audio.duration, played_to_seconds:audio.currentTime,
        decoded:true, playback_advanced:true});
    }
  }
  require(files.length === expectedFiles && new Set(files.map(x => x.url)).size === expectedFiles, 'Incomplete WAV coverage');
  stem.value = 'bass'; stem.dispatchEvent(new Event('change'));
  button('working').click(); await until(ready, 'initial switching clip');
  audio.currentTime = 4.25; await until(() => !audio.seeking && Math.abs(audio.currentTime - 4.25) < 0.01, 'switching seek');
  await audio.play();
  const beforeSource = audio.currentTime;
  button('candidate').click();
  await until(() => ready() && !audio.seeking && !audio.paused, 'source resume');
  const afterSource = audio.currentTime;
  require(beforeSource >= 4.24 && Math.abs(afterSource - beforeSource) < 0.25, 'Source switch lost position');
  const beforeStem = audio.currentTime;
  stem.value = 'vocals'; stem.dispatchEvent(new Event('change'));
  await until(() => ready() && !audio.seeking && !audio.paused, 'stem resume');
  const afterStem = audio.currentTime;
  require(beforeStem >= 4.24 && Math.abs(afterStem - beforeStem) < 0.25, 'Stem switch lost position');
  const beforeRapid = audio.currentTime;
  button('candidate').click(); button('parent').click(); button('working').click();
  await until(() => ready() && !audio.seeking && !audio.paused, 'rapid source resume');
  const afterRapid = audio.currentTime;
  require(beforeRapid >= 4.24 && Math.abs(afterRapid - beforeRapid) < 0.25 && button('working').getAttribute('aria-pressed') === 'true',
    'Rapid switches lost the final source or position');
  document.body.dispatchEvent(new KeyboardEvent('keydown', {key:String(data.sources.findIndex(x => x.id === 'candidate') + 1), bubbles:true}));
  await until(() => ready() && button('candidate').getAttribute('aria-pressed') === 'true', 'keyboard source');
  const loop = document.getElementById('loop');
  loop.checked = false; loop.dispatchEvent(new Event('change'));
  require(!audio.loop, 'Loop disable failed');
  loop.checked = true; loop.dispatchEvent(new Event('change'));
  require(audio.loop, 'Loop enable failed');
  audio.pause();
  require(document.getElementById('error').hidden, 'Player displayed an error');
  return {status:'pass', files, source_switch:{before:beforeSource, after:afterSource},
    stem_switch:{before:beforeStem, after:afterStem}, rapid_source_switch:{before:beforeRapid, after:afterRapid},
    strict_nonzero_seek_verified:true,
    keyboard_source_switch:true, loop_toggle:true, browser_audio_muted:true,
    human_listening_completed:false, human_listening_verdict:null};
})()
'@
    $result = Invoke-Cdp "Runtime.evaluate" @{
        expression=$expression; awaitPromise=$true; returnByValue=$true; userGesture=$true
    }
    if ($null -ne $result.exceptionDetails) {
        throw ($result.exceptionDetails | ConvertTo-Json -Compress -Depth 15)
    }
    $profileBytes = (Get-ChildItem -LiteralPath $profile -File -Recurse -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    $report = @{schema="latency58-listening-browser-check-v1"; status="pass";
        page_url=$PageUrl; browser=$version; checks=$result.result.value;
        elapsed_seconds=$began.Elapsed.TotalSeconds; temporary_profile_bytes=$profileBytes;
        isolated_profile=$true; human_listening_completed=$false; human_listening_verdict=$null}
} catch {
    $report = @{schema="latency58-listening-browser-check-v1"; status="failed";
        page_url=$PageUrl; error=$_.Exception.Message; elapsed_seconds=$began.Elapsed.TotalSeconds;
        human_listening_completed=$false; human_listening_verdict=$null}
} finally {
    if ($null -ne $socket) {
        try { Invoke-Cdp "Browser.close" | Out-Null } catch {}
        $socket.Dispose()
    }
    # Only processes carrying the fresh, unique profile path belong to this run.
    Get-CimInstance Win32_Process -Filter "name = 'chrome.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($profile) } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    for ($attempt = 0; $attempt -lt 20 -and (Test-Path -LiteralPath $profile); $attempt++) {
        try { Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction Stop } catch {
            Start-Sleep -Milliseconds 100
        }
    }
    if ($null -ne $report) {
        $report.temporary_profile_removed = -not (Test-Path -LiteralPath $profile)
        $json = $report | ConvertTo-Json -Depth 20
        [IO.File]::WriteAllText($ReceiptPath, $json + "`n", [Text.UTF8Encoding]::new($false))
        Write-Output $json
    }
    $deadline.Dispose()
}
if ($report.status -ne "pass" -or -not $report.temporary_profile_removed) { exit 1 }



