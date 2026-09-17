"""One owned PowerShell process with bounded, authenticated request/response IO."""
import base64
import json
import os
import secrets
import select
import subprocess
import time


WORKER = r"""
$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$session='SESSION_TOKEN'
$self=[System.Diagnostics.Process]::GetCurrentProcess()
$started=$self.StartTime.ToUniversalTime().ToString('o')
[Console]::Out.WriteLine(([pscustomobject]@{kind='ready';session=$session;pid=$PID;started=$started}|ConvertTo-Json -Compress))
[Console]::Out.Flush()
while (($line=[Console]::In.ReadLine()) -ne $null) {
  try {
    $request=ConvertFrom-Json -InputObject $line
    if ($request.session -ne $session -or $request.id -le 0) { throw 'Invalid request identity' }
    $program=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($request.program))
    $answer=@(& ([scriptblock]::Create($program))) -join "`n"
    [Console]::Out.WriteLine(([pscustomobject]@{kind='response';session=$session;id=$request.id;pid=$PID;stdout=$answer}|ConvertTo-Json -Depth 4 -Compress))
    [Console]::Out.Flush()
  } catch {
    [Console]::Error.WriteLine($_.Exception.ToString())
    exit 2
  }
}
"""


class PersistentWindowsQuery:
    """Never restart a worker after a failed query; the supervisor must stop."""

    def __init__(self, powershell):
        self.powershell = str(powershell)
        self.session = secrets.token_hex(24)
        self.process = None
        self.worker_pid = None
        self.worker_started = None
        self.request_id = 0
        self.buffer = bytearray()
        self.broken = False
        self.closed = False

    def _start(self):
        program = WORKER.replace("SESSION_TOKEN", self.session)
        argv = [self.powershell, "-NoProfile", "-NonInteractive", "-EncodedCommand",
                base64.b64encode(program.encode("utf-16le")).decode()]
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            os.set_blocking(stream.fileno(), False)

    def query(self, program, timeout):
        if self.broken or self.closed:
            raise RuntimeError("Event worker is failed or closed; automatic restart is forbidden")
        if not isinstance(program, str) or not 0 < timeout <= 15:
            raise ValueError("Require a program and the bounded event-query deadline")
        began = time.monotonic()
        deadline = began + timeout
        stderr = bytearray()
        stdout_observed = bytearray()
        try:
            if self.process is None:
                self._start()
            process = self.process
            if process.poll() is not None:
                raise RuntimeError(f"Event worker exited with code {process.returncode}")
            if self.buffer:
                raise RuntimeError("Unconsumed event-worker output before a new request")
            self.request_id += 1
            request_id = self.request_id
            request = (json.dumps({"session": self.session, "id": request_id,
                       "program": base64.b64encode(program.encode()).decode()}, separators=(",", ":")) + "\n").encode()
            sent = 0
            eof = False
            while True:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("persistent_windows_system", timeout,
                        output=bytes(stdout_observed), stderr=bytes(stderr))
                while b"\n" in self.buffer:
                    line, _, remainder = self.buffer.partition(b"\n")
                    self.buffer = bytearray(remainder)
                    frame = json.loads(line.decode("utf-8-sig"))
                    if frame.get("session") != self.session:
                        raise RuntimeError("Event-worker session identity mismatch")
                    if frame.get("kind") == "ready":
                        if self.worker_pid is not None or type(frame.get("pid")) is not int or frame["pid"] <= 0:
                            raise RuntimeError("Duplicate or malformed event-worker handshake")
                        if not isinstance(frame.get("started"), str) or not frame["started"]:
                            raise RuntimeError("Missing event-worker process creation identity")
                        self.worker_pid, self.worker_started = frame["pid"], frame["started"]
                        continue
                    if (frame.get("kind") != "response" or self.worker_pid is None
                            or frame.get("pid") != self.worker_pid or type(frame.get("id")) is not int
                            or frame["id"] != request_id or sent != len(request)
                            or not isinstance(frame.get("stdout"), str) or self.buffer):
                        raise RuntimeError("Malformed, stale or unsolicited event-worker response")
                    if process.poll() is not None:
                        raise RuntimeError("Event worker exited while returning a response")
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired("persistent_windows_system", timeout,
                            output=bytes(stdout_observed), stderr=bytes(stderr))
                    return {"stdout": frame["stdout"], "stderr": stderr.decode("utf-8", errors="replace"),
                            "worker_pid": self.worker_pid, "worker_started": self.worker_started,
                            "request_id": request_id, "elapsed_seconds": time.monotonic() - began,
                            "actual_exit_code": None, "worker_still_running": True}
                if eof or process.poll() is not None:
                    raise RuntimeError(f"Event worker exited before responding: {process.poll()}")
                readers, writers, _ = select.select(
                    [process.stdout, process.stderr], [process.stdin] if sent < len(request) else [], [],
                    max(0, deadline - time.monotonic()))
                if writers:
                    try:
                        sent += os.write(process.stdin.fileno(), request[sent:])
                    except BlockingIOError:
                        pass
                for stream in readers:
                    try:
                        chunk = os.read(stream.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if stream is process.stdout:
                        if not chunk:
                            eof = True
                        self.buffer.extend(chunk)
                        stdout_observed.extend(chunk)
                    else:
                        stderr.extend(chunk)
                if len(stdout_observed) + len(stderr) > 64 * 1024 * 1024:
                    raise RuntimeError("Event-worker response exceeds the bounded diagnostic buffer")
        except BaseException:
            self.broken = True
            raise

    def close(self, timeout=5):
        """Send EOF and drain pipes; signal only this Popen-owned worker if needed."""
        self.closed = True
        process = self.process
        if process is None:
            return {"started": False, "closed": True, "forced": False, "actual_exit_code": None}
        began = time.monotonic()
        process.stdin.close()
        forced = False
        streams = [process.stdout, process.stderr]
        while process.poll() is None and time.monotonic() - began < timeout:
            readable, _, _ = select.select(streams, [], [], min(.1, max(0, timeout - (time.monotonic() - began))))
            for stream in readable:
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    streams.remove(stream)
        if process.poll() is None:
            forced = True
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for stream in (process.stdout, process.stderr):
            stream.close()
        return {"started": True, "closed": process.poll() is not None, "forced": forced,
                "actual_exit_code": process.returncode, "worker_pid": self.worker_pid,
                "worker_started": self.worker_started, "linux_pid": process.pid,
                "elapsed_seconds": time.monotonic() - began}
