"""Supervise one GPU child using owned persistent NVML and Windows workers.

No GPU workload starts on import. Root reviews source and exact launch spec
before executing. The supervisor never imports Torch or changes driver,
registry, power, clock, or operating-system settings.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
SMI = "/usr/lib/wsl/lib/nvidia-smi"
TRANSPORT_SOURCE = HERE.parents[3] / "latency58_windows_event_transport.py"
TRANSPORT_SHA256 = "27c42b89faa7fc5eb9d84f195501945eb6aa1c5c3da520b094fb295ab77d4dab"
GPU_TRANSPORT_SOURCE = HERE.parents[3] / "latency58_nvml_transport.py"
GPU_WORKER_SOURCE = HERE.parents[3] / "latency58_nvml_worker.py"
GPU_READER_SOURCE = HERE.parents[3] / "latency58_nvml_reader.py"
GPU_LIBRARY = Path("/usr/lib/wsl/lib/libnvidia-ml.so.1")
GPU_BINDINGS = {
    str(GPU_TRANSPORT_SOURCE): "a8373cc3aa4e26e94a84b50a6ec8492e68bf33023f836863a4d2c5f8f085b35c",
    str(GPU_WORKER_SOURCE): "f3a93b09e6cc46329479ff262e6d1b34b7a3f5d95f3db9aa7deb1a23617ea15a",
    str(GPU_READER_SOURCE): "6470883f05e07db05b9ce3910104715e39963ca124d949fc3762bfef05556f10",
    str(GPU_LIBRARY): "f1853527c3738e9632695dddbb6e3129440c886d70a5c270b42089d3a3ddf4cb",
}
SCAN_LIMIT = 512
MAX_SCANNED_RECORDS = 4096
SENTINEL = 63827  # Preserved 2026-09-06T03:09:20.0097296Z nvlddmkm BusReset TDR.
SMI_FIELDS = ("uuid", "name", "driver_version", "memory.total", "memory.used",
              "temperature.gpu", "power.draw", "power.limit", "utilization.gpu")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def event_query(previous=None, *, verify_sentinel=False):
    """Read all providers by record order, never use StartTime filtering."""
    require(previous is None or type(previous) is int and previous >= 0, "Invalid event high-water mark")
    marker = -1 if previous is None else previous
    sentinel = ""
    if verify_sentinel:
        sentinel = f'''
$sentinel = @(Get-WinEvent -LogName System -FilterXPath "*[System[EventRecordID = {SENTINEL}]]" -MaxEvents 2 -ErrorAction Stop)
if ($sentinel.Count -ne 1 -or $sentinel[0].ProviderName -ne 'nvlddmkm' -or $sentinel[0].Id -ne 153 -or $sentinel[0].ToXml() -notmatch 'BusReset TDR') {{ throw 'Known TDR sentinel was not recovered' }}
'''
    return f'''
[Console]::Error.WriteLine('telemetry_phase:start:' + [DateTime]::UtcNow.ToString('o'))
$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='Stop'
try {{
{sentinel}
  [Console]::Error.WriteLine('telemetry_phase:before_initial_read:' + [DateTime]::UtcNow.ToString('o'))
  $records=@(Get-WinEvent -LogName System -MaxEvents {SCAN_LIMIT} -ErrorAction Stop)
  [Console]::Error.WriteLine('telemetry_phase:after_initial_read:' + [DateTime]::UtcNow.ToString('o'))
  if ($records.Count -eq 0) {{ throw 'Empty System event log' }}
  $previous=[long]{marker}
  while ($previous -ge 0 -and $records[-1].RecordId -gt $previous -and $records.Count -lt {MAX_SCANNED_RECORDS}) {{
    $before=[long]$records[-1].RecordId
    $older=@(Get-WinEvent -LogName System -FilterXPath "*[System[EventRecordID < $before]]" -MaxEvents {SCAN_LIMIT} -ErrorAction Stop)
    if ($older.Count -eq 0) {{ throw 'Event coverage gap while paging older records' }}
    $records += $older
  }}
  [Console]::Error.WriteLine('telemetry_phase:coverage_loaded:' + [DateTime]::UtcNow.ToString('o'))
  $new=@($records | Where-Object {{ $_.RecordId -gt $previous }} | ForEach-Object {{
    [pscustomobject]@{{RecordId=[long]$_.RecordId; Id=[int]$_.Id; Provider=$_.ProviderName;
      Level=$_.Level; Utc=$_.TimeCreated.ToUniversalTime().ToString('o'); Xml=$_.ToXml()}}
  }})
  [Console]::Error.WriteLine('telemetry_phase:records_encoded:' + [DateTime]::UtcNow.ToString('o'))
  [pscustomobject]@{{Status='ok'; CheckedUtc=[DateTime]::UtcNow.ToString('o');
    Count=$records.Count; RecordIds=@($records | ForEach-Object {{[long]$_.RecordId}});
    NewRecords=$new; SentinelVerified={'$true' if verify_sentinel else '$false'}}} |
    ConvertTo-Json -Depth 6 -Compress
}} catch {{
  [pscustomobject]@{{Status='error'; Error=$_.Exception.Message; ErrorId=$_.FullyQualifiedErrorId}} |
    ConvertTo-Json -Compress
  exit 2
}}
'''


def validate_events(payload, previous=None):
    require(payload.get("Status") == "ok", "Windows event query failed")
    ids = payload.get("RecordIds")
    require(isinstance(ids, list) and 1 <= len(ids) <= MAX_SCANNED_RECORDS
            and payload.get("Count") == len(ids) and all(type(value) is int and value > 0 for value in ids),
            "Malformed event scan coverage")
    require(all(left == right + 1 for left, right in zip(ids, ids[1:])),
            "System record scan has an interior gap, duplicate, or ordering change")
    if previous is not None:
        require(ids[0] >= previous and previous in ids, "System event coverage gap or log reset")
    wanted = set(ids if previous is None else [value for value in ids if value > previous])
    rows = payload.get("NewRecords")
    require(isinstance(rows, list) and {row.get("RecordId") for row in rows} == wanted
            and len(rows) == len(wanted), "New-record inventory differs from covered IDs")
    require(all(isinstance(row.get("Xml"), str) and "<Event " in row["Xml"]
                and isinstance(row.get("Provider"), str) and type(row.get("Id")) is int
                for row in rows), "A raw event record is missing")
    return ids[0], rows


def reset_event(row):
    provider, event_id = row["Provider"], row["Id"]
    level = row.get("Level")
    warning_or_error = type(level) is int and 1 <= level <= 3
    return (provider == "nvlddmkm" and (event_id in (13, 14, 153) or warning_or_error)
            or provider == "Display" and (event_id == 4101 or warning_or_error)
            or provider == "Microsoft-Windows-WHEA-Logger"
            or provider == "Microsoft-Windows-Resource-Exhaustion-Detector"
            or provider == "Microsoft-Windows-Kernel-Power" and event_id == 41
            or provider == "EventLog" and event_id == 6008)


def parse_gpu(stdout):
    rows = list(csv.reader(io.StringIO(stdout.strip())))
    require(len(rows) == 1 and len(rows[0]) == len(SMI_FIELDS), "Expected one complete GPU telemetry row")
    fields = dict(zip(SMI_FIELDS, (value.strip() for value in rows[0]), strict=True))
    require(fields["uuid"].startswith("GPU-") and fields["driver_version"] not in ("", "N/A", "[N/A]"),
            "GPU identity or driver telemetry unavailable")
    for key in ("memory.total", "memory.used", "temperature.gpu"):
        fields[key] = float(fields[key])
        require(math.isfinite(fields[key]) and fields[key] >= 0, f"Invalid required GPU telemetry: {key}")
    require(fields["memory.total"] > 0 and fields["memory.used"] <= fields["memory.total"], "Invalid GPU memory accounting")
    # WSL NVML can omit these supplemental counters; retain the raw strings.
    return fields


def gpu_alert(gpu, baseline, *, max_temperature, memory_headroom):
    if (gpu["uuid"], gpu["driver_version"], gpu["memory.total"]) != (
            baseline["uuid"], baseline["driver_version"], baseline["memory.total"]):
        return "GPU identity, driver, or total memory changed"
    if gpu["temperature.gpu"] >= max_temperature:
        return "GPU temperature reached the configured guard"
    if gpu["memory.total"] - gpu["memory.used"] < memory_headroom:
        return "GPU free memory fell below the configured headroom"
    return None


def progress_step(path):
    """Read a bounded tail; ignore an incomplete append still being written."""
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 16384))
        data = stream.read()
    lines = data.split(b"\n")[:-1]
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(row, dict) and type(row.get("step")) is int and row["step"] >= 0:
            return row["step"]
    return None


def load_event_transport():
    require(sha(TRANSPORT_SOURCE) == TRANSPORT_SHA256, "Persistent event transport source changed")
    spec = importlib.util.spec_from_file_location("latency58_windows_event_transport", TRANSPORT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PersistentWindowsQuery(POWERSHELL)


def load_gpu_transport():
    require(all(sha(path) == digest for path, digest in GPU_BINDINGS.items()),
            "Persistent NVML source or library changed")
    spec = importlib.util.spec_from_file_location("latency58_nvml_transport", GPU_TRANSPORT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PersistentNvmlQuery(
        worker_source=GPU_WORKER_SOURCE, worker_sha256=GPU_BINDINGS[str(GPU_WORKER_SOURCE)],
        reader_source=GPU_READER_SOURCE, reader_sha256=GPU_BINDINGS[str(GPU_READER_SOURCE)],
        library=GPU_LIBRARY, library_sha256=GPU_BINDINGS[str(GPU_LIBRARY)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-spec", type=Path, required=True)
    parser.add_argument("--launch-spec-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-runtime-seconds", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--query-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=120.0)
    parser.add_argument("--progress-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--finalization-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--stop-grace-seconds", type=float, default=15.0)
    parser.add_argument("--post-exit-quiet-seconds", type=float, default=10.0)
    parser.add_argument("--max-temperature-c", type=float, default=80.0)
    parser.add_argument("--memory-headroom-mib", type=float, default=4096.0)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, float):
            require(math.isfinite(value) and value > 0, f"Require finite positive {key}")
    require(0.5 <= args.poll_seconds <= 10 and args.query_timeout_seconds <= 10
            and args.stop_grace_seconds <= 30 and args.max_temperature_c <= 80
            and args.memory_headroom_mib >= 4096 and 10 <= args.post_exit_quiet_seconds <= 30
            and args.finalization_timeout_seconds <= 300,
            "Do not weaken the reviewed supervision limits")
    require(sha(args.launch_spec) == args.launch_spec_sha256, "Launch spec changed after review")
    spec = json.loads(args.launch_spec.read_text())
    require(spec.get("schema") == "gpu-watchdog-launch-finalization-v1", "Unexpected launch spec schema")
    final_step = spec.get("expected_final_step")
    require(type(final_step) is int and final_step > 0, "Require a positive planned final update")
    command, cwd, overrides = spec["argv"], Path(spec["cwd"]), spec["environment"]
    require(isinstance(command, list) and len(command) > 1 and all(isinstance(part, str) and part for part in command)
            and Path(command[0]).is_absolute() and cwd.is_absolute() and cwd.is_dir(), "Use an explicit executable, argv, and cwd")
    require(isinstance(overrides, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in overrides.items())
            and overrides.get("CUDA_VISIBLE_DEVICES") == "0", "The reviewed child must explicitly select GPU 0")
    progress = Path(spec["progress_path"])
    require(progress.is_absolute(), "Progress log must be an absolute path")
    out = args.output_dir.resolve()
    require(out.parent == HERE and not out.exists(), "Use a new direct child result directory")
    out.mkdir()
    evidence = (out / "watchdog.jsonl").open("x", buffering=1)
    lock = (HERE / "gpu-watchdog.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    child_log = None
    launched = None
    status, reason = "preflight", None
    high_water = None
    last_step = progress_step(progress)
    require(last_step is None, "Use a fresh owned progress journal")
    last_progress = None
    saw_progress = False
    finalization_started = None
    exited_at = None
    interrupted = False
    started = time.monotonic()
    source_before = sha(__file__)
    event_worker = load_event_transport()
    gpu_worker = None

    def record(event, **fields):
        evidence.write(json.dumps({"utc": utc(), "monotonic": time.monotonic(), "event": event, **fields},
                                  allow_nan=False) + "\n")
        evidence.flush()
        os.fsync(evidence.fileno())

    def windows_query(program):
        began = time.monotonic()
        try:
            result = event_worker.query(program, args.query_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            record("query_timeout", kind="windows_system", duration=time.monotonic() - began,
                   stdout=repr(error.output), stderr=repr(error.stderr), transport="persistent")
            raise RuntimeError("windows_system telemetry timed out") from error
        except BaseException as error:
            record("event_worker_error", reason=repr(error), duration=time.monotonic() - began,
                   worker_pid=event_worker.worker_pid)
            raise
        record("query", kind="windows_system", request_status="response_received",
               duration=time.monotonic() - began, stdout=result["stdout"], stderr=result["stderr"],
               transport="persistent", worker_pid=result["worker_pid"], request_id=result["request_id"],
               worker_exit_code=None, worker_still_running=result["worker_still_running"])
        return result["stdout"]

    def events(previous, *, sentinel=False):
        ps = event_query(previous, verify_sentinel=sentinel)
        payload = json.loads(windows_query(ps))
        newest, rows = validate_events(payload, previous)
        if sentinel:
            require(payload.get("SentinelVerified") is True, "Known event-query regression check did not pass")
        record("event_coverage", previous=previous, newest=newest, oldest=payload["RecordIds"][-1],
               scanned=payload["Count"], new_records=len(rows))
        return newest, rows

    def gpu():
        nonlocal gpu_worker
        began = time.monotonic()
        try:
            if gpu_worker is None:
                gpu_worker = load_gpu_transport()
            remaining = args.query_timeout_seconds - (time.monotonic() - began)
            require(remaining > 0, "NVML worker construction exceeded the query deadline")
            result = gpu_worker.query(remaining)
        except subprocess.TimeoutExpired as error:
            record("query_timeout", kind="nvml", duration=time.monotonic() - began,
                   stdout=repr(error.output), stderr=repr(error.stderr), transport="persistent_nvml",
                   worker_pid=getattr(gpu_worker, "worker_pid", None))
            raise RuntimeError("nvml telemetry timed out") from error
        except BaseException as error:
            record("gpu_worker_error", reason=repr(error), duration=time.monotonic() - began,
                   worker_pid=getattr(gpu_worker, "worker_pid", None))
            raise
        require(time.monotonic() - began < args.query_timeout_seconds, "NVML telemetry exceeded its query deadline")
        record("query", kind="nvml", request_status="response_received",
               duration=time.monotonic() - began, stdout=result["stdout"], stderr=result["stderr"],
               transport="persistent_nvml", worker_pid=result["worker_pid"], request_id=result["request_id"],
               worker_exit_code=None, worker_still_running=result["worker_still_running"], nvml=result["nvml"])
        fields = parse_gpu(result["stdout"])
        memory = result["nvml"]["memory_bytes"]
        require(isinstance(memory, dict) and set(memory) == {"total", "reserved", "free", "used"}
                and all(type(value) is int and 0 <= value < 2**64 - 1 for value in memory.values())
                and memory["total"] > 0
                and memory["total"] == memory["reserved"] + memory["free"] + memory["used"]
                and fields["memory.total"] == memory["total"] // 2**20
                and fields["memory.used"] == (memory["used"] + 2**20 - 1) // 2**20,
                "NVML raw memory accounting changed")
        # Preserve the original total-minus-allocated guard, and also account
        # for driver-reserved memory using the independently reported raw free bytes.
        require(memory["free"] >= args.memory_headroom_mib * 2**20,
                "GPU physical free memory fell below the configured headroom")
        return fields

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        record("configuration", argv=sys.argv, source_sha256=source_before, launch_spec=spec,
               launch_spec_sha256=args.launch_spec_sha256, limits={key: value for key, value in vars(args).items()
                   if isinstance(value, float)}, gpu_bindings=GPU_BINDINGS,
               scope="one owned process session; no global GPU-job discovery guarantee")
        high_water, old_rows = events(None, sentinel=True)
        baseline = gpu()
        reason = gpu_alert(baseline, baseline, max_temperature=args.max_temperature_c,
                           memory_headroom=args.memory_headroom_mib)
        require(reason is None, reason)
        record("baseline", high_water=high_water, gpu=baseline, previous_progress_step=last_step,
               prior_fault_records=[row["RecordId"] for row in old_rows if reset_event(row)])
        require(not interrupted, "Stop requested during preflight")
        require(sha(args.launch_spec) == args.launch_spec_sha256, "Launch spec changed during preflight")
        high_water, prelaunch_rows = events(high_water)
        require(not any(reset_event(row) for row in prelaunch_rows), "Fresh host fault during preflight")
        child_log = (out / "child.log").open("xb")
        child = subprocess.Popen(command, cwd=cwd, env={**os.environ, **overrides},
                                 stdout=child_log, stderr=subprocess.STDOUT, start_new_session=True)
        launched = last_progress = time.monotonic()
        status = "running"
        record("child_started", pid=child.pid, process_group=child.pid)
        while True:
            require(sha(__file__) == source_before and sha(args.launch_spec) == args.launch_spec_sha256
                    and all(sha(path) == digest for path, digest in GPU_BINDINGS.items()),
                    "Watchdog source or launch spec changed while supervising")
            # Even a just-exited child receives one final event/telemetry check.
            high_water, rows = events(high_water)
            faults = [row for row in rows if reset_event(row)]
            if faults:
                record("fresh_host_fault", records=faults)
                raise RuntimeError("Fresh NVIDIA/Display reset, WHEA, resource, or host reboot event")
            current_gpu = gpu()
            reason = gpu_alert(current_gpu, baseline, max_temperature=args.max_temperature_c,
                               memory_headroom=args.memory_headroom_mib)
            record("gpu", values=current_gpu)
            require(reason is None, reason)
            now = time.monotonic()
            step = progress_step(progress)
            if step is not None:
                require(step <= final_step, "Owned progress exceeded the planned final update")
                require(last_step is None or step >= last_step, "Owned progress counter moved backward")
                if last_step is None or step > last_step:
                    last_step, last_progress = step, now
                    saw_progress = True
                    record("progress", completed_step=step)
                if step == final_step and finalization_started is None:
                    finalization_started = now
                    record("finalization_started", completed_step=step,
                           timeout_seconds=args.finalization_timeout_seconds)
            code = child.poll()
            if code is not None:
                require(code != 0 or last_step == final_step, "Child exited successfully before its planned final update")
                if exited_at is None:
                    require(finalization_started is None or now - finalization_started < args.finalization_timeout_seconds,
                            "Child exit was not observed within its bounded finalization period")
                    exited_at = now
                    record("child_exited", numeric_exit_code=code,
                           quiet_interval_seconds=args.post_exit_quiet_seconds)
                if now - exited_at >= args.post_exit_quiet_seconds:
                    status = "pass" if code == 0 else "child_failed"
                    reason = None if code == 0 else f"Child exit code {code}"
                    break
                time.sleep(args.poll_seconds)
                continue
            require(not interrupted, "External stop requested")
            require(now - launched < args.max_runtime_seconds, "Bounded GPU stage reached its runtime limit")
            if finalization_started is not None:
                require(now - finalization_started < args.finalization_timeout_seconds,
                        "Owned child did not exit within its bounded finalization period")
            elif saw_progress or now - launched >= args.startup_grace_seconds:
                require(now - last_progress < args.progress_timeout_seconds, "Owned child stopped reporting completed updates")
            time.sleep(args.poll_seconds)
    except BaseException as error:
        status = ("blocked_preflight" if child is None else
                  "health_failed_after_child_exit" if child.poll() is not None else "stopped_by_watchdog")
        reason = repr(error)
        record("stop_reason", reason=reason, latest_completed_step_seen=last_step)
    finally:
        if child is not None and child.poll() is None:
            # Only this Popen-created session can be signalled. No PID/name scan.
            record("owned_stop", pid=child.pid, signal="SIGTERM", grace_seconds=args.stop_grace_seconds)
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=args.stop_grace_seconds)
            except subprocess.TimeoutExpired:
                if child.poll() is None:
                    try:
                        owned_group = os.getpgid(child.pid) == child.pid
                    except ProcessLookupError:
                        owned_group = False
                    if owned_group:
                        record("owned_stop", pid=child.pid, signal="SIGKILL", scope="owned process group")
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    elif child.poll() is None:
                        record("owned_stop", pid=child.pid, signal="SIGKILL", scope="owned PID; group changed")
                        child.kill()
                    child.wait(timeout=10)
        if child_log is not None:
            child_log.flush()
            os.fsync(child_log.fileno())
            child_log.close()
        try:
            event_worker_close = event_worker.close()
        except BaseException as error:
            event_worker_close = {"closed": False, "reason": repr(error)}
        record("event_worker_closed", **event_worker_close)
        if status in ("pass", "child_failed") and (not event_worker_close.get("closed")
                or event_worker_close.get("forced") or event_worker_close.get("actual_exit_code") != 0):
            status, reason = "event_worker_cleanup_failed", "Persistent event worker did not close normally"
        try:
            gpu_worker_close = (gpu_worker.close() if gpu_worker is not None else
                                {"closed": True, "not_started": True})
        except BaseException as error:
            gpu_worker_close = {"closed": False, "reason": repr(error)}
        record("gpu_worker_closed", **gpu_worker_close)
        if status in ("pass", "child_failed") and (not gpu_worker_close.get("closed")
                or gpu_worker_close.get("forced") or gpu_worker_close.get("actual_exit_code") != 0
                or not gpu_worker_close.get("identities_unchanged")):
            status, reason = "gpu_worker_cleanup_failed", "Persistent NVML worker did not close normally"
        identities_before = {"source": source_before, "launch_spec": args.launch_spec_sha256,
                             "event_transport": TRANSPORT_SHA256, **GPU_BINDINGS}
        identities_after = {}
        for name, path in (("source", Path(__file__)), ("launch_spec", args.launch_spec),
                           ("event_transport", TRANSPORT_SOURCE), *((p, p) for p in GPU_BINDINGS)):
            try:
                identities_after[name] = sha(path)
            except OSError as error:
                identities_after[name] = repr(error)
        identities_unchanged = identities_after == identities_before
        if not identities_unchanged:
            status, reason = "identity_failed", "Watchdog source or launch spec changed"
        summary = {"schema": "gpu-process-watchdog-v1", "status": status, "reason": reason,
                   "child_exit_code": child.returncode if child else None,
                   "supervisor_health": "pass" if status in ("pass", "child_failed") else "failed",
                   "child_pid": child.pid if child else None, "last_event_record_id": high_water,
                   "event_transport_sha256": TRANSPORT_SHA256, "event_worker_close": event_worker_close,
                   "gpu_transport": "persistent_nvml", "gpu_bindings": GPU_BINDINGS,
                   "gpu_worker_close": gpu_worker_close,
                   "latest_completed_step_seen": last_step, "wall_seconds": time.monotonic() - started,
                   "expected_final_step": final_step, "finalization_started": finalization_started is not None,
                   "finalization_timeout_seconds": args.finalization_timeout_seconds,
                   "finalization_elapsed_seconds": (None if finalization_started is None else
                       (exited_at if exited_at is not None else time.monotonic()) - finalization_started),
                   "source_sha256": source_before, "source_sha256_after": identities_after["source"],
                   "launch_spec_sha256": args.launch_spec_sha256,
                   "launch_spec_sha256_after": identities_after["launch_spec"], "identities_unchanged": identities_unchanged,
                   "post_exit_quiet_seconds": args.post_exit_quiet_seconds,
                   "post_exit_quiet_completed": status in ("pass", "child_failed"),
                   "no_driver_registry_power_changes": True, "host_stability_proven": False,
                   "post_alert_checkpoint_policy": "Retain artifacts; audit separately before treating any post-alert checkpoint as trustworthy"}
        record("finished", **summary)
        evidence.close()
        summary["artifacts"] = {name: {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
                                for name, path in (("watchdog_log", out / "watchdog.jsonl"), ("child_log", out / "child.log"))
                                if path.exists()}
        with (out / "result.json").open("x") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        lock.close()
        print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
