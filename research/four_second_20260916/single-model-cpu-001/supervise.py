"""Bound CPU qualification time, process-tree memory and scalar artifacts."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs

out = Path(__file__).resolve().parent
plan_path, command_path = out / 'plan.json', out / 'command.json'
plan, command = read(plan_path), read(command_path)
require(command['plan_sha256'] == sha(plan_path), 'Qualification plan changed')
require(not (out / 'qualification-execution.json').exists() and not (out / 'console.log').exists(),
        'Preserve earlier qualification execution')
verify_inputs(plan)
require(psutil.virtual_memory().available >= plan['minimum_launch_available_bytes'],
        'Insufficient available host memory for the bounded CPU check')
root_command = {'argv': ['/home/axel/miniforge3/bin/python', '-u', str(Path(__file__).resolve())],
                'cwd': str(ROOT), 'environment': command['environment'],
                'plan_sha256': sha(plan_path), 'command_sha256': sha(command_path),
                'supervisor_sha256': sha(Path(__file__)),
                'limits': {key: plan[key] for key in ('timeout_seconds', 'maximum_process_tree_rss_bytes',
                    'minimum_host_available_bytes', 'minimum_launch_available_bytes', 'maximum_new_artifact_bytes')}}
require(not (out / 'root-command.json').exists(), 'Preserve enclosing command')
write(out / 'root-command.json', root_command)
began = time.monotonic()
peak = 0
minimum_available = psutil.virtual_memory().available
reason = None
child = None
samples = 0
try:
    with (out / 'console.log').open('xb') as stream:
        child = subprocess.Popen(command['argv'], cwd=command['cwd'], env={**os.environ, **command['environment']},
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'event': 'cpu_qualification_child_started', 'pid': child.pid,
                          'memory_limit_bytes': plan['maximum_process_tree_rss_bytes']}), flush=True)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                processes = [process, *process.children(recursive=True)]
                resident = sum(p.memory_info().rss for p in processes if p.is_running())
            except psutil.NoSuchProcess:
                resident = 0
            available = psutil.virtual_memory().available
            peak = max(peak, resident); minimum_available = min(minimum_available, available); samples += 1
            total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            if time.monotonic() - began > plan['timeout_seconds']:
                reason = 'qualification timeout'
            elif resident > plan['maximum_process_tree_rss_bytes']:
                reason = 'CPU process-tree memory limit'
            elif available < plan['minimum_host_available_bytes']:
                reason = 'host available-memory floor'
            elif total > plan['maximum_new_artifact_bytes'] - 100_000:
                reason = 'scalar artifact limit'
            if reason:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
                break
            time.sleep(1)
        code = child.wait()
except BaseException:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
    raise
verify_inputs(plan)
execution = {'schema': 'latency58-bounded-cpu-child-execution-v1', 'argv': command['argv'],
    'cwd': command['cwd'], 'plan_sha256': sha(plan_path), 'command_sha256': sha(command_path),
    'actual_exit_code': code, 'timed_out': reason == 'qualification timeout', 'termination_reason': reason,
    'elapsed_seconds': time.monotonic() - began, 'peak_process_tree_rss_bytes': peak,
    'minimum_available_host_bytes': minimum_available, 'memory_samples': samples,
    'source_bindings_unchanged': True, 'finished_utc': datetime.now(timezone.utc).isoformat(),
    'gpu_used': False}
write(out / 'qualification-execution.json', execution)
print(json.dumps(execution), flush=True)
require(code == 0 and reason is None, 'CPU qualification failed or exceeded its resource bound')
result = read(out / 'result.json')
require(result['status'] == 'pass' and result['plan_sha256'] == sha(plan_path), 'Missing complete CPU qualification')
