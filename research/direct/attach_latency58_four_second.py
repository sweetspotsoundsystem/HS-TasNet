"""Recover supervision of an explicitly identified, paused trainer.

Reuses the frozen health transports. Never claims an exit code for a process
that is not our child; endpoint artifacts still require independent audit.
On monitoring failure, pause the exact training group to preserve its state.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import time

from research.direct import watch_latency58_four_second as health


def open_pidfd(pid):
    # This Conda Python omits os.pidfd_open; glibc exposes the same Linux API.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def identity(pid, ticks, argv):
    path = Path('/proc') / str(pid)
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    health.require(fields[19] == str(ticks), 'Trainer PID was reused')
    health.require(int(fields[2]) == pid and int(fields[3]) == pid,
                   'Trainer no longer leads its original group/session')
    actual = (path / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    health.require(actual == [part.encode() for part in argv], 'Trainer command changed')
    return fields[0]


def signal_group(pid, ticks, argv, signum):
    identity(pid, ticks, argv)
    os.killpg(pid, signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attachment', type=Path, required=True)
    parser.add_argument('--attachment-sha256', required=True)
    args = parser.parse_args()
    health.require(health.sha(args.attachment) == args.attachment_sha256, 'Attachment changed')
    cfg = json.loads(args.attachment.read_text())
    bindings = cfg['bindings']
    health.require(all(health.sha(p) == digest for p, digest in bindings.items()), 'Inputs changed')
    spec = json.loads(Path(cfg['launch_spec']).read_text())
    pid, ticks, argv = cfg['trainer_pid'], cfg['trainer_start_ticks'], spec['argv']
    health.require(identity(pid, ticks, argv) in ('T', 't'), 'Trainer must be paused before attaching')
    pidfd = open_pidfd(pid)
    identity(pid, ticks, argv)
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    lock = (health.HERE / 'gpu-watchdog.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = Path(cfg['output_dir'])
    out.mkdir(exist_ok=False)
    evidence = (out / 'watchdog.jsonl').open('x', buffering=1)
    source_sha = health.sha(__file__)
    event_worker = gpu_worker = None
    high_water = cfg['previous_event_high_water']
    progress_path = Path(spec['progress_path'])
    last_step = health.progress_step(progress_path)
    health.require(last_step == cfg['paused_step'], 'Paused training journal changed')
    final_step = spec['expected_final_step']
    last_progress = time.monotonic()
    finalization_started = exited_at = None
    interrupted = False
    resumed = False
    status, reason = 'preflight', None

    def record(event, **fields):
        evidence.write(json.dumps({'utc': health.utc(), 'monotonic': time.monotonic(),
                                   'event': event, **fields}, allow_nan=False) + '\n')
        evidence.flush()
        os.fsync(evidence.fileno())

    def check_bindings():
        health.require(health.sha(__file__) == source_sha
                       and health.sha(args.attachment) == args.attachment_sha256
                       and all(health.sha(p) == digest for p, digest in bindings.items()),
                       'Attachment or bound source changed')

    def sample(*, sentinel=False):
        nonlocal high_water
        began = time.monotonic()
        response = event_worker.query(health.event_query(high_water, verify_sentinel=sentinel), 10)
        payload = json.loads(response['stdout'])
        newest, rows = health.validate_events(payload, high_water)
        health.require(not sentinel or payload.get('SentinelVerified') is True,
                       'Known event sentinel not verified')
        record('event_query', response=response, elapsed_seconds=time.monotonic() - began)
        record('event_coverage', previous=high_water, newest=newest,
               oldest=payload['RecordIds'][-1], scanned=payload['Count'], new_records=len(rows))
        high_water = newest
        faults = [row for row in rows if health.reset_event(row)]
        health.require(not faults, f'Fresh host faults: {faults!r}')
        response = gpu_worker.query(10)
        gpu = health.parse_gpu(response['stdout'])
        # The bound NVML transport validates all raw memory accounting.
        health.require(response['nvml']['memory_bytes']['free'] >= 4096 * 2**20,
                       'GPU physical free memory is below 4096 MiB')
        alert = health.gpu_alert(gpu, cfg['original_gpu_baseline'],
                                 max_temperature=80, memory_headroom=4096)
        record('gpu', values=gpu, response=response)
        health.require(alert is None, alert)

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupt)
    try:
        record('configuration', attachment=cfg, attachment_sha256=args.attachment_sha256,
               source_sha256=source_sha, monitor_pid=os.getpid(),
               monitor_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
               training_restarted=False, trainer_exit_code_available=False,
               monitoring_gap_acknowledged=True, quality_claim=False)
        event_worker = health.load_event_transport()
        gpu_worker = health.load_gpu_transport()
        sample(sentinel=True)
        check_bindings()
        health.require(not interrupted and not poller.poll(0), 'Trainer exited or stop requested')
        health.require(identity(pid, ticks, argv) in ('T', 't'), 'Trainer resumed unexpectedly')
        signal_group(pid, ticks, argv, signal.SIGCONT)
        resumed = True
        last_progress = time.monotonic()
        record('trainer_resumed', pid=pid, start_ticks=ticks, completed_step=last_step,
               original_launch_monotonic=cfg['original_launch_monotonic'],
               no_checkpoint_reload=True, model_optimizer_rng_preserved_in_memory=True)
        while True:
            check_bindings()
            sample()
            now = time.monotonic()
            step = health.progress_step(progress_path)
            health.require(step is not None and last_step <= step <= final_step,
                           'Training journal is missing, regressed, or exceeded the plan')
            if step > last_step:
                last_step, last_progress = step, now
                record('progress', completed_step=step)
            if step == final_step and finalization_started is None:
                finalization_started = now
                record('finalization_started', completed_step=step, timeout_seconds=180)
            health.require(not interrupted, 'External stop requested')
            if poller.poll(0):
                if exited_at is None:
                    exited_at = now
                    record('trainer_exit_observed', actual_exit_code=None,
                           reason='pidfd readable; trainer is not a child of this monitor')
                    health.require(finalization_started is None or now - finalization_started < 180,
                                   'Trainer exit exceeded finalization bound')
                if now - exited_at >= 10:
                    health.require(last_step == final_step, 'Trainer exited before planned final update')
                    status = 'supervision_complete_trainer_exit_code_unknown'
                    break
            else:
                identity(pid, ticks, argv)
                health.require(now - cfg['original_launch_monotonic'] < 203000,
                               'Original production runtime limit reached')
                health.require(now - (finalization_started or last_progress)
                               < (180 if finalization_started is not None else 240),
                               'Trainer finalization or progress timed out')
            time.sleep(2)
    except BaseException as error:
        status, reason = 'paused_by_attachment_monitor', repr(error)
        # Preserve training memory on a telemetry failure. Never auto-restart.
        try:
            if not poller.poll(0):
                signal_group(pid, ticks, argv, signal.SIGSTOP)
                record('trainer_paused', pid=pid, start_ticks=ticks)
        except BaseException as stop_error:
            record('pause_failed', reason=repr(stop_error))
        record('stop_reason', reason=reason, latest_completed_step_seen=last_step)
    finally:
        cleanup = {}
        for name, worker in (('events', event_worker), ('gpu', gpu_worker)):
            try:
                cleanup[name] = worker.close() if worker is not None else {'not_started': True}
            except BaseException as error:
                cleanup[name] = {'closed': False, 'reason': repr(error)}
            record('worker_closed', kind=name, result=cleanup[name])
        if status == 'supervision_complete_trainer_exit_code_unknown':
            if any(not r.get('closed') or r.get('forced') or r.get('actual_exit_code') != 0
                   for r in cleanup.values()) or not cleanup['gpu'].get('identities_unchanged'):
                status, reason = 'worker_cleanup_failed', 'Health workers did not close normally'
        summary = {'status': status, 'reason': reason, 'trainer_pid': pid,
                   'trainer_start_ticks': ticks, 'trainer_actual_exit_code': None,
                   'trainer_exit_observed': exited_at is not None, 'resumed': resumed,
                   'training_restarted': False, 'latest_completed_step_seen': last_step,
                   'last_event_record_id': high_water, 'cleanup': cleanup,
                   'monitoring_gap_acknowledged': True, 'quality_claim': False,
                   'endpoint_requires_independent_checkpoint_audit': True}
        record('finished', **summary)
        evidence.close()
        with (out / 'result.json').open('x') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.close(pidfd)
        lock.close()
        print(json.dumps(summary), flush=True)
    return 0 if status == 'supervision_complete_trainer_exit_code_unknown' else 1


if __name__ == '__main__':
    raise SystemExit(main())
