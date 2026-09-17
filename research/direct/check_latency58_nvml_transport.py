"""CPU-only protocol and cleanup controls for the persistent NVML transport.

Owned fixture workers emit synthetic telemetry. Neither these workers nor the
reader error controls load the NVIDIA library or run a GPU workload.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from research.direct.latency58_nvml_transport import PersistentNvmlQuery
from research.direct.latency58_nvml_reader import ReadOnlyNvml, validate_memory

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/direct/runs/latency58/nvml-transport-check-001'
FAKE = r'''
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--mode');p.add_argument('--session');p.add_argument('--bindings')
a=p.parse_args();bindings=json.loads(a.bindings);pid=os.getpid()
ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
common={'kind':'ready','session':a.session,'pid':pid,'started':ticks}
if a.mode=='exit_before_ready':raise SystemExit(7)
if a.mode=='startup_timeout':
 print('fixture_phase:startup',file=sys.stderr,flush=True);time.sleep(.5)
if a.mode=='wrong_session':common['session']='wrong'
if a.mode=='wrong_pid':common['pid']=pid+1
if a.mode=='wrong_start':common['started']='wrong'
print(json.dumps(common),flush=True)
for line in sys.stdin:
 r=json.loads(line);received=time.monotonic_ns()
 if a.mode=='exit_before_reply':raise SystemExit(7)
 if a.mode in ('query_timeout','query_hang'):
  print('fixture_phase:sample',file=sys.stderr,flush=True)
  time.sleep(.5 if a.mode=='query_timeout' else 10)
 unit=2**20
 memory={'total':16384*unit,'reserved':512*unit,'used':1024*unit,'free':14848*unit}
 values={'uuid':'GPU-fixture','name':'Fixture','driver_version':'fixture-driver',
         'memory.total':16384,'memory.used':1024,'temperature.gpu':40,
         'power.draw':'N/A','power.limit':'N/A','utilization.gpu':'N/A'}
 payload={'schema':'latency58-persistent-nvml-v1','pid':common['pid'],
          'start_ticks':common['started'],'request_id':r['id'],'bindings':dict(bindings),
          'request_received_ns':received,'sample_started_ns':received,
          'sample_completed_ns':time.monotonic_ns(),'memory_bytes':memory,
          'values':values,'api_timings':[]}
 if a.mode=='bad_schema':payload['schema']='wrong'
 if a.mode=='bad_binding':payload['bindings']['reader_sha256']='0'*64
 if a.mode=='stale_sample':
  for k in ('request_received_ns','sample_started_ns','sample_completed_ns'):payload[k]=0
 if a.mode=='bad_memory':memory['free']-=1
 if a.mode=='bad_conversion':values['memory.used']+=1
 response={'kind':'response','session':a.session,'pid':common['pid'],
           'id':r['id']-1 if a.mode=='stale_id' else r['id'],'stdout':json.dumps(payload)}
 encoded=json.dumps(response)+'\n'
 if a.mode=='duplicate_reply':encoded+=encoded
 sys.stdout.write(encoded);sys.stdout.flush()
 if a.mode=='exit_after_reply':raise SystemExit(7)
if a.mode=='close_nonzero':raise SystemExit(7)
if a.mode=='close_hang':time.sleep(10)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


class FixtureQuery(PersistentNvmlQuery):
    def __init__(self, mode, case):
        self.mode = mode
        self.fixture = OUT / 'fixture_worker.py'
        library = case / 'synthetic-library.txt'
        library.write_text('CPU fixture; never loaded as a library.\n')
        super().__init__(worker_source=ROOT / 'research/direct/latency58_nvml_worker.py',
                         worker_sha256=sha(ROOT / 'research/direct/latency58_nvml_worker.py'),
                         reader_source=ROOT / 'research/direct/latency58_nvml_reader.py',
                         reader_sha256=sha(ROOT / 'research/direct/latency58_nvml_reader.py'),
                         library=library, library_sha256=sha(library))

    def _worker_command(self):
        return [sys.executable, '-u', str(self.fixture), '--mode', self.mode,
                '--session', self.session, '--bindings', json.dumps(self.bindings)]


def run_case(mode):
    case = OUT / mode
    case.mkdir()
    query = FixtureQuery(mode, case)
    responses = []
    error = None
    began = time.monotonic()
    timeout = .2 if mode in ('startup_timeout', 'query_timeout', 'query_hang') else 2
    try:
        if mode == 'source_changed':
            responses.append(query.query(timeout))
            query.library.write_text('Intentional fixture identity change.\n')
        responses.append(query.query(timeout))
        if mode == 'success':
            responses.extend(query.query(timeout) for _ in range(2))
        if mode == 'exit_after_reply':
            query.process.wait(timeout=2)
            responses.append(query.query(timeout))
    except Exception as caught:
        error = caught
    query_seconds = time.monotonic() - began
    pid = query.process.pid if query.process else None
    rejected_retry = False
    if error is not None:
        old_id = query.request_id
        try:
            query.query(1)
        except Exception:
            rejected_retry = True
        assert rejected_retry and query.request_id == old_id
        assert query.process is None or query.process.pid == pid
    closed = query.close(timeout=.1 if mode in ('query_hang', 'close_hang') else 1)
    assert closed['closed'] and (pid is None or not Path('/proc', str(pid)).exists())
    success_modes = ('success', 'close_nonzero', 'close_hang')
    assert (error is None) == (mode in success_modes), (mode, repr(error))
    if mode == 'success':
        assert [r['request_id'] for r in responses] == [1, 2, 3]
        assert len({r['worker_pid'] for r in responses}) == 1
        assert closed['actual_exit_code'] == 0 and not closed['forced'] and closed['identities_unchanged']
        try:
            query.query(1)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Closed worker was restarted')
    if mode in ('startup_timeout', 'query_timeout', 'query_hang'):
        assert isinstance(error, subprocess.TimeoutExpired)
        assert b'fixture_phase:' in error.stderr and .15 <= query_seconds < 1.5
    if mode in ('query_hang', 'close_hang'):
        assert closed['forced'] and closed['actual_exit_code'] == -15
    if mode == 'close_nonzero':
        assert closed['actual_exit_code'] == 7 and not closed['forced']
    if mode == 'source_changed':
        assert not closed['identities_unchanged']
    record = {'case': mode, 'status': 'pass', 'query_error': repr(error) if error else None,
              'query_seconds': query_seconds, 'responses': len(responses), 'worker_pid': pid,
              'failed_worker_retry_rejected': rejected_retry, 'close': closed,
              'actual_owned_child_reaped': True, 'telemetry_simulated': True, 'gpu_queried': False}
    write(case / 'result.json', record)
    print(json.dumps({'case': mode, 'status': 'pass', 'actual_worker_exit': closed['actual_exit_code']}), flush=True)
    return record


def reader_controls():
    cases = []
    for name, code, allowed in (('nvmlDeviceGetPowerUsage', 3, True),
                                ('nvmlDeviceGetTemperature', 3, False),
                                ('nvmlDeviceGetPowerUsage', 15, False),
                                ('nvmlDeviceGetTemperature', 999, False)):
        reader = ReadOnlyNvml.__new__(ReadOnlyNvml)
        reader.functions = {name: lambda *args, result=code: result}
        reader.phase = lambda *_args: None
        reader.calls = []
        try:
            result = reader._call(name)
        except RuntimeError:
            assert not allowed
        else:
            assert allowed and result is False
        assert reader.calls[0]['returncode'] == code
        cases.append({'api': name, 'returncode': code, 'optional_unsupported_only': allowed})
    for memory in ({'total': 10, 'reserved': 2, 'used': 3, 'free': 6},
                   {'total': 0, 'reserved': 0, 'used': 0, 'free': 0},
                   {'total': 10, 'reserved': 2, 'used': -1, 'free': 9},
                   {'total': 2**64 - 1, 'reserved': 0, 'used': 0, 'free': 2**64 - 1}):
        try:
            validate_memory(memory)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Invalid physical memory accepted')
    return {'api_error_cases': cases, 'invalid_memory_cases_rejected': 4,
            'nvidia_library_loaded': False}


def main():
    assert Path.cwd() == ROOT and not OUT.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert all(os.environ.get(name) == '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    OUT.mkdir()
    (OUT / 'fixture_worker.py').write_text(FAKE)
    paths = [Path(__file__).resolve(), OUT / 'fixture_worker.py']
    paths.extend(ROOT / 'research/direct' / name for name in
                 ('latency58_nvml_reader.py', 'latency58_nvml_worker.py', 'latency58_nvml_transport.py',
                  'latency58_windows_event_transport.py'))
    bindings = {str(path): sha(path) for path in paths}
    write(OUT / 'plan.json', {'schema': 'latency58-nvml-transport-cpu-control-v1',
          'source_bindings': bindings, 'gpu_queried': False, 'telemetry_simulated': True})
    # A test accidentally reaching ctypes.CDLL would be a scope violation.
    original_cdll = ctypes.CDLL
    ctypes.CDLL = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('CPU fixture tried to load a native library'))
    try:
        modes = ('success', 'startup_timeout', 'query_timeout', 'query_hang', 'exit_before_ready',
                 'exit_before_reply', 'exit_after_reply', 'wrong_session', 'wrong_pid', 'wrong_start',
                 'stale_id', 'duplicate_reply', 'bad_schema', 'bad_binding', 'stale_sample',
                 'bad_memory', 'bad_conversion', 'source_changed', 'close_nonzero', 'close_hang')
        cases = [run_case(mode) for mode in modes]
        reader = reader_controls()
    finally:
        ctypes.CDLL = original_cdll
    assert all(sha(path) == digest for path, digest in bindings.items())
    result = {'status': 'pass', 'plan_sha256': sha(OUT / 'plan.json'), 'source_bindings': bindings,
              'source_bindings_unchanged': True, 'cases': cases, 'reader_controls': reader,
              'all_actual_workers_reaped': True, 'gpu_queried': False, 'telemetry_simulated': True}
    write(OUT / 'result.json', result)
    print(json.dumps({'status': 'pass', 'transport_cases': len(cases), 'reader_cases': 8}), flush=True)


if __name__ == '__main__':
    main()
