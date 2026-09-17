"""Persistent NVML sampling over the already qualified bounded pipe transport.

The reused IO implementation is byte-bound and unchanged. The NVML worker
accepts only the literal sample operation, never an executable program.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROTOCOL = 'latency58-persistent-nvml-v1'
BASE_SOURCE = Path(__file__).with_name('latency58_windows_event_transport.py')
BASE_SHA256 = '27c42b89faa7fc5eb9d84f195501945eb6aa1c5c3da520b094fb295ab77d4dab'
FIELDS = ('uuid', 'name', 'driver_version', 'memory.total', 'memory.used',
          'temperature.gpu', 'power.draw', 'power.limit', 'utilization.gpu')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def start_ticks(pid):
    return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]


require(sha(BASE_SOURCE) == BASE_SHA256, 'Qualified bounded IO source changed')
_spec = importlib.util.spec_from_file_location('nvml_bounded_io_base', BASE_SOURCE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class PersistentNvmlQuery(_base.PersistentWindowsQuery):
    """One owned worker per supervisor; failed workers are never restarted."""

    def __init__(self, *, worker_source, worker_sha256, reader_source, reader_sha256,
                 library, library_sha256):
        super().__init__('unused-by-nvml-worker')
        self.worker_source = Path(worker_source).resolve(strict=True)
        self.reader_source = Path(reader_source).resolve(strict=True)
        self.library = Path(library).resolve(strict=True)
        self.bindings = {'worker_sha256': worker_sha256, 'reader_sha256': reader_sha256,
                         'library_sha256': library_sha256}
        self.sources = {str(self.worker_source): worker_sha256, str(self.reader_source): reader_sha256,
                        str(self.library): library_sha256, str(BASE_SOURCE): BASE_SHA256}
        self.expected_start_ticks = None
        self._verify()

    def _verify(self):
        require(all(sha(path) == digest for path, digest in self.sources.items()),
                'NVML transport input identity changed')

    def _worker_command(self):
        return [sys.executable, '-u', str(self.worker_source), '--session', self.session,
                '--worker-sha256', self.bindings['worker_sha256'], '--reader', str(self.reader_source),
                '--reader-sha256', self.bindings['reader_sha256'], '--library', str(self.library),
                '--library-sha256', self.bindings['library_sha256']]

    def _start(self):
        self._verify()
        self.process = subprocess.Popen(self._worker_command(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        self.expected_start_ticks = start_ticks(self.process.pid)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            os.set_blocking(stream.fileno(), False)

    def query(self, timeout):
        require(type(timeout) in (int, float) and 0 < timeout <= 10,
                'Require the unchanged NVML query deadline of at most ten seconds')
        began_ns = time.monotonic_ns()
        try:
            self._verify()
            response = super().query('sample', timeout)
            payload = json.loads(response['stdout'])
            now_ns = time.monotonic_ns()
            require(payload.get('schema') == PROTOCOL
                    and payload.get('pid') == self.worker_pid == self.process.pid
                    and payload.get('start_ticks') == self.worker_started == self.expected_start_ticks
                    == start_ticks(self.process.pid)
                    and payload.get('request_id') == response['request_id'] == self.request_id
                    and payload.get('bindings') == self.bindings, 'NVML response identity differs')
            times = [payload.get(key) for key in
                     ('request_received_ns', 'sample_started_ns', 'sample_completed_ns')]
            require(all(type(value) is int for value in times)
                    and began_ns <= times[0] <= times[1] <= times[2] <= now_ns
                    and now_ns - began_ns < timeout * 1_000_000_000, 'NVML response is stale or late')
            values, memory = payload.get('values'), payload.get('memory_bytes')
            require(isinstance(values, dict) and set(values) == set(FIELDS), 'Incomplete NVML response')
            require(isinstance(memory, dict) and set(memory) == {'total', 'reserved', 'free', 'used'}
                    and all(type(value) is int and 0 <= value < 2**64 - 1 for value in memory.values())
                    and memory['total'] > 0
                    and memory['total'] == memory['reserved'] + memory['free'] + memory['used'],
                    'NVML memory accounting is incomplete')
            require(type(values['memory.total']) is int and type(values['memory.used']) is int
                    and values['memory.total'] == memory['total'] // 2**20
                    and values['memory.used'] == (memory['used'] + 2**20 - 1) // 2**20,
                    'NVML memory conversion differs from the reviewed contract')
            self._verify()
            require(time.monotonic_ns() - began_ns < timeout * 1_000_000_000,
                    'NVML response validation exceeded its query deadline')
            stream = io.StringIO()
            csv.writer(stream, lineterminator='\n').writerow([values[key] for key in FIELDS])
            return {**response, 'stdout': stream.getvalue(), 'nvml': payload,
                    'elapsed_seconds': (time.monotonic_ns() - began_ns) / 1_000_000_000}
        except subprocess.TimeoutExpired as error:
            self.broken = True
            raise subprocess.TimeoutExpired('persistent_nvml', timeout,
                                            output=error.output, stderr=error.stderr) from error
        except BaseException:
            self.broken = True
            raise

    def close(self, timeout=5):
        require(0 < timeout <= 5, 'Require bounded NVML worker cleanup')
        result = super().close(timeout=timeout)
        try:
            self._verify()
            unchanged = True
        except (OSError, RuntimeError):
            unchanged = False
        return {**result, 'identities_unchanged': unchanged, 'input_bindings': self.bindings,
                'transport': 'persistent_nvml'}
