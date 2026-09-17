"""Owned JSON worker for read-only NVML queries; no arbitrary programs or setters."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

PROTOCOL = 'latency58-persistent-nvml-v1'
MAX_REQUEST_BYTES = 4096


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def start_ticks(pid):
    return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--reader', required=True, type=Path)
    parser.add_argument('--reader-sha256', required=True)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--library-sha256', required=True)
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{48}', args.session) is not None, 'Invalid NVML session')
    sources = {str(Path(__file__).resolve()): args.worker_sha256,
               str(args.reader.resolve(strict=True)): args.reader_sha256,
               str(args.library.resolve(strict=True)): args.library_sha256}
    bindings = {'worker_sha256': args.worker_sha256, 'reader_sha256': args.reader_sha256,
                'library_sha256': args.library_sha256}

    def verify():
        require(all(sha(path) == digest for path, digest in sources.items()),
                'NVML worker, reader or library changed')

    verify()
    spec = importlib.util.spec_from_file_location('owned_nvml_reader', args.reader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pid, ticks = os.getpid(), start_ticks(os.getpid())
    common = {'schema': PROTOCOL, 'session': args.session, 'pid': pid, 'started': ticks,
              'bindings': bindings}

    def emit(kind, **fields):
        print(json.dumps({**common, 'kind': kind, **fields}, allow_nan=False, separators=(',', ':')),
              flush=True)

    current_id = 0

    def phase(stage, api, returncode=None, duration=None):
        print(json.dumps({'kind': 'nvml_api_phase', 'pid': pid, 'request_id': current_id,
                          'stage': stage, 'api': api, 'returncode': returncode,
                          'seconds': duration, 'monotonic_ns': time.monotonic_ns()},
                         allow_nan=False, separators=(',', ':')), file=sys.stderr, flush=True)

    emit('ready')
    reader = None
    last_id = 0
    try:
        while True:
            line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            if not line:
                break
            received_ns = time.monotonic_ns()
            require(len(line) <= MAX_REQUEST_BYTES and line.endswith(b'\n'), 'Unbounded NVML request')
            request = json.loads(line)
            require(set(request) == {'session', 'id', 'program'}
                    and request['session'] == args.session
                    and type(request['id']) is int and request['id'] == last_id + 1
                    and isinstance(request['program'], str)
                    and base64.b64decode(request['program'], validate=True) == b'sample',
                    'Malformed or noncontiguous NVML request')
            verify()
            current_id = request['id']
            if reader is None:
                reader = module.ReadOnlyNvml(library=args.library, library_sha256=args.library_sha256,
                                             phase=phase)
            sample = reader.sample()
            verify()
            last_id = current_id
            payload = {'schema': PROTOCOL, 'pid': pid, 'start_ticks': ticks,
                       'request_id': last_id, 'bindings': bindings,
                       'request_received_ns': received_ns, **sample}
            emit('response', id=last_id, stdout=json.dumps(payload, allow_nan=False, separators=(',', ':')))
    finally:
        if reader is not None:
            reader.close()
    verify()
    emit('closed', last_id=last_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
