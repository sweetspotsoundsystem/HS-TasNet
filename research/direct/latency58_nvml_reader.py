"""Strict read-only NVML sampling for one locally supervised WSL GPU.

The library is opened only when ReadOnlyNvml is constructed, never on import.
ABI definitions and version encoding follow NVIDIA's header:
https://raw.githubusercontent.com/NVIDIA/go-nvml/main/pkg/nvml/nvml.h
https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html
"""
from __future__ import annotations

import ctypes as c
import hashlib
from pathlib import Path
import time

LIBRARY = Path('/usr/lib/wsl/lib/libnvidia-ml.so.1')
LIBRARY_SHA256 = 'f1853527c3738e9632695dddbb6e3129440c886d70a5c270b42089d3a3ddf4cb'
MIB = 2**20
FIELDS = ('uuid', 'name', 'driver_version', 'memory.total', 'memory.used',
          'temperature.gpu', 'power.draw', 'power.limit', 'utilization.gpu')


class MemoryV2(c.Structure):
    _fields_ = [('version', c.c_uint), ('total', c.c_ulonglong), ('reserved', c.c_ulonglong),
                ('free', c.c_ulonglong), ('used', c.c_ulonglong)]


class Utilization(c.Structure):
    _fields_ = [('gpu', c.c_uint), ('memory', c.c_uint)]


SIGNATURES = {
    'nvmlInit_v2': [], 'nvmlShutdown': [],
    'nvmlDeviceGetCount_v2': [c.POINTER(c.c_uint)],
    'nvmlDeviceGetHandleByIndex_v2': [c.c_uint, c.POINTER(c.c_void_p)],
    'nvmlSystemGetDriverVersion': [c.POINTER(c.c_char), c.c_uint],
    'nvmlDeviceGetUUID': [c.c_void_p, c.POINTER(c.c_char), c.c_uint],
    'nvmlDeviceGetName': [c.c_void_p, c.POINTER(c.c_char), c.c_uint],
    'nvmlDeviceGetMemoryInfo_v2': [c.c_void_p, c.POINTER(MemoryV2)],
    'nvmlDeviceGetTemperature': [c.c_void_p, c.c_uint, c.POINTER(c.c_uint)],
    'nvmlDeviceGetPowerUsage': [c.c_void_p, c.POINTER(c.c_uint)],
    'nvmlDeviceGetPowerManagementLimit': [c.c_void_p, c.POINTER(c.c_uint)],
    'nvmlDeviceGetUtilizationRates': [c.c_void_p, c.POINTER(Utilization)],
}
OPTIONAL = frozenset(('nvmlDeviceGetPowerUsage', 'nvmlDeviceGetPowerManagementLimit',
                      'nvmlDeviceGetUtilizationRates'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_memory(memory):
    require(set(memory) == {'total', 'reserved', 'free', 'used'}
            and all(type(value) is int and 0 <= value < 2**64 - 1 for value in memory.values())
            and memory['total'] > 0
            and memory['total'] == memory['reserved'] + memory['free'] + memory['used'],
            'Malformed NVML physical-memory accounting')


class ReadOnlyNvml:
    """Only getter functions plus balanced library initialization and cleanup."""

    def __init__(self, *, library=LIBRARY, library_sha256=LIBRARY_SHA256, phase=None):
        require(c.sizeof(c.c_void_p) == 8 and c.sizeof(MemoryV2) == 40
                and MemoryV2.total.offset == 8 and MemoryV2.used.offset == 32
                and c.sizeof(Utilization) == 8, 'Unsupported NVML ABI')
        self.library = Path(library).resolve(strict=True)
        self.library_sha256 = library_sha256
        require(sha(self.library) == library_sha256, 'NVML library identity changed')
        self.phase = phase or (lambda *_args: None)
        self.initialized = False
        self.closed = False
        self.calls = []
        self.lib = c.CDLL(str(self.library))
        self.functions = {}
        for name, arguments in SIGNATURES.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, c.c_int
            self.functions[name] = function
        self._call('nvmlInit_v2')
        self.initialized = True

    def _call(self, name, *arguments):
        require(name in self.functions, 'Unreviewed NVML operation')
        self.phase('begin', name)
        began = time.monotonic()
        code = self.functions[name](*arguments)
        duration = time.monotonic() - began
        self.calls.append({'api': name, 'returncode': code, 'seconds': duration})
        self.phase('end', name, code, duration)
        if code == 3 and name in OPTIONAL:
            return False
        require(code == 0, f'{name} failed with NVML return code {code}')
        return True

    def _text(self, name, *arguments):
        buffer = c.create_string_buffer(128)
        self._call(name, *arguments, buffer, len(buffer))
        value = buffer.value.decode('ascii')
        require(value and '\n' not in value and '\r' not in value, 'Invalid NVML identity text')
        return value

    def sample(self):
        require(self.initialized and not self.closed, 'NVML reader is not open')
        require(sha(self.library) == self.library_sha256, 'NVML library changed while sampling')
        self.calls = []
        began_ns = time.monotonic_ns()
        count, device = c.c_uint(), c.c_void_p()
        self._call('nvmlDeviceGetCount_v2', c.byref(count))
        require(count.value == 1, 'Require exactly one NVML GPU')
        self._call('nvmlDeviceGetHandleByIndex_v2', 0, c.byref(device))
        require(bool(device.value), 'NVML returned an empty GPU handle')
        values = {'uuid': self._text('nvmlDeviceGetUUID', device),
                  'name': self._text('nvmlDeviceGetName', device),
                  'driver_version': self._text('nvmlSystemGetDriverVersion')}
        require(values['uuid'].startswith('GPU-'), 'NVML GPU UUID unavailable')
        memory = MemoryV2(version=c.sizeof(MemoryV2) | (2 << 24))
        self._call('nvmlDeviceGetMemoryInfo_v2', device, c.byref(memory))
        memory_bytes = {key: getattr(memory, key) for key in ('total', 'reserved', 'free', 'used')}
        validate_memory(memory_bytes)
        temperature, power, limit, utilization = c.c_uint(), c.c_uint(), c.c_uint(), Utilization()
        self._call('nvmlDeviceGetTemperature', device, 0, c.byref(temperature))
        # The unchanged legacy guard receives conservatively converted MiB.
        # The supervisor additionally checks raw free bytes, excluding reserved memory.
        values.update({'memory.total': memory.total // MIB,
                       'memory.used': (memory.used + MIB - 1) // MIB,
                       'temperature.gpu': temperature.value})
        power_ok = self._call('nvmlDeviceGetPowerUsage', device, c.byref(power))
        limit_ok = self._call('nvmlDeviceGetPowerManagementLimit', device, c.byref(limit))
        utilization_ok = self._call('nvmlDeviceGetUtilizationRates', device, c.byref(utilization))
        require(not utilization_ok or 0 <= utilization.gpu <= 100, 'Invalid NVML utilization')
        values.update({'power.draw': f'{power.value / 1000:.2f}' if power_ok else 'N/A',
                       'power.limit': f'{limit.value / 1000:.2f}' if limit_ok else 'N/A',
                       'utilization.gpu': str(utilization.gpu) if utilization_ok else 'N/A'})
        require(set(values) == set(FIELDS), 'Incomplete NVML sample')
        return {'values': values, 'memory_bytes': memory_bytes, 'api_timings': list(self.calls),
                'sample_started_ns': began_ns, 'sample_completed_ns': time.monotonic_ns()}

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.initialized:
            self._call('nvmlShutdown')
            self.initialized = False
