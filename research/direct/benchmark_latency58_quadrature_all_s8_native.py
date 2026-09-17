"""Native Linux CPU1 comparison using sealed graph bytes held in memory.

The unchanged portable C++ probe receives read-only memfd paths. This avoids
saving an experimental ONNX graph before native performance selection.
This Linux experiment is not a Mac or DAW qualification.
"""
from contextlib import contextmanager, ExitStack
import ctypes
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def require_quiet():
    from research.direct.run_latency58_quadrature_quiet_native import require_no_training_or_scoring
    require_no_training_or_scoring()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if argv and Path(os.fsdecode(argv[0])).name.startswith("python"):
            require(not any(re.search(rb"\b(?:train|evaluate)_latency58\w*", value) for value in argv[1:]),
                    "A Python training or scoring process is still active")


@contextmanager
def sealed_graph(label, data):
    require(isinstance(data, bytes) and data, "Require immutable graph bytes")
    # This environment's Python was built against older headers and omits
    # os.memfd_create. Use the installed libc entry point and Linux UAPI
    # constants from linux/memfd.h, linux/fcntl.h and asm-generic/fcntl.h.
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes, create.restype = (ctypes.c_char_p, ctypes.c_uint), ctypes.c_int
    fd = create(label.encode("ascii"), 0x0001 | 0x0002)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        remaining = memoryview(data)
        while remaining:
            count = os.write(fd, remaining)
            require(count > 0, "Incomplete in-memory graph write")
            remaining = remaining[count:]
        seals = 0x0008 | 0x0004 | 0x0002 | 0x0001
        fcntl.fcntl(fd, 1024 + 9, seals)
        require(fcntl.fcntl(fd, 1024 + 10) == seals and os.fstat(fd).st_size == len(data)
                and hashlib.sha256(os.pread(fd, len(data), 0)).digest() == hashlib.sha256(data).digest(),
                "Graph sealing or readback failed")
        yield fd, f"/proc/self/fd/{fd}"
    finally:
        os.close(fd)


def main():
    import onnxruntime as ort
    import torch
    import onnx
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_quadrature_checkpoint import load_model
    from research.direct.latency58_quadrature_all_s8 import build
    from research.direct.train_latency58 import state_sha256
    require(Path.cwd() == ROOT and platform.system() == "Linux" and ort.__version__ == "1.26.0"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require Linux CPU1 and the reviewed shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "temporal-attention-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    require_space(source, 2_000_000)
    training_root = Path(source["output_directory"])
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(training_root / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
                "Wait for completed training and scoring")
    lock = (training_root / "production-run/trainer.lock").open("r")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require_quiet()
    quality_root = PHASE / "quadrature-all-s8-full14-memory-001"
    review_path = quality_root / "selection-review.json"
    review, quality = read(review_path), read(quality_root / "result.json")
    execution_path = PHASE / "quadrature-all-s8-full14-memory-stage-001/execution.json"
    execution = read(execution_path)
    require(review["status"] == "advance_to_native_comparison" and review["actual_root_exit_code"] == 0
            and quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and not quality["graph_saved"] and execution["actual_exit_code"] == 0
            and not execution["timed_out"] and execution["source_bindings_unchanged"],
            "Wait for completed and reviewed CPU quality scoring")
    verify_inputs(review)
    verify_inputs(read(quality_root / "plan.json"))
    screen_path = PHASE / "quadrature-all-s8-screen-001/result.json"
    screen = read(screen_path)
    parent = screen["checkpoint"]
    native, _ = load_model(parent)
    fingerprint = state_sha256(native.state_dict())
    unsigned_path = PHASE / "m4-quadrature-magint8-saved-001/model.onnx"
    original = onnx.load(unsigned_path)
    original_proof_path = PHASE / "m4-quadrature-magint8-screen-001/graph.json"
    graph, proof = build(native, original, read(original_proof_path)["graph_conversion"])
    unsigned, signed = unsigned_path.read_bytes(), graph.SerializeToString()
    digests = {"quadrature_unsigned": hashlib.sha256(unsigned).hexdigest(),
               "quadrature_signed_reduced": hashlib.sha256(signed).hexdigest()}
    require(digests["quadrature_signed_reduced"] == screen["graph_sha256"]
            == quality["results"][0]["checkpoint"]["sha256"]
            and digests["quadrature_unsigned"] == proof["baseline_graph_sha256"]
            and state_sha256(native.state_dict()) == fingerprint and not torch.cuda.is_initialized(),
            "Timing graph or saved model differs from the independent checks")
    del native, original, graph
    c204 = PHASE / "m4-int8-precise-core-full14-001/model.onnx"
    require(sha(c204) == "a550c904ef501fe98f3afa010eca5a53abd5bf66d63906e018511a01c5076d61", "C204 graph changed")
    sdk = Path("/home/axel/autoresearch/codex/HS-TasNet-latency11-v1-state/workspaces/c118-native-timing/stemgen-rt/libs/onnxruntime")
    cpp = ROOT / "research/direct/benchmark_latency58_native.cpp"
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    require(compiler is not None, "Missing C++ compiler")
    libraries = sorted({p.resolve() for p in (sdk / "lib").glob("libonnxruntime*.so*") if p.is_file()})
    headers = sorted((sdk / "include").rglob("*.h"))
    require(libraries and headers, "Native SDK is incomplete")
    paths = [source_path, review_path, execution_path, quality_root / "result.json", quality_root / "plan.json",
             screen_path, cpp, c204, unsigned_path, original_proof_path, Path(__file__).resolve(), Path(compiler), *libraries, *headers]
    paths.extend(Path(path) for path in ("/usr/include/linux/memfd.h", "/usr/include/linux/fcntl.h",
                                        "/usr/include/asm-generic/fcntl.h"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_quadrature_all_s8.py", "benchmark_latency58_quadrature_memory_native.py",
        "run_latency58_quadrature_quiet_native.py", "latency58_int8_precise_ten.py",
        "latency58_quadrature_int8.py", "latency58_quadrature_onnx.py"))
    bindings = {**read(quality_root / "plan.json")["source_bindings"], **{str(p): sha(p) for p in paths}}
    out = PHASE / "quadrature-all-s8-quiet-native-001"
    require(not out.exists(), "Preserve earlier timings")
    out.mkdir()
    write(out / "plan.json", {"schema": "latency58-native-sealed-memory-three-model-v1", "source_bindings": bindings,
          "runtime_graphs": {"c204_integer": {"path": str(c204), "sha256": sha(c204)},
              **{label: {"sha256": digests[label], "bytes": len(data), "saved": False, "sealed_memfd": True}
                 for label, data in (("quadrature_unsigned", unsigned), ("quadrature_signed_reduced", signed))}},
          "system": platform.platform(), "machine": platform.machine(), "sdk": str(sdk),
          "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
          "warmup_hops": 256, "measured_hops_per_cycle": 2048, "cycles": 6,
          "ordering": "All six permutations; every model occupies every position twice",
          "scope": "Linux native preallocated CPU1 inference; all ten weight matrices signed; no concurrent training or quality scoring",
          "our_training_and_scoring_active_at_start": False,
          "speed_gate": {"maximum_signed_over_unsigned_median_ratio": .97, "minimum_faster_cycles": 4},
          "native_host_qualified": False, "mac_execution_performed": False, "onnx_files_written": False})
    shutil.copyfile(cpp, out / cpp.name)
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    began, rows = time.monotonic(), []
    with ExitStack() as resources, tempfile.TemporaryDirectory(prefix="latency58-native-") as temporary:
        fd9, path9 = resources.enter_context(sealed_graph("quadrature-unsigned", unsigned))
        fd10, path10 = resources.enter_context(sealed_graph("quadrature-signed", signed))
        models = {"c204_integer": str(c204), "quadrature_unsigned": path9, "quadrature_signed_reduced": path10}
        binary = Path(temporary) / "benchmark"
        subprocess.run([compiler, "-std=c++20", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror",
                        str(cpp), "-I" + str(sdk / "include"), "-L" + str(sdk / "lib"),
                        "-Wl,-rpath," + str(sdk / "lib"), "-lonnxruntime", "-o", str(binary)], check=True, env=environment)
        for cycle, order in enumerate(itertools.permutations(models), start=1):
            for label in order:
                require_quiet()
                completed = subprocess.run([str(binary), models[label], "256", "2048"], pass_fds=(fd9, fd10),
                                           check=True, text=True, capture_output=True, env=environment)
                row = {"cycle": cycle, "variant": label, **json.loads(completed.stdout), "runtime_stderr": completed.stderr}
                require(row["status"] == "pass" and row["onnxruntime_version"] == "1.26.0", "Native execution failed")
                write(out / f"cycle-{cycle}-{label}.json", row)
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("cycle", "variant", "run")}), flush=True)
    verify_inputs({"source_bindings": bindings})
    require_quiet()
    medians = {label: statistics.median(row["run"]["p50_ms"] for row in rows if row["variant"] == label) for label in models}
    ratio = medians["quadrature_signed_reduced"] / medians["quadrature_unsigned"]
    faster = sum(next(row["run"]["p50_ms"] for row in rows if row["cycle"] == cycle and row["variant"] == "quadrature_signed_reduced")
                 < next(row["run"]["p50_ms"] for row in rows if row["cycle"] == cycle and row["variant"] == "quadrature_unsigned")
                 for cycle in range(1, 7))
    write(out / "result.json", {"status": "pass", "cycles": rows, "plan_sha256": sha(out / "plan.json"),
          "median_of_cycle_p50_ms": medians, "signed_over_unsigned_median_ratio": ratio, "faster_cycles": faster,
          "native_speed_gate_passed": ratio <= .97 and faster >= 4,
          "source_bindings_unchanged": True, "elapsed_seconds": time.monotonic() - began,
          "native_host_qualified": False, "mac_execution_performed": False, "onnx_files_written": False,
          "our_training_and_scoring_active_at_end": False, "other_user_programs_idle_claimed": False,
          "counted_bytes_after": require_space(source, 0)})
    print(json.dumps({"status": "pass", "median_of_cycle_p50_ms": medians, "signed_over_unsigned_median_ratio": ratio,
                      "faster_cycles": faster}), flush=True)


if __name__ == "__main__":
    main()
