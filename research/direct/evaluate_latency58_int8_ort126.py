"""Repeat the integer full14 score with the shipping ORT 1.26.0 runtime."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct import evaluate_latency58_int8 as common
from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs


def initialize(plan):
    import onnxruntime as ort
    require(ort.__version__ == "1.26.0" and str(Path(ort.__file__).resolve()) == plan["runtime"]["python_module"],
            "Worker loaded the wrong runtime")
    common.initialize(plan)


def main():
    import onnxruntime as ort
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and
            all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU1 workers with CUDA hidden")
    require(ort.__version__ == "1.26.0", "Supply the preserved shipping ORT Python package via PYTHONPATH")
    source = read(PHASE / "full-magnitude-001/plan.json")
    require_space(source, 375_000_000)
    parent_path = PHASE / "m4-int8-full14-001/result.json"
    parent = read(parent_path)
    require(parent["status"] == "pass" and parent["track_count"] == 14, "Initial full14 score is incomplete")
    old_plan = read(parent_path.parent / "plan.json")
    reference_path = Path(old_plan["reference_result"])
    reference = read(reference_path)["results"][0]
    module = Path(ort.__file__).resolve()
    runtime_paths = [module, *sorted((module.parent / "capi").glob("*.so*"))]
    paths = [parent_path, parent_path.parent / "plan.json", reference_path, Path(__file__).resolve(), *runtime_paths]
    bindings = {**old_plan["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    plan = {**old_plan, "source_bindings": bindings, "parent_result": str(parent_path),
            "runtime": {"version": ort.__version__, "python_module": str(module)}, "workers": 2}
    out = PHASE / "m4-int8-ort126-full14-001"
    require(not out.exists(), "Preserve validation evidence")
    out.mkdir()
    write(out / "plan.json", plan)
    began, scored = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(plan,)) as pool:
        futures = {pool.submit(common.score, index): index for index in range(14)}
        for future in as_completed(futures):
            index, track, stream, closure = future.result()
            require(index == futures[future] and index not in scored, "Unexpected validation index")
            scored[index] = track, stream, closure
            row = {"index": index, "track": track["name"], "full_sdr_db": track["full_sdr_db"],
                   "elapsed_seconds": time.monotonic() - began}
            progress.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    tracks = [scored[index][0] for index in range(14)]
    require([row["name"] for row in tracks] == [row["name"] for row in reference["tracks"]], "Track order differs")
    candidate = {"model": {**parent["results"][0]["model"], "runtime_version": "1.26.0"},
                 "checkpoint": old_plan["checkpoint"], "tracks": tracks, "aggregate": legacy._aggregate_tracks(tracks),
                 "stream_batches": [scored[index][1] for index in range(14)],
                 "reconstruction_max_abs": max(value[2] for value in scored.values())}
    verify_inputs(plan)
    result = {"status": "pass", "results": [candidate], "comparison": compare(reference, candidate),
              "runtime_version_comparison": compare(parent["results"][0], candidate),
              "all_track_stem_cells": music_cells(reference, candidate), "track_count": 14, "excerpt_count": 28,
              "source_bindings": bindings, "source_bindings_unchanged": True, "plan_sha256": sha(out / "plan.json"),
              "runtime": plan["runtime"], "elapsed_seconds": time.monotonic() - began, "graph_delay_samples": 128,
              "host_queue_samples": 128, "native_host_qualified": False, "plugin_modified": False, "gpu_used": False,
              "target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.0,
              "counted_bytes_after": require_space(source, 370_000_000)}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
                      "delta_from_ort127": candidate["aggregate"]["full_sdr_db"] -
                        parent["results"][0]["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
