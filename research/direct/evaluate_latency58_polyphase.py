"""Unchanged full14 scoring of the untrained four-phase recurrent initializer."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute

_MODEL = _IDENTITY = None


def initialize(plan):
    global _MODEL, _IDENTITY
    import torch
    from research.direct.latency58_direct_sdr_checkpoint import load_model
    from research.direct.latency58_evaluate import model_state_sha256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from research.direct.latency58_polyphase import Latency58InterleavedModel
    parent, _ = load_model(plan["parent_checkpoint"])
    _MODEL = Latency58InterleavedModel.from_parent(parent, phases=4)
    _IDENTITY = {"label": plan["label"], "state_kind": "untrained_initialization", "training_updates": 0,
                 "provenance": _MODEL.provenance, "architecture": _MODEL.architecture_metadata,
                 "model_state_sha256": model_state_sha256(_MODEL)}


def score(index):
    from research.direct.latency58_evaluate import evaluate_latency58_music
    report = evaluate_latency58_music(_MODEL, identity=_IDENTITY, track_indices=[index])
    require(report["excerpt_count"] == 2 and report["graph_alignment_samples"] == 128, "Protocol changed")
    return index, report["results"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.plan is None:
        prepare_and_run()
        return
    require(sha(args.plan) == args.plan_sha256, "Direct-SDR scoring plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-polyphase-initializer-full14-plan-v1" and plan["track_indices"] == list(range(14))
            and plan["workers"] == 2 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Require frozen CPU full14 scoring")
    out = Path(plan["output_directory"])
    began, reports = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as progress, ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=initialize, initargs=(plan,),
    ) as pool:
        futures = {pool.submit(score, i): i for i in plan["track_indices"]}
        for future in as_completed(futures):
            index, report = future.result()
            require(index == futures[future] and index not in reports, "Wrong or repeated track")
            reports[index] = report
            row = {"event": "track", "index": index, "track": report["tracks"][0]["name"],
                   "full_sdr_db": report["aggregate"]["full_sdr_db"]}
            progress.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    from research import evaluate as legacy
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    tracks = [reports[i]["tracks"][0] for i in plan["track_indices"]]
    candidate = {"model": reports[0]["model"], "checkpoint": None, "tracks": tracks,
                 "aggregate": legacy._aggregate_tracks(tracks),
                 "stream_batches": [reports[i]["stream_batches"][0] for i in plan["track_indices"]],
                 "reconstruction_max_abs": max(r["reconstruction_max_abs"] for r in reports.values())}
    reference = read(plan["reference_result"])["results"][0]
    require([t["name"] for t in tracks] == [t["name"] for t in reference["tracks"]]
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Track panel or evaluation inputs changed")
    write(out / "result.json", {"schema": "latency58-polyphase-initializer-full14-result-v1", "status": "pass",
          "results": [candidate], "comparison": compare(reference, candidate),
          "all_track_stem_cells": music_cells(reference, candidate), "plan_sha256": args.plan_sha256,
          "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "track_count": 14, "excerpt_count": 28, "elapsed_seconds": time.monotonic() - began,
          "target_full_sdr_db": 5.0, "numeric_threshold_reached": candidate["aggregate"]["full_sdr_db"] >= 5.0,
          "saved_checkpoint_target_reached": False, "state_kind": "untrained_initialization", "new_training_updates": 0,
          "graph_delay_samples": 128, "host_queue_samples": 128, "new_checkpoint_host_qualified": False})
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
                      "numeric_threshold_reached": candidate["aggregate"]["full_sdr_db"] >= 5.0}), flush=True)


def prepare_and_run():
    from research.direct.latency58_sdr_checkpoint import require_space
    prototype = PHASE / "polyphase-functional-001/result.json"
    proof = read(prototype)
    require(proof["status"] == "pass" and not proof["quality_measured"] and proof["training_updates"] == 0
            and proof["host_total_delay_samples"] == 256 and proof["future_input_independence"],
            "Qualify the untrained cadence prototype before scoring")
    out = PHASE / "polyphase-init-full14-001"
    require(not out.exists(), "Preserve existing evaluations")
    reference = PHASE / "c204-residual-model-full14-001/result.json"
    qualification = read(PHASE / "c204-residual-model-001/qualification.json")
    paths = [Path(__file__).resolve(), ROOT / "research/direct/latency58_polyphase.py", prototype,
             ROOT / "research/direct/latency58_direct_sdr_checkpoint.py", reference,
             Path(proof["checkpoint_parent"]["path"])]
    bindings = {**qualification["source_bindings"], **proof["source_bindings"], **{str(p): sha(p) for p in paths}}
    require(all(sha(p) == h for p, h in bindings.items()), "Prototype scoring inputs changed")
    budget = read(PHASE / "direct-sdr-musdb-001/plan.json")
    require_space(budget, 350_000_000)
    plan = {"schema": "latency58-polyphase-initializer-full14-plan-v1", "label": "four-phase-untrained-initializer",
            "parent_checkpoint": proof["checkpoint_parent"], "reference_result": str(reference),
            "workers": 2, "track_indices": list(range(14)), "source_bindings": bindings, "output_directory": str(out)}
    out.mkdir()
    write(out / "plan.json", plan)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_polyphase", "--plan", str(out / "plan.json"),
             "--plan-sha256", sha(out / "plan.json")], out, "evaluation", 1800, bindings,
            {"plan_sha256": sha(out / "plan.json"), "new_training_updates": 0, "state_kind": "untrained_initialization"})


if __name__ == "__main__":
    main()
