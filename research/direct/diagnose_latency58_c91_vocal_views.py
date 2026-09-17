"""Inspect frozen C91 labels on the two already-scored Skelpolu passages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.latency58_vocal_focus_checkpoint import require_space


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Different diagnostic plan or cwd")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-c91-skelpolu-vocal-views-plan-v1" and plan["track_index"] == 10
            and plan["reference_intervals"] == [[1323000, 1984500], [3307500, 3969000]]
            and plan["maximum_new_artifact_bytes"] == 2_000_000
            and not plan["confirmation_material_used"] and not plan["audio_export"]
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the fixed CPU1, metrics-only development diagnostic")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve diagnostic evidence")
    before = require_space(plan, 2_000_000)
    for key in ("training_plan", "teacher", "teacher_qualification", "teacher_qualification_execution"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
                "Unbound teacher prerequisite")
    training = read(plan["training_plan"]["path"])
    require(training["teacher_kind"] == "c91" and training["teacher"] == plan["teacher"]
            and training["teacher_model_state_sha256"] == plan["teacher_model_state_sha256"], "Different pilot teacher")
    verify_inputs(training)
    proof, proof_execution = (read(plan[k]["path"]) for k in ("teacher_qualification", "teacher_qualification_execution"))
    require(proof["status"] == "pass" and proof["source_bindings_unchanged"]
            and proof_execution["actual_exit_code"] == 0 and not proof_execution["timed_out"]
            and proof_execution["source_bindings_unchanged"]
            and proof["plan_sha256"] == proof_execution["plan_sha256"]
            and proof["teachers"]["c91"]["context_oracle_max_abs"] <= 1e-5
            and proof["teachers"]["c91"]["identity"]["model_state_sha256"] == plan["teacher_model_state_sha256"],
            "Teacher physical-target oracle was not qualified")
    import numpy as np
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_sdr_teacher import load_teacher, physical_targets
    from research.direct.latency58_vocal_views import VIEWS, combine_sources, score_views
    from research.direct.report_latency58_vocal_focus import load_views
    from research.metrics import MetricConfig, SOURCE_ORDER
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260920)
    torch.use_deterministic_algorithms(True)
    began, rng = time.monotonic(), torch.get_rng_state().clone()
    teacher, identity = load_teacher("c91", plan["teacher"])
    require(identity["model_state_sha256"] == plan["teacher_model_state_sha256"]
            and torch.equal(rng, torch.get_rng_state()), "Teacher state or loader RNG differs")
    manifest, config = (read(plan[k]["path"]) for k in ("manifest", "config"))
    tracks, config = shared.select_panel(manifest, config, panel="full", track_indices=[10],
                                        excerpt_starts=None, duration=15.0, alignment_samples=512)
    require(len(tracks) == 1 and tracks[0]["name"] == "Skelpolu - Human Mistakes", "Different diagnostic track")
    track = tracks[0]
    intervals = legacy._reference_intervals(track, config)
    require([[r["reference_start"], r["reference_end"]] for r in intervals] == plan["reference_intervals"],
            "Physical passages changed")
    # Read genuine future input through the last C91 callback. The helper's
    # final zero flush is beyond both captured intervals and cannot affect them.
    read_end = ((max(r["reference_end"] for r in intervals) + 512 + 511) // 512) * 512
    paths = [legacy._safe_dataset_path(Path(manifest["root"]), track["stems"][stem]) for stem in SOURCE_ORDER]
    require(all(plan["source_bindings"].get(str(p)) == sha(p) for p in paths), "Unbound original source audio")
    source = np.stack([legacy._read_excerpt(p, 0, read_end, expected_frames=int(track["frames"])) for p in paths])
    source = np.ascontiguousarray(source, dtype=np.float32)
    refs = [np.ascontiguousarray(source[..., r["reference_start"]:r["reference_end"]]) for r in intervals]
    for values, row in zip(refs, intervals, strict=True):
        independent = np.stack([legacy._read_excerpt(p, row["reference_start"], row["reference_end"],
                                expected_frames=int(track["frames"])) for p in paths]).astype(np.float32)
        require(np.array_equal(values, independent), "Prefix source differs from independent physical excerpt")
    evidence, references = {}, {}
    for label, directory in plan["comparison_directories"].items():
        report = load_views(Path(directory), evidence)
        candidate = report["tracks"][10]
        require(candidate["index"] == 10 and candidate["name"] == track["name"]
                and [[r["reference_start"], r["reference_end"]] for r in candidate["intervals"]]
                    == plan["reference_intervals"], "Comparison passage differs")
        references[label] = candidate
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound comparison evidence")
    outputs, mixtures, streams = {}, {}, {}
    for view in VIEWS:
        print({"event": "teacher_view_started", "view": view, "real_prefix_samples": read_end}, flush=True)
        mixture = combine_sources(source, view)
        common_hashes = {}
        for label, row in references.items():
            common_end = row["stream"]["real_input_samples"]
            require(common_end <= read_end, "Teacher prefix does not cover the comparison input")
            digest = hashlib.sha256(np.ascontiguousarray(mixture[:, :common_end].T).tobytes()).hexdigest()
            require(digest == row["stream"]["input_stream_sha256"][view], "Common source view bytes differ")
            common_hashes[label] = {"samples": common_end, "sha256": digest}
        target = physical_targets(teacher, torch.from_numpy(mixture[None]), kind="c91")[0].numpy()
        outputs[view] = [np.ascontiguousarray(target[..., r["reference_start"]:r["reference_end"]]) for r in intervals]
        mixtures[view] = [combine_sources(value, view) for value in refs]
        error = max(float(np.max(np.abs(estimate.sum(axis=0, dtype=np.float32) - mixed)))
                    for estimate, mixed in zip(outputs[view], mixtures[view], strict=True))
        require(error <= 1e-6, "Native teacher residual does not close to its physical input")
        streams[view] = {"real_input_interval": [0, read_end], "graph_alignment_samples": 512,
                         "teacher_callbacks_including_final_flush": read_end // 512 + 1,
                         "physical_reference_intervals": plan["reference_intervals"],
                         "received_capture_intervals": [[r["estimate_start"], r["estimate_end"]] for r in intervals],
                         "final_flush_outside_captured_intervals": True, "reset_once_at_track_origin": True,
                         "source_subset_fixed_through_prefix": True, "reconstruction_max_abs": error,
                         "common_input_prefix_hashes": common_hashes,
                         "captured_output_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in outputs[view]]}
        del target, mixture
        print({"event": "teacher_view_completed", "view": view, "reconstruction_max_abs": error}, flush=True)
    scores = score_views(track["name"], intervals, refs, outputs, mixtures, MetricConfig.from_mapping(config["metrics"]))
    comparisons = {}
    for label, reference in references.items():
        comparisons[label] = {}
        for view in VIEWS:
            a, b = reference["views"][view], scores[view]
            require(a["input_active_windows"] == b["input_active_windows"] and a["total_windows"] == b["total_windows"],
                    "Input support differs")
            for left, right in zip(a["windows"], b["windows"], strict=True):
                require(all(left[k] == right[k] for k in ("physical_start", "physical_end", "input_rms_dbfs", "input_active")),
                        "Physical diagnostic window or native input differs")
            per_stem = {}
            for stem in SOURCE_ORDER:
                values = {}
                for field in ("output_rms_dbfs", "output_to_input_db", "signed_desired_projection_gain"):
                    x, y = a["native_output_levels"][stem][field], b["native_output_levels"][stem][field]
                    require((x is None) == (y is None), "Native metric support differs")
                    values[field] = {"reference": x, "teacher": y, "delta": None if x is None else y - x}
                ac, bc = (v["standard_scores_on_remixed_references"]["per_stem"][stem] for v in (a, b))
                for field, x, y in [("full_sdr_db", ac["full_sdr_db"], bc["full_sdr_db"]),
                                    *[(band, ac["band_sdr_db"][band], bc["band_sdr_db"][band]) for band in ac["band_sdr_db"]]]:
                    require((x is None) == (y is None), "Desired fidelity support differs")
                    values[field] = {"reference": x, "teacher": y, "delta": None if x is None else y - x}
                per_stem[stem] = values
            comparisons[label][view] = per_stem
    require(state_sha256(teacher.state_dict()) == identity["model_state_sha256"]
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(not p.requires_grad and p.grad is None for p in teacher.parameters()), "Teacher, RNG or CPU scope changed")
    verify_inputs(plan)
    result = {"schema": "latency58-c91-skelpolu-vocal-views-v1", "status": "pass", "plan_sha256": args.plan_sha256,
              "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
              "teacher": identity, "track_index": 10, "track_name": track["name"], "intervals": intervals,
              "views": scores, "stream": streams, "comparisons": comparisons,
              "model_and_rng_unchanged": True, "cuda_initialized": False, "training_updates_executed": 0,
              "quality_selected": False, "confirmation_material_used": False, "audio_exported": False,
              "counted_bytes_before": before, "counted_bytes_after": require_space(plan, 0),
              "elapsed_seconds": time.monotonic() - began,
              "limitations": ["Two previously inspected development passages from one track; no panel-wide or causal training conclusion.",
                              "Continuous native FP32 CPU C91 inference uses its own 512-sample graph alignment; not a 256-sample plugin candidate.",
                              "Pilot GPU teachers reset each independent training crop; this full-prefix diagnostic is not their exact training-target journal.",
                              "Recorded stems may contain bleed; these controlled views do not establish full-mixture or human listening quality."]}
    require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            + len(json.dumps(result, indent=2, allow_nan=False).encode()) + 100_000 < plan["maximum_new_artifact_bytes"],
            "Diagnostic metrics exceed their separate allowance")
    write(out / "result.json", result)
    print({"status": "pass", "elapsed_seconds": result["elapsed_seconds"],
           "native_levels": {view: values["native_output_levels"] for view, values in scores.items()}}, flush=True)


if __name__ == "__main__":
    main()
