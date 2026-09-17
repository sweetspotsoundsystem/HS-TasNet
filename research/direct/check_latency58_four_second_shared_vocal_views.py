"""Qualify unchanged source views on an explicit role in a packed checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.latency58_four_second_shared_evaluation import evaluation_storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Vocal-view qualification plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-packed-branch-vocal-views-functional-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    verify_inputs(plan)
    before = evaluation_storage(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve qualification")
    import numpy as np
    import soundfile as sf
    import torch
    from research.direct.latency58_four_second_shared_evaluation import load_model
    from research.direct.latency58_evaluate import plan_latency58_stream
    from research.direct.latency58_vocal_views import VIEWS, VERSION, combine_sources, stream_views, score_views
    from research.metrics import MetricConfig, SOURCE_ORDER
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260920)
    torch.use_deterministic_algorithms(True)
    model, _ = load_model(plan, plan)
    require(state_sha256(model.state_dict()) == plan["model_state_sha256"]
            and len(model.initial_state(1, device=torch.device("cpu"))) == 8
            and model.algorithmic_latency_samples == 256, "Wrong branch-memory model or delay")
    fingerprint, rng = state_sha256(model.state_dict()), torch.get_rng_state().clone()
    began = time.monotonic()
    generator = np.random.default_rng(20260920)
    sources = generator.normal(0, .02, (4, 2, 16384)).astype(np.float32)
    sources[2, :, 700:850] = 0
    from tempfile import TemporaryDirectory
    fixture = TemporaryDirectory(prefix="latency58-packed-vocal-views-")
    fixture_root = Path(fixture.name)
    paths, fixture_bindings = [], {}
    for stem, value in zip(SOURCE_ORDER, sources, strict=True):
        path = fixture_root / ("synthetic-" + stem + ".wav")
        require(not path.exists(), "Preserve fixture audio")
        sf.write(path, value.T, 44100, subtype="FLOAT")
        decoded, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 44100 and np.array_equal(decoded.T, value), "Fixture file changed samples")
        paths.append(path)
        fixture_bindings[str(path)] = sha(path)
    intervals = [{"id": str(i), "reference_start": start, "reference_end": stop,
                  "estimate_start": start + 128, "estimate_end": stop + 128}
                 for i, (start, stop) in enumerate(((33, 321), (9901, 14497)))]
    parity, digests = [], []
    for group in (1, 3, 64):
        stream_plan = plan_latency58_stream(intervals, sources.shape[-1], unroll_hops=group, io_block_hops=2)
        refs, outputs, mixtures, metadata = stream_views(model, paths, stream_plan)
        require(all(np.array_equal(ref, sources[..., row["reference_start"]:row["reference_end"]])
                    for ref, row in zip(refs, intervals, strict=True)), "Source capture differs from independent array")
        digests.append(metadata["input_stream_sha256"])
        with torch.inference_mode():
            for view in VIEWS:
                physical = combine_sources(sources[..., :stream_plan.receive_end], view)
                expected_sha = hashlib.sha256(np.ascontiguousarray(physical.T).tobytes()).hexdigest()
                require(metadata["input_stream_sha256"][view] == expected_sha, "Input digest or remix differs")
                direct = model.render(torch.from_numpy(physical[None]))
                errors = []
                for index, row in enumerate(intervals):
                    expected = direct.deployed[0, ..., row["estimate_start"]:row["estimate_end"]].numpy()
                    error = float(np.max(np.abs(outputs[view][index] - expected)))
                    require(error <= 1e-6 and np.array_equal(mixtures[view][index],
                        physical[:, row["reference_start"]:row["reference_end"]]),
                        "Grouped source-view streaming differs from independent full-prefix rendering")
                    errors.append(error)
                parity.append({"group_hops": group, "view": view, "max_absolute_error": max(errors),
                               "input_stream_sha256": expected_sha,
                               "physical_alignment_exact": True, "metadata": metadata})
    require(digests[0] == digests[1] == digests[2], "Input digest depends on grouping")
    # First second is active; the second is deliberately below the fixed
    # activity threshold. Use distinct frequencies and stereo phases.
    t = np.arange(88200, dtype=np.float64) / 44100
    refs = np.stack([np.stack([.03 * np.sin(2 * np.pi * f * t + phase)
                    for phase in (.2, .9)]) for f in (83., 151., 307., 719.)]).astype(np.float32)
    refs[..., 44100:] *= 1e-5
    metric_intervals = [{"reference_start": 0, "reference_end": 88200,
                         "estimate_start": 128, "estimate_end": 88328}]
    mixtures = {view: [combine_sources(refs, view)] for view in VIEWS}
    ideal = {}
    for view, included in VIEWS.items():
        output = np.zeros_like(refs)
        output[list(included)] = refs[list(included)]
        ideal[view] = [output]
    ideal_score = score_views("synthetic", metric_intervals, [refs], ideal, mixtures, MetricConfig())
    damaged = {view: [values[0].copy()] for view, values in ideal.items()}
    damaged["vocals_only"][0][3] += .1 * refs[2]
    damaged["vocals_only"][0][2] -= .1 * refs[2]
    damaged["instrumental"][0][2] += .2 * refs[1]
    damaged["instrumental"][0][1] -= .2 * refs[1]
    damaged_score = score_views("synthetic", metric_intervals, [refs], damaged, mixtures, MetricConfig())
    silent = {view: [np.zeros_like(refs)] for view in VIEWS}
    silent_score = score_views("synthetic", metric_intervals, [refs], silent, mixtures, MetricConfig())
    for view, included in VIEWS.items():
        require(ideal_score[view]["input_active_windows"] == damaged_score[view]["input_active_windows"]
                == silent_score[view]["input_active_windows"] == 1,
                "Leakage support incorrectly includes a quiet input or depends on output")
        for source in included:
            stem = SOURCE_ORDER[source]
            desired = ideal_score[view]["standard_scores_on_remixed_references"]["per_stem"][stem]
            silenced = silent_score[view]["standard_scores_on_remixed_references"]["per_stem"][stem]
            require(desired["full_sdr_db"] == 60 and abs(silenced["full_sdr_db"]) < 1e-10
                    and abs(ideal_score[view]["native_output_levels"][stem]["signed_desired_projection_gain"] - 1) < 1e-10
                    and silent_score[view]["native_output_levels"][stem]["signed_desired_projection_gain"] == 0,
                    "Desired fidelity or signed gain failed to expose muting")
    for view, stem in (("vocals_only", "other"), ("instrumental", "vocals")):
        require(damaged_score[view]["native_output_levels"][stem]["output_to_input_db"]
                > ideal_score[view]["native_output_levels"][stem]["output_to_input_db"] + 10,
                "Known cross-source leakage did not worsen its metric")
    require(state_sha256(model.state_dict()) == fingerprint and torch.equal(torch.get_rng_state(), rng)
            and not torch.cuda.is_initialized() and all(sha(p) == s for p, s in fixture_bindings.items()),
            "Qualification changed weights, RNG, fixture files or CPU scope")
    fixture_bytes = sum(path.stat().st_size for path in paths)
    require(fixture_bytes < 1_000_000, "Synthetic audio exceeds its small outside allowance")
    fixture.cleanup()
    require(not fixture_root.exists(), "Temporary synthetic audio was not removed")
    verify_inputs(plan)
    result = {"schema": "latency58-packed-branch-vocal-views-functional-result-v1", "status": "pass", "version": VERSION,
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "fixture_bindings": fixture_bindings,
              "parity": parity, "metric_fixtures": {"ideal": ideal_score, "known_leakage": damaged_score, "muted": silent_score},
              "elapsed_seconds": time.monotonic() - began, "storage_before": before,
              "storage_after": evaluation_storage(plan), "cuda_initialized": False,
              "training_updates_executed": 0, "validation_material_used": False, "quality_selected": False,
              "checkpoint": plan["checkpoint"], "model_state_sha256": fingerprint, "public_stream_states": 8,
              "graph_plus_host_delay_samples": 256, "tested_group_hops": [1, 3, 64],
              "checkpoint_role": plan["checkpoint_role"], "packed_training_plan": plan["packed_training_plan"],
              "synthetic_audio_removed": True, "synthetic_audio_bytes": fixture_bytes}
    write(out / "result.json", result)
    print({"status": "pass", "max_absolute_stream_error": max(r["max_absolute_error"] for r in parity),
           "metric_support_and_muting_checks": "pass"}, flush=True)


if __name__ == "__main__":
    main()
