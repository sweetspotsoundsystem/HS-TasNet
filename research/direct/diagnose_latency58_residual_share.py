"""Check a fixed raw-Other correction and score one existing Actions excerpt.

The exact unchanged parent is streamed once. Both policies use the same raw
captures and original metric functions; the zero correction must reproduce
the stored ordinary track report exactly. No fitted coefficient, checkpoint,
audio export, training, GPU access or confirmation material is involved.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Residual-share plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-residual-share-actions-diagnostic-v1"
            and plan["primary_share"] == 1 / 16 and plan["track_indices"] == [1]
            and plan["excerpt_starts"] == [60.0] and plan["duration_seconds"] == 15.0
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and all(sha(p) == s for p, s in plan["source_bindings"].items()),
            "Require frozen fixed-share Actions diagnostic on CPU1")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve prior diagnostics")
    began = time.monotonic()
    import numpy as np
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_evaluate import (
        HOP, SOURCE_ORDER, model_state_sha256, plan_latency58_stream, stream_latency58_track,
    )
    from research.direct.latency58_log_relative_checkpoint import load_parent
    from research.direct.latency58_residual_share import PRIMARY_SHARE, VERSION, residual_share
    from research.metrics import MetricConfig

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rng_before = torch.get_rng_state().clone()
    # Independent affine reference: each DBV row is its raw source minus
    # 1/16 of each raw source, plus 1/16 of the physical mixture. Other's
    # row is its raw source minus 13/16 of every source, plus 13/16 mixture.
    toy = np.linspace(-.4, .7, 4 * 2 * 257, dtype=np.float32).reshape(4, 2, 257)
    mixture = np.sin(np.arange(514, dtype=np.float64) * .03).astype(np.float32).reshape(2, 257)
    old_toy, old_mix = toy.copy(), mixture.copy()
    weights = np.array([1 / 16, 1 / 16, 1 / 16, 13 / 16], dtype=np.float64)
    matrix = np.eye(4) - weights[:, None]
    independent = np.einsum("ij,jct->ict", matrix, toy.astype(np.float64)) + weights[:, None, None] * mixture
    actual = residual_share(toy, mixture)
    affine_error = float(np.max(np.abs(actual.astype(np.float64) - independent)))
    closure = float(np.max(np.abs(actual.sum(axis=0, dtype=np.float32) - mixture)))
    require(affine_error < 3e-7 and closure < 3e-7
            and np.array_equal(residual_share(toy, mixture, share=0), shared.shipping_residual(toy, mixture))
            and np.array_equal(toy, old_toy) and np.array_equal(mixture, old_mix)
            and np.array_equal(residual_share(toy[..., :91], mixture[..., :91]), actual[..., :91])
            and np.array_equal(residual_share(np.zeros_like(toy), np.zeros_like(mixture)), np.zeros_like(toy)),
            "Independent affine, closure, zero identity, input, prefix or silence check failed")
    print({"event": "functional_pass", "affine_max_abs": affine_error, "closure_max_abs": closure}, flush=True)

    parent_binding = plan["parent_training_plan"]
    require(sha(parent_binding["path"]) == parent_binding["sha256"], "Parent loader plan changed")
    model = load_parent(read(parent_binding["path"])).eval().requires_grad_(False)
    fingerprint = model_state_sha256(model)
    require(fingerprint == plan["parent_model_state_sha256"], "Different diagnostic parent")
    manifest, config_source = read(plan["manifest"]), read(plan["evaluation_config"])
    tracks, config = shared.select_panel(manifest, config_source, panel="full", track_indices=[1],
                                        excerpt_starts=[60.0], duration=15.0, alignment_samples=HOP)
    require(len(tracks) == 1, "Screen covers exactly one track")
    track = tracks[0]
    rows = legacy._reference_intervals(track, config)
    require(len(rows) == 1, "Screen covers exactly one excerpt")
    streaming = plan_latency58_stream(rows, int(track["frames"]))
    root = Path(manifest["root"])
    raw_outputs, delayed_mixtures, stream_info = stream_latency58_track(
        model, legacy._safe_dataset_path(root, track["mixture"]), streaming)

    def excerpt(relative, row):
        return legacy._read_excerpt(legacy._safe_dataset_path(root, relative),
                                    int(row["reference_start"]), int(row["reference_end"]),
                                    expected_frames=int(track["frames"]))

    mixtures = [excerpt(track["mixture"], row) for row in rows]
    references = [np.stack([excerpt(track["stems"][stem], row) for stem in SOURCE_ORDER]) for row in rows]
    require(all(np.array_equal(delayed, physical.astype(np.float32)) for delayed, physical in
                zip(delayed_mixtures, mixtures, strict=True)), "Physical mixture alignment differs")
    config_metric = MetricConfig.from_mapping(config["metrics"])
    scores = {}
    for name, share in (("working_policy", 0), ("fixed_share", PRIMARY_SHARE)):
        estimates = [residual_share(raw, physical.astype(np.float32), share=share)
                     for raw, physical in zip(raw_outputs, mixtures, strict=True)]
        scores[name] = legacy._score_track(track["name"], rows, mixtures, references, estimates, config_metric)
        closure = max(closure, max(float(np.max(np.abs(estimate.sum(axis=0, dtype=np.float32) - physical)))
                                   for estimate, physical in zip(estimates, mixtures, strict=True)))
    reference_binding = plan["parent_actions_result"]
    require(sha(reference_binding["path"]) == reference_binding["sha256"], "Stored ordinary score changed")
    stored = read(reference_binding["path"])
    require(stored["results"][0]["model"]["model_state_sha256"] == fingerprint
            and stored["results"][0]["tracks"] == [scores["working_policy"]],
            "Zero correction does not reproduce the exact stored ordinary track report")
    aggregates = {name: legacy._aggregate_tracks([score]) for name, score in scores.items()}
    deltas = {metric: aggregates["fixed_share"][metric] - aggregates["working_policy"][metric]
              for metric in ("full_sdr_db", "low_sdr_db", "bleed_sir_db")}
    per_stem = {stem: {metric: aggregates["fixed_share"]["per_stem"][stem][metric]
                             - aggregates["working_policy"]["per_stem"][stem][metric]
                      for metric in ("full_sdr_db", "sir_db")}
                for stem in SOURCE_ORDER}
    require(model_state_sha256(model) == fingerprint and torch.equal(rng_before, torch.get_rng_state())
            and not torch.cuda.is_initialized() and all(sha(p) == s for p, s in plan["source_bindings"].items()),
            "Model, RNG, CPU scope or bound input changed")
    result = {"schema": "latency58-residual-share-actions-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "parent_model_state_sha256": fingerprint,
              "version": VERSION, "primary_share": PRIMARY_SHARE, "fixed_before_music_scoring": True,
              "affine_reference_max_abs": affine_error, "mixture_closure_max_abs": closure,
              "zero_share_exact_stored_track_report": True, "stream_metadata": stream_info,
              "scores": scores, "aggregates": aggregates, "aggregate_deltas": deltas,
              "per_stem_deltas": per_stem, "model_unchanged": True, "rng_unchanged": True,
              "cuda_initialized": False, "checkpoint_written": False, "audio_written": False,
              "training_updates_executed": 0, "quality_selected": False,
              "elapsed_seconds": time.monotonic() - began,
              "limitations": ["One previously used development excerpt; no full-panel or uncertainty claim.",
                              "This changes output mixing, while keeping all neural weights and native gains fixed.",
                              "No listening, ONNX, native runtime or host-latency qualification."]}
    write(out / "result.json", result)
    print({"event": "diagnostic_complete", "aggregate_deltas": deltas}, flush=True)


if __name__ == "__main__":
    main()
