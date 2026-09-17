"""Test the previously fixed residual share on C204 with the original full14 metrics."""
from __future__ import annotations

import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, write, sha, require, execute
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space


def binding(path):
    return {"path": str(path.resolve()), "sha256": sha(path)}


def main():
    out = PHASE / "c204-residual-share-full14-001"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve existing fixed-share evaluations")
    sources = [Path(__file__).resolve(), ROOT / "research/direct/diagnose_latency58_c204_residual_share.py",
               ROOT / "research/direct/latency58_residual_share.py",
               ROOT / "research/direct/report_latency58_vocal_focus.py",
               ROOT / "research/direct/compare.py",
               PHASE / "latency58-reduced-teacher-001/sdr-comparison.json",
               PHASE / "latency58-reduced-teacher-001/training-plan.json",
               PHASE / "sdr-residual-share-full14-diagnostic-001/result.json"]
    bindings = {str(p): sha(p) for p in sources}
    quality_path = PHASE / "leader-cleanup-250-full14-001/plan.json"
    quality, baseline = load_completed(quality_path.parent, bindings)
    bindings.update(quality["source_bindings"])
    model_state = baseline["results"][0]["model"]["model_state_sha256"]
    require(model_state == "c204b0fcb9627ca7fecd287db42fb869a1ae6783a1bc24cf2d8864c3b4a565fb",
            "Unexpected reference checkpoint")
    recent = read(PHASE / "latency58-reduced-teacher-001/sdr-comparison.json")
    require(recent["status"] == "pass" and recent["source_bindings_unchanged"]
            and recent["full_mixture_aggregate"]["full_sdr_db"] < baseline["results"][0]["aggregate"]["full_sdr_db"],
            "Review the new teacher checkpoint before this follow-up")
    budget = read(PHASE / "latency58-reduced-teacher-001/training-plan.json")
    require_space(budget, 20_000_000)
    import numpy as np
    from research.direct.latency58_residual_share import residual_share
    from research.direct.evaluate import shipping_residual
    raw = np.linspace(-.4, .7, 4 * 2 * 257, dtype=np.float32).reshape(4, 2, 257)
    mixture = np.sin(np.arange(514, dtype=np.float64) * .03).astype(np.float32).reshape(2, 257)
    raw_copy, mix_copy = raw.copy(), mixture.copy()
    corrected = residual_share(raw, mixture)
    weights = np.array([1 / 16, 1 / 16, 1 / 16, 13 / 16], dtype=np.float64)
    independent = np.einsum("ij,jct->ict", np.eye(4) - weights[:, None], raw.astype(np.float64)) \
                  + weights[:, None, None] * mixture
    error = float(np.max(np.abs(corrected.astype(np.float64) - independent)))
    closure = float(np.max(np.abs(corrected.sum(axis=0, dtype=np.float32) - mixture)))
    require(error < 3e-7 and closure < 3e-7
            and np.array_equal(residual_share(raw, mixture, share=0), shipping_residual(raw, mixture))
            and np.array_equal(residual_share(raw[..., :91], mixture[..., :91]), corrected[..., :91])
            and np.array_equal(raw, raw_copy) and np.array_equal(mixture, mix_copy)
            and np.array_equal(residual_share(np.zeros_like(raw), np.zeros_like(mixture)), np.zeros_like(raw)),
            "Fixed correction failed its independent arithmetic or prefix check")
    verify_inputs({"source_bindings": bindings})
    out.mkdir()
    write(out / "functional.json", {"status": "pass", "primary_share": 1 / 16,
          "zero_identity_and_prefix_exact": True, "affine_max_abs_error": error,
          "closure_max_abs_error": closure, "input_tensors_unchanged": True,
          "coefficient_selection": "Unchanged 1/16 coefficient from the earlier frozen experiment; no coefficient search.",
          "added_audio_buffering_samples": 0})
    bindings[str(out / "functional.json")] = sha(out / "functional.json")
    plan = {"schema": "latency58-c204-residual-share-parallel-diagnostic-v1", "workers": 2,
            "primary_share": 1 / 16, "track_indices": list(range(14)),
            "parent_quality_plan": binding(quality_path), "parent_model_state_sha256": model_state,
            "parent_full14_result": binding(quality_path.parent / "result.json"),
            "manifest": str(ROOT / "research/manifests/valid.json"),
            "evaluation_config": str(ROOT / "research/eval_config.json"),
            "functional_check": binding(out / "functional.json"), "output_directory": str(out),
            "source_bindings": bindings, "timeout_seconds": 1800}
    verify_inputs(plan)
    write(out / "plan.json", plan)
    execute([PYTHON, "-u", "-m", "research.direct.diagnose_latency58_c204_residual_share",
             "--plan", str(out / "plan.json"), "--plan-sha256", sha(out / "plan.json")],
            out, "diagnostic", 1800, bindings, {"plan_sha256": sha(out / "plan.json")})
    result = read(out / "result.json")
    require(result["status"] == "pass" and result["exact_stored_track_reports_and_aggregate"],
            "Full14 correction diagnostic failed")
    print(json.dumps({"status": "pass", "full_sdr_db": result["policies"]["fixed_share"]["aggregate"]["full_sdr_db"],
                      "comparison": result["comparison"]["metrics"], "model_integrated": False}), flush=True)


if __name__ == "__main__":
    main()
