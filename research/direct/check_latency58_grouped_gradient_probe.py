"""Qualify separated group gradients against untouched accumulation on CPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import resource
import time

import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.latency58_grouped_gradient_probe import collect, summarize, policy
from research.direct.latency58_grouped_vocal_canonical import accumulate_groups
from research.direct.latency58_grouped_vocal_auxiliary import source_views, AUXILIARY_WEIGHT
from research.direct.latency58_branch_memory_context import render_scored_context
from research.direct.latency58_branch_sdr_blend import objective as whole_objective


def arithmetic_checks():
    def fixture(ordinary, instrumental, vocals):
        vectors = {"ordinary": torch.tensor(ordinary, dtype=torch.float32),
                   "instrumental": torch.tensor(instrumental, dtype=torch.float32),
                   "vocals_only": torch.tensor(vocals, dtype=torch.float32)}
        vectors["auxiliary"] = vectors["instrumental"] + vectors["vocals_only"]
        return summarize({group: {"fixture": tensor} for group, tensor in vectors.items()})

    aligned = fixture([1, 2], [3, 0], [-1, 4])
    require(math.isclose(aligned["pairs"]["ordinary_vs_auxiliary"]["cosine"], 1., rel_tol=0., abs_tol=1e-15)
            and aligned["pairs"]["ordinary_vs_auxiliary"]["dot"] == 10
            and aligned["pairs"]["ordinary_vs_auxiliary"]["right_to_left_l2_ratio"] == 2
            and aligned["ordinary_directional_derivative_along_negative_combined_gradient"] == -15
            and aligned["auxiliary_directional_derivative_along_negative_combined_gradient"] == -30,
            "Known aligned-vector arithmetic differs")
    opposed = fixture([1, 0], [-1, 0], [0, 0])
    require(opposed["pairs"]["ordinary_vs_auxiliary"]["cosine"] == -1
            and opposed["ordinary_directional_derivative_along_negative_combined_gradient"] == 0
            and opposed["auxiliary_directional_derivative_along_negative_combined_gradient"] == 0,
            "Known cancelling-vector arithmetic differs")
    orthogonal = fixture([1, 0], [0, 2], [0, 0])
    require(orthogonal["pairs"]["ordinary_vs_auxiliary"]["cosine"] == 0
            and orthogonal["pairs"]["ordinary_vs_auxiliary"]["dot"] == 0, "Orthogonal vectors differ")
    zero = fixture([0, 0], [0, 2], [0, 0])
    require(zero["pairs"]["ordinary_vs_auxiliary"]["cosine"] is None
            and zero["pairs"]["ordinary_vs_auxiliary"]["right_to_left_l2_ratio"] is None,
            "Zero-vector undefined statistics were not retained")
    return {"aligned": aligned["pairs"], "cancelling": opposed["pairs"],
            "orthogonal": orthogonal["pairs"], "zero": zero["pairs"]}


def compare_gradients(actual, expected, *, exact):
    require(list(actual) == list(expected) and len(actual) == 40, "Parameter inventory differs")
    rows = {}
    for name in actual:
        a, b = actual[name], expected[name]
        require(a.shape == b.shape and a.dtype == b.dtype == torch.float32
                and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
                "Nonfinite or mismatched reference gradient: " + name)
        difference = (a - b).double()
        norm = float(b.double().norm())
        relative = float(difference.norm()) / norm if norm else (0. if torch.equal(a, b) else None)
        equal = torch.equal(a, b)
        require(equal if exact else (relative is not None and relative < 5e-5
                and torch.allclose(a, b, atol=1e-7, rtol=1e-4)), "Probe gradient differs: " + name)
        rows[name] = {"bitwise_equal": equal, "maximum_absolute_error": float(difference.abs().max()),
                      "relative_l2_error": relative, "reference_l2": norm}
    return rows


def auxiliary_reference(model, mixture, targets, *, warmup_samples, progress):
    """Differentiate scalar joint losses with one neural graph detached at a time.

    Both one-example graphs are retained, so this control does not use the
    probe's observation callbacks or detached-output derivative replay.
    """
    audio, truth = source_views(mixture, targets)
    outputs = [render_scored_context(model, audio[i:i + 1], warmup_samples=warmup_samples,
                                     carry_state=True) for i in (0, 1)]
    named = list(model.named_parameters())
    gradients, losses = {}, {}
    for name, active in (("auxiliary", (0, 1)), ("instrumental", (0,)), ("vocals_only", (1,))):
        raw = torch.cat([value.raw if i in active else value.raw.detach()
                         for i, value in enumerate(outputs)])
        deployed = torch.cat([value.deployed if i in active else value.deployed.detach()
                              for i, value in enumerate(outputs)])
        value = AUXILIARY_WEIGHT * whole_objective(
            raw, deployed, truth[..., warmup_samples:], audio[..., warmup_samples:]).total
        losses[name] = float(value.detach())
        result = torch.autograd.grad(value, [p for _, p in named], retain_graph=name != "vocals_only")
        gradients[name] = {key: gradient.detach().clone() for (key, _), gradient in zip(named, result, strict=True)}
        del raw, deployed, value, result
        progress("direct_scalar_auxiliary_reference", name, 0)
    require(len(set(losses.values())) == 1 and all(p.grad is None for _, p in named),
            "Detaching a graph changed the joint loss value or leaf gradients")
    return gradients, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="grouped-gradient-probe-cpu-001")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    require(args.output_prefix and all(c.isalnum() or c in "-_" for c in args.output_prefix), "Invalid prefix")
    out = PHASE / args.output_prefix
    require(not out.exists(), "Preserve previous probe qualifications")
    source_path = PHASE / "branch-grouped-vocal-013/plan.json"
    source = read(source_path)
    review = PHASE / "grouped-continuation-review-013"
    decision = read(review / "continuation-review-decision.json")
    require(sha(review / "continuation-review-decision.json")
            == "2a01953bbdb0e012dcbb49202991e17a3507eff9a5e4c7d607190a47846dfcc4"
            and decision["status"] == "review_complete" and not decision["unchanged_recipe_continuation_selected"]
            and source["parent_checkpoint"] == decision["preserved_checkpoints"]["starting_parent"],
            "Require the completed continuation review and retained starting checkpoint")
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    from research.direct.latency58_branch_memory_checkpoint import load_model
    before_budget = budget_snapshot(source["storage_budget"])
    require(before_budget["headroom_bytes"] > 2_000_000, "Reserve scalar-only qualification artifacts")
    bindings = {p: digest for p, digest in source["source_bindings"].items() if Path(p).suffix == ".py"}
    paths = [Path(__file__).resolve(), ROOT / "research/direct/latency58_grouped_gradient_probe.py",
             source_path, review / "continuation-review-decision.json", review / "result.json", review / "execution.json"]
    bindings.update({str(p): sha(p) for p in paths})
    bindings[source["parent_checkpoint"]["path"]] = source["parent_checkpoint"]["sha256"]
    plan = {"schema": "latency58-group-gradient-probe-cpu-qualification-v1", "policy": policy(),
            "source_bindings": bindings, "fixture_checkpoint": source["parent_checkpoint"],
            "fixture_model_state_sha256": source["parent_model_state_sha256"],
            "warmup_samples": 512, "scored_samples": 44160, "synthetic_seed": 202611015,
            "output_allowance_bytes": 2_000_000, "maximum_observed_rss_bytes": 18_000_000_000,
            "storage_budget": source["storage_budget"], "budget_before": before_budget,
            "training_material_used": False, "validation_audio_decoded": False, "gpu_used": False,
            "gradient_absolute_tolerance": 1e-7, "gradient_relative_tolerance": 1e-4,
            "gradient_relative_l2_tolerance": 5e-5, "reference_accumulation_requires_bit_exact": True}
    verify_inputs(plan)
    out.mkdir(); write(out / "plan.json", plan)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    model, _ = load_model(plan["fixture_checkpoint"])
    require(state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"]
            and not torch.cuda.is_initialized(), "Fixture checkpoint or device differs")
    model.train().requires_grad_(True); model.training_precision = "fp32"
    rng = torch.get_rng_state().clone()
    started = time.monotonic()
    arithmetic = arithmetic_checks()
    generator = torch.Generator().manual_seed(plan["synthetic_seed"])
    truth = .02 * torch.randn(16, 4, 2, 512 + 44160, generator=generator)
    truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
    mixture = truth.sum(1)
    original = state_sha256(model.state_dict())

    def progress(phase, group, offset):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        require(rss < plan["maximum_observed_rss_bytes"], "CPU probe qualification exceeded its memory bound")
        print(json.dumps({"event": "gradient_probe_cpu_progress", "phase": phase, "group": group,
                          "offset": offset, "elapsed_seconds": time.monotonic() - started,
                          "peak_rss_bytes": rss}), flush=True)

    rejected = []
    for name in ("dirty_gradients", "wrong_precision", "short_batch", "invalid_warmup"):
        parameter = next(model.parameters())
        if name == "dirty_gradients":
            parameter.grad = torch.ones_like(parameter)
        if name == "wrong_precision":
            model.training_precision = "bf16"
        try:
            collect(model, mixture[:15] if name == "short_batch" else mixture, truth,
                    warmup_samples=513 if name == "invalid_warmup" else 512)
        except RuntimeError as error:
            require(str(error).startswith("Require a fresh CPU FP32") or str(error).startswith("Require complete fixed FP32"),
                    "Unexpected invalid-input rejection")
            rejected.append(name)
        else:
            raise RuntimeError("Invalid probe input was accepted: " + name)
        finally:
            model.zero_grad(set_to_none=True); model.training_precision = "fp32"
        require(state_sha256(model.state_dict()) == original, "Invalid-input check changed weights")

    gradients, metadata = collect(model, mixture, truth, warmup_samples=512, progress=progress)
    statistics = summarize(gradients)
    write(out / "probe-statistics.json", statistics)
    expected_rows = accumulate_groups(model, mixture, truth, warmup_samples=512,
                                     ordinary_microbatch=1, auxiliary_microbatch=1,
                                     verify_input_gradients=True, progress=progress)
    expected = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    combined = {name: (gradients["ordinary"][name] + gradients["instrumental"][name])
                      + gradients["vocals_only"][name] for name in expected}
    combined_errors = compare_gradients(combined, expected, exact=True)
    require(metadata["groups"] == expected_rows, "Probe changed whole-group loss or activity counts")
    del expected, combined
    reference, reference_losses = auxiliary_reference(model, mixture, truth, warmup_samples=512, progress=progress)
    reference_errors = {group: compare_gradients(gradients[group], reference[group], exact=False)
                        for group in reference}
    require(abs(reference_losses["auxiliary"] - metadata["groups"]["auxiliary"]["weighted_loss"]) < 3e-6,
            "Independent scalar auxiliary objective differs")
    del gradients, reference
    gc.collect()

    class RequestedStop(Exception):
        pass

    # Stop after the first backward to exercise cleanup of a partial ordinary
    # gradient without completing another probe.
    def stop_early(phase, group, offset):
        if phase == "canonical_backward" and group == "ordinary" and offset == 0:
            raise RequestedStop()

    try:
        collect(model, mixture, truth, warmup_samples=512, progress=stop_early)
    except RequestedStop:
        require(all(p.grad is None for p in model.parameters()) and state_sha256(model.state_dict()) == original,
                "Interrupted probe did not clear partial gradients or preserve weights")
    else:
        raise RuntimeError("Requested probe interruption was not observed")
    require(state_sha256(model.state_dict()) == original == plan["fixture_model_state_sha256"]
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(p.grad is None for p in model.parameters()), "Qualification changed its endpoint or device")
    verify_inputs(plan)
    result = {"schema": plan["schema"], "status": "pass", "plan_sha256": sha(out / "plan.json"),
              "source_bindings_unchanged": True, "policy": policy(), "arithmetic_fixtures": arithmetic,
              "rejected_inputs": rejected, "probe_metadata": metadata,
              "combined_all_40_gradients": combined_errors, "direct_scalar_auxiliary_references": reference_errors,
              "reference_losses": reference_losses, "probe_statistics_sha256": sha(out / "probe-statistics.json"),
              "joint_auxiliary_denominators_preserved": True, "interrupted_partial_gradients_cleared": True,
              "warmup_input_gradients_zero": True, "parent_and_rng_unchanged": True,
              "gradient_buffers_clear": True, "optimizer_updates": 0, "checkpoint_files_written": False,
              "training_material_used": False, "validation_audio_decoded": False, "gpu_used": False,
              "quality_measured": False, "elapsed_seconds": time.monotonic() - started,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "budget_after": budget_snapshot(source["storage_budget"])}
    require(sum(p.stat().st_size for p in out.iterdir() if p.is_file())
            + len(json.dumps(result, indent=2).encode()) + 100_000 < plan["output_allowance_bytes"],
            "Qualification exceeded its scalar artifact allowance")
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "all_40_combined_gradients_bit_exact": True,
                      "independent_auxiliary_scalar_references_pass": True,
                      "elapsed_seconds": result["elapsed_seconds"], "peak_rss_bytes": result["peak_rss_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
