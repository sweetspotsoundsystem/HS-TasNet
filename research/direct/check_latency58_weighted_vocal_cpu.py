"""Qualify quarter-vocal scalar supervision, canonical gradients and restart."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.check_latency58_grouped_vocal_restart import fingerprint, RequestedStop
from research.direct.latency58_weighted_vocal_auxiliary import objective, policy as objective_policy, VIEW_WEIGHTS
from research.direct.latency58_weighted_vocal_canonical import grouped_update, policy as accumulation_policy
from research.direct.latency58_grouped_vocal_auxiliary import source_views, AUXILIARY_WEIGHT
from research.direct.latency58_branch_sdr_blend import objective as original_objective, SDR_WEIGHT
from research.direct.latency58_logical_batch_loss import activity_counts
from research.direct.latency58_branch_memory_context import render_scored_context
from research.direct.check_latency58_grouped_gradient_probe import compare_gradients


def independent_scalar(raw, deployed, targets, mixture, *, weights):
    """Use the old whole objective with each other view fixed at its target.

    Exact predictions contribute zero reconstruction/absence/anchor loss and
    -60 dB on active windows. Removing that known constant isolates each
    view's scalar contribution while retaining the old whole-group reductions.
    This reference never calls the new loss or the microbatch contribution.
    """
    active, _ = activity_counts(targets)
    per_view = [activity_counts(targets[i:i + 1])[0] for i in (0, 1)]
    constants = [-60 * SDR_WEIGHT * (counts.float() / active.clamp_min(1)).sum()
                 / (active > 0).sum().clamp_min(1) for counts in per_view]
    values = []
    for view in (0, 1):
        estimates = [torch.cat([value[i:i + 1] if i == view else targets[i:i + 1] for i in (0, 1)])
                     for value in (raw, deployed)]
        total = original_objective(*estimates, targets, mixture).total
        values.append(total - constants[1 - view])
    return sum(weight * value for weight, value in zip(weights, values, strict=True))


def compare_outputs(actual, expected, name):
    require(actual.shape == expected.shape and bool(torch.isfinite(actual).all())
            and bool(torch.isfinite(expected).all()), "Invalid output-coordinate gradients")
    error = (actual - expected).double()
    norm = float(expected.double().norm())
    relative = float(error.norm()) / norm if norm else 0. if torch.equal(actual, expected) else float("inf")
    require(torch.allclose(actual, expected, atol=1e-7, rtol=1e-4) and relative < 5e-5,
            "Weighted output derivative differs: " + name)
    return {"maximum_absolute_error": float(error.abs().max()), "relative_l2_error": relative,
            "reference_norm": norm, "bitwise_equal": torch.equal(actual, expected)}


def scalar_checks(progress):
    generator = torch.Generator().manual_seed(202611016)
    truth = .02 * torch.randn(2, 4, 2, 88320, generator=generator)
    truth[0, 2] = 0; truth[1, [0, 1, 3]] = 0
    truth[0, 1, :, :44100] = 0
    truth[1, 2, :, 44100:88200] *= .01
    reports = []
    for fixture, targets in (("mixed_activity", truth), ("all_silent", torch.zeros_like(truth))):
        mixture = targets.sum(1)
        raw = (targets + .003 * torch.randn(targets.shape, generator=generator)).requires_grad_()
        deployed = (targets + .004 * torch.randn(targets.shape, generator=generator)).requires_grad_()
        original = AUXILIARY_WEIGHT * original_objective(raw, deployed, targets, mixture).total
        original_gradients = torch.autograd.grad(original, (raw, deployed))
        for weights in ((1., 1.), VIEW_WEIGHTS, (1., 0.)):
            actual = AUXILIARY_WEIGHT * objective(raw, deployed, targets, mixture, weights=weights).total
            reference = AUXILIARY_WEIGHT * independent_scalar(raw, deployed, targets, mixture, weights=weights)
            require(abs(float((actual - reference).detach())) < 3e-6, "Weighted scalar differs from independent whole-objective reference")
            if weights == (1., 1.):
                require(abs(float((actual - original).detach())) < 3e-6, "Unit view weights changed the original scalar")
            actual_gradients = torch.autograd.grad(actual, (raw, deployed))
            reference_gradients = torch.autograd.grad(reference, (raw, deployed))
            multiplier = raw.new_tensor(weights)[:, None, None, None]
            errors = {}
            for label, actual_gradient, reference_gradient, old in zip(
                    ("raw", "deployed"), actual_gradients, reference_gradients, original_gradients, strict=True):
                errors[label + "_independent_scalar"] = compare_outputs(actual_gradient, reference_gradient, label)
                errors[label + "_scaled_original"] = compare_outputs(actual_gradient, old * multiplier, label)
            reports.append({"fixture": fixture, "weights": list(weights), "weighted_loss": float(actual.detach()),
                            "independent_loss": float(reference.detach()), "gradient_errors": errors})
            progress("scalar_reference", fixture, len(reports))
        perfect = objective(targets, targets, targets, mixture)
        perfect_reference = independent_scalar(targets, targets, targets, mixture, weights=VIEW_WEIGHTS)
        require(abs(float(perfect.total - perfect_reference)) < 3e-6, "Perfect-reference constant differs")
        if fixture == "all_silent":
            require(float(perfect.total) == 0., "All-silent perfect objective is not zero")
    rejected = []
    for weights in ((1., -.25), (1., float("nan")), (.25, 1.), (1.,), [1., .25]):
        try:
            objective(raw, deployed, targets, mixture, weights=weights)
        except RuntimeError as error:
            require(str(error).startswith("Require exactly the ordered source-view pair"), "Unexpected weight rejection")
            rejected.append(str(weights))
        else:
            raise RuntimeError("Invalid view weights accepted")
    try:
        objective(truth.flip(0), truth.flip(0), truth.flip(0), truth.flip(0).sum(1))
    except RuntimeError as error:
        require(str(error) == "Source-view target ordering or removed stems changed", "Unexpected view-order rejection")
    else:
        raise RuntimeError("Reversed source-view targets accepted")
    return {"status": "pass", "cases": reports, "rejected_weights": rejected,
            "reversed_view_order_rejected": True, "complete_group_denominators_preserved": True,
            "instrumental_output_derivative_preserved_within_original_tolerances": True,
            "vocals_output_derivative_scaled_by_exact_quarter_within_original_tolerances": True}


def neural_scalar_reference(model, mixture, targets, *, progress):
    from research.direct.latency58_weighted_vocal_canonical import accumulate_group
    audio, truth = source_views(mixture, targets)
    before = state_sha256(model.state_dict())
    model.zero_grad(set_to_none=True)
    row = accumulate_group(model, audio, truth, group="auxiliary", microbatch=1, warmup_samples=512,
                           verify_input_gradients=True, progress=progress)
    actual = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    outputs = [render_scored_context(model, audio[i:i + 1], warmup_samples=512, carry_state=True) for i in (0, 1)]
    raw, deployed = (torch.cat([getattr(value, name) for value in outputs]) for name in ("raw", "deployed"))
    loss = AUXILIARY_WEIGHT * independent_scalar(raw, deployed, truth[..., 512:], audio[..., 512:], weights=VIEW_WEIGHTS)
    gradients = torch.autograd.grad(loss, [p for _, p in model.named_parameters()])
    expected = {name: value.detach() for (name, _), value in zip(model.named_parameters(), gradients, strict=True)}
    errors = compare_gradients(actual, expected, exact=False)
    require(abs(float(loss.detach()) - row["weighted_loss"]) < 3e-6
            and state_sha256(model.state_dict()) == before and all(p.grad is None for p in model.parameters()),
            "Neural scalar reference changed its parent or loss")
    progress("independent_neural_scalar_complete", "auxiliary", 0)
    return {"status": "pass", "all_40_parameter_errors": errors,
            "independent_loss": float(loss.detach()), "actual_loss": row["weighted_loss"],
            "parent_unchanged": True, "warmup_gradients_zero": True}


def exercise(model, optimizer, ema, inputs, *, step, stop_after=None):
    before = fingerprint(model, optimizer, ema)
    groups, gradients, optimizer_calls, stop = [], {}, [], False

    def check_continue():
        if stop:
            raise RequestedStop("Requested stop before committing grouped update")

    def after_group(group, row):
        nonlocal stop
        require(fingerprint(model, optimizer, ema) == before, "Endpoint advanced before both groups completed")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0
                    for p in model.parameters()), "A group did not reach all 40 parameters")
        if group == "ordinary":
            gradients.update({name: p.grad.detach().clone() for name, p in model.named_parameters()})
        else:
            require(all(float((p.grad - gradients[name]).norm()) > 0 for name, p in model.named_parameters()),
                    "Auxiliary group did not add to every parameter gradient")
            require(row["view_contribution_multipliers"] == [1., .25], "Restart lost the selected view weights")
            gradients.clear()
        groups.append(group); stop = group == stop_after
        print(json.dumps({"event": "weighted_restart_progress", "step": step, "group": group,
                          "stop_after": stop_after, "weighted_loss": row["weighted_loss"]}), flush=True)

    hook = optimizer.register_step_post_hook(lambda *args: optimizer_calls.append(1))
    try:
        with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clips, \
                patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zeroes:
            try:
                result = grouped_update(model, optimizer, ema, *inputs, step=step, warmup_samples=512,
                                        check_continue=check_continue, after_group=after_group)
            except RequestedStop:
                require(stop_after is not None and fingerprint(model, optimizer, ema) == before
                        and not optimizer_calls and clips.call_count == 0 and zeroes.call_count == 1,
                        "Interrupted accumulation changed an optimizer endpoint")
                require(groups == (["ordinary"] if stop_after == "ordinary" else ["ordinary", "auxiliary"]),
                        "Wrong interruption boundary")
                return {"stop_after": stop_after, "groups_completed": groups,
                        "weights_adam_ema_unchanged": True, "optimizer_calls": 0, "clip_calls": 0}
            require(stop_after is None and groups == ["ordinary", "auxiliary"] and optimizer_calls == [1]
                    and clips.call_count == zeroes.call_count == 1 and ema.updates == before["ema_updates"] + 1,
                    "Weighted update was not committed exactly once")
            require(result["raw_model_state_sha256"] != before["model"]
                    and result["ema_parameters_sha256"] != before["ema"]
                    and len(result["groups"]["ordinary"]["microbatches"]) == 4
                    and len(result["groups"]["auxiliary"]["microbatches"]) == 1,
                    "Weighted update or microbatch geometry differs")
            result.update(observed_optimizer_calls=len(optimizer_calls), observed_clip_calls=clips.call_count,
                          observed_zero_grad_calls=zeroes.call_count,
                          both_group_gradients_reach_all_40_parameters=True,
                          endpoint_unchanged_at_both_group_boundaries=True)
            return result
    finally:
        hook.remove()


def main():
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.check_latency58_weighted_vocal_device import compare_group_gradients, check_restart, check_gpu
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    out = PHASE / "weighted-vocal-quarter-cpu-001"
    require(not out.exists(), "Preserve previous weighted-loss qualifications")
    decision_path = PHASE / "grouped-gradient-probe-training-review-001/next-experiment-decision.json"
    decision = read(decision_path)
    require(sha(decision_path) == "f258eb12e90c5212dae95ab7cb96e3eeee750415a01491bdf291c8b5d631d33e"
            and decision["scientific_change"]["view_contribution_multipliers"] == list(VIEW_WEIGHTS),
            "Selected scientific change differs")
    verify_inputs(decision)
    source_path = PHASE / "branch-grouped-vocal-013/plan.json"
    source = read(source_path)
    require(source["parent_checkpoint"] == decision["parent_checkpoint"]
            and source["config"] == decision["proposed_training_config"], "Controlled reference trial changed")
    bindings = {p: digest for p, digest in source["source_bindings"].items() if Path(p).suffix == ".py"}
    bindings.update(decision["source_bindings"])
    paths = [Path(__file__).resolve(), source_path, decision_path,
             ROOT / "research/direct/latency58_weighted_vocal_auxiliary.py",
             ROOT / "research/direct/latency58_weighted_vocal_canonical.py",
             ROOT / "research/direct/check_latency58_weighted_vocal_device.py",
             ROOT / "research/direct/check_latency58_grouped_gradient_probe.py",
             ROOT / "research/direct/latency58_grouped_gradient_probe.py"]
    bindings.update({str(p): sha(p) for p in paths})
    bindings[source["parent_checkpoint"]["path"]] = source["parent_checkpoint"]["sha256"]
    budget = budget_snapshot(source["storage_budget"])
    require(budget["headroom_bytes"] > 2_000_000, "Reserve scalar qualification artifacts")
    plan = {"schema": "latency58-quarter-vocal-cpu-qualification-v1", "source_bindings": bindings,
            "fixture_checkpoint": source["parent_checkpoint"], "fixture_model_state_sha256": source["parent_model_state_sha256"],
            "objective_policy": objective_policy(), "accumulation_policy": accumulation_policy(),
            "synthetic_seed": 202611016, "warmup_samples": 512, "scored_samples": 44160,
            "restart_synthetic_seed": 202610309,
            "scalar_reference_scored_samples": 88320, "maximum_observed_rss_bytes": 18_000_000_000,
            "gradient_absolute_tolerance": 1e-7, "gradient_relative_tolerance": 1e-4,
            "gradient_relative_l2_tolerance": 5e-5, "scalar_absolute_tolerance": 3e-6,
            "storage_budget": source["storage_budget"], "budget_before": budget,
            "gpu_used": False, "training_audio_decoded": False, "validation_audio_decoded": False}
    verify_inputs(plan); out.mkdir(); write(out / "plan.json", plan)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    began = time.monotonic()

    def progress(phase, group, offset):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        require(rss < plan["maximum_observed_rss_bytes"], "Weighted CPU qualification exceeded its memory bound")
        print(json.dumps({"event": "weighted_vocal_cpu_progress", "phase": phase, "group": group,
                          "offset": offset, "elapsed_seconds": time.monotonic() - began, "peak_rss_bytes": rss}), flush=True)

    rng = torch.get_rng_state().clone()
    scalar = scalar_checks(progress); write(out / "scalar-reference.json", scalar)
    model, payload = load_model(plan["fixture_checkpoint"]); del payload
    require(state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"], "Fixture parent differs")
    model.train().requires_grad_(True); model.training_precision = "fp32"
    try:
        check_gpu(model, source)
    except RuntimeError as error:
        require(str(error) == "Require the selected BF16 parent before updates" and not torch.cuda.is_initialized(),
                "Unexpected GPU-entry rejection")
    else:
        raise RuntimeError("CPU fixture accepted at GPU resource entry")
    generator = torch.Generator().manual_seed(plan["synthetic_seed"])
    truth = .02 * torch.randn(16, 4, 2, 512 + 44160, generator=generator)
    truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
    neural = neural_scalar_reference(model, truth.sum(1), truth, progress=progress)
    write(out / "neural-scalar-reference.json", neural); gc.collect()
    gradients = compare_group_gradients(model, truth.sum(1), truth, warmup_samples=512, progress=progress)
    require(all(row["bitwise_equal"] for row in gradients["all_40_gradients"].values()),
            "Canonical weighted gradients did not match their reference bit-for-bit")
    write(out / "gradients.json", gradients); del truth; gc.collect()
    restart = check_restart(model, source, progress=progress); write(out / "restart.json", restart)
    require(state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"]
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(p.grad is None for p in model.parameters()), "CPU qualification changed its parent or RNG")
    verify_inputs(plan)
    result = {"schema": plan["schema"], "status": "pass", "plan_sha256": sha(out / "plan.json"),
              "source_bindings_unchanged": True, "objective_policy": objective_policy(),
              "scalar_reference_sha256": sha(out / "scalar-reference.json"),
              "neural_scalar_reference_sha256": sha(out / "neural-scalar-reference.json"),
              "gradients_sha256": sha(out / "gradients.json"), "restart_sha256": sha(out / "restart.json"),
              "independent_scalar_and_neural_references_pass": True,
              "ordinary_all_40_gradients_bit_exact": True, "canonical_combined_all_40_gradients_bit_exact": True,
              "interrupted_update_preserves_raw_adam_ema": True,
              "third_update_raw_adam_ema_and_accounting_bit_exact": restart["third_update_raw_adam_ema_and_accounting_bit_exact"],
              "parent_and_rng_unchanged": True, "gpu_used": False, "gpu_execution_qualified": False,
              "checkpoint_files_written": False, "quality_measured": False,
              "elapsed_seconds": time.monotonic() - began,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "completed_utc": datetime.now(timezone.utc).isoformat(), "budget_after": budget_snapshot(source["storage_budget"])}
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "peak_rss_bytes", "gpu_used")}), flush=True)


if __name__ == "__main__":
    main()
