"""Compare grouped model gradients with an independent whole-batch output VJP."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def recorded_batch(snapshot, expected):
    import torch
    from research.direct import check_latency58_branch_long_context_data as original
    from research.direct.latency58_long_context_data import CROP_SAMPLES, EXPANDED_SAMPLES, LongContextCropDataset
    selection = original.selection_contract()
    require(selection == snapshot["selection"], "Training selection differs")
    config = read(original.PRODUCTION / "full_config.json")
    _, tracks, _, _ = original.production.load_corpus_manifest(original.PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=selection["source_manifest_sha256"], config=config)
    tracks = original.select_tracks(tracks, selection)
    kwargs = dict(root_weights=original.ROOT_WEIGHTS, seed=config["seed"],
        vocal_active_probability=config["sampling"]["vocal_active_probability"],
        final_sample_index=snapshot["stop_sample_index"])
    dataset = LongContextCropDataset(original.production.CounterAddressedCropDataset(tracks, crop_samples=CROP_SAMPLES, **kwargs),
        original.production.CounterAddressedCropDataset(tracks, crop_samples=EXPANDED_SAMPLES, **kwargs),
        seed=snapshot["augmentation_seed"])
    first = snapshot["first_sample_index"]
    crops = [dataset[index] for index in range(first, first + 16)]
    mixture, targets = (torch.stack([pair[index] for pair in crops]) for index in range(2))
    require(original.audio_sha(mixture, targets) == expected["input_sha256"], "Recorded source batch differs")
    mixture, targets, _, _ = original.augment(mixture, targets, seed=snapshot["augmentation_seed"], first_sample_index=first)
    require(original.audio_sha(mixture, targets) == expected["after_remix_sha256"], "Augmented ordinary batch differs")
    return mixture, targets


def progress(phase, group, index, count):
    print(json.dumps({"event": "gradient_check_progress", "phase": phase,
                      "group": group, "completed_examples": index + 1, "group_examples": count}), flush=True)


def capture_outputs(model, inputs, warmup):
    import torch
    from research.direct.latency58_branch_memory_context import render_scored_context
    result = {}
    for group, (mixture, _) in inputs.items():
        raw, deployed = [], []
        for index in range(mixture.shape[0]):
            output = render_scored_context(model, mixture[index:index + 1], warmup_samples=warmup, carry_state=True)
            raw.append(output.raw.detach().clone())
            deployed.append(output.deployed.detach().clone())
            require(output.initial_state_detached and output.flush_hops == 1, "Warmup/flush contract changed")
            del output
            if (index + 1) % 4 == 0 or index + 1 == mixture.shape[0]:
                progress("capture", group, index, mixture.shape[0])
        result[group] = (torch.cat(raw).requires_grad_(), torch.cat(deployed).requires_grad_())
    return result


def accumulate(model, inputs, outputs, groups, reference_vjps, *, warmup, reference):
    import torch
    from research.direct.latency58_grouped_vocal_auxiliary import contribution
    from research.direct.latency58_branch_memory_context import render_scored_context
    model.zero_grad(set_to_none=True)
    totals, counts = {}, {}
    for group, (mixture, targets) in inputs.items():
        total = 0.
        active = torch.zeros(4, dtype=torch.int64)
        absent = torch.zeros(4, dtype=torch.int64)
        for index in range(mixture.shape[0]):
            audio = mixture[index:index + 1].clone().requires_grad_()
            result = render_scored_context(model, audio, warmup_samples=warmup, carry_state=True)
            expected_raw, expected_deployed = outputs[group]
            require(torch.equal(result.raw, expected_raw[index:index + 1])
                    and torch.equal(result.deployed, expected_deployed[index:index + 1]),
                    "Gradient passes evaluated different model outputs")
            if reference:
                raw_vjp, deployed_vjp = reference_vjps[group]
                torch.autograd.backward((result.raw, result.deployed),
                    (raw_vjp[index:index + 1], deployed_vjp[index:index + 1]))
            else:
                value, terms = contribution(group, result.raw, result.deployed,
                    targets[index:index + 1, ..., warmup:], mixture[index:index + 1, ..., warmup:], groups)
                value.backward()
                total += float(value.detach())
                active += terms.active_window_counts
                absent += terms.absent_window_counts
                del value, terms
            require(audio.grad is not None and torch.count_nonzero(audio.grad[..., :warmup]) == 0
                    and torch.count_nonzero(audio.grad[..., warmup:]) > 0
                    and bool(torch.isfinite(audio.grad).all()), "Detached warmup input gradient changed")
            del audio, result
            if (index + 1) % 4 == 0 or index + 1 == mixture.shape[0]:
                progress("independent_vjp" if reference else "grouped_loss", group, index, mixture.shape[0])
        totals[group] = total
        counts[group] = {"active": active.tolist(), "absent": absent.tolist()}
    gradients = {}
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                and float(parameter.grad.norm()) > 0, "Missing, zero or nonfinite gradient: " + name)
        gradients[name] = parameter.grad.detach().clone()
    require(len(gradients) == 40, "Wrong trained parameter inventory")
    model.zero_grad(set_to_none=True)
    gc.collect()
    return gradients, totals, counts


def main():
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_branch_sdr_blend import objective as whole_objective
    from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups, policy, AUXILIARY_WEIGHT
    from research.direct.latency58_long_context_data import WARMUP_SAMPLES, SCORED_SAMPLES
    from research.direct.check_latency58_branch_long_context_data import audio_sha
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    states = PHASE / "grouped-vocal-auxiliary-states-001"
    old_plan, old_result, execution = (read(states / name) for name in ("plan.json", "result.json", "execution.json"))
    require(old_result["status"] == "pass" and old_result["source_bindings_unchanged"]
            and old_result["plan_sha256"] == sha(states / "plan.json")
            and execution["actual_exit_code"] == execution["actual_enclosing_exit_code"] == 0
            and execution["source_bindings_unchanged"] and not execution["timed_out"],
            "Recorded state qualification is incomplete")
    bindings = dict(old_plan["source_bindings"])
    paths = [Path(__file__).resolve()]
    paths.extend(states / name for name in ("plan.json", "result.json", "execution.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_grouped_vocal_auxiliary.py", "latency58_branch_sdr_blend.py",
        "latency58_direct_sdr.py", "latency58_wave_spectral.py", "latency58_logical_batch_loss.py"))
    bindings.update({str(path): sha(path) for path in paths})
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "grouped-vocal-auxiliary-gradients-001"
    require(not out.exists(), "Preserve previous parameter-gradient checks")
    budget_plan = read(PHASE / "branch-gru-int8-post-pr-storage-001.json")
    plan = {"schema": "latency58-grouped-vocal-whole-objective-vjp-cpu-v1", "source_bindings": bindings,
        "fixture_checkpoint": old_plan["fixture_checkpoint"], "fixture_model_state_sha256": old_plan["fixture_model_state_sha256"],
        "policy": policy(), "ordinary_examples": 16, "auxiliary_examples": 2, "microbatch_size": 1,
        "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
        "reference": "Original whole-B16 objective plus 0.1 original whole-B2 objective in independent output coordinates; propagate those derivatives through replayed model outputs",
        "gradient_absolute_tolerance": 1e-7, "gradient_relative_tolerance": 1e-4,
        "gradient_relative_l2_tolerance": 5e-5, "loss_absolute_tolerance": 3e-6,
        "precision": "CPU FP32 training arithmetic", "training_updates": 0,
        "future_parent_selected": False, "production_recipe_selected": False, "gpu_used": False,
        "budget_before": budget_snapshot(budget_plan)}
    out.mkdir(); write(out / "plan.json", plan)
    began = time.monotonic()
    model, _ = load_model(plan["fixture_checkpoint"])
    require(state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"], "Model fixture differs")
    model.train().requires_grad_(True); model.training_precision = "fp32"
    rng = torch.get_rng_state().clone()
    snapshot = read(PHASE / "branch-long-context-data-001/inputs.json")
    expected = read(PHASE / "grouped-vocal-auxiliary-data-001/result.json")["batches"][0]
    mixture, targets = recorded_batch(snapshot, expected)
    auxiliary_mix, auxiliary_targets = source_views(mixture, targets)
    require(audio_sha(auxiliary_mix, auxiliary_targets) == expected["auxiliary_full_context_sha256"],
            "Source-view data differs from qualified recorded inputs")
    inputs = {"ordinary": (mixture, targets), "auxiliary": (auxiliary_mix, auxiliary_targets)}
    groups = prepare_groups(targets[..., WARMUP_SAMPLES:], auxiliary_targets[..., WARMUP_SAMPLES:])
    outputs = capture_outputs(model, inputs, WARMUP_SAMPLES)
    values, vjps = {}, {}
    for group, (raw, deployed) in outputs.items():
        audio, truth = inputs[group]
        weight = 1. if group == "ordinary" else AUXILIARY_WEIGHT
        value = weight * whole_objective(raw, deployed, truth[..., WARMUP_SAMPLES:], audio[..., WARMUP_SAMPLES:]).total
        values[group] = float(value.detach())
        vjps[group] = tuple(v.detach() for v in torch.autograd.grad(value, (raw, deployed)))
        outputs[group] = (raw.detach(), deployed.detach())
        del value
    candidate, totals, counts = accumulate(model, inputs, outputs, groups, vjps, warmup=WARMUP_SAMPLES, reference=False)
    reference, _, _ = accumulate(model, inputs, outputs, groups, vjps, warmup=WARMUP_SAMPLES, reference=True)
    rows = {}
    for name, expected_gradient in reference.items():
        actual = candidate[name]
        relative_l2 = float((actual - expected_gradient).double().norm() / expected_gradient.double().norm())
        rows[name] = {"maximum_absolute_error": float((actual - expected_gradient).abs().max()),
                      "reference_norm": float(expected_gradient.norm()), "relative_l2_error": relative_l2}
        require(torch.allclose(actual, expected_gradient, atol=plan["gradient_absolute_tolerance"],
                               rtol=plan["gradient_relative_tolerance"])
                and relative_l2 < plan["gradient_relative_l2_tolerance"], "Whole-objective parameter gradient differs: " + name)
    for group in inputs:
        reduction = getattr(groups, group)
        require(abs(totals[group] - values[group]) < plan["loss_absolute_tolerance"]
                and counts[group] == {"active": reduction.active.tolist(), "absent": reduction.absent.tolist()},
                "Grouped loss scalar or eligibility counts differ")
    require(audio_sha(mixture, targets) == expected["after_remix_sha256"]
            and state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"]
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "Gradient check changed weights, ordinary data, RNG or initialized CUDA")
    verify_inputs(plan)
    result = {"schema": plan["schema"], "status": "pass", "observed_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(out / "plan.json"), "source_bindings_unchanged": True,
        "all_40_parameter_gradients": rows, "independent_whole_group_losses": values, "accumulated_group_losses": totals,
        "activity_counts": counts, "warmup_input_gradients_zero": True, "scored_input_gradients_exercised": True,
        "model_weights_data_and_rng_unchanged": True, "training_updates": 0, "gpu_used": False,
        "checkpoint_files_written": False, "validation_audio_decoded": False,
        "future_parent_selected": False, "production_recipe_selected": False, "quality_measured": False,
        "elapsed_seconds": time.monotonic() - began,
        "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "budget_after": budget_snapshot(budget_plan),
        "limitations": "One recorded batch and the retained fixture checkpoint, with CPU FP32 microbatch one. "
                       "The eventual parent still needs restart and GPU BF16/resource qualification, followed by actual saved-checkpoint quality evaluation."}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "parameters_checked": len(rows),
        "maximum_gradient_absolute_error": max(row["maximum_absolute_error"] for row in rows.values()),
        "maximum_gradient_relative_l2_error": max(row["relative_l2_error"] for row in rows.values())}), flush=True)


if __name__ == "__main__":
    main()
