"""Qualify independent source-view warmup on recorded training audio and a saved model."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def state_errors(reference, candidate):
    from research.direct.latency58 import PUBLIC_FUSION_SCALE
    from research.direct.latency58_branch_memory import BranchMemoryState
    import torch
    require(type(reference) is type(candidate) is BranchMemoryState, "Wrong state family")
    rows = {}
    for index, (name, a, b) in enumerate(zip(BranchMemoryState._fields, reference, candidate, strict=True)):
        require(a.shape == b.shape and a.dtype == b.dtype == torch.float32
                and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
                and not a.requires_grad and not b.requires_grad
                and a.grad_fn is None and b.grad_fn is None, "Invalid or live warmup state")
        scale = PUBLIC_FUSION_SCALE if index in (1, 6, 7) else 1.
        rows[name] = {"maximum_public_error": float((a - b).abs().max()),
                      "maximum_unscaled_error": float((a - b).abs().max()) / scale,
                      "bit_exact": torch.equal(a.view(torch.int32), b.view(torch.int32))}
    return rows


def slice_state(state, index):
    return type(state)(*(v[:, index:index + 1].clone() if i in (1, 6, 7)
                         else v[index:index + 1].clone() for i, v in enumerate(state)))


def render_suffix(model, suffix, state):
    from torch.nn import functional as F
    import torch
    count = suffix.shape[-1]
    result = model.render(F.pad(suffix, (0, (-count) % 128 + 128)), state)
    values = {key: getattr(result, key)[..., 128:128 + count].clone()
              for key in ("raw", "deployed", "delayed_mixture")}
    require(torch.equal(values["delayed_mixture"], suffix)
            and all(bool(torch.isfinite(value).all()) for value in values.values()),
            "Scored source view is nonfinite or physically misaligned")
    return values


def compare_audio(reference, candidate):
    return {key: float((reference[key] - candidate[key]).abs().max()) for key in reference}


def check_batch(model, mixture, targets, expected, warmup):
    import torch
    from research.direct.latency58_grouped_vocal_auxiliary import source_views
    from research.direct.check_latency58_branch_long_context_data import audio_sha
    from research.direct.latency58_branch_memory_context import render_scored_context
    fingerprint, rng = audio_sha(mixture, targets), torch.get_rng_state().clone()
    require(fingerprint == expected["after_remix_sha256"], "Ordinary recorded batch changed")
    auxiliary_mix, auxiliary = source_views(mixture, targets)
    require(audio_sha(auxiliary_mix, auxiliary) == expected["auxiliary_full_context_sha256"],
            "Previously qualified source-view bytes changed")
    prefix, suffix = auxiliary_mix[..., :warmup], auxiliary_mix[..., warmup:]
    with torch.no_grad():
        # The full prefix decodes every received frame. The optimized path only
        # decodes the final warmup frame. Neither introduces an intermediate flush.
        reference_state = model.render(prefix).state.detached()
        fast_state = model.warm_state(prefix).detached()
        optimized_errors = state_errors(reference_state, fast_state)
        require(max(row["maximum_unscaled_error"] for row in optimized_errors.values()) < 5e-4,
                "Optimized source-view warmup differs from full-prefix reference")
        reference_audio = render_suffix(model, suffix, reference_state)
        current = render_scored_context(model, auxiliary_mix, warmup_samples=warmup, carry_state=True)
        fast_audio = {"raw": current.raw, "deployed": current.deployed,
                      "delayed_mixture": current.physical_mixture}
        optimized_audio = compare_audio(reference_audio, fast_audio)
        require(max(optimized_audio.values()) < 1e-4 and current.flush_hops == 1
                and current.initial_state_detached, "Fresh source-view rendering differs")
        closure = float((current.deployed.sum(1) - suffix).abs().max())
        require(torch.allclose(current.deployed.sum(1), suffix, atol=1e-6, rtol=1e-6),
                "Source-view reconstruction changed")

        # Each lane also receives an independent reset, prefix and suffix render.
        lanes = []
        for index in range(2):
            independent = model.render(prefix[index:index + 1]).state.detached()
            lane_state_errors = state_errors(slice_state(reference_state, index), independent)
            lane_audio = render_suffix(model, suffix[index:index + 1], independent)
            lane_audio_errors = compare_audio({k: v[index:index + 1] for k, v in reference_audio.items()}, lane_audio)
            require(max(row["maximum_unscaled_error"] for row in lane_state_errors.values()) < 5e-4
                    and max(lane_audio_errors.values()) < 1e-4, "Batch lanes share source-view history")
            lanes.append({"view": ("instrumental", "vocals_only")[index],
                          "state_errors": lane_state_errors, "audio_errors": lane_audio_errors})

        ordinary_prefix = mixture[[14, 15], :, :warmup]
        ordinary_state = model.warm_state(ordinary_prefix).detached()
        wrong_audio = render_suffix(model, suffix, ordinary_state)
        wrong_state_errors = state_errors(reference_state, ordinary_state)
        controls = []
        for index in range(2):
            prefix_changed = not torch.equal(prefix[index], ordinary_prefix[index])
            error = float((wrong_audio["deployed"][index] - reference_audio["deployed"][index]).abs().max())
            controls.append({"view": ("instrumental", "vocals_only")[index],
                "removed_source_changes_warmup": prefix_changed,
                "removed_source_prefix_rms": float((prefix[index] - ordinary_prefix[index]).square().mean().sqrt()),
                "maximum_output_error_from_wrong_ordinary_state": error})
        require(any(c["removed_source_changes_warmup"]
                    and c["maximum_output_error_from_wrong_ordinary_state"] > 1e-4 for c in controls),
                "Wrong-history negative control did not exercise state isolation")

        # An intervening render of ordinary audio must leave no hidden state.
        repeated_state = model.warm_state(prefix).detached()
        repeat_errors = state_errors(fast_state, repeated_state)
        repeated_audio = render_suffix(model, suffix, repeated_state)
        require(all(row["bit_exact"] for row in repeat_errors.values())
                and all(torch.equal(a.view(torch.int32), repeated_audio[k].view(torch.int32))
                        for k, a in fast_audio.items()),
                "Interleaving ordinary input changed reset replay")
    require(audio_sha(mixture, targets) == fingerprint and torch.equal(rng, torch.get_rng_state()),
            "Warmup qualification changed ordinary audio or RNG")
    return {"first_sample_index": expected["first_index"], "ordinary_sha256": fingerprint,
        "auxiliary_sha256": audio_sha(auxiliary_mix, auxiliary),
        "optimized_warmup_errors": optimized_errors, "optimized_scored_audio_errors": optimized_audio,
        "independent_lanes": lanes, "wrong_ordinary_history_state_differences": wrong_state_errors,
        "wrong_ordinary_history_controls": controls, "interleaved_reset_replay_bit_exact": True,
        "mixture_reconstruction_max_abs_error": closure, "ordinary_data_and_rng_unchanged": True}


def main():
    import torch
    from research.direct import check_latency58_branch_long_context_data as original
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_grouped_vocal_auxiliary import policy
    from research.direct.latency58_long_context_data import (
        WARMUP_SAMPLES, SCORED_SAMPLES, CROP_SAMPLES, EXPANDED_SAMPLES, LongContextCropDataset)
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CUDA-hidden CPU1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    old = PHASE / "grouped-vocal-auxiliary-data-001"
    old_plan, old_result, old_execution = (read(old / name) for name in ("plan.json", "result.json", "execution.json"))
    require(old_result["status"] == "pass" and old_result["source_bindings_unchanged"]
            and old_result["plan_sha256"] == sha(old / "plan.json") and old_execution["actual_exit_code"] == 0
            and old_execution["source_bindings_unchanged"], "Complete recorded-data qualification first")
    parent_plan_path = PHASE / "branch-long-context-006/plan.json"
    parent_plan = read(parent_plan_path)
    bindings = {**parent_plan["source_bindings"], **old_plan["source_bindings"]}
    paths = [Path(__file__).resolve(), parent_plan_path]
    paths.extend(old / name for name in ("plan.json", "result.json", "execution.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_branch_memory.py", "latency58_branch_memory_context.py", "latency58_sdr_context.py",
        "latency58_branch_memory_checkpoint.py", "check_latency58_branch_memory.py"))
    bindings.update({str(path): sha(path) for path in paths})
    verify_inputs({"source_bindings": bindings})
    out = PHASE / "grouped-vocal-auxiliary-states-001"
    require(not out.exists(), "Preserve previous state checks")
    budget_plan = read(PHASE / "branch-gru-int8-post-package-storage-001.json")
    plan = {"schema": "latency58-grouped-vocal-recorded-states-cpu-v1", "source_bindings": bindings,
        "fixture_checkpoint": parent_plan["parent_checkpoint"],
        "fixture_model_state_sha256": parent_plan["parent_model_state_sha256"], "policy": policy(),
        "recorded_batch_indices": [0, 1], "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
        "state_tolerance_unscaled": 5e-4, "waveform_tolerance": 1e-4,
        "tolerance_source": "Existing branch-memory CPU partition/state checks; fixed before rendering",
        "precision": "CPU FP32 inference", "future_parent_selected": False, "production_recipe_selected": False,
        "gpu_used": False, "budget_before": budget_snapshot(budget_plan)}
    out.mkdir()
    write(out / "plan.json", plan)
    began = time.monotonic()
    model, _ = load_model(plan["fixture_checkpoint"])
    require(state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"], "Saved fixture model differs")
    rng = torch.get_rng_state().clone()
    snapshot = read(PHASE / "branch-long-context-data-001/inputs.json")
    selection = original.selection_contract()
    require(selection == snapshot["selection"], "Training-corpus selection changed")
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
    rows = []
    for batch in plan["recorded_batch_indices"]:
        expected = old_result["batches"][batch]
        first = snapshot["first_sample_index"] + 16 * batch
        crops = [dataset[i] for i in range(first, first + 16)]
        mixture, targets = (torch.stack([crop[i] for crop in crops]) for i in range(2))
        require(original.audio_sha(mixture, targets) == expected["input_sha256"], "Recorded input hash changed")
        mixture, targets, _, _ = original.augment(mixture, targets, seed=snapshot["augmentation_seed"], first_sample_index=first)
        row = check_batch(model, mixture, targets, expected, WARMUP_SAMPLES)
        rows.append(row)
        write(out / ("batch-%02d.json" % batch), row)
        print(json.dumps({"event": "source_view_states_checked", "first_sample_index": first,
                          "optimized_audio_errors": row["optimized_scored_audio_errors"],
                          "negative_controls": row["wrong_ordinary_history_controls"]}), flush=True)
    require(len(rows) == 2 and state_sha256(model.state_dict()) == plan["fixture_model_state_sha256"]
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(p.grad is None for p in model.parameters()), "Qualification mutated model or RNG, or used CUDA")
    verify_inputs(plan)
    result = {"schema": "latency58-grouped-vocal-recorded-states-cpu-v1", "status": "pass",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(out / "plan.json"),
        "source_bindings_unchanged": True, "fixture_model_state_sha256": plan["fixture_model_state_sha256"],
        "batches": rows, "recorded_training_crops": 32, "source_views_rendered": 4,
        "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
        "independent_per_view_warmup_qualified_on_fixture": True, "model_and_rng_unchanged": True,
        "training_updates": 0, "checkpoint_files_written": False, "validation_audio_decoded": False,
        "model_parameter_gradients_qualified": False, "production_recipe_selected": False,
        "future_parent_selected": False, "gpu_used": False, "quality_measured": False,
        "elapsed_seconds": time.monotonic() - began, "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "budget_after": budget_snapshot(budget_plan),
        "limitations": "CPU FP32 state transport on the retained checkpoint and recorded training audio. "
                       "The eventual parent needs actual objective/parameter-gradient, restart and monitored GPU resource qualification. "
                       "No listening, quality, M4 timing or new training result is established."}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "source_views_rendered": 4,
                      "independent_per_view_warmup_qualified_on_fixture": True,
                      "model_parameter_gradients_qualified": False}), flush=True)


if __name__ == "__main__":
    main()
