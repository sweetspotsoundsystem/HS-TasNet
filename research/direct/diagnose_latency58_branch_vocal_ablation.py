"""Measure trained branch memories on fixed training mixes and vocal-removed views.

These CPU ablations use the saved parent, never the live training model. They
measure local behavior on 64 training examples, not held-out checkpoint quality.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs
from research.direct.diagnose_latency58_quadrature_continuation_learning import selected_rows

STEPS = [1, 25, 50, 100]
VARIANTS = ("saved_memories", "without_spectral_memory", "without_waveform_memory", "without_both_memories")
VIEWS = ("all_sources", "vocals_removed")


def main():
    import torch
    from torch.nn import functional as F
    from research.direct.latency58 import SOURCE_ORDER
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.latency58_remix_augmentation import augment
    from research.direct.latency58_branch_sdr_blend import objective
    from research.direct.latency58_direct_sdr import WINDOW, ACTIVITY_POWER
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU inference with one numerical thread")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training = PHASE / "branch-sdr-blend-001"
    source_path = training / "plan.json"
    source = read(source_path)
    verify_inputs(source)
    config = source["config"]
    require(source["parent_kind"] == "saved_trained_branch_memory" and source["parent_training_updates"] == 30250
            and config["batch_size"] == config["microbatch_size"] == 16
            and source["warmup_samples"] == 88064 and source["scored_samples"] == 44160
            and source["artifact_cap_bytes"] == 90_000_000_000, "Use the declared trained parent and geometry")
    require_space(source, 450_000_000)
    journal = training / "production-run/metrics.jsonl"
    selected = selected_rows(journal, STEPS)
    model, _ = load_model(source["parent_checkpoint"])
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == source["parent_model_state_sha256"] and not model.training,
            "Use the unchanged saved parent in evaluation mode")
    projections = (model.spec_memory_output.weight, model.waveform_memory_output.weight)
    original = [value.detach().clone() for value in projections]
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    corpus = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=corpus)
    require(manifest_sha == source["manifest_sha256"], "Training corpus changed")
    tracks = select_tracks(tracks, source["training_selection"])
    dataset = production.CounterAddressedCropDataset(tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"], vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=selected[-1]["next_sample_index"])
    out = PHASE / "branch-parent-vocal-ablation-001"
    require(not out.exists(), "Preserve existing diagnostics")
    out.mkdir()
    write(out / "selected-training-rows.json", selected)
    paths = [source_path, Path(__file__).resolve(), out / "selected-training-rows.json",
             ROOT / "research/direct/diagnose_latency58_quadrature_continuation_learning.py"]
    bindings = {**source["source_bindings"], **{str(path): sha(path) for path in paths}}
    plan = {"source_bindings": bindings, "training_steps": STEPS, "examples_per_batch": 16,
            "variants": VARIANTS, "views": VIEWS, "parent_checkpoint": source["parent_checkpoint"],
            "training_only": True, "validation_or_test_used": False, "checkpoint_quality_measured": False,
            "precision": "CPU FP32", "pending_training_reservation_bytes": 440_000_000,
            "selection": "Updates 1, 25, 50 and 100, fixed before reading their audio or ablation outcomes",
            "all_sources_view": "Sum the four augmented targets, so both synthetic views share the same source-defined mixture. Preserve and report the logged mixture/reference mismatch separately.",
            "vocal_removed_view": "Zero the vocal target for the entire warmup and scored crop, then sum the remaining targets into the physical mixture",
            "state_policy": "Recompute warmup per variant; six upstream states must match, while both synthesis tails may differ",
            "vocal_addition_response_scope": "Output difference between ordinary and vocal-removed training mixtures; includes nonlinear source interactions and is not a separated estimate of actual bleed"}
    write(out / "plan.json", plan)
    vocal = list(SOURCE_ORDER).index("vocals")
    warmup, scored = source["warmup_samples"], source["scored_samples"]

    def window_power(value):
        windows = value.shape[-1] // WINDOW
        return value[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW)).square().mean(dim=(-3, -1))

    def restore():
        for value, saved in zip(projections, original, strict=True):
            value.copy_(saved)

    began, rows = time.monotonic(), []
    with torch.no_grad(), (out / "progress.jsonl").open("x", buffering=1) as progress:
        try:
            for logged in selected:
                first = logged["first_sample_index"]
                require(first == config["data_start"] + (logged["step"] - 1) * 16, "Training index differs")
                inputs = [dataset[i] for i in range(first, first + 16)]
                mixture, truth = torch.stack([x for x, _ in inputs]), torch.stack([y for _, y in inputs])
                mixture, truth, changed, factors = augment(mixture, truth, seed=config["seed"], first_sample_index=first)
                digest = hashlib.sha256()
                for value in (mixture, truth, changed, factors):
                    digest.update(value.contiguous().numpy().tobytes())
                require(digest.hexdigest() == logged["augmented_inputs_sha256"], "Training audio hash differs")
                view_results, emitted = {}, {}
                for view in VIEWS:
                    targets = truth.clone()
                    if view == "vocals_removed":
                        targets[:, vocal] = 0
                    physical = targets.sum(1)
                    truth_score, physical_score = targets[..., warmup:], physical[..., warmup:]
                    reference_power = window_power(truth_score)
                    active = reference_power > ACTIVITY_POWER
                    require(view != "vocals_removed" or not bool(active[:, vocal].any()),
                            "Vocal-removed references must have no active vocal windows")
                    baseline_state, baseline_warm, baseline_output = None, None, None
                    variants = {}
                    for variant in VARIANTS:
                        restore()
                        if variant in ("without_spectral_memory", "without_both_memories"):
                            projections[0].zero_()
                        if variant in ("without_waveform_memory", "without_both_memories"):
                            projections[1].zero_()
                        state = model.warm_state(physical[..., :warmup]).detached()
                        state_digest = state_sha256(state._asdict())
                        rendered = model.render(F.pad(physical_score, (0, 128)), state)
                        require(state_sha256(state._asdict()) == state_digest
                                and all(bool(torch.isfinite(value).all()) for value in rendered.state),
                                "Rendering mutated its initial state or produced a nonfinite state")
                        require(torch.equal(rendered.delayed_mixture[..., 128:128 + scored], physical_score),
                                "Physical alignment differs")
                        raw, deployed = (value[..., 128:128 + scored] for value in (rendered.raw, rendered.deployed))
                        require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(deployed).all())
                                and float((deployed.sum(1) - physical_score).abs().max()) < 1e-6,
                                "Nonfinite output or mixture reconstruction failure")
                        if baseline_state is None:
                            baseline_state = tuple(value.clone() for value in rendered.state)
                            baseline_warm = tuple(value.clone() for value in state)
                            baseline_output = deployed.clone()
                        upstream = (0, 1, 4, 5, 6, 7)
                        require(all(torch.equal(state[i], baseline_warm[i])
                                    and torch.equal(rendered.state[i], baseline_state[i]) for i in upstream),
                                "Output-projection ablation changed an upstream recurrent state")
                        terms = objective(raw, deployed, truth_score, physical_score)
                        require(terms.active_window_counts.tolist() == active.sum(dim=(0, 2)).tolist()
                                and terms.absent_window_counts.tolist() == (~active).sum(dim=(0, 2)).tolist(),
                                "Diagnostic and objective activity windows disagree")
                        output_power = window_power(deployed)
                        absent = {}
                        for index, name in enumerate(SOURCE_ORDER):
                            eligible = ~active[:, index]
                            values = output_power[:, index][eligible]
                            absent[name] = {"windows": int(eligible.sum()),
                                "mean_window_output_dbfs": float((10 * torch.log10(values + 1e-12)).mean()) if values.numel() else None,
                                "mean_output_power": float(values.mean()) if values.numel() else None}
                        row = {"training_batch_sdr_db": -float(terms.negative_sdr_db),
                               "per_stem_training_sdr_db": dict(zip(SOURCE_ORDER, (-terms.per_stem_negative_sdr_db).tolist(), strict=True)),
                               "reconstruction_loss": float(terms.reconstruction_loss), "blended_loss": float(terms.total),
                               "mixture_relative_absence_penalty_db": float(terms.absence_db), "absent_output": absent,
                               "active_windows": terms.active_window_counts.tolist(), "absent_windows": terms.absent_window_counts.tolist(),
                               "max_output_abs_change": float((deployed - baseline_output).abs().max()),
                               "six_upstream_states_bit_exact": True,
                               "warm_synthesis_tail_max_abs_change": [float((state[i] - baseline_warm[i]).abs().max()) for i in (2, 3)],
                               "final_synthesis_tail_max_abs_change": [float((rendered.state[i] - baseline_state[i]).abs().max()) for i in (2, 3)]}
                        variants[variant] = row
                        emitted[(view, variant)] = deployed.clone()
                        print(json.dumps({"step": logged["step"], "view": view, "variant": variant,
                              "training_sdr_db": row["training_batch_sdr_db"],
                              "absent_vocal_output_dbfs": absent["vocals"]["mean_window_output_dbfs"]}), flush=True)
                        del rendered, raw, deployed, terms, state
                    view_results[view] = variants
                vocal_target = truth[:, vocal, :, warmup:][..., :WINDOW]
                vocal_energy = vocal_target.square().sum(dim=(1, 2))
                eligible = vocal_energy / (2 * WINDOW) > ACTIVITY_POWER
                responses = {}
                for variant in VARIANTS:
                    response = (emitted[("all_sources", variant)] - emitted[("vocals_removed", variant)])[..., :WINDOW]
                    require(float((response.sum(1) - vocal_target).abs().max()) < 2e-6,
                            "Vocal-addition responses lost mixture closure")
                    energy = response.square().sum(dim=(2, 3))
                    dot = (response * vocal_target[:, None]).sum(dim=(2, 3))
                    responses[variant] = {name: {
                        "active_vocal_examples": int(eligible.sum()),
                        "response_energy_db_relative_to_vocal": float((10 * torch.log10((energy[:, i][eligible] + 1e-12) / (vocal_energy[eligible] + 1e-12))).mean()) if bool(eligible.any()) else None,
                        "mean_projection_gain_on_vocal": float((dot[:, i][eligible] / vocal_energy[eligible]).mean()) if bool(eligible.any()) else None,
                    } for i, name in enumerate(SOURCE_ORDER)}
                restore()
                require(state_sha256(model.state_dict()) == fingerprint, "Diagnostic failed to restore the saved weights")
                item = {"step": logged["step"], "augmented_inputs_sha256": digest.hexdigest(), "views": view_results,
                        "logged_mixture_minus_source_sum_max_abs": float((mixture - truth.sum(1)).abs().max()),
                        "logged_mixture_minus_source_sum_rms": float((mixture - truth.sum(1)).square().mean().sqrt()),
                        "vocal_addition_response": responses, "elapsed_seconds": time.monotonic() - began}
                rows.append(item)
                progress.write(json.dumps(item, allow_nan=False) + "\n")
                del inputs, mixture, truth, changed, factors, emitted, view_results, targets, physical
        finally:
            restore()
    require(selected_rows(journal, STEPS) == selected and state_sha256(model.state_dict()) == fingerprint
            and all(parameter.grad is None for parameter in model.parameters()) and not torch.cuda.is_initialized(),
            "Journal, saved model, gradients or CPU scope changed")
    verify_inputs(plan)
    require_space(source, 445_000_000)
    write(out / "result.json", {"status": "pass", "source_bindings": bindings, "source_bindings_unchanged": True,
          "comparisons": rows, "all_four_augmented_batch_hashes_verified": True,
          "training_examples_covered_by_batch_hashes": 64, "all_six_upstream_states_unchanged_across_variants": True,
          "warmup_recomputed_for_every_variant": True, "saved_parent_parameters_and_gradients_unchanged": True,
          "all_projection_weights_restored": True, "training_only": True,
          "checkpoint_quality_measured": False, "gpu_used": False, "checkpoint_written": False,
          "elapsed_seconds": time.monotonic() - began,
          "limitation": "Four fixed training batches with temporary output-projection ablations and synthetic vocal removal. No variant is retrained or saved. Vocal-addition responses include nonlinear interactions. These results neither establish held-out quality nor select a deployment model."})
    print(json.dumps({"status": "pass", "batches": len(rows), "result_sha256": sha(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
