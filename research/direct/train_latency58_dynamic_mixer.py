"""Train a dynamic output head on frozen recurrent features from recorded training crops."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU1 with CUDA hidden")
    import torch
    from research.direct.latency58_direct_sdr_checkpoint import load_model as load_parent
    from research.direct.latency58_musdb_sdr_data import selection_contract, select_tracks
    from research.direct.latency58_sdr_context import render_scored_context
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_dynamic_mixer import (
        VERSION, Latency58DynamicMixer, frame_covariances, load_model, loss_from_frames,
        make_head, apply_correction, coefficients, mixing_scale, mixing_ramp,
    )
    from research.direct.fit_latency58_dynamic_mixer import fit_head
    from research.direct.check_latency58_dynamic_mixer import check
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = PHASE / "dynamic-mixer-training-001"
    require(not out.exists(), "Preserve previous calibration trials")
    budget = read(PHASE / "magnitude-sdr-001/plan.json")
    counted = require_space(budget, 10_000_000)
    parent_binding = read(PHASE / "c204-residual-model-001/checkpoint.json")
    parent, _ = load_parent(parent_binding)
    parent_state = state_sha256(parent.state_dict())
    contract = selection_contract()
    reference = PHASE / "c204-residual-model-full14-001/result.json"
    bindings = {**read(PHASE / "c204-residual-model-001/qualification.json")["source_bindings"]}
    for path in (Path(__file__).resolve(), ROOT / "research/direct/latency58_dynamic_mixer.py",
                 ROOT / "research/direct/evaluate_latency58_dynamic_mixer.py", ROOT / "research/direct/latency58_direct_sdr_checkpoint.py",
                 ROOT / "research/direct/fit_latency58_dynamic_mixer.py", ROOT / "research/direct/check_latency58_dynamic_mixer.py",
                 ROOT / "research/direct/latency58_musdb_sdr_data.py", ROOT / "research/direct/latency58_sdr_context.py",
                 PRODUCTION / "train_production.py", PRODUCTION / "full_config.json", PRODUCTION / "manifests/combined.manifest.json",
                 Path(parent_binding["path"]), reference):
        bindings[str(path)] = sha(path)
    plan = {"schema": VERSION, "parent_checkpoint": parent_binding, "parent_model_state_sha256": parent_state,
            "training_selection": contract, "fit_sample_indices": list(range(1_400_000, 1_400_512)),
            "probe_sample_indices": list(range(1_500_000, 1_500_128)), "data_seed": 60,
            "vocal_active_probability": .85, "crop_samples": 176384, "warmup_samples": 88064,
            "cpu_batch_size": 4, "optimization_steps": 2000, "maximum_lr": .001, "minimum_lr": .0001,
            "warmup_updates": 25, "fit_batch_size": 16, "training_seed": 20261002, "cache_in_memory_only": True,
            "diagonal_bounds": [0., 2.], "off_diagonal_bounds": [-.25, .25],
            "loss": "equal-stem exact 44100-sample SD-SDR + 0.5 relative absence energy + 0.01 mean dynamic coefficient displacement squared",
            "probe_minimum_improvement_db_before_full14": .05, "quality_coefficient_refitting": False,
            "fit_used_validation_or_test_audio": False, "probe_scope": "Different crops within the recorded training split; not unseen-track validation",
            "source_bindings": bindings, "output_directory": str(out)}
    verify_inputs(plan)
    out.mkdir()
    write(out / "plan.json", plan)
    write(out / "functional.json", check())
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    source_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_hash, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=contract["source_manifest_sha256"], config=source_config)
    require(manifest_hash == contract["source_manifest_sha256"] and source_config["seed"] == plan["data_seed"], "Corpus differs")
    tracks = select_tracks(tracks, contract)
    dataset = production.CounterAddressedCropDataset(tracks, root_weights={"musdb18hq_train": 1.}, seed=plan["data_seed"],
        crop_samples=plan["crop_samples"], vocal_active_probability=plan["vocal_active_probability"],
        final_sample_index=max(plan["probe_sample_indices"]) + 1)
    statistics = {}
    began = time.monotonic()
    with (out / "progress.jsonl").open("x", buffering=1) as journal, torch.inference_mode():
        for role in ("fit", "probe"):
            rows = []
            indices = plan[role + "_sample_indices"]
            for first in range(0, len(indices), 4):
                addresses = indices[first:first + 4]
                inputs = [dataset[index] for index in addresses]
                mixture = torch.stack([x for x, y in inputs])
                truth = torch.stack([y for x, y in inputs])
                digest = hashlib.sha256()
                for tensor in (mixture, truth):
                    digest.update(tensor.numpy().tobytes())
                captured = []
                handle = parent.fusion_branch.register_forward_hook(lambda module, inputs, output: captured.append(output[0]))
                try:
                    output = render_scored_context(parent, mixture, warmup_samples=plan["warmup_samples"], carry_state=True)
                finally:
                    handle.remove()
                require(len(captured) == 2 and captured[1].shape == (len(addresses), 691, 1000),
                        "Frozen recurrent features do not cover the previous and current scored hops")
                require(torch.equal(output.physical_mixture, mixture[..., plan["warmup_samples"]:]), "Physical training alignment differs")
                if role == "fit" and first == 0:
                    test_head = make_head()
                    test_head.weight.normal_(0, .002)
                    test_head.bias.normal_(0, .005)
                    test_model = Latency58DynamicMixer.from_parent(parent, test_head.state_dict())
                    actual = render_scored_context(test_model, mixture, warmup_samples=plan["warmup_samples"], carry_state=True)
                    cached = apply_correction(output.deployed, output.physical_mixture,
                        coefficients(test_head, captured[1], mixing_scale()), mixing_ramp())
                    difference = float((actual.deployed - cached).abs().max())
                    require(difference < 1e-6, "Cached recurrent features are shifted relative to the deployed audio")
                    write(out / "feature-alignment-check.json", {"status": "pass", "maximum_waveform_difference": difference,
                          "sample_indices": addresses, "input_sha256": digest.hexdigest(),
                          "scope": "Nonzero synthetic head on actual recorded training crops; no quality claim"})
                    del test_head, test_model, actual, cached
                rows.append({**frame_covariances(output.deployed, truth[..., plan["warmup_samples"]:], output.physical_mixture),
                             "features": captured[1].clone()})
                row = {"role": role, "sample_indices": addresses, "input_sha256": digest.hexdigest(),
                       "completed_examples": first + len(addresses), "elapsed_seconds": time.monotonic() - began}
                journal.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            statistics[role] = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    # Clone outside inference_mode so these constants can participate in autograd.
    statistics = {role: {key: value.clone() for key, value in rows.items()} for role, rows in statistics.items()}
    del rows, inputs, mixture, truth, output, captured
    # These derived caches are regenerated from the bound data and retained only
    # in RAM. Persist their identities without creating multi-gigabyte artifacts.
    cache_identity = {role: {name: {"shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": hashlib.sha256(memoryview(value.contiguous().numpy()).cast("B")).hexdigest()}
        for name, value in values.items()} for role, values in statistics.items()}
    write(out / "cache-digest.json", cache_identity)
    head, resume = fit_head(statistics["fit"]["features"],
        {k: v for k, v in statistics["fit"].items() if k != "features"}, plan, out / "training-metrics.jsonl")
    initial_head = make_head().eval().requires_grad_(False)
    summaries = {}
    for role, values in statistics.items():
        moments = {k: v for k, v in values.items() if k != "features"}
        with torch.no_grad():
            base_loss, base_stems, base_absence = loss_from_frames(initial_head, values["features"], moments)
            loss, stems, absence = loss_from_frames(head, values["features"], moments)
        summaries[role] = {"parent_full_sdr_db": float(base_stems.mean()), "candidate_full_sdr_db": float(stems.mean()),
                           "full_sdr_improvement_db": float((stems - base_stems).mean()),
                           "per_stem_parent_db": base_stems.tolist(), "per_stem_candidate_db": stems.tolist(),
                           "per_stem_delta_db": (stems - base_stems).tolist(), "parent_objective": float(base_loss),
                           "candidate_objective": float(loss), "parent_absence": float(base_absence), "candidate_absence": float(absence)}
    model = Latency58DynamicMixer.from_parent(parent, head.state_dict())
    model.provenance.update(dynamic_mixer_optimization_steps=plan["optimization_steps"], dynamic_mixer_fit_examples=512,
                            dynamic_mixer_probe_examples=128, dynamic_mixer_fit_used_validation_or_test_audio=False,
                            dynamic_mixer_plan_sha256=sha(out / "plan.json"), dynamic_mixer_inherited_neural_tensors_frozen=True)
    fingerprint = state_sha256(model.state_dict())
    payload = {"schema": VERSION, "parent_checkpoint": parent_binding, "head": dict(head.state_dict()),
               "model_state_sha256": fingerprint, "architecture": model.architecture_metadata, "provenance": model.provenance,
               "fit_used_validation_or_test_audio": False, "training_plan_sha256": sha(out / "plan.json")}
    with (out / "model.pt").open("xb") as stream:
        torch.save(payload, stream)
    with (out / "optimizer.pt").open("xb") as stream:
        torch.save({**resume, "model_state_sha256": fingerprint, "training_plan_sha256": sha(out / "plan.json")}, stream)
    checkpoint = {"path": str(out / "model.pt"), "sha256": sha(out / "model.pt")}
    restored, _ = load_model(checkpoint)
    with torch.inference_mode():
        test = torch.linspace(-.05, .05, 2 * 11 * 128).reshape(1, 2, -1)
        require(torch.equal(model.render(test).deployed, restored.render(test).deployed), "Saved calibration replay differs")
        state, outputs = None, []
        for chunk in test.split(128, -1):
            audio, state = restored.forward_chunk(chunk, state)
            outputs.append(audio)
        flush, _ = restored.flush(state)
        host = torch.cat([torch.zeros_like(outputs[0]), *outputs, flush], -1)
        closure = float((host[..., 256:256 + test.shape[-1]].sum(1) - test).abs().max())
        require(closure < 1e-6 and restored.algorithmic_latency_samples == 256, "Host physical delay differs")
    require(state_sha256(parent.state_dict()) == parent_state and not torch.cuda.is_initialized(), "Parent or CPU scope changed")
    verify_inputs(plan)
    write(out / "checkpoint-audit.json", {"status": "pass", "checkpoint": checkpoint, "model_state_sha256": fingerprint,
          "inherited_neural_tensors_unchanged": True, "exact_saved_replay": True, "host_delay_samples": 256,
          "closure_max_abs": closure, "source_bindings_unchanged": True,
          "optimizer_sha256": sha(out / "optimizer.pt"),
          "checkpoint_format": "trained output-head checkpoint with a SHA-bound preserved parent"})
    probe_pass = summaries["probe"]["full_sdr_improvement_db"] >= plan["probe_minimum_improvement_db_before_full14"]
    write(out / "fit-result.json", {"status": "complete", "head_parameters": sum(p.numel() for p in head.parameters()), "summaries": summaries,
          "probe_passed": probe_pass, "checkpoint": checkpoint, "elapsed_seconds": time.monotonic() - began,
          "gpu_used": False, "fit_used_validation_or_test_audio": False, "source_bindings_unchanged": True,
          "counted_bytes_before": counted, "counted_bytes_after": require_space(budget, 2_000_000)})
    print(json.dumps(read(out / "fit-result.json")), flush=True)
    if not probe_pass:
        write(out / "result.json", {"status": "training_probe_rejected", "target_reached": False,
              "full14_executed": False, "checkpoint": checkpoint, "fit_result_sha256": sha(out / "fit-result.json")})
        return
    del statistics, moments, values
    quality = out / "full14"
    quality.mkdir()
    quality_bindings = {**bindings, checkpoint["path"]: checkpoint["sha256"],
                        **{str(out / name): sha(out / name) for name in ("plan.json", "checkpoint-audit.json", "fit-result.json", "cache-digest.json", "optimizer.pt", "training-metrics.jsonl", "feature-alignment-check.json")}}
    quality_plan = {"schema": "latency58-direct-sdr-full14-plan-v1", "label": out.name, "checkpoint": checkpoint,
                    "reference_result": str(reference), "workers": 2, "track_indices": list(range(14)),
                    "source_bindings": quality_bindings, "output_directory": str(quality)}
    write(quality / "plan.json", quality_plan)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_dynamic_mixer", "--plan", str(quality / "plan.json"),
             "--plan-sha256", sha(quality / "plan.json")], quality, "evaluation", 1800, quality_bindings,
            {"plan_sha256": sha(quality / "plan.json")})
    result = read(quality / "result.json")
    write(out / "result.json", {"status": "training_probe_and_full14_complete", "full14_executed": True,
          "checkpoint": checkpoint, "full_sdr_db": result["results"][0]["aggregate"]["full_sdr_db"],
          "target_reached": result["target_reached"], "quality_result_sha256": sha(quality / "result.json"), "plugin_changed": False})
    print(json.dumps(read(out / "result.json")), flush=True)


if __name__ == "__main__":
    main()
