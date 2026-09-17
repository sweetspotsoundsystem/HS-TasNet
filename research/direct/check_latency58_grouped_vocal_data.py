"""Replay recorded training crops before adding the prospective source views."""
from __future__ import annotations

import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write


def replay(dataset, *, workers, snapshot, expected):
    import torch
    from torch.utils.data import DataLoader
    from research.direct import check_latency58_branch_long_context_data as original
    from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups
    from research.direct.latency58_long_context_data import WARMUP_SAMPLES, CROP_SAMPLES
    from research.direct.latency58_remix_augmentation import augment
    first, stop, seed = (snapshot[k] for k in ("first_sample_index", "stop_sample_index", "augmentation_seed"))
    kwargs = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}
    loader = DataLoader(dataset, batch_size=16, num_workers=workers,
        sampler=original.production.AbsoluteIndexSampler(first, stop), worker_init_fn=original.production.worker_init,
        generator=torch.Generator().manual_seed(seed + 1), **kwargs)
    rows = []
    for batch, (mixture, targets) in enumerate(loader):
        require(batch < 4 and mixture.shape == (16, 2, CROP_SAMPLES)
                and targets.shape == (16, 4, 2, CROP_SAMPLES), "Wrong bounded recorded batch")
        index = first + 16 * batch
        before = original.audio_sha(mixture, targets)
        mixture, targets, _, _ = augment(mixture, targets, seed=seed, first_sample_index=index)
        after = original.audio_sha(mixture, targets)
        require(expected[batch] == {"first_index": index, "input_sha256": before, "after_remix_sha256": after},
                "Recorded ordinary batch differs from the qualified long-context input")
        rng = torch.get_rng_state().clone()
        auxiliary_mix, auxiliary = source_views(mixture, targets)
        require(original.audio_sha(mixture, targets) == after and torch.equal(torch.get_rng_state(), rng),
                "Adding source views changed ordinary samples or RNG")
        for begin, end in ((0, WARMUP_SAMPLES), (WARMUP_SAMPLES, CROP_SAMPLES)):
            desired = auxiliary[..., begin:end]
            require(torch.count_nonzero(desired[0, 2]) == 0 and torch.count_nonzero(desired[1, [0, 1, 3]]) == 0
                    and torch.equal(desired[0, [0, 1, 3]], targets[14, [0, 1, 3], :, begin:end])
                    and torch.equal(desired[1, 2], targets[15, 2, :, begin:end])
                    and torch.equal(auxiliary_mix[..., begin:end], desired.sum(1)),
                    "Warmup or scored physical source view is wrong")
        groups = prepare_groups(targets[..., WARMUP_SAMPLES:], auxiliary[..., WARMUP_SAMPLES:])
        rows.append({"first_index": index, "input_sha256": before, "after_remix_sha256": after,
            "auxiliary_full_context_sha256": original.audio_sha(auxiliary_mix, auxiliary),
            "auxiliary_warmup_sha256": original.audio_sha(auxiliary_mix[..., :WARMUP_SAMPLES], auxiliary[..., :WARMUP_SAMPLES]),
            "auxiliary_scored_sha256": original.audio_sha(auxiliary_mix[..., WARMUP_SAMPLES:], auxiliary[..., WARMUP_SAMPLES:]),
            "ordinary_active_counts": groups.ordinary.active.tolist(), "ordinary_absent_counts": groups.ordinary.absent.tolist(),
            "auxiliary_active_counts": groups.auxiliary.active.tolist(), "auxiliary_absent_counts": groups.auxiliary.absent.tolist()})
        print(json.dumps({"event": "recorded_source_views_checked", "workers": workers, "first_index": index}), flush=True)
    require(len(rows) == 4, "Incomplete recorded replay")
    return rows


def main():
    import torch
    from research.direct import check_latency58_branch_long_context_data as original
    from research.direct.latency58_grouped_vocal_auxiliary import policy
    from research.direct.latency58_long_context_data import CROP_SAMPLES, EXPANDED_SAMPLES, LongContextCropDataset
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    from research.direct.train_latency58 import verify_inputs
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "1"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Require CUDA-hidden CPU1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    old = PHASE / "branch-long-context-data-001"
    snapshot, result = read(old / "inputs.json"), read(old / "result.json")
    execution = read(PHASE / "branch-long-context-data-stage-001/execution.json")
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and result["inputs_sha256"] == sha(old / "inputs.json"),
            "Original recorded-crop qualification incomplete")
    check = PHASE / "grouped-vocal-auxiliary-cpu-001"
    require(read(check / "result.json")["status"] == "pass" and read(check / "execution.json")["actual_exit_code"] == 0,
            "Complete group-loss CPU qualification first")
    bindings = dict(snapshot["source_bindings"])
    paths = [Path(__file__).resolve(), old / "inputs.json", old / "result.json",
             PHASE / "branch-long-context-data-stage-001/execution.json"]
    paths.extend(check / n for n in ("plan.json", "result.json", "execution.json"))
    paths.extend(ROOT / "research/direct" / n for n in ("latency58_grouped_vocal_auxiliary.py",
        "latency58_logical_batch_loss.py", "run_latency58_deployed_vocal_views.py"))
    bindings.update({str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    budget_plan = {**read(PHASE / "deployed-vocal-views-001/plan.json"), "diagnostic_artifact_allowance_bytes": 2_050_000_000}
    before = budget_snapshot(budget_plan)
    out = PHASE / "grouped-vocal-auxiliary-data-001"
    require(not out.exists(), "Preserve recorded source-view evidence")
    out.mkdir()
    plan = {"schema": "latency58-grouped-vocal-data-cpu-v1", "source_bindings": bindings,
        "policy": policy(), "budget_before": before, "original_qualification": str(old / "result.json"),
        "training_crop_count": 64, "warmup_samples": 88064, "scored_samples": 88320,
        "model_parent_selected": False, "production_recipe_selected": False, "gpu_used": False}
    write(out / "plan.json", plan)
    selection = original.selection_contract()
    require(selection == snapshot["selection"], "Training corpus selection changed")
    config = read(original.PRODUCTION / "full_config.json")
    _, tracks, _, _ = original.production.load_corpus_manifest(original.PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=selection["source_manifest_sha256"], config=config)
    tracks = original.select_tracks(tracks, selection)
    kwargs = dict(root_weights=original.ROOT_WEIGHTS, seed=config["seed"],
        vocal_active_probability=config["sampling"]["vocal_active_probability"], final_sample_index=snapshot["stop_sample_index"])
    dataset = LongContextCropDataset(original.production.CounterAddressedCropDataset(tracks, crop_samples=CROP_SAMPLES, **kwargs),
        original.production.CounterAddressedCropDataset(tracks, crop_samples=EXPANDED_SAMPLES, **kwargs), seed=snapshot["augmentation_seed"])
    expected = result["augmented_loaders"][0]["batches"]
    require(expected == result["augmented_loaders"][1]["batches"], "Original zero/two-worker references differ")
    runs = [replay(dataset, workers=w, snapshot=snapshot, expected=expected) for w in (0, 2)]
    require(runs[0] == runs[1], "Worker count changed ordinary or auxiliary data/support")
    verify_inputs(plan)
    require(not torch.cuda.is_initialized(), "CPU data qualification initialized CUDA")
    write(out / "result.json", {"status": "pass", "plan_sha256": sha(out / "plan.json"), "source_bindings_unchanged": True,
        "recorded_training_crops": 64, "independent_original_batch_hashes_match": True,
        "ordinary_samples_and_rng_unchanged": True, "zero_and_two_worker_replay_exact": True,
        "source_views_exact_through_warmup_and_scored_suffix": True, "batches": runs[0],
        "production_recipe_selected": False, "model_warmup_states_qualified": False, "training_updates": 0,
        "validation_audio_decoded": False, "gpu_used": False, "budget_after": budget_snapshot(budget_plan),
        "limitations": "Recorded data and reduction support only. Fresh per-view model warmup, actual parameter gradients, resource rehearsal and saved-checkpoint evaluation remain required."})
    print(json.dumps({"status": "pass", "recorded_training_crops": 64, "zero_and_two_worker_replay_exact": True,
                      "production_recipe_selected": False}))


if __name__ == "__main__":
    main()
