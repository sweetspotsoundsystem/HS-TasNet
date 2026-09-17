"""Train the selected four-second weighted objective with allocated exact recovery."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from research.direct.train_latency58 import ROOT, PRODUCTION, read, require, sha, state_sha256, verify_inputs, load_source, continuity
from research.direct.run_latency58_quality import write


from research.direct.train_latency58_branch_sdr_ema import compare_ema_to_fp64
from research.direct.latency58_four_second_data import policy as data_policy, dataset as make_dataset, batch_recipes, audio_sha
from research.direct.latency58_logical_batch_loss import policy as reduction_policy
from research.direct.latency58_weighted_vocal_auxiliary import policy as proposal_policy, VERSION
from research.direct.latency58_grouped_vocal_auxiliary import source_views
from research.direct.latency58_weighted_vocal_canonical import grouped_update, policy as accumulation_policy
from research.direct.latency58_four_second_storage import snapshot as complete_budget_snapshot, ARTIFACT_ROOT, POLICY
from research.direct.latency58_grouped_vocal_recovery import (
    policy as recovery_policy, make_snapshot, restore_training)
from research.direct.latency58_four_second_recovery_files import publish_snapshot, read_snapshot, finalize
from research.direct.latency58_lossless_recovery_codec_v3 import policy as packed_policy


def applied_policy():
    # Proposal status is a historical annotation; the authenticated training
    # plan and its stage determine whether this policy is being executed.
    return {key: value for key, value in proposal_policy().items() if key != "production_recipe_selected"}


def runtime_policy():
    return {"version": "four-second-weighted-packed-save-runtime-v1", "resource_max_seconds": 1200,
            "production_seconds_per_update": 100, "production_fixed_allowance_seconds": 600,
            "save_seconds_per_generation": 60, "progress_timeout_seconds": 120,
            "finalization_timeout_seconds": 180}


def budget_snapshot(plan):
    require(plan["new_storage_policy_binding"] == {"path": str(POLICY), "sha256": sha(POLICY)},
            "New artifact allocation differs")
    return complete_budget_snapshot()


def validate_recipe(plan):
    decision_binding = plan["scientific_decision"]
    require(sha(decision_binding["path"]) == decision_binding["sha256"]
            == "2d518bb6067914779e8adcd1903f53c6d52a361b44352093cf890f618a4e295c",
            "Four-second scientific decision changed")
    decision = read(decision_binding["path"])
    require(plan["config"] == decision["proposed_training_config"]
            and plan["parent_checkpoint"] == decision["parent_checkpoint"]
            and plan["parent_model_state_sha256"] == decision["parent_model_state_sha256"]
            and plan["parent_training_updates"] == decision["parent_training_updates"]
            and plan["parameter_names"] == decision["trainable_parameter_names"]
            and plan["inference_architecture"] == decision["inference_architecture"],
            "Selected parent, schedule, parameter set or inference architecture changed")
    require(plan["schema"] == "latency58-grouped-vocal-training-plan-v1"
            and plan["packed_recovery"] == packed_policy() and plan["recovery_checkpoint"] == recovery_policy()
            and plan["runtime_allowance"] == runtime_policy()
            and plan["new_storage_policy_binding"] == {"path": str(POLICY), "sha256": sha(POLICY)},
            "Unprepared recovery, runtime or storage policy")
    from research.direct.latency58_asymmetric import PRECISION_POLICY
    from research.direct.latency58_branch_ema_checkpoint import policy as ema_policy
    config = plan["config"]
    require(plan["objective_version"] == VERSION and plan["grouped_vocal_loss"] == applied_policy()
            and plan["logical_batch_loss"] == reduction_policy() and plan["direct_sdr_weight"] == .2
            and plan["precision_policy"] == PRECISION_POLICY and config["augmentation"] == data_policy()
            and plan["ema"] == ema_policy(.995) and plan["accumulation_policy"] == accumulation_policy(),
            "Weighted objective, whole-group reduction, precision, data or EMA policy differs")
    require(config["batch_size"] == 16 and config["microbatch_size"] == 2
            and config["auxiliary_microbatch_size"] == 2 and plan["accumulation_steps"] == 8
            and config["crop_samples"] == 264576 and plan["warmup_samples"] == 88064
            and plan["scored_samples"] == 176512 and config["device"] == "cuda" and config["precision"] == "bf16"
            and config["workers"] == 2 and config["steps"] == 2000 and config["warmup"] == 100
            and config["seed"] == 20261102 and config["data_start"] == 4132000
            and config["lr"] == 3e-5 and config["min_lr"] == 3e-6,
            "Selected four-second geometry or schedule differs")
    root = ARTIFACT_ROOT / "branch-four-second-015"
    require(Path(plan["output_directory"]) == root
            and plan["packed_publication_directory"] == str(root / "production-run"),
            "Training or recovery directory escaped its disjoint allocation")
    require(len(plan["qualified_data_prefix"]) == 4 and all(
        row["first_index"] == config["data_start"] + 16 * index
        and all(isinstance(row.get(key), str) and len(row[key]) == 64
                and all(c in "0123456789abcdef" for c in row[key])
                for key in ("input_sha256", "after_remix_sha256", "auxiliary_full_context_sha256"))
        and all(len(row[group + "_active_counts"]) == len(row[group + "_absent_counts"]) == 4
                and all(type(a) is type(b) is int and a >= 0 and b >= 0 and a + b == count
                        for a, b in zip(row[group + "_active_counts"], row[group + "_absent_counts"], strict=True))
                for group, count in (("ordinary", 64), ("auxiliary", 8)))
        for index, row in enumerate(plan["qualified_data_prefix"])), "Selected data prefix is incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--stage-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256 and sha(args.stage) == args.stage_sha256,
            "Direct-SDR plan or stage changed")
    plan, stage = read(args.plan), read(args.stage)
    validate_recipe(plan)
    verify_inputs(plan)
    verify_inputs(stage)
    config = plan["config"]
    require(plan["schema"] == "latency58-grouped-vocal-training-plan-v1" and stage["plan_sha256"] == args.plan_sha256
            and stage["stop_step"] == (2 if stage["resource_only"] else config["steps"])
            and all(os.environ.get(k) == v for k, v in plan["environment"].items()), "Training recipe or environment differs")
    monitor = load_source("direct_sdr_watchdog", plan["watchdog_source"])
    _, events = continuity(stage, monitor)
    run = Path(stage["run_directory"])
    require(run == Path(plan["output_directory"]) / ("resource-run-003" if stage["resource_only"] else "production-run")
            and run.is_relative_to(ARTIFACT_ROOT) and not run.exists(), "Preserve allocated four-second runs")
    budget_snapshot(plan)
    run.mkdir()
    write(run / "event-continuity.json", events)
    lock = (run / "trainer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from torch.utils.data import DataLoader
    from research.direct.latency58_remix_augmentation import augment, recipe
    from research.direct.latency58_branch_memory_checkpoint import load_model as load_parent
    from research.direct.latency58_branch_memory import Latency58BranchMemoryModel
    from research.direct.latency58_branch_memory_checkpoint import audit_live
    from research.direct.latency58_branch_ema import BranchParameterEMA
    from research.direct.latency58_branch_ema_checkpoint import policy
    from research.direct.latency58_asymmetric import PRECISION_POLICY
    require(plan["ema"] == policy(plan["ema"]["decay"])
            and VERSION == plan["objective_version"] and plan["direct_sdr_weight"] == .2
            and PRECISION_POLICY == plan["precision_policy"]
            and plan["logical_batch_loss"] == reduction_policy(), "Objective or precision source differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parent, _ = load_parent(plan["parent_checkpoint"])
    require(plan["parent_kind"] == "saved_trained_branch_memory"
            and plan["optimizer_initialization"] == "fresh_adam"
            and plan["parent_weight_role"] in ("ema", "raw")
            and parent.provenance.get("checkpoint_weight_role") == {
                "ema": "averaged_inference", "raw": "raw_optimizer_endpoint"}[plan["parent_weight_role"]]
            and parent.provenance["training_updates"] == plan["parent_training_updates"]
            and type(parent) is Latency58BranchMemoryModel
            and state_sha256(parent.state_dict()) == plan["parent_model_state_sha256"],
            "Saved attention parent or continuation initialization changed")
    model = parent
    model.provenance = {**model.provenance,
                        "branch_memory_previous_provenance": dict(model.provenance),
                        "branch_memory_parent_model_state_sha256": plan["parent_model_state_sha256"]}
    del parent
    require(state_sha256(model.state_dict()) == plan["initialized_model_state_sha256"]
            and state_sha256(dict(model.named_buffers())) == plan["fixed_buffers_sha256"]
            and not torch.cuda.is_initialized(), "CPU parent initialization changed")
    require(torch.__version__ == plan["torch_version"] and torch.cuda.is_available()
            and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "Require the reviewed BF16 GPU")
    torch.cuda.set_per_process_memory_fraction(.75)
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    production.configure_determinism(config["seed"])
    from research.direct.latency58_recorded301_data import select_tracks, ROOT_WEIGHTS
    production_config = read(PRODUCTION / "full_config.json")
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"], config=production_config)
    require(manifest_sha == plan["manifest_sha256"] and production_config["seed"] == config["data_seed"]
            and production_config["sampling"]["root_weights"] == config["source_corpus_root_weights"]
            and production_config["sampling"]["vocal_active_probability"] == config["vocal_active_probability"],
            "Training corpus or sampling distribution changed")
    tracks = select_tracks(tracks, plan["training_selection"])
    require(config["root_weights"] == ROOT_WEIGHTS
            and config["augmentation"] == data_policy()
            and not plan["online_teacher_used"] and not plan["teacher_generated_targets_in_current_stage"],
            "Unexpected recorded301 training recipe")
    model.provenance = {**model.provenance, "branch_memory_current_stage_corpus": "301 recorded MUSDB18-HQ and MoisesDB training tracks",
                        "branch_memory_current_stage_augmentation": data_policy(),
                        "branch_memory_online_teacher_used": False,
                        "branch_memory_teacher_generated_targets_in_current_stage": False,
                        "branch_memory_training_selection": plan["training_selection"],
                        "branch_memory_current_stage_training_context": {"warmup_samples": plan["warmup_samples"],
                            "scored_samples": plan["scored_samples"], "logical_batch_size": 16, "microbatch_size": 2,
                            "logical_batch_loss": reduction_policy(), "grouped_vocal_loss": applied_policy(),
                            "auxiliary_examples": 2, "auxiliary_microbatch_size": 2,
                            "accumulation_policy": accumulation_policy()}}
    model.cuda().train().requires_grad_(True)
    model.training_precision = "bf16"
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], foreach=False)
    audit_live(model, optimizer, 0, frozen)
    require(config["microbatch_size"] == 2 and plan["accumulation_steps"] == 8
            and config["batch_size"] == 16 and config["crop_samples"] == 264576
            and plan["warmup_samples"] == 88064 and plan["scored_samples"] == 176512,
            "Branch-memory training geometry changed")
    require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"
            and torch.cuda.memory.get_allocator_backend() == "native", "Require the selected native allocator retry")
    def allocator_record(phase):
        snapshot = torch.cuda.memory._snapshot()
        segments = snapshot["segments"]
        inactive = [block["size"] for segment in segments for block in segment["blocks"]
                    if block["state"] == "inactive"]
        result = {"phase": phase, "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "memory_stats": dict(torch.cuda.memory_stats()),
            "allocator_settings": snapshot.get("allocator_settings"),
            "segment_count": len(segments), "expandable_segment_count": sum(bool(s.get("is_expandable")) for s in segments),
            "inactive_block_count": len(inactive), "largest_inactive_block_bytes": max(inactive, default=0),
            "total_inactive_block_bytes": sum(inactive), "configured_process_fraction": .75,
            "environment_setting": os.environ["PYTORCH_CUDA_ALLOC_CONF"]}
        write(run / ("allocator-" + phase + ".json"), result)
        return result
    initial_allocator = allocator_record("before-qualification")
    require(initial_allocator["expandable_segment_count"] > 0,
            "Expandable allocation was requested but not observed on the selected device")
    if stage["resource_only"]:
        from research.direct.check_latency58_four_second_device import check_gpu
        def qualification_progress(phase, group, offset):
            print(json.dumps({"event": "resource_qualification_progress", "phase": phase,
                              "group": group, "offset": offset}), flush=True)
        try:
            write(run / "grouped-vocal-gpu-parity.json", check_gpu(model, plan, progress=qualification_progress))
        except torch.OutOfMemoryError:
            allocator_record("qualification-oom")
            raise
        allocator_record("after-qualification")
        audit_live(model, optimizer, 0, frozen)
        from research.direct.latency58_four_second_device_recovery import exercise_recovery
        import random
        import numpy as np
        recovery_check = run / "recovery-qualification"
        recovery_check.mkdir()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        try:
            with torch.random.fork_rng(devices=[next(model.parameters()).device.index]):
                recovery_result = exercise_recovery(model, plan, recovery_check,
                    disk_directory=recovery_check / "disk-fixture")
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        require(torch.equal(cpu_rng, torch.get_rng_state())
                and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
                "Packed recovery qualification changed production RNG")
        recovery_result["production_rng_restored"] = True
        write(run / "recovery-gpu-parity.json", recovery_result)
        audit_live(model, optimizer, 0, frozen)
        print(json.dumps({"event": "recovery_packing_qualification_pass", "device": recovery_result["device"]}), flush=True)
    ema = BranchParameterEMA(model, decay=plan["ema"]["decay"],
                             base_state_sha256=plan["parent_model_state_sha256"])
    start_step, previous_journal, recovered_prefix = 0, b"", []
    if not stage["resource_only"] and plan.get("resume_checkpoint"):
        recovery = plan["resume_checkpoint"]
        old_plan_path = Path(recovery["training_plan"]["path"])
        require(sha(old_plan_path) == recovery["training_plan"]["sha256"], "Recovery training plan changed")
        old_plan = read(old_plan_path)
        verify_inputs(old_plan)
        require(all(plan[key] == old_plan[key] for key in ("config", "parent_checkpoint",
            "parent_model_state_sha256", "parent_training_updates", "ema", "fixed_buffers_sha256",
            "objective_version", "grouped_vocal_loss", "accumulation_policy", "qualified_data_prefix",
            "inference_architecture", "recovery_checkpoint")), "Recovery changed the trained trajectory")
        snapshot, audited = read_snapshot(recovery["snapshot"], old_plan, recovery["training_plan"]["sha256"])
        start_step = snapshot["step"]
        require(0 < start_step < stage["stop_step"], "Recovery must precede the requested endpoint")
        previous_journal = snapshot["journal"]
        recovered_prefix = [{k: v for k, v in row.items() if k not in
            ("data_wait_seconds", "compute_and_audit_seconds", "elapsed_seconds", "peak_vram_gib")}
            for row in [json.loads(line) for line in previous_journal.splitlines()][:2]]
        require(recovered_prefix == read(stage["resource_result"]["path"])["matching_production_updates"],
                "Recovered trajectory differs from the fresh resource prefix")
        del model, optimizer, ema
        gc.collect()
        model, optimizer, ema = restore_training(snapshot, audited, old_plan, device="cuda", precision="bf16")
        del snapshot, audited, old_plan
        gc.collect()
        audit_live(model, optimizer, start_step, frozen)
    reference = ({name: parameter.detach().cpu().double().clone() for name, parameter in model.named_parameters()}
                 if stage["resource_only"] else None)
    arithmetic_checks = []
    final_index = config["data_start"] + stage["stop_step"] * config["batch_size"]
    dataset = make_dataset(production, tracks, config, final_index)
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["workers"],
        sampler=production.AbsoluteIndexSampler(config["data_start"] + start_step * config["batch_size"], final_index), pin_memory=True,
        worker_init_fn=production.worker_init, generator=torch.Generator().manual_seed(config["seed"] + 1),
        multiprocessing_context="spawn", prefetch_factor=2)
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    rehearsal = None if stage["resource_only"] else read(stage["resource_result"]["path"])
    began, step, prefix = time.monotonic(), start_step, recovered_prefix
    recovery_observations = []
    previous_batch_done = began
    torch.cuda.reset_peak_memory_stats()
    with (run / "metrics.jsonl").open("x", buffering=1) as journal:
        if previous_journal:
            journal.write(previous_journal.decode())
            journal.flush()
            os.fsync(journal.fileno())
        for mixture_cpu, truth_cpu in loader:
            batch_arrived = time.monotonic()
            data_wait = batch_arrived - previous_batch_done
            require(not stopped and mixture_cpu.shape[0] == config["batch_size"], "Incomplete or interrupted batch")
            if step < config["warmup"]:
                lr = config["lr"] * (step + 1) / config["warmup"]
            else:
                phase = (step - config["warmup"]) / max(1, config["steps"] - 1 - config["warmup"])
                lr = config["min_lr"] + .5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * phase))
            optimizer.param_groups[0]["lr"] = lr
            first_index = config["data_start"] + step * config["batch_size"]
            pitch_rows = batch_recipes(config, first_index)
            before_remix_sha = audio_sha(mixture_cpu, truth_cpu)
            recipe_row = {key: value.tolist() for key, value in recipe(
                seed=config["seed"], first_sample_index=first_index).items()}
            mixture_cpu, truth_cpu, changed_cpu, factors_cpu = augment(
                mixture_cpu, truth_cpu, seed=config["seed"], first_sample_index=first_index)
            after_remix_sha = audio_sha(mixture_cpu, truth_cpu)
            if step < 4:
                expected = plan["qualified_data_prefix"][step]
                require(expected["first_index"] == first_index
                        and expected["input_sha256"] == before_remix_sha
                        and expected["after_remix_sha256"] == after_remix_sha,
                        "Actual trainer data differs from CPU-qualified pitch/remix pipeline")
            auxiliary_mix, auxiliary_truth = source_views(mixture_cpu, truth_cpu)
            auxiliary_sha = audio_sha(auxiliary_mix, auxiliary_truth)
            del auxiliary_mix, auxiliary_truth
            if step < 4:
                require(auxiliary_sha == plan["qualified_data_prefix"][step]["auxiliary_full_context_sha256"],
                        "Auxiliary history differs from its CPU data qualification")
            def check_continue():
                require(not stopped, "Interrupted grouped accumulation")
            update = grouped_update(model, optimizer, ema, mixture_cpu, truth_cpu, step=step + 1,
                warmup_samples=plan["warmup_samples"], ordinary_microbatch=config["microbatch_size"],
                auxiliary_microbatch=config["auxiliary_microbatch_size"], check_continue=check_continue)
            if step < 4:
                expected = plan["qualified_data_prefix"][step]
                for group in ("ordinary", "auxiliary"):
                    require(update["groups"][group]["active_windows"] == expected[group + "_active_counts"]
                            and update["groups"][group]["absent_windows"] == expected[group + "_absent_counts"],
                            "GPU group activity differs from qualified data")
            if step < 2:
                require(len(update["parameter_gradient_norms"]) == 40
                        and all(value > 0 for value in update["parameter_gradient_norms"].values()),
                        "Grouped update did not exercise every trained parameter")
            if reference is not None:
                arithmetic_checks.append(compare_ema_to_fp64(model, ema, reference))
            torch.cuda.synchronize()
            step += 1
            row = {**update, "lr": lr,
                   "first_sample_index": config["data_start"] + (step - 1) * config["batch_size"],
                   "next_sample_index": config["data_start"] + step * config["batch_size"],
                   "augmentation_recipe": recipe_row, "pitch_tempo_recipes": pitch_rows,
                   "pitch_tempo_selected_examples": sum(r["selected"] for r in pitch_rows),
                   "before_remix_audio_sha256": before_remix_sha, "after_remix_audio_sha256": after_remix_sha,
                   "auxiliary_full_context_sha256": auxiliary_sha, "grad_norm": update["gradient_norm_before_clip"],
                   "loss": update["weighted_loss"], "adam_steps_this_update": 1, "gradient_clips_this_update": 1,
                   "grouped_vocal_loss": applied_policy()}
            if step <= 2:
                prefix.append(row)
                if rehearsal is not None:
                    require(row == rehearsal["matching_production_updates"][step - 1], "Production prefix differs from resource rehearsal")
                if step == 2:
                    audit_live(model, optimizer, step, frozen)
                    fingerprint = state_sha256(model.state_dict())
                    if rehearsal is not None:
                        require(fingerprint == rehearsal["final_raw_model_state_sha256"]
                                and update["ema_parameters_sha256"] == rehearsal["final_ema_parameters_sha256"],
                                "Rehearsed raw or EMA update differs")
            row = {**row, "data_wait_seconds": data_wait, "compute_and_audit_seconds": time.monotonic() - batch_arrived, "elapsed_seconds": time.monotonic() - began, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps({k: row[k] for k in ("step", "loss", "grad_norm", "elapsed_seconds", "peak_vram_gib")}), flush=True)
            if step % 25 == 0:
                audit_live(model, optimizer, step, frozen)
                budget_snapshot(plan)
            if (not stage["resource_only"] and step < stage["stop_step"]
                    and step % recovery_policy()["interval_updates"] == 0):
                checkpoint_began = time.monotonic()
                snapshot = make_snapshot(model, optimizer, ema, step, plan, args.plan_sha256,
                                         (run / "metrics.jsonl").read_bytes())
                saved = publish_snapshot(snapshot, run, plan, args.plan_sha256, before_replace=check_continue)
                del snapshot
                saved["complete_save_seconds"] = time.monotonic() - checkpoint_began
                require(saved["complete_save_seconds"] < runtime_policy()["save_seconds_per_generation"],
                        "Packed recovery save exceeded its prepared progress allowance")
                recovery_observations.append(saved)
                print(json.dumps({"event": "recovery_saved", **saved}), flush=True)
            del mixture_cpu, truth_cpu, changed_cpu, factors_cpu, update
            previous_batch_done = time.monotonic()
    require(not stopped and step == stage["stop_step"], "Training did not reach its requested endpoint")
    audit_live(model, optimizer, step, frozen)
    verify_inputs(plan)
    verify_inputs(stage)
    ema_state = ema.state_dict(model)
    averaged_state = {**ema_state["parameters"], **ema_state["buffers"]}
    summary = {"status": "pass", "updates": step,
               "final_raw_model_state_sha256": state_sha256(model.state_dict()),
               "final_model_state_sha256": state_sha256(averaged_state),
               "final_ema_parameters_sha256": ema_state["ema_parameters_sha256"],
               "ema_updates": ema.updates, "ema_policy": plan["ema"],
               "resource_ema_arithmetic_checks": arithmetic_checks,
               "fixed_buffers_unchanged": True, "matching_production_updates": prefix,
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "elapsed_seconds": time.monotonic() - began,
               "source_bindings_unchanged": True, "plan_sha256": args.plan_sha256, "checkpoint_written": not stage["resource_only"]}
    summary["recovery_checkpoints"] = recovery_observations
    summary["resumed_from_step"] = start_step
    if not stage["resource_only"]:
        budget_snapshot(plan)
        checkpoint_began = time.monotonic()
        snapshot = make_snapshot(model, optimizer, ema, step, plan, args.plan_sha256,
                                 (run / "metrics.jsonl").read_bytes())
        saved = publish_snapshot(snapshot, run, plan, args.plan_sha256,
            before_replace=lambda: require(not stopped, "Interrupted final packed publication"))
        del snapshot
        summary["checkpoint"] = finalize(saved, plan, args.plan_sha256)
        summary["final_save_seconds"] = time.monotonic() - checkpoint_began
        require(summary["final_save_seconds"] < runtime_policy()["save_seconds_per_generation"],
                "Final packed publication exceeded its prepared bound")
        summary["checkpoint_roles"] = {"raw": summary["final_raw_model_state_sha256"],
                                       "ema": summary["final_model_state_sha256"]}
        summary["packed_final_reuses_rolling_file"] = True
        summary["storage_after"] = budget_snapshot(plan)
    write(run / "result.json", summary)
    print(json.dumps({"status": "pass", "updates": step, "checkpoint_written": summary["checkpoint_written"]}), flush=True)


if __name__ == "__main__":
    main()
