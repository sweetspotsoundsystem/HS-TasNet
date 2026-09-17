"""Check accumulated drum journal guards and provenance without training."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import load_source, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Runner qualification plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-drum-accum-runner-qualification-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use a CUDA-hidden CPU1 qualification")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve qualification outputs")
    fixture = copy.deepcopy(read(plan["accumulation_template"]["path"]))
    parent_fixture = read(plan["parent_fixture"]["path"])
    functional = read(plan["functional"]["path"])
    functional_execution = read(plan["functional_execution"]["path"])
    for key in ("accumulation_template", "parent_fixture", "functional", "functional_execution"):
        binding = plan[key]
        require(sha(binding["path"]) == binding["sha256"]
                and plan["source_bindings"].get(binding["path"]) == binding["sha256"], "Unbound fixture: " + key)
    require(functional["status"] == "pass" and functional["all_21_parameter_gradients_match"]
            and functional["source_bindings_unchanged"] and functional_execution["actual_exit_code"] == 0
            and not functional_execution["timed_out"] and functional_execution["source_bindings_unchanged"]
            and functional_execution["plan_sha256"] == functional["plan_sha256"]
            and functional["model_state_sha256"] == parent_fixture["parent"]["model_state_sha256"],
            "Require completed independent gradient qualification for this parent")
    from research.direct.latency58_sdr_drum_accum_checkpoint import LOSS_KEYS, expected_provenance, validate_journal
    from research.direct.latency58_drum_emphasis import DrumEmphasisLoss, VERSION
    fixture.update(parent=parent_fixture["parent"], drum_weight=2, objective_version=VERSION)
    helpers = load_source("drum_accum_qualification_helpers", fixture["helper_source"])
    config = fixture["config"]
    # These are explicitly synthetic journal values, not training observations.
    micros = []
    for index in range(4):
        raw, raw_drum = .01 + index * .001, .02 + index * .002
        teacher, teacher_drum = .005 + index * .001, .008 + index * .001
        waveform, weighted_teacher = (4 * raw + raw_drum) / 5, (4 * teacher + teacher_drum) / 5
        cap = .0001
        first = config["data_start"] + 4 * index
        micros.append({
            "micro_index": index, "batch_size": 4, "first_sample_index": first, "next_sample_index": first + 4,
            "initial_state_detached": True, "data_hops": fixture["scored_samples"] // 128, "flush_hops": 1,
            "deranged_examples": 2, "loss": waveform + cap + .5 * weighted_teacher,
            "supervised_loss": waveform + cap, "waveform_l1": waveform, "projection": .01,
            "projection_contribution": cap, "teacher_l1": weighted_teacher, "unweighted_waveform_l1": raw,
            "unweighted_teacher_l1": teacher, "raw_drum_l1": raw_drum, "teacher_drum_l1": teacher_drum,
            "augmented_batch_sha256": hashlib.sha256(f"synthetic-input-{index}".encode()).hexdigest(),
            "teacher_targets_sha256": hashlib.sha256(f"synthetic-teacher-{index}".encode()).hexdigest(),
        })
    row = {
        "step": 1, "lr": helpers.learning_rate(0, config), "first_sample_index": config["data_start"],
        "next_sample_index": config["data_start"] + 16, "carry_state": True, "initial_state_detached": True,
        "teacher_kind": fixture["teacher_kind"], "teacher_weight": .5, "warmup_samples": fixture["warmup_samples"],
        "scored_samples": fixture["scored_samples"], "data_hops": fixture["scored_samples"] // 128,
        "flush_hops": 1, "grad_norm": .1, "microbatch_size": 4, "accumulation_steps": 4,
        "gradient_clips_this_update": 1, "adam_steps_this_update": 1, "microbatches": micros,
        "drum_weight": 2, "objective_version": VERSION, "deranged_examples": 8,
    }

    def aggregate(value):
        for key in LOSS_KEYS:
            value[key] = sum(m[key] for m in value["microbatches"]) / 4
        for key in ("augmented_batch_sha256", "teacher_targets_sha256"):
            value[key] = hashlib.sha256("".join(m[key] for m in value["microbatches"]).encode()).hexdigest()

    def check(value):
        return validate_journal((json.dumps(value, allow_nan=False) + "\n").encode(), 1, fixture, helpers)

    aggregate(row)
    require(check(row) == [row], "Valid synthetic journal did not pass")
    rejected = []

    def reject(name, change, *, recompute=False):
        damaged = copy.deepcopy(row)
        change(damaged)
        if recompute:
            aggregate(damaged)
        try:
            check(damaged)
        except (RuntimeError, KeyError):
            rejected.append(name)
        else:
            raise RuntimeError("Journal accepted corruption: " + name)

    reject("missing_microbatch", lambda r: r["microbatches"].pop())
    reject("extra_adam_step", lambda r: r.update(adam_steps_this_update=4))
    reject("extra_gradient_clip", lambda r: r.update(gradient_clips_this_update=4))
    reject("skipped_data_address", lambda r: r["microbatches"][2].update(first_sample_index=0))
    reject("wrong_drum_normalization", lambda r: r["microbatches"][1].update(raw_drum_l1=.5), recompute=True)
    reject("wrong_teacher_normalization", lambda r: r["microbatches"][1].update(teacher_drum_l1=.5), recompute=True)
    reject("wrong_mean", lambda r: r.update(waveform_l1=r["waveform_l1"] * 4))
    reject("last_microbatch_identity_only", lambda r: r.update(augmented_batch_sha256=r["microbatches"][-1]["augmented_batch_sha256"]))
    reject("wrong_weight_version", lambda r: r.update(objective_version="wrong"))
    reject("wrong_activity_sum", lambda r: r.update(deranged_examples=2))
    provenance = expected_provenance(fixture, 250, "0" * 64)
    require(provenance["parent_provenance"] == fixture["parent"]["provenance"]
            and provenance["training_updates"] == fixture["parent"]["provenance"]["training_updates"] + 250
            and provenance["accum_trial_updates"] == provenance["drum_accum_trial_updates"]
            == provenance["drum_emphasis_trial_updates"] == 250
            and provenance["gradient_accumulation_steps"] == provenance["training_microbatch_size"] == 4
            and provenance["trial_augmented_examples"] == 4000 and provenance["trial_microbatches"] == 1000
            and provenance["drum_weight"] == 2 and provenance["objective_version"] == VERSION,
            "Current-trial counters or parent history differ")
    sources = [Path(p) for p in plan["adapter_sources"]]
    for path in sources:
        require(plan["source_bindings"].get(str(path)) == sha(path), "Unbound adapter source")
        ast.parse(path.read_text())
    trainer = ast.parse((ROOT / "research/direct/train_latency58_sdr_drum_accum.py").read_text())
    attributes = {node.attr for node in ast.walk(trainer) if isinstance(node, ast.Attribute)
                  and isinstance(node.value, ast.Name) and node.value.id == "terms"}
    require(attributes <= set(DrumEmphasisLoss.__dataclass_fields__), "Trainer reads an absent loss field")
    unchanged_scorers = []
    for suffix in ("_parallel", "_probes"):
        old = ROOT / f"research/direct/evaluate_latency58_sdr_accum{suffix}.py"
        new = ROOT / f"research/direct/evaluate_latency58_sdr_drum_accum{suffix}.py"
        expected = old.read_text().replace("sdr_accum", "sdr_drum_accum").replace("sdr-accum", "sdr-drum-accum") \
            .replace("accum_candidate", "drum_accum_candidate")
        require(new.read_text() == expected, "Scoring changed beyond loader and schema mapping")
        unchanged_scorers.append(str(new))
    import torch
    require(not torch.cuda.is_initialized(), "Qualification initialized CUDA")
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-drum-accum-runner-qualification-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
        "source_bindings_unchanged": True, "valid_synthetic_journal_accepted": True,
        "corrupt_synthetic_journals_rejected": rejected, "parent_history_and_current_counters_verified": True,
        "adapter_syntax_count": len(sources), "trainer_loss_fields_verified": sorted(attributes),
        "scorers_unchanged_after_loader_schema_mapping": unchanged_scorers,
        "training_updates_executed": 0, "optimizer_instances": 0, "checkpoint_written": False,
        "cuda_initialized": False, "quality_selected": False,
        "limitations": ["Synthetic journal guards and source checks do not establish GPU resource use or saved-state correctness.",
                        "The associated independent CPU gradient check is separate; a two-update GPU rehearsal is still required."],
    })
    print({"status": "pass", "rejected_corrupt_journals": len(rejected), "syntax_checked_sources": len(sources)}, flush=True)


if __name__ == "__main__":
    main()
