"""Measure two/eight-second history effects on four authenticated training crops.

This is a CPU preparation diagnostic with the accepted student and C91 teacher.
It does not train, select a checkpoint, or score validation material.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "History diagnostic plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-long-context-diagnostic-plan-v1"
            and plan["sample_indices"] == list(range(963000, 963004))
            and plan["history_samples"] == [88064, 352256]
            and plan["scored_samples"] == 88064
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Diagnostic scope differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use bounded CUDA-hidden CPU1")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve diagnostic evidence")
    import torch
    from research import experiment
    from research.direct.check_latency58_long_context_data import digest
    from research.direct.latency58_sdr_context import render_scored_context, physical_context_teacher
    from research.direct.latency58_sdr_teacher import load_initial_student, load_teacher, STUDENT_STATE_SHA256
    from research.direct.train_latency58 import PRODUCTION, state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260915)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    config = read(PRODUCTION / "full_config.json")
    _, tracks, _, _ = production.load_corpus_manifest(
        PRODUCTION / "manifests/combined.manifest.json", expected_file_sha256=plan["manifest_sha256"], config=config)
    dataset = production.CounterAddressedCropDataset(
        tracks, root_weights=config["sampling"]["root_weights"], seed=config["seed"], crop_samples=440320,
        vocal_active_probability=config["sampling"]["vocal_active_probability"], final_sample_index=963004)
    batch = [dataset[i] for i in plan["sample_indices"]]
    mixture, targets = (torch.stack([row[k] for row in batch]) for k in (0, 1))
    expected = read(plan["data_proof"]["path"])["batches"][0]
    require(sha(plan["data_proof"]["path"]) == plan["data_proof"]["sha256"]
            and expected["unaugmented_batch_sha256"] == digest((("mixture", mixture), ("targets", targets))),
            "Training crop identity changed")
    mixture, targets, flags = experiment._augment_training_distribution(mixture=mixture, targets=targets)
    augmented_sha = digest(zip(("mixture", "targets", "deranged"), (mixture, targets, flags)))
    require(augmented_sha == expected["augmented_batch_sha256_cpu"], "Augmentation replay differs")
    rng = torch.get_rng_state().clone()
    model = load_initial_student().eval().requires_grad_(False)
    teacher, identity = load_teacher("c91", plan["teacher"])
    short_start = 352256 - 88064
    short_mixture = mixture[..., short_start:]
    scored_mixture, scored_targets = mixture[..., 352256:], targets[..., 352256:]
    require(torch.equal(short_mixture[..., 88064:], scored_mixture), "Scored coordinates differ")
    student = {}
    with torch.no_grad():
        for name, audio, warm in (("short", short_mixture, 88064), ("long", mixture, 352256)):
            output = render_scored_context(model, audio, warmup_samples=warm, carry_state=True)
            require(output.initial_state_detached and torch.equal(output.physical_mixture, scored_mixture)
                    and all(bool(torch.isfinite(x).all()) for x in (output.raw, output.deployed))
                    and float((output.deployed.sum(dim=1) - scored_mixture).abs().max()) <= 1e-6,
                    "Student state, finite output, physical alignment or mixture closure differs")
            student[name] = {"raw": output.raw, "deployed": output.deployed}
            print(json.dumps({"event": "student_rendered", "history": name,
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
        teacher_long = physical_context_teacher(teacher, mixture, kind="c91", warmup_samples=352256)
        teacher_short = physical_context_teacher(teacher, short_mixture, kind="c91", warmup_samples=88064)
    require(all(float((v.sum(dim=1) - scored_mixture).abs().max()) <= 1e-6
                for v in (teacher_long, teacher_short)), "Teacher physical mixture closure differs")

    def measures(left, right):
        error = (left - right).double()
        return {"max_abs": float(error.abs().max()), "mean_abs": float(error.abs().mean()),
                "rms": float(error.square().mean().sqrt())}

    def per_stem(left, right):
        return {stem: measures(left[:, i], right[:, i])
                for i, stem in enumerate(("drums", "bass", "vocals", "other"))}

    # Compare the exact same teacher tensor to each student. The short-teacher
    # output is diagnostic only: a matched study must not silently swap targets.
    shared_teacher_sha = digest((("teacher_targets", teacher_long),))
    rows = {}
    for name, estimates in student.items():
        rows[name] = {"raw_vs_source": per_stem(estimates["raw"], scored_targets),
                      "deployed_vs_shared_long_teacher": per_stem(estimates["deployed"], teacher_long),
                      "teacher_targets_sha256": shared_teacher_sha}
    windows = []
    for start in range(0, 88064, 22016):
        region = slice(start, start + 22016)
        windows.append({"start": start, "end": start + 22016,
                        "student_deployed_long_minus_short": per_stem(
                            student["long"]["deployed"][..., region], student["short"]["deployed"][..., region]),
                        "teacher_long_minus_short": per_stem(teacher_long[..., region], teacher_short[..., region])})
    require(augmented_sha == digest(zip(("mixture", "targets", "deranged"), (mixture, targets, flags)))
            and torch.equal(rng, torch.get_rng_state())
            and state_sha256(model.state_dict()) == STUDENT_STATE_SHA256
            and state_sha256(teacher.state_dict()) == identity["model_state_sha256"]
            and all(not p.requires_grad and p.grad is None for m in (model, teacher) for p in m.parameters())
            and not torch.cuda.is_initialized()
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Diagnostic inputs or models changed")
    result = {"schema": "latency58-long-context-diagnostic-result-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
              "sample_indices": plan["sample_indices"], "augmented_batch_sha256_cpu": augmented_sha,
              "student_model_state_sha256": STUDENT_STATE_SHA256, "teacher_identity": identity,
              "student_history_samples": plan["history_samples"], "teacher_history_samples_for_both_students": 352256,
              "shared_teacher_targets_sha256_cpu": shared_teacher_sha, "scored_samples": 88064,
              "physical_scored_suffix_exact": True, "model_teacher_data_and_rng_unchanged": True,
              "comparisons": rows, "student_raw_long_minus_short": per_stem(student["long"]["raw"], student["short"]["raw"]),
              "student_deployed_long_minus_short": per_stem(student["long"]["deployed"], student["short"]["deployed"]),
              "teacher_long_minus_short": per_stem(teacher_long, teacher_short), "scored_windows": windows,
              "optimizer_updates": 0, "saved_weights": False, "cuda_initialized": False,
              "quality_selected": False, "confirmation_excerpts_used": False, "training_recipe_frozen": False,
              "elapsed_seconds": time.monotonic() - started,
              "limitations": ["Four preparation training crops and accepted student only; no validation or quality conclusion.",
                              "CPU FP32 diagnostic does not qualify GPU memory, mixed precision, gradients or a production recipe.",
                              "Both student histories use the same long teacher target; teacher-history differences are diagnostic only."]}
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "elapsed_seconds": result["elapsed_seconds"],
                      "shared_teacher_targets_sha256_cpu": shared_teacher_sha}), flush=True)


if __name__ == "__main__":
    main()
