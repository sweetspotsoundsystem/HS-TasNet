"""Compare matched student-history endpoints after authenticating every training microbatch."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed, probe_comparison
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Endpoint comparison plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-history-comparison-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the frozen CPU1 comparison")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous comparison")
    from research.direct.compare import compare

    audit_binding, execution_binding = plan["match_audit"], plan["match_execution"]
    require(sha(audit_binding["path"]) == audit_binding["sha256"]
            and sha(execution_binding["path"]) == execution_binding["sha256"], "Match audit changed")
    audit, execution = read(audit_binding["path"]), read(execution_binding["path"])
    command = execution["argv"]
    require(audit["schema"] == "latency58-history-matched-batches-v1" and audit["status"] == "pass"
            and audit["source_bindings_unchanged"] and audit["all_augmentation_and_teacher_targets_exact"]
            and audit["final_rng_states_exact"] and execution["actual_exit_code"] == 0
            and not execution["timed_out"] and execution["source_bindings_unchanged"]
            and plan["reference_step"] == plan["candidate_step"] == audit["step"]
            and audit["trained_model_states"]["short"] == plan["reference_model_state_sha256"]
            and audit["trained_model_states"]["long"] == plan["candidate_model_state_sha256"]
            and command[command.index("-m") + 1] == "research.direct.audit_latency58_history_match"
            and Path(command[command.index("--output") + 1]).resolve() == Path(audit_binding["path"]).resolve()
            and int(command[command.index("--step") + 1]) == audit["step"],
            "Training arms were not authenticated as matched")
    verify_inputs(audit)
    evidence, comparisons, states = {}, {}, {"reference": {}, "candidate": {}}
    for mode in ("full14", "actions60", "probes"):
        reports = {}
        for side in ("reference", "candidate"):
            prefix = plan[side + "_prefix"]
            require(prefix and all(c.isalnum() or c in "-_" for c in prefix), "Invalid endpoint prefix")
            evaluation_plan, report = load_completed(PHASE / (prefix + "-" + mode + "-001"), evidence)
            require(evaluation_plan["training_plan"] == plan[side + "_training_plan"]
                    and evaluation_plan["step"] == plan[side + "_step"],
                    "Quality bundle belongs to another training plan or endpoint")
            if mode == "probes":
                require(report["status"] == "pass" and report["source_bindings_unchanged"], "Probe run failed")
                state = report["model_state_sha256"]
            else:
                require(report["inputs_unchanged"], "Music inputs changed")
                state = report["results"][0]["model"]["model_state_sha256"]
            require(state == plan[side + "_model_state_sha256"], "Endpoint learned state differs")
            states[side][mode], reports[side] = state, report
        reference, candidate = reports["reference"], reports["candidate"]
        if mode == "probes":
            comparisons[mode] = probe_comparison(reference, candidate)
        else:
            require(all(reference[k] == candidate[k] for k in (
                "manifest_sha256", "output_policy", "precision", "metrics", "metric_source_sha256")),
                "Music protocols differ")
            comparisons[mode] = compare(reference["results"][0], candidate["results"][0])
    require(all(len(set(values.values())) == 1 for values in states.values())
            and all(plan["source_bindings"].get(p) == s for p, s in evidence.items()),
            "Mixed endpoints or unbound completed quality evidence")
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-history-comparison-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
        "source_bindings_unchanged": True, "match_audit": audit_binding,
        "reference_training_plan": plan["reference_training_plan"],
        "candidate_training_plan": plan["candidate_training_plan"],
        "reference_prefix": plan["reference_prefix"], "candidate_prefix": plan["candidate_prefix"],
        "reference_step": plan["reference_step"], "candidate_step": plan["candidate_step"],
        "reference_model_state_sha256": plan["reference_model_state_sha256"],
        "candidate_model_state_sha256": plan["candidate_model_state_sha256"],
        "comparisons": comparisons, "quality_selected": False, "human_listening_completed": False,
        "limitations": [
            "Primary development panel; confirmation excerpts remain excluded.",
            "One matched training seed; track intervals omit training-seed uncertainty and checkpoint-selection correction.",
            "Actions is one track; its degenerate interval is not a sampling-uncertainty estimate.",
            "Probe energy and level measurements do not establish audibility.",
        ],
    })
    print({"status": "pass", "reference_step": plan["reference_step"], "candidate_step": plan["candidate_step"],
           "full14": comparisons["full14"]["metrics"]}, flush=True)


if __name__ == "__main__":
    main()
