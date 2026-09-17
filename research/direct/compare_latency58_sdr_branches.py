"""Compare completed quality bundles from different, explicitly bound recipes."""
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
    require(plan["schema"] == "latency58-sdr-branch-comparison-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the frozen CPU1 comparison")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous comparison")
    from research.direct.compare import compare

    require(plan["reference_training_plan"] != plan["candidate_training_plan"],
            "Use the same-trial comparator for one shared recipe")
    for side in ("reference", "candidate"):
        binding = plan[side + "_training_plan"]
        require(sha(binding["path"]) == binding["sha256"], "Training plan changed")
        training = read(binding["path"])
        require(0 < plan[side + "_step"] <= training["config"]["steps"], "Invalid saved endpoint")
    evidence, comparisons, states = {}, {}, {"reference": {}, "candidate": {}}
    for mode in ("full14", "actions60", "probes"):
        reports = {}
        for side in ("reference", "candidate"):
            training_binding = plan[side + "_training_plan"]
            prefix = plan[side + "_prefix"]
            require(prefix and all(c.isalnum() or c in "-_" for c in prefix), "Invalid endpoint prefix")
            evaluation_plan, report = load_completed(PHASE / (prefix + "-" + mode + "-001"), evidence)
            require(evaluation_plan["training_plan"] == training_binding
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
        "schema": "latency58-sdr-branch-comparison-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
        "source_bindings_unchanged": True,
        "reference_training_plan": plan["reference_training_plan"],
        "candidate_training_plan": plan["candidate_training_plan"],
        "reference_prefix": plan["reference_prefix"], "candidate_prefix": plan["candidate_prefix"],
        "reference_step": plan["reference_step"], "candidate_step": plan["candidate_step"],
        "reference_model_state_sha256": plan["reference_model_state_sha256"],
        "candidate_model_state_sha256": plan["candidate_model_state_sha256"],
        "comparisons": comparisons, "quality_selected": False, "human_listening_completed": False,
        "limitations": [
            "Different recipes may differ in examples, context, objective and averaging; this comparison cannot isolate those effects.",
            "Primary development panel; confirmation excerpts remain excluded.",
            "Track intervals omit training-seed uncertainty and correction for checkpoint selection.",
            "Actions is one track; its degenerate interval is not a sampling-uncertainty estimate.",
            "Probe energy and level measurements do not establish audibility.",
        ],
    })
    print({"status": "pass", "reference_step": plan["reference_step"], "candidate_step": plan["candidate_step"],
           "full14": comparisons["full14"]["metrics"]}, flush=True)


if __name__ == "__main__":
    main()
