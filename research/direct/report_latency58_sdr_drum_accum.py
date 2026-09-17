"""Compare accumulated drum training with its authenticated parent and working model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed, probe_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.prefix and all(c.isalnum() or c in "-_" for c in args.prefix)
            and not args.output.exists() and args.output.parent.is_dir(), "Invalid prefix or existing output")
    from research.direct.compare import compare
    bindings = {str(p): sha(p) for p in (Path(__file__).resolve(), ROOT / "research/direct/report_latency58_sdr.py", ROOT / "research/direct/compare.py")}
    comparisons, plans, states, candidates = {}, {}, {}, {}
    for mode in ("full14", "actions60", "probes"):
        plan, candidate = load_completed(PHASE / (args.prefix + "-" + mode + "-001"), bindings)
        _, reference = load_completed(PHASE / ("teacher-half250-" + mode + "-001"), bindings,
                                      canonical_baseline=True)
        plans[mode], candidates[mode] = plan, candidate
        if mode == "probes":
            require(candidate["status"] == "pass" and candidate["source_bindings_unchanged"], "Probe run failed")
            states[mode] = candidate["model_state_sha256"]
            comparisons[mode] = probe_comparison(reference, candidate)
        else:
            require(candidate["inputs_unchanged"] and all(candidate[k] == reference[k] for k in
                    ("manifest_sha256", "output_policy", "precision", "metrics", "metric_source_sha256")),
                    "Music protocols differ")
            states[mode] = candidate["results"][0]["model"]["model_state_sha256"]
            comparisons[mode] = compare(reference["results"][0], candidate["results"][0])
    require(len(set(states.values())) == 1 and all(plans[m]["generation"] == plans["full14"]["generation"]
            and plans[m]["step"] == plans["full14"]["step"] for m in plans), "Bundle contains different endpoints")
    training_binding = plans["full14"]["training_plan"]
    require(sha(training_binding["path"]) == training_binding["sha256"], "Training plan changed")
    training = read(training_binding["path"])
    require(training["schema"] == "latency58-sdr-drum-accum-training-v1", "Unknown accumulated-gradient training plan")
    parent_prefix = training["parent_prefix"]
    parent_comparisons = {}
    for mode in ("full14", "actions60", "probes"):
        _, parent = load_completed(PHASE / (parent_prefix + "-" + mode + "-001"), bindings,
                                   canonical_baseline=parent_prefix == "teacher-half250")
        parent_state = parent["model_state_sha256"] if mode == "probes" else parent["results"][0]["model"]["model_state_sha256"]
        require(parent_state == training["parent"]["model_state_sha256"], "Different parent in accumulated-gradient comparison")
        parent_comparisons[mode] = probe_comparison(parent, candidates[mode]) if mode == "probes" else compare(
            parent["results"][0], candidates[mode]["results"][0])
    require(all(sha(p) == s for p, s in bindings.items()), "A report input changed")
    report = {
        "schema": "latency58-sdr-drum-accum-quality-summary-v1", "prefix": args.prefix, "step": plans["full14"]["step"],
        "model_state_sha256": states["full14"], "source_bindings": bindings,
        "versus_working_baseline": comparisons,
        "versus_parent": parent_comparisons, "parent_prefix": parent_prefix, "carry_state": training["carry_state"],
        "learning_rate_peak": training["config"]["lr"], "learning_rate_floor": training["config"]["min_lr"],
        "accumulation_steps": training["accumulation_steps"], "microbatch_size": training["microbatch_size"],
        "drum_weight": training["drum_weight"], "objective_version": training["objective_version"],
        "examples_per_optimizer_update": training["config"]["batch_size"],
        "trial_augmented_examples": plans["full14"]["step"] * training["config"]["batch_size"],
        "quality_selected": False, "listening_completed": False,
        "limitations": ["Primary development-panel results; confirmation intervals remain outside selection.",
                        "No matched unweighted or B4-update control; gains cannot isolate drum weighting, data exposure or update averaging.",
                        "Each optimizer update averages four B4 objectives, retaining their separate nonlinear projection caps.",
                        "Track-bootstrap intervals omit training-seed uncertainty and checkpoint-selection correction.",
                        "The single Actions track has a degenerate bootstrap interval, not a sampling-uncertainty estimate.",
                        "Synthetic probes describe unexplained energy and native level; they do not establish audibility."],
    }
    write(args.output, report)
    print(json.dumps({"result": str(args.output), "step": report["step"],
                      "full14": comparisons["full14"]["metrics"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
