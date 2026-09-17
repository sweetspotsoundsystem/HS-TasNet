"""Authenticated context-trial comparisons against its parent and reset control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed, probe_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--matched-prefix")
    parser.add_argument("--match-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.prefix and all(c.isalnum() or c in "-_" for c in args.prefix)
            and not args.output.exists() and args.output.parent.is_dir(), "Invalid prefix or existing output")
    require(bool(args.matched_prefix) == bool(args.match_audit), "Match comparison requires its training audit")
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
    require(sha(training_binding["path"]) == training_binding["sha256"], "Context training plan changed")
    training = read(training_binding["path"])
    require(training["schema"] == "latency58-context-training-v1", "Unknown context training plan")
    parent_prefix = training["parent_prefix"]
    parent_comparisons = {}
    for mode in ("full14", "actions60", "probes"):
        _, parent = load_completed(PHASE / (parent_prefix + "-" + mode + "-001"), bindings,
                                   canonical_baseline=parent_prefix == "teacher-half250")
        parent_state = parent["model_state_sha256"] if mode == "probes" else parent["results"][0]["model"]["model_state_sha256"]
        require(parent_state == training["parent"]["model_state_sha256"], "Different parent in context comparison")
        parent_comparisons[mode] = probe_comparison(parent, candidates[mode]) if mode == "probes" else compare(
            parent["results"][0], candidates[mode]["results"][0])
    matched = None
    if args.matched_prefix:
        require(all(c.isalnum() or c in "-_" for c in args.matched_prefix), "Invalid matched prefix")
        plan, reference = load_completed(PHASE / (args.matched_prefix + "-full14-001"), bindings)
        audit = read(args.match_audit)
        require(audit["schema"] == "latency58-context-matched-batches-v1"
                and audit["status"] == "pass" and audit["source_bindings_unchanged"]
                and audit["step"] == plan["step"] == plans["full14"]["step"]
                and all(sha(p) == s for p, s in audit["source_bindings"].items())
                and set(audit["different_trained_model_states"].values())
                == {states["full14"], reference["results"][0]["model"]["model_state_sha256"]},
                "Matched training audit differs from these two endpoints")
        require(all(reference[k] == candidates["full14"][k] for k in
                    ("manifest_sha256", "output_policy", "precision", "metrics", "metric_source_sha256")),
                "Matched music protocols differ")
        bindings[str(args.match_audit.resolve())] = sha(args.match_audit)
        matched = compare(reference["results"][0], candidates["full14"]["results"][0])
    require(all(sha(p) == s for p, s in bindings.items()), "A report input changed")
    report = {
        "schema": "latency58-context-quality-summary-v1", "prefix": args.prefix, "step": plans["full14"]["step"],
        "model_state_sha256": states["full14"], "source_bindings": bindings,
        "versus_working_baseline": comparisons, "versus_matched_context": matched,
        "versus_parent": parent_comparisons, "parent_prefix": parent_prefix, "carry_state": training["carry_state"],
        "matched_reference_prefix": args.matched_prefix, "quality_selected": False, "listening_completed": False,
        "limitations": ["Primary development-panel results; confirmation intervals remain outside selection.",
                        "Track-bootstrap intervals omit training-seed uncertainty and checkpoint-selection correction.",
                        "The single Actions track has a degenerate bootstrap interval, not a sampling-uncertainty estimate.",
                        "Synthetic probes describe unexplained energy and native level; they do not establish audibility."],
    }
    write(args.output, report)
    print(json.dumps({"result": str(args.output), "step": report["step"],
                      "full14": comparisons["full14"]["metrics"],
                      "matched": matched["metrics"] if matched else None}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
