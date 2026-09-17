"""Collect authenticated music, absence and probe comparisons for one endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write


def load_completed(directory, bindings, *, canonical_baseline=False):
    paths = [directory / name for name in ("plan.json", "execution.json", "result.json")]
    plan, execution, report = [read(p) for p in paths]
    current_bindings = dict(plan["source_bindings"])
    if canonical_baseline:
        # The old serialized duplicate was retired only after an exact tensor
        # comparison. Authenticate that explicit replacement, not a missing file.
        proof_paths = [PHASE / "teacher-half-canonical-001/receipt.json",
                       PHASE / "teacher-canonical-prep/half/canonicalization-execution.json",
                       PHASE / "teacher-canonical-prep/half/plan.json"]
        proof, proof_execution, _ = [read(p) for p in proof_paths]
        fingerprint = report.get("model_state_sha256") or report["results"][0]["model"]["model_state_sha256"]
        require(proof["status"] == "pass" and proof["every_model_tensor_bit_exact"]
                and proof["source_bindings_unchanged"] and proof_execution["actual_exit_code"] == 0
                and not proof_execution["timed_out"] and proof_execution["source_bindings_unchanged"]
                and proof["plan_sha256"] == proof_execution["plan_sha256"] == sha(proof_paths[2])
                and proof["model_state_sha256"] == fingerprint
                and all(plan["checkpoint"][k] == proof["input"][k] for k in ("path", "sha256"))
                and current_bindings.get(proof["input"]["path"]) == proof["input"]["sha256"]
                and sha(proof["output"]["path"]) == proof["output"]["sha256"],
                "Stored baseline lacks its exact canonical replacement proof")
        del current_bindings[proof["input"]["path"]]
        current_bindings[proof["output"]["path"]] = proof["output"]["sha256"]
        bindings.update({str(p): sha(p) for p in proof_paths})
        bindings[proof["output"]["path"]] = proof["output"]["sha256"]
    require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == report["plan_sha256"] == sha(paths[0])
            and all(sha(p) == s for p, s in current_bindings.items()),
            "Require a completed evaluation with unchanged inputs: " + str(directory))
    bindings.update({str(p): sha(p) for p in paths})
    return plan, report


def probe_comparison(reference, candidate):
    require(reference["metrics_sha256"] == candidate["metrics_sha256"], "Probe scorer differs")
    geometry = lambda report: [(r["id"], [(s["start"], s["end"], s["frequencies_hz"])
                                for s in r["segments"]]) for r in report["results"]]
    require(geometry(reference) == geometry(candidate)
            and [(s["id"], s["decoded_input_sha256"]) for s in reference["streaming"]]
            == [(s["id"], s["decoded_input_sha256"]) for s in candidate["streaming"]],
            "Probe stimuli or scored physical intervals differ")
    return {probe: {stem: {metric: {
        "reference": reference["segment_means"][probe][stem][metric], "candidate": value,
        "delta": None if value is None or reference["segment_means"][probe][stem][metric] is None
        else value - reference["segment_means"][probe][stem][metric],
    } for metric, value in metrics.items()} for stem, metrics in stems.items()}
        for probe, stems in candidate["segment_means"].items()}


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
    bindings = {str(p): sha(p) for p in (Path(__file__).resolve(), ROOT / "research/direct/compare.py")}
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
    matched = None
    if args.matched_prefix:
        require(all(c.isalnum() or c in "-_" for c in args.matched_prefix), "Invalid matched prefix")
        plan, reference = load_completed(PHASE / (args.matched_prefix + "-full14-001"), bindings)
        audit = read(args.match_audit)
        require(audit["status"] == "pass" and audit["source_bindings_unchanged"]
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
        "schema": "latency58-sdr-quality-summary-v2", "prefix": args.prefix, "step": plans["full14"]["step"],
        "model_state_sha256": states["full14"], "source_bindings": bindings,
        "versus_working_baseline": comparisons, "versus_matched_teacher": matched,
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
