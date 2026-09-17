"""Paired primary-panel comparisons against the working model and two teachers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write


REFERENCE_PATHS = {
    "working_5_8ms": PHASE / "teacher-half250-full14-001/result.json",
    "accepted_11_6ms": ROOT / ("research/direct/runs/latency11/"
        "cropped1024-matched-raw4_control-b4-bf16-lr3e-5/evaluation-cpu-fp32-pilot000250/music/result.json"),
    "c91_23_2ms": ROOT / "research/direct/runs/latency11/baselines/refined-v1-full14.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    from research.direct.compare import compare
    path = args.candidate / "result.json"
    candidate, execution = read(path), read(args.candidate / "execution.json")
    require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and candidate["inputs_unchanged"]
            and execution["plan_sha256"] == candidate["plan_sha256"] == sha(args.candidate / "plan.json"),
            "Require a completed, authenticated quality evaluation")
    bindings = {str(p): sha(p) for p in (path, args.candidate / "execution.json",
                                        args.candidate / "plan.json", Path(__file__).resolve(),
                                        ROOT / "research/direct/compare.py", *REFERENCE_PATHS.values())}
    comparisons = {}
    for label, ref_path in REFERENCE_PATHS.items():
        reference = read(ref_path)
        reference = reference.get("music", reference)
        for key in ("manifest_sha256", "output_policy", "precision", "metrics", "metric_source_sha256"):
            require(reference[key] == candidate[key], "Metric or input protocol differs: " + key)
        comparisons[label] = compare(reference["results"][0], candidate["results"][0])
    require(all(sha(p) == s for p, s in bindings.items()), "Comparison input changed")
    report = {"schema": "latency58-sdr-primary-comparison-v1", "source_bindings": bindings,
              "comparisons": comparisons, "candidate_quality_accepted": False,
              "selection_scope": "Predeclared primary development panel; confirmation excerpts are excluded"}
    write(args.candidate / "paired-sdr-summary.json", report)
    print(json.dumps({k: v["metrics"] for k, v in comparisons.items()}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
