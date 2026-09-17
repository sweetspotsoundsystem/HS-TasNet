"""Compare completed hop128 probe reports at their unchanged native gains.

This reads saved metrics only. Lower unexplained energy is not by itself a
fidelity verdict; output levels and DC offsets remain alongside the deltas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.latency58_checkpoint import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve previous comparisons")
    reports = []
    bindings = {str(Path(__file__).resolve()): sha(__file__)}
    for directory in (args.reference, args.candidate):
        paths = [directory / name for name in ("result.json", "execution.json", "plan.json")]
        result, execution, plan = [json.loads(path.read_text()) for path in paths]
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"]
                and result["plan_sha256"] == execution["plan_sha256"] == sha(paths[2])
                and result["checkpoint"] == plan["checkpoint"],
                "Require completed, identified probe executions")
        bindings.update({str(path.resolve()): sha(path) for path in paths})
        reports.append(result)
    reference, candidate = reports
    for field in ("native_output_source_scales", "metrics_sha256", "stimulus_policy", "normalization"):
        require(reference[field] == candidate[field], "Probe protocol differs: " + field)
    require(reference["normalization"] == "none", "Use native output levels")
    def protocol(result):
        streams = [{key: value for key, value in row.items() if key != "absolute_peak"}
                   for row in result["streaming"]]
        geometry = [{"id": row["id"], "segments": [
            {key: segment[key] for key in ("start", "end", "frequencies_hz")}
            for segment in row["segments"]]} for row in result["results"]]
        return streams, geometry
    require(protocol(reference) == protocol(candidate), "Stimuli or physical scoring geometry differ")
    require(all(row["all_finite"] and row["physical_alignment_verified"]
                for result in reports for row in result["streaming"]), "Probe stream validation failed")
    means = [result["segment_means"] for result in reports]
    require(means[0].keys() == means[1].keys(), "Probe identities differ")
    deltas = {}
    for probe, stems in means[0].items():
        require(stems.keys() == means[1][probe].keys(), "Stem identities differ")
        deltas[probe] = {}
        for stem, metrics in stems.items():
            require(metrics.keys() == means[1][probe][stem].keys(), "Metric identities differ")
            deltas[probe][stem] = {metric: means[1][probe][stem][metric] - value
                                   for metric, value in metrics.items()}
    require(all(sha(path) == digest for path, digest in bindings.items()), "Comparison inputs changed")
    report = {"protocol_identity_verified": True, "segment_mean_delta": deltas,
              "reference_result": str((args.reference / "result.json").resolve()),
              "reference_result_sha256": sha(args.reference / "result.json"),
              "candidate_result_sha256": sha(args.candidate / "result.json"),
              "reference_segment_means": means[0], "candidate_segment_means": means[1],
              "source_bindings": bindings, "source_bindings_unchanged": True,
              "candidate_quality_accepted": False,
              "interpretation": "Candidate minus reference, equal segment weighting. Unexplained energy excludes fitted DC and input-frequency sinusoids. Keep native level and DC changes alongside it; no listening verdict."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({probe: stems["bass"] for probe, stems in deltas.items()}, allow_nan=False))


if __name__ == "__main__":
    main()
