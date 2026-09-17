"""Compare completed matched trials, their parent and preserved bass probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.train_latency58 import ROOT, read, require, sha, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Comparison plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-teacher-paired-comparison-v1", "Wrong comparison plan")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == ROOT / "research/direct/runs/latency58" and out.is_dir()
            and not (out / "result.json").exists(), "Preserve previous comparison")
    training_proof = read(plan["matched_training_proof"]["path"])
    require(training_proof["status"] == "pass" and training_proof["completed_updates_each"] == 250
            and training_proof["all_augmented_batches_and_learning_rates_exact"], "Matched training was not verified")
    reports = {}
    for label, paths in plan["endpoints"].items():
        bundle = {}
        for mode in ("full14", "actions60", "probes"):
            directory = Path(paths[mode])
            execution = read(directory / "execution.json")
            require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
                    and execution["source_bindings_unchanged"], "Incomplete endpoint evaluation")
            bundle[mode] = read(directory / "result.json")
        model = bundle["full14"]["results"][0]["model"]
        fingerprint = model["model_state_sha256"]
        require(len(bundle["full14"]["results"][0]["tracks"]) == 14
                and bundle["full14"]["inputs_unchanged"]
                and model["model_state_sha256_before"] == model["model_state_sha256_after"] == fingerprint
                and bundle["actions60"]["results"][0]["model"]["model_state_sha256"] == fingerprint
                and bundle["probes"]["status"] == "pass" and bundle["probes"]["model_state_sha256"] == fingerprint,
                "Panel, audition and probe weights differ")
        bundle["audio_summary"] = read(Path(paths["actions60"]) / "paired-audio-summary.json")
        if label in ("half", "control"):
            canonical = read(paths["canonical_receipt"])
            require(canonical["status"] == "pass" and canonical["every_model_tensor_bit_exact"]
                    and canonical["model_state_sha256"] == fingerprint
                    and canonical["input"] == bundle["full14"]["results"][0]["checkpoint"], "Canonical weights differ")
        reports[label] = bundle
    require(set(reports) == {"parent", "half", "control"}, "Require the exact matched triplet")
    from research.direct.compare import compare
    comparisons = {}
    for reference, candidate in (("parent", "half"), ("parent", "control"), ("control", "half")):
        a, b = reports[reference], reports[candidate]
        for field in ("manifest_sha256", "output_policy", "precision", "metrics"):
            require(a["full14"][field] == b["full14"][field], "Panel protocol differs: " + field)
        panel = compare(a["full14"]["results"][0], b["full14"]["results"][0])
        audio = {}
        for stem in ("drums", "bass", "vocals", "other"):
            left, right = (r["audio_summary"]["per_stem"][stem]["candidate"] for r in (a, b))
            audio[stem] = {key: {"reference": left[key], "candidate": right[key], "delta": right[key] - left[key]}
                           for key in ("raw_waveform_sdr_db", "rms_dbfs", "error_phase128_first_harmonic_fraction")}
        require(a["audio_summary"]["physical_interval_samples"] == b["audio_summary"]["physical_interval_samples"]
                and a["probes"]["accepted_probe_report"] == b["probes"]["accepted_probe_report"],
                "Audio/probe reference protocol differs")
        probes = {}
        for probe, stems in a["probes"]["paired_accepted_segment_means"].items():
            probes[probe] = {}
            for stem, metrics in stems.items():
                probes[probe][stem] = {}
                for metric, left in metrics.items():
                    right = b["probes"]["paired_accepted_segment_means"][probe][stem][metric]
                    require(left["reference"] == right["reference"], "Accepted probe means differ")
                    probes[probe][stem][metric] = {"reference": left["candidate"], "candidate": right["candidate"],
                                                  "delta": right["candidate"] - left["candidate"]}
        comparisons[candidate + "_minus_" + reference] = {"full14": panel, "actions60": audio, "probes": probes}
    verify_inputs(plan)
    result = {"schema": "latency58-teacher-paired-comparison-result-v1", "status": "pass",
              "comparisons": comparisons, "source_bindings_unchanged": True, "plan_sha256": args.plan_sha256,
              "matched_training_proof": plan["matched_training_proof"], "quality_selected": False,
              "interpretation": "Differences are at native levels on the fixed protocol. Bootstrap scope is track sampling only, not training-seed or model-selection uncertainty. Probes do not prescribe stem routing or establish listening acceptance."}
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: {metric: {name: row[name] for name in ("delta", "paired_track_bootstrap_95_percent")}
                           for metric, row in value["full14"]["metrics"].items()}
                      for key, value in comparisons.items()}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
