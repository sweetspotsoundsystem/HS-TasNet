"""Close the predeclared midpoint when it does not improve the best component."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Require the frozen CPU closure plan")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-weight-average-closure-plan-v1", "Unknown decision plan")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve prior decision")
    binding = plan["summary"]
    require(sha(binding["path"]) == binding["sha256"], "Summary changed")
    summary = read(binding["path"])
    execution = read(plan["summary_execution"]["path"])
    command = execution["argv"]
    require(sha(plan["summary_execution"]["path"]) == plan["summary_execution"]["sha256"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and summary["source_bindings_unchanged"]
            and command[command.index("-m") + 1] == "research.direct.report_latency58_weight_average"
            and Path(command[command.index("--output") + 1]) == Path(binding["path"])
            and summary["component_steps"] == [500, 1000] and summary["component_weights"] == [.5, .5]
            and not summary["coefficient_search"] and not summary["quality_selected"], "Incomplete fixed comparison")
    verify_inputs(summary)
    evidence, reports = {}, {}
    for prefix in ("sdr-weight-average-500-1000", "sdr-drum-accum-500", "sdr-drum-accum-1000", "teacher-half250"):
        reports[prefix] = {}
        for mode in ("full14", "actions60", "probes"):
            _, reports[prefix][mode] = load_completed(PHASE / (prefix + "-" + mode + "-001"), evidence,
                                                      canonical_baseline=prefix == "teacher-half250")
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound quality evidence")
    candidate = reports["sdr-weight-average-500-1000"]
    full = candidate["full14"]["results"][0]
    state = full["model"]["model_state_sha256"]
    require(state == summary["model_state_sha256"] == candidate["probes"]["model_state_sha256"]
            == candidate["actions60"]["results"][0]["model"]["model_state_sha256"], "Mixed candidate weights")
    component_scores = {prefix: reports[prefix]["full14"]["results"][0]["aggregate"]["full_sdr_db"]
                        for prefix in ("sdr-drum-accum-500", "sdr-drum-accum-1000")}
    score = full["aggregate"]["full_sdr_db"]
    require(score <= max(component_scores.values()), "This closure does not apply to an improved candidate")
    detailed, support = {}, {}
    for prefix in ("sdr-drum-accum-500", "sdr-drum-accum-1000", "teacher-half250"):
        detailed[prefix] = {}
        reference = reports[prefix]["full14"]["results"][0]
        for a, b in zip(full["tracks"], reference["tracks"], strict=True):
            require(a["name"] == b["name"] and a["excerpts"] == b["excerpts"], "Primary intervals differ")
            row = {}
            for stem, values in a["per_stem"].items():
                prior = b["per_stem"][stem]
                require(all(values[k] == prior[k] for k in ("active_windows", "absent_windows")), "Scoring support differs")
                row[stem] = {k: None if values[k] is None or prior[k] is None else values[k] - prior[k]
                             for k in ("full_sdr_db", "sir_db", "absent_fp_dbfs", "absent_fp_ratio_db")}
                row[stem]["band_sdr_db"] = {k: None if v is None or prior["band_sdr_db"][k] is None
                                             else v - prior["band_sdr_db"][k] for k, v in values["band_sdr_db"].items()}
                support.setdefault(a["name"], {})[stem] = {k: values[k] for k in ("active_windows", "absent_windows")}
            detailed[prefix][a["name"]] = row
    verify_inputs(plan)
    write(out / "result.json", {
        "schema": "latency58-weight-average-closure-v1", "status": "pass", "plan_sha256": args.plan_sha256,
        "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "decision": "close_without_adoption", "quality_selected": False,
        "reason": "The one predeclared midpoint does not improve the best component full-SDR point estimate.",
        "model_state_sha256": state, "full_sdr_db": score, "component_full_sdr_db": component_scores,
        "delta_from_best_component_db": score - max(component_scores.values()),
        "full_aggregate": full["aggregate"], "actions_aggregate": candidate["actions60"]["results"][0]["aggregate"],
        "track_stem_deltas": detailed, "activity_support": support,
        "summary": binding, "component_weights": [.5, .5], "coefficient_search": False,
        "new_training_updates": 0, "history_training_plan_changed": False,
        "model_and_quality_evidence_retained": True, "actions_audio_retained": True,
        "additional_primary_audio_captured": False, "confirmation_excerpts_used": False,
        "human_listening_verdict": None, "working_plugin_changed": False,
        "limitations": ["Failure to raise the best point estimate is not evidence of an audible difference.",
                        "Paired track intervals omit training-seed uncertainty and model-selection correction.",
                        "This one midpoint does not establish that all checkpoint averaging would fail."]})
    print({"decision": "close_without_adoption", "full_sdr_db": score,
           "delta_from_best_component_db": score - max(component_scores.values())}, flush=True)


if __name__ == "__main__":
    main()
