"""Record the reviewed 500-update matched-history endpoints and close both schedules."""
import argparse
import math
from pathlib import Path
import statistics

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_history_checkpoint import read_generation, require_space


def cell_deltas(reference, candidate):
    rows = {}
    for left, right in zip(reference["tracks"], candidate["tracks"], strict=True):
        require(left["name"] == right["name"] and left["excerpts"] == right["excerpts"], "Different physical excerpts")
        cells = {}
        for stem in ("drums", "bass", "vocals", "other"):
            a, b = left["per_stem"][stem], right["per_stem"][stem]
            require(all(a[k] == b[k] for k in ("active_windows", "absent_windows")), "Different activity support")
            require(a["band_sdr_db"].keys() == b["band_sdr_db"].keys(), "Different bands")
            metrics = {k: (a[k], b[k]) for k in ("full_sdr_db", "sir_db", "absent_fp_dbfs")}
            metrics.update({k: (a["band_sdr_db"][k], b["band_sdr_db"][k]) for k in a["band_sdr_db"]})
            values = {}
            for key, (x, y) in metrics.items():
                require((x is None) == (y is None), "Different metric support")
                require(x is None or (math.isfinite(x) and math.isfinite(y)), "Nonfinite metric")
                values[key] = None if x is None else y - x
            cells[stem] = {"active_windows": b["active_windows"], "absent_windows": b["absent_windows"], "delta": values}
        require(left["name"] not in rows, "Duplicate track")
        rows[left["name"]] = cells
    require(len(rows) == 14, "Incomplete primary panel")
    return rows


def completed(directory, label):
    result, execution = read(directory / "result.json"), read(directory / f"{label}-execution.json")
    require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"], "Incomplete report execution")
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--plan-sha256", required=True)
args = parser.parse_args()
require(sha(args.plan) == args.plan_sha256, "Final review plan changed")
plan = read(args.plan)
require(plan["schema"] == "latency58-history-500-review-plan-v1", "Unexpected final review")
verify_inputs(plan)
out = Path(plan["output_directory"])
require(out.is_dir() and not (out / "result.json").exists(), "Preserve completed final review")
require(plan["all_track_stem_band_absence_probe_dc_actions_reviewed"] is True
        and all(plan["arm_reviews"][arm] for arm in ("short", "long"))
        and plan["matched_review"] and plan["rationale"] and plan["limitations"], "Missing substantive review")
phase = out.parent
pair = completed(phase / "sdr-history-long-versus-short-500-001", "comparison")
require(pair["status"] == "pass" and pair["source_bindings_unchanged"], "Incomplete paired comparison")
match = read(pair["match_audit"]["path"])
require(sha(pair["match_audit"]["path"]) == pair["match_audit"]["sha256"]
        and match["status"] == "pass" and match["matching_updates"] == 500
        and match["matching_microbatches"] == 2000 and match["matching_augmented_examples"] == 8000
        and match["all_augmentation_and_teacher_targets_exact"] and match["final_rng_states_exact"], "Unmatched training")
players = {}
for panel, count in (("skelpolu", 50), ("actions", 25)):
    browser_dir = phase / f"sdr-history-500-{panel}-browser-001"
    browser = read(browser_dir / "result.json")
    inventory = read(browser_dir / "inventory-check.json")
    execution = read(browser_dir / "inventory-execution.json")
    require(browser["status"] == inventory["status"] == "pass"
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and inventory["source_bindings_unchanged"]
            and inventory["all_served_bytes_exact"] and inventory["all_byte_ranges_exact"]
            and len(inventory["files"]) == count and not inventory["human_listening_completed"], "Incomplete native players")
    players[panel] = {"url": browser["page_url"], "files": count,
                      "inventory_sha256": sha(browser_dir / "inventory-check.json"), "human_listening_completed": False}
reports = {name: read(phase / f"{prefix}-full14-001/result.json")["results"][0]
           for name, prefix in (("parent", "sdr-drum-accum-1000"), ("leader", "sdr-drum-accum-500"),
                                ("working", "teacher-half250"), ("short", "sdr-history-short-500"),
                                ("long", "sdr-history-long-500"), ("short250", "sdr-history-short-250"),
                                ("long250", "sdr-history-long-250"))}
arms, training_plans = {}, {}
for arm, side, history in (("short", "reference", 88064), ("long", "candidate", 352256)):
    training_path = phase / "sdr-history-prep-001" / f"{arm}-training-plan.json"
    training = read(training_path)
    training_plans[arm] = training
    binding = {"path": str(training_path), "sha256": sha(training_path)}
    summary = completed(phase / f"sdr-history-{arm}-500-summary-001", "summary")
    trend = completed(phase / f"sdr-history-{arm}-500-versus-250-001", "comparison")
    require(summary["step"] == 500 and summary["model_state_sha256"] == pair[side + "_model_state_sha256"]
            and pair[side + "_training_plan"] == binding
            and summary["student_history_samples"] == training["warmup_samples"] == history,
            "Different reviewed arm")
    require(trend["status"] == "pass" and trend["source_bindings_unchanged"]
            and trend["training_plan"] == binding and trend["reference_step"] == 250
            and trend["candidate_step"] == 500 and trend["candidate_model_state_sha256"] == summary["model_state_sha256"]
            and trend["reference_model_state_sha256"] == reports[arm + "250"]["model"]["model_state_sha256"],
            "Different endpoint trend")
    generation = Path(training["run_dir"]) / "checkpoints/step-000500"
    receipt = read_generation(generation, expected_plan_sha=sha(training_path), require_optimizer=False)
    require(receipt["model_state_sha256"] == summary["model_state_sha256"], "Saved state differs from quality endpoint")
    rules = training["continuation_rules"]
    require(rules["review_points"] == [250] and rules["maximum_step"] == training["config"]["steps"] == 500,
            "Original schedule limit differs")
    comparisons = {key: summary[key] for key in ("versus_parent", "versus_sdr_leader", "versus_working_baseline")}
    comparisons["versus_250"] = trend["comparisons"]
    cells = {ref: cell_deltas(reports[ref], reports[arm]) for ref in ("parent", "leader", "working", arm + "250")}
    for ref, key in (("parent", "versus_parent"), ("leader", "versus_sdr_leader"),
                     ("working", "versus_working_baseline"), (arm + "250", "versus_250")):
        for metric, cell_key in (("full_sdr_db", "full_sdr_db"), ("low_sdr_db", "low_20_250"), ("bleed_sir_db", "sir_db")):
            for stem, expected in comparisons[key]["full14"]["metrics"][metric]["per_stem_delta"].items():
                actual = statistics.mean(row[stem]["delta"][cell_key] for row in cells[ref].values())
                require(abs(actual - expected) < 1e-12, "Independent stem delta differs")
    arms[arm] = {"training_plan_sha256": sha(training_path), "model_state_sha256": summary["model_state_sha256"],
                 "original_maximum_step": 500, "completed_step": 500, "training_closed": True,
                 "further_optimizer_updates": 0, "comparisons": comparisons,
                 "all_track_stem_band_absence_deltas": cells, "review": plan["arm_reviews"][arm]}
require(training_plans["short"]["config"] == training_plans["long"]["config"], "Matched schedules differ")
counted = require_space(training_plans["long"], 10_000_000)
review = {"schema": "latency58-history-500-review-v1", "status": "pass", "plan_sha256": args.plan_sha256,
          "source_bindings": {**plan["source_bindings"], str(args.plan): args.plan_sha256}, "source_bindings_unchanged": True,
          "arms": arms, "long_versus_short": pair["comparisons"],
          "matched_track_stem_band_absence_deltas": cell_deltas(reports["short"], reports["long"]),
          "matched_review": plan["matched_review"], "players": players,
          "decision": "Close both matched-history training schedules at their original 500-update limit.",
          "rationale": plan["rationale"], "limitations": plan["limitations"],
          "all_track_stem_band_absence_probe_dc_actions_reviewed": True,
          "human_listening_completed": False, "quality_selected": False, "goal_complete": False,
          "current_counted_bytes": counted, "reserved_additional_bytes": 10_000_000}
verify_inputs(plan)
write(out / "result.json", review)
print({"status": "pass", "decision": review["decision"], "counted_bytes": counted}, flush=True)
