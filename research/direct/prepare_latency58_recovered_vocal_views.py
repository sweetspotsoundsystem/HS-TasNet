"""Prepare the existing vocal-view runner for independently recovered 006 weights.

The original failed supervisor remains failed. A separate completed CPU audit
and both unchanged full14 evaluations establish the saved artifact evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_paired_vocal_views import binding, merge_bindings, cpu_environment, ROLES, SCHEMA, RELEASED
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot

SOURCE = PHASE / "branch-long-context-006"
RECOVERY = PHASE / "branch-long-context-recovery-006"
CONTROL = PHASE / "paired-vocal-recovery-check-006"
SOURCE_SHA = "c6228b7a6d021728fc7b42397d53f4da4c1958a5130abb2dd5b30145c1ac8b22"
RECOVERY_SHA = "33b7ce544a30fa44afab1af89dc995d07bc9808c1a186359dc669531fc6676c4"


def validate_records(r):
    plan, terminal, execution, audit = (r[k] for k in ("plan", "terminal", "execution", "audit"))
    source, training, receipt = (r[k] for k in ("source", "training", "receipt"))
    require(plan["schema"] == "latency58-completed-generation-recovery-v1"
            and r["plan_sha256"] == execution["plan_sha256"] == terminal["plan_sha256"] == RECOVERY_SHA
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution.get("timed_out", False)
            and execution["result_sha256"] == r["terminal_sha256"]
            and execution["root_command_sha256"] == r["command_sha256"], "Recovery execution is incomplete")
    require(terminal["status"] == "recovered_saved_generation_audited_and_paired_full14_complete"
            and terminal["source_bindings_unchanged"] and terminal["original_root_exit_code"] == 1
            and terminal["original_child_exit_code"] == -15 and not terminal["original_monitor_successful"]
            and not terminal["host_stability_proven"] and not terminal["training_replayed"]
            and not terminal["gpu_used"] and not terminal["quality_selected"] and not terminal["plugin_replaced"],
            "Recovery scope or original failure was changed")
    require(r["original_execution"]["actual_exit_code"] == r["production_execution"]["actual_exit_code"] == 1
            and r["original_execution"]["actual_root_session"] == 27228
            and r["original_execution"]["actual_tool_output_chunk"] == "1b6049"
            and r["production_execution"]["source_bindings_unchanged"]
            and r["monitor"]["child_exit_code"] == -15
            and r["monitor"]["status"] == "stopped_by_watchdog"
            and r["monitor"]["reason"] == "RuntimeError('Owned child stopped reporting completed updates')"
            and r["monitor"]["latest_completed_step_seen"] == 4000,
            "Original failed monitor evidence differs")
    require(r["source_sha256"] == plan["source_plan_sha256"] == training["plan_sha256"] == receipt["plan_sha256"] == SOURCE_SHA
            and training["status"] == "pass" and training["checkpoint_written"] and training["source_bindings_unchanged"]
            and training["updates"] == training["ema_updates"] == receipt["step"] == source["config"]["steps"] == 4000,
            "Original saved endpoint is incomplete")
    require(audit["status"] == "pass" and audit["source_bindings_unchanged"]
            and audit["recovery_plan_sha256"] == RECOVERY_SHA
            and audit["step"] == audit["training_journal_updates_verified"] == 4000
            and audit["saved_generation_preceded_alert"] and audit["cpu_only"]
            and not audit["original_monitor_successful"] and not audit["host_stability_proven"]
            and terminal["checkpoint_audit"] == r["audit_binding"], "Recovery artifact audit is incomplete")
    require(audit["optimizer_owner"] == receipt["optimizer_owner"] == "raw-model.pt"
            and audit["saved_optimizer_tensor_count"] == len(receipt["parameter_names"]) == 40
            and audit["algorithmic_latency_samples"] == 256
            and audit["averaged_outputs_and_eight_states_bit_exact"] and audit["ema_reconstructed_without_duplicate_tensors"]
            and receipt["inference_model"] == "model.pt" and receipt["ema_policy"] == source["ema"],
            "Recovered tensor ownership or latency differs")
    states = {"raw": audit["raw_model_state_sha256"], "ema": audit["model_state_sha256"]}
    require(states["raw"] == receipt["raw_model_state_sha256"] == training["final_raw_model_state_sha256"]
            and states["ema"] == receipt["model_state_sha256"] == training["final_model_state_sha256"],
            "Recovered saved tensor identities differ")
    require(set(terminal["checkpoints"]) == set(terminal["quality_results"]) == set(ROLES), "Recovered pair is incomplete")
    models = {}
    for role, filename in (("raw", "raw-model.pt"), ("ema", "model.pt")):
        q = r["quality"][role]
        qp, report, qe = (q[k] for k in ("plan", "result", "execution"))
        checkpoint = terminal["checkpoints"][role]
        require(checkpoint["path"] == str(SOURCE / "production-run/checkpoint" / filename)
                and checkpoint["sha256"] == receipt["files"][filename]["sha256"]
                and checkpoint == qp["checkpoint"] == report["results"][0]["checkpoint"]
                and report["results"][0]["model"]["model_state_sha256_after"] == states[role],
                "Recovered checkpoint role or tensor identity differs")
        require(qp["schema"] == "latency58-branch-memory-full14-plan-v1" and qp["workers"] == 2
                and qp["track_indices"] == list(range(14)) and qp["reference_result"] == source["reference_result"]
                and report["status"] == "pass" and report["source_bindings_unchanged"]
                and report["track_count"] == 14 and report["excerpt_count"] == 28
                and len(report["results"]) == 1 and len(report["results"][0]["tracks"]) == 14
                and report["graph_delay_samples"] == report["host_queue_samples"] == 128
                and report["plan_sha256"] == qe["plan_sha256"] == q["plan_sha256"]
                and terminal["quality_results"][role] == q["result_binding"]
                and terminal["full_sdr_db"][role] == report["results"][0]["aggregate"]["full_sdr_db"],
                "Recovered full14 protocol is incomplete or mismatched")
        require(qe["actual_exit_code"] == 0 and qe["source_bindings_unchanged"] and not qe["timed_out"],
                "Recovered full14 execution failed")
        models[role] = {"kind": "branch_memory", "label": source["name"] + "-" + role,
                        "checkpoint": checkpoint, "model_state_sha256": states[role],
                        "original_full_mixture_report": q["result_binding"]}
    return models


def load_endpoint():
    locations = {"source": SOURCE / "plan.json", "training": SOURCE / "production-run/result.json",
        "receipt": SOURCE / "production-run/checkpoint/receipt.json", "original_execution": SOURCE / "root-execution.json",
        "production_execution": SOURCE / "production-stage/execution.json", "plan": RECOVERY / "plan.json",
        "terminal": RECOVERY / "result.json", "execution": RECOVERY / "root-execution.json",
        "command": RECOVERY / "root-command.json", "audit": RECOVERY / "checkpoint-audit.json"}
    paths = list(locations.values())
    paths.extend(RECOVERY / ("full14-" + role) / (name + ".json") for role in ROLES for name in ("plan", "result", "execution"))
    require(all(p.is_file() for p in paths), "Finish and authenticate both recovery evaluations before vocal views")
    r = {k: read(p) for k, p in locations.items()}
    monitor_path = Path(r["production_execution"]["monitor_result"])
    paths.append(monitor_path)
    r.update(monitor=read(monitor_path), source_sha256=sha(locations["source"]), plan_sha256=sha(locations["plan"]),
             terminal_sha256=sha(locations["terminal"]), command_sha256=sha(locations["command"]),
             audit_binding=binding(locations["audit"]), quality={})
    for role in ROLES:
        directory = RECOVERY / ("full14-" + role)
        r["quality"][role] = {**{k: read(directory / (k + ".json")) for k in ("plan", "result", "execution")},
            "plan_sha256": sha(directory / "plan.json"), "result_binding": binding(directory / "result.json")}
    models = validate_records(r)
    bindings = {str(p): sha(p) for p in paths}
    generation = SOURCE / "production-run/checkpoint"
    require({p.name for p in generation.iterdir()} == set(r["receipt"]["files"]) | {"receipt.json"}, "Saved generation inventory changed")
    for name, record in r["receipt"]["files"].items():
        p = generation / name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == record["bytes"]
                and sha(p) == record["sha256"], "Recovered generation bytes changed")
        merge_bindings(bindings, {str(p): record["sha256"]})
    merge_bindings(bindings, {str(SOURCE / "production-run/metrics.jsonl"): r["receipt"]["metrics_sha256"]})
    for record in (r["source"], r["plan"], *(r["quality"][role]["result"] for role in ROLES)):
        merge_bindings(bindings, {p: digest for p, digest in record["source_bindings"].items() if Path(p).suffix == ".py"})
    return r, models, bindings


def qualify():
    cpu_environment()
    require(not CONTROL.exists(), "Preserve the existing recovery-adapter qualification")
    records, models, bindings = load_endpoint()
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    cases = [
        ("failed_recovery", ("execution", "actual_exit_code"), 1, "Recovery execution"),
        ("timed_out_recovery", ("execution", "timed_out"), True, "Recovery execution"),
        ("concealed_original_failure", ("terminal", "original_monitor_successful"), True, "Recovery scope"),
        ("wrong_original_failure", ("monitor", "child_exit_code"), 0, "Original failed monitor"),
        ("incomplete_update_count", ("training", "updates"), 3999, "Original saved endpoint"),
        ("save_after_alert", ("audit", "saved_generation_preceded_alert"), False, "Recovery artifact audit"),
        ("wrong_optimizer_owner", ("audit", "optimizer_owner"), "model.pt", "Recovered tensor ownership"),
        ("changed_latency", ("audit", "algorithmic_latency_samples"), 512, "Recovered tensor ownership"),
        ("ema_in_raw_role", ("terminal", "checkpoints", "raw"), models["ema"]["checkpoint"], "Recovered checkpoint role"),
        ("wrong_tensor_state", ("quality", "raw", "result", "results", 0, "model", "model_state_sha256_after"),
         models["ema"]["model_state_sha256"], "Recovered checkpoint role"),
        ("incomplete_panel", ("quality", "ema", "result", "track_count"), 13, "Recovered full14 protocol"),
        ("failed_scoring", ("quality", "raw", "execution", "actual_exit_code"), 1, "Recovered full14 execution"),
    ]
    verify_inputs({"source_bindings": bindings})
    original = copy.deepcopy(records)
    require(validate_records(records) == models and records == original, "Validation changed its inputs")
    results = []
    for name, location, value, expected in cases:
        changed = copy.deepcopy(records)
        node = changed
        for key in location[:-1]:
            node = node[key]
        node[location[-1]] = value
        try:
            validate_records(changed)
        except RuntimeError as error:
            require(expected in str(error), "Negative control failed for an unrelated reason: " + name)
            results.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise RuntimeError("Invalid recovered endpoint was accepted: " + name)
    CONTROL.mkdir()
    write(CONTROL / "plan.json", {"schema": "latency58-recovered-vocal-adapter-control-v1", "source_bindings": bindings,
          "negative_cases": [c[0] for c in cases], "model_inference": False, "source_audio_decoded": False})
    verify_inputs({"source_bindings": bindings})
    write(CONTROL / "result.json", {"status": "pass", "source_bindings_unchanged": True,
          "plan_sha256": sha(CONTROL / "plan.json"), "preparer_sha256": sha(__file__), "negative_cases": results,
          "original_failure_preserved": True, "model_inference": False, "source_audio_decoded": False})
    print(json.dumps({"status": "pass", "negative_cases": len(results)}), flush=True)


def prepare(prefix):
    cpu_environment()
    require(prefix and all(c.isalnum() or c in "-_" for c in prefix), "Invalid output prefix")
    out = PHASE / prefix
    require(not out.exists(), "Preserve existing source-view diagnostics")
    check, execution = (read(CONTROL / (n + ".json")) for n in ("result", "execution"))
    require(check["status"] == "pass" and check["source_bindings_unchanged"]
            and check["preparer_sha256"] == sha(__file__) and check["original_failure_preserved"]
            and len(check["negative_cases"]) == 12 and all(c["rejected"] for c in check["negative_cases"])
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution.get("timed_out", False)
            and check["plan_sha256"] == execution["plan_sha256"] == sha(CONTROL / "plan.json")
            and execution["result_sha256"] == sha(CONTROL / "result.json"), "Recovery adapter qualification is incomplete")
    records, models, bindings = load_endpoint()
    template_path = PHASE / "vocal-views-working-001/plan.json"
    template = read(template_path)
    reference_root = PHASE / "deployed-vocal-views-001"
    reference_plan, reference, reference_execution = (read(reference_root / (n + ".json")) for n in ("plan", "result", "execution"))
    require(reference["status"] == "pass" and reference["source_bindings_unchanged"]
            and reference["checkpoint"]["sha256"] == reference_plan["checkpoint"]["sha256"] == RELEASED
            and reference["plan_sha256"] == sha(reference_root / "plan.json")
            and reference_execution["actual_exit_code"] == 0 and reference_execution["source_bindings_unchanged"]
            and not reference_execution["timed_out"] and reference["track_count"] == 14
            and reference["excerpt_count_per_view"] == 28, "Released reference is incomplete")
    fields = ("workers", "track_indices", "track_intervals", "audio_export", "manifest", "config")
    require(all(template[k] == reference_plan[k] for k in fields)
            and template["workers"] == 2 and template["track_indices"] == list(range(14))
            and not template["audio_export"], "Fixed vocal-view protocol differs")
    for role in ROLES:
        tracks = records["quality"][role]["result"]["results"][0]["tracks"]
        require(all(t["name"] == template["track_intervals"][str(i)]["name"]
                    and t["excerpts"] == template["track_intervals"][str(i)]["intervals"]
                    for i, t in enumerate(tracks)), "Recovered full14 and source-view excerpts differ")
    merge_bindings(bindings, reference_plan["source_bindings"])
    paths = [Path(__file__).resolve(), template_path]
    paths.extend(CONTROL / (n + ".json") for n in ("plan", "result", "execution"))
    paths.extend(reference_root / (n + ".json") for n in ("plan", "result", "execution"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "run_latency58_paired_vocal_views.py", "check_latency58_branch_vocal_views.py",
        "evaluate_latency58_branch_vocal_views.py", "compare_latency58_vocal_views.py",
        "report_latency58_branch_gru_int8.py", "run_latency58_deployed_vocal_views.py", "latency58_vocal_views.py"))
    # Keep the previously qualified runner and its qualification immutable.
    old_control = PHASE / "paired-vocal-launcher-check-001"
    old_check, old_execution = (read(old_control / (n + ".json")) for n in ("result", "execution"))
    require(old_check["status"] == "pass" and old_check["source_bindings_unchanged"]
            and old_check["launcher_sha256"] == sha(ROOT / "research/direct/run_latency58_paired_vocal_views.py")
            and old_execution["actual_exit_code"] == old_execution["actual_enclosing_exit_code"] == 0
            and old_execution["source_bindings_unchanged"] and not old_execution["timed_out"]
            and old_execution["result_sha256"] == sha(old_control / "result.json"), "Existing runner qualification changed")
    paths.extend(old_control / (n + ".json") for n in ("plan", "result", "execution"))
    for name in ("manifest", "config"):
        merge_bindings(bindings, {template[name]["path"]: template[name]["sha256"]})
    budget_path = PHASE / "branch-gru-int8-post-ci-storage-001.json"
    paths.append(budget_path)
    merge_bindings(bindings, {str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    budget = read(budget_path)
    before = budget_snapshot(budget)
    outside = (before["external_git_common_bytes"] + budget["other_outside_allowance_bytes"]
               + budget["live_training_save_reservation_bytes"] + budget["diagnostic_artifact_allowance_bytes"])
    plan = {"schema": SCHEMA, "output_directory": str(out), "source_bindings": bindings,
        "source_training_root": str(SOURCE), "source_recovery_root": str(RECOVERY), "models": models,
        "source_recovery_result": binding(RECOVERY / "result.json"),
        "original_training_monitor_successful": False, "host_stability_proven": False,
        "storage_budget": budget, "budget_before": before,
        "checker_budget": {"counted_roots": list(budget["counted_roots"]),
                          "stop_counted_bytes": 90_000_000_000 - outside, "outside_roots_reservation_bytes": outside},
        **{k: template[k] for k in fields}, "protocol_template": binding(template_path),
        "released_reference": {n: binding(reference_root / (n + ".json")) for n in ("plan", "result", "execution")},
        "protocol_version": reference_plan["protocol_version"], "roles": list(ROLES),
        "qualification_timeout_seconds": 600, "evaluation_timeout_seconds": 3000,
        "total_diagnostic_artifact_allowance_bytes": 50_000_000,
        "training_audit_scope": "Original failed teardown retained; independently audited 4000-update generation and completed CPU full14 pair. No training replay or host-health acceptance.",
        "gpu_used": False, "quality_selected": False, "plugin_replaced": False}
    out.mkdir()
    write(out / "plan.json", plan)
    print(json.dumps({"event": "recovered_paired_source_views_prepared", "plan": str(out / "plan.json"),
                      "plan_sha256": sha(out / "plan.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qualify-only", action="store_true")
    group.add_argument("--prefix")
    args = parser.parse_args()
    if args.qualify_only:
        qualify()
    else:
        prepare(args.prefix)
