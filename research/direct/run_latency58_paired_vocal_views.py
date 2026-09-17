"""Qualify and compare vocal views for a completed raw/EMA checkpoint pair.

The original source-view checker, streamer and metrics are reused unchanged.
Training must have exited, published its generation and finished both full14
scores before this launcher can prepare a diagnostic. It never launches GPU
work, chooses weights, changes a checkpoint, or replaces a deployment graph.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot

SCHEMA = "latency58-paired-vocal-views-plan-v1"
ROLES = ("raw", "ema")
RELEASED = "d2945742d27fe23469614aef4f5b79e46fb1a11696ee2c8e6055c494163bcffa"


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def merge_bindings(destination, incoming):
    for path, digest in incoming.items():
        require(path not in destination or destination[path] == digest, "Conflicting frozen input: " + path)
        destination[path] = digest


def cpu_environment():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1 processes")


def validate_records(records):
    """Check identities and terminal evidence before any output or inference."""
    source, terminal, training, audit, receipt = (records[k] for k in ("source", "terminal", "training", "audit", "receipt"))
    root_execution, production_execution, monitor = (records[k] for k in ("root_execution", "production_execution", "monitor"))
    require(source["schema"] in ("latency58-branch-memory-training-plan-v1", "latency58-grouped-vocal-training-plan-v1")
            and terminal["status"] == "training_audit_and_paired_full14_complete"
            and set(terminal["checkpoints"]) == set(terminal["quality_results"]) == set(ROLES), "Incomplete paired endpoint")
    require(root_execution["actual_exit_code"] == 0 and root_execution["source_bindings_unchanged"]
            and not root_execution.get("timed_out", False)
            and root_execution["plan_sha256"] == records["plan_sha256"]
            and root_execution["result_sha256"] == records["terminal_sha256"]
            and root_execution["root_command_sha256"] == records["root_command_sha256"], "Root execution is not authenticated")
    require(production_execution["actual_exit_code"] == 0 and production_execution["source_bindings_unchanged"]
            and not production_execution.get("timed_out", False)
            and production_execution["plan_sha256"] == records["plan_sha256"]
            and monitor["status"] == monitor["supervisor_health"] == "pass"
            and monitor["child_exit_code"] == 0 and monitor["post_exit_quiet_completed"], "Training monitor did not finish cleanly")
    require(training["status"] == "pass" and training["checkpoint_written"] and training["source_bindings_unchanged"]
            and training["plan_sha256"] == records["plan_sha256"]
            and audit["status"] == "pass" and audit["source_bindings_unchanged"]
            and receipt["schema"] == "latency58-branch-ema-generation-v1"
            and receipt["plan_sha256"] == records["plan_sha256"]
            and training["updates"] == training["ema_updates"] == audit["step"] == receipt["step"] == source["config"]["steps"],
            "Training, save or audit endpoint differs")
    require(audit["saved_optimizer_tensor_count"] == len(receipt["parameter_names"]) == 40
            and audit["algorithmic_latency_samples"] == 256
            and audit["averaged_outputs_and_eight_states_bit_exact"]
            and audit["ema_reconstructed_without_duplicate_tensors"]
            and audit["optimizer_owner"] == receipt["optimizer_owner"] == "raw-model.pt"
            and receipt["inference_model"] == "model.pt" and receipt["ema_policy"] == source["ema"], "Raw/EMA ownership or audit differs")
    require(training["checkpoint"] == audit["checkpoint"] == terminal["checkpoints"]["ema"], "EMA checkpoint identity differs")
    states = {"raw": audit["raw_model_state_sha256"], "ema": audit["model_state_sha256"]}
    require(states["raw"] == receipt["raw_model_state_sha256"] == training["final_raw_model_state_sha256"]
            and states["ema"] == receipt["model_state_sha256"] == training["final_model_state_sha256"], "Saved tensor identities differ")
    models = {}
    for role, filename in (("raw", "raw-model.pt"), ("ema", "model.pt")):
        q = records["quality"][role]
        plan, result, execution = (q[k] for k in ("plan", "result", "execution"))
        checkpoint = terminal["checkpoints"][role]
        require(checkpoint["path"] == str(Path(source["output_directory"]) / "production-run/checkpoint" / filename)
                and checkpoint["sha256"] == receipt["files"][filename]["sha256"]
                and plan["checkpoint"] == checkpoint and result["results"][0]["checkpoint"] == checkpoint
                and result["results"][0]["model"]["model_state_sha256_after"] == states[role], "Raw/EMA role or tensor identity differs")
        require(plan["schema"] == "latency58-branch-memory-full14-plan-v1" and plan["workers"] == 2
                and plan["track_indices"] == list(range(14))
                and result["status"] == "pass" and result["source_bindings_unchanged"]
                and result["track_count"] == 14 and result["excerpt_count"] == 28
                and len(result["results"]) == 1 and len(result["results"][0]["tracks"]) == 14
                and result["graph_delay_samples"] == result["host_queue_samples"] == 128
                and result["plan_sha256"] == execution["plan_sha256"] == q["plan_sha256"]
                and terminal["quality_results"][role] == q["result_binding"]
                and terminal["full_sdr_db"][role] == result["results"][0]["aggregate"]["full_sdr_db"], "Original full14 score is incomplete or mismatched")
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "Full14 execution did not finish successfully")
        models[role] = {"kind": "branch_memory", "label": source["name"] + "-" + role,
                        "checkpoint": checkpoint, "model_state_sha256": states[role],
                        "original_full_mixture_report": q["result_binding"]}
    return models


def load_endpoint(root):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(PHASE), "Use a retained local endpoint")
    locations = {"source": "plan.json", "terminal": "result.json", "root_execution": "root-execution.json",
        "root_command": "root-command.json", "training": "production-run/result.json",
        "production_execution": "production-stage/execution.json", "audit": "checkpoint-audit.json",
        "receipt": "production-run/checkpoint/receipt.json"}
    required = [root / name for name in locations.values()]
    required.extend(root / ("full14-" + role) / name for role in ROLES for name in ("plan.json", "result.json", "execution.json"))
    missing = [str(p) for p in required if not p.is_file()]
    require(not missing, "Finish the saved endpoint and record actual root completion before preparing vocal views: " + str(missing))
    records = {key: read(root / name) for key, name in locations.items()}
    require(Path(records["source"]["output_directory"]).resolve() == root, "Source plan belongs to another root")
    records.update(plan_sha256=sha(root / "plan.json"), terminal_sha256=sha(root / "result.json"),
                   root_command_sha256=sha(root / "root-command.json"), quality={})
    monitor_path = Path(records["production_execution"]["monitor_result"])
    records["monitor"] = read(monitor_path)
    required.append(monitor_path)
    for role in ROLES:
        directory = root / ("full14-" + role)
        records["quality"][role] = {**{k: read(directory / (k + ".json")) for k in ("plan", "result", "execution")},
            "plan_sha256": sha(directory / "plan.json"), "result_binding": binding(directory / "result.json")}
    models = validate_records(records)
    generation = root / "production-run/checkpoint"
    receipt = records["receipt"]
    require(set(receipt["files"]) == {"raw-model.pt", "model.pt", "raw-optimizer.pt", "ema.json"}
            and {p.name for p in generation.iterdir()} == set(receipt["files"]) | {"receipt.json"}, "Saved generation inventory changed")
    bindings = {str(p): sha(p) for p in required}
    for name, expected in receipt["files"].items():
        path = generation / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected["bytes"], "Saved generation file differs")
        merge_bindings(bindings, {str(path): expected["sha256"]})
    merge_bindings(bindings, {str(root / "production-run/metrics.jsonl"): receipt["metrics_sha256"]})
    # The completed training audit owns training-corpus verification. Retain
    # its complete frozen documents and verify all code dependencies here;
    # inference needs only this generation and the fixed validation sources.
    for record in (records["source"], *(records["quality"][r]["result"] for r in ROLES)):
        merge_bindings(bindings, {p: digest for p, digest in record["source_bindings"].items() if Path(p).suffix == ".py"})
    return records, models, bindings


def prepare(quality_root, output_prefix):
    cpu_environment()
    require(output_prefix and all(c.isalnum() or c in "-_" for c in output_prefix), "Invalid output prefix")
    out = PHASE / output_prefix
    require(not out.exists(), "Preserve earlier diagnostics")
    control = PHASE / "paired-vocal-launcher-check-001"
    check, check_execution = read(control / "result.json"), read(control / "execution.json")
    require(check["status"] == "pass" and check["source_bindings_unchanged"]
            and check["launcher_sha256"] == sha(__file__) and check["live_endpoint_refused"]
            and check["plan_sha256"] == sha(control / "plan.json")
            and len(check["negative_cases"]) == 10 and all(c["rejected"] for c in check["negative_cases"])
            and check_execution["actual_exit_code"] == 0 and check_execution["actual_enclosing_exit_code"] == 0
            and check_execution["source_bindings_unchanged"] and not check_execution["timed_out"]
            and check_execution["result_sha256"] == sha(control / "result.json"), "Launcher control qualification is incomplete")
    records, models, bindings = load_endpoint(quality_root)
    template_path = PHASE / "vocal-views-working-001/plan.json"
    template = read(template_path)
    reference_root = PHASE / "deployed-vocal-views-001"
    reference_plan, reference, reference_execution = (read(reference_root / name) for name in ("plan.json", "result.json", "execution.json"))
    require(reference["status"] == "pass" and reference["source_bindings_unchanged"]
            and reference["checkpoint"]["sha256"] == reference_plan["checkpoint"]["sha256"] == RELEASED
            and reference["plan_sha256"] == sha(reference_root / "plan.json")
            and reference_execution["actual_exit_code"] == 0 and reference_execution["source_bindings_unchanged"]
            and not reference_execution["timed_out"] and reference["track_count"] == 14
            and reference["excerpt_count_per_view"] == 28, "Released source-view baseline is incomplete")
    fields = ("workers", "track_indices", "track_intervals", "audio_export", "manifest", "config")
    require(all(template[k] == reference_plan[k] for k in fields)
            and template["workers"] == 2 and template["track_indices"] == list(range(14))
            and not template["audio_export"], "Fixed source-view protocol differs")
    for role in ROLES:
        tracks = records["quality"][role]["result"]["results"][0]["tracks"]
        require(all(t["name"] == template["track_intervals"][str(i)]["name"]
                    and t["excerpts"] == template["track_intervals"][str(i)]["intervals"]
                    for i, t in enumerate(tracks)), "Original full14 and vocal-view excerpts differ")
    merge_bindings(bindings, reference_plan["source_bindings"])
    paths = [Path(__file__).resolve(), template_path]
    paths.extend(control / name for name in ("plan.json", "result.json", "execution.json"))
    paths.append(ROOT / "research/direct/check_latency58_paired_vocal_views_launcher.py")
    paths.extend(reference_root / name for name in ("plan.json", "result.json", "execution.json"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "check_latency58_branch_vocal_views.py", "evaluate_latency58_branch_vocal_views.py",
        "compare_latency58_vocal_views.py", "report_latency58_branch_gru_int8.py",
        "run_latency58_deployed_vocal_views.py", "latency58_vocal_views.py"))
    for name in ("manifest", "config"):
        merge_bindings(bindings, {template[name]["path"]: template[name]["sha256"]})
    budget_path = PHASE / "branch-gru-int8-post-ci-storage-001.json"
    paths.append(budget_path)
    merge_bindings(bindings, {str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    budget = read(budget_path)
    before = budget_snapshot(budget)
    # Adapt the old checker's two budget fields conservatively. The enclosing
    # launcher also measures the complete external Git directory before/after
    # every stage instead of treating the old counted-root limit as the cap.
    outside = (before["external_git_common_bytes"] + budget["other_outside_allowance_bytes"]
               + budget["live_training_save_reservation_bytes"] + budget["diagnostic_artifact_allowance_bytes"])
    legacy_budget = {"counted_roots": list(budget["counted_roots"]),
                     "stop_counted_bytes": 90_000_000_000 - outside, "outside_roots_reservation_bytes": outside}
    plan = {"schema": SCHEMA, "output_directory": str(out), "source_bindings": bindings,
        "source_training_root": str(Path(quality_root).resolve()), "models": models,
        "storage_budget": budget, "budget_before": before, "checker_budget": legacy_budget,
        **{key: template[key] for key in fields}, "protocol_template": binding(template_path),
        "released_reference": {name: binding(reference_root / (name + ".json")) for name in ("plan", "result", "execution")},
        "protocol_version": reference_plan["protocol_version"], "roles": list(ROLES),
        "qualification_timeout_seconds": 600, "evaluation_timeout_seconds": 3000,
        "total_diagnostic_artifact_allowance_bytes": 50_000_000,
        "training_audit_scope": "Authenticated completed root/monitor/save/full14 evidence and generation bytes; no training replay or training-audio decoding.",
        "gpu_used": False, "quality_selected": False, "plugin_replaced": False}
    out.mkdir()
    write(out / "plan.json", plan)
    print(json.dumps({"event": "paired_source_views_prepared", "plan": str(out / "plan.json"),
                      "plan_sha256": sha(out / "plan.json"), "roles": list(ROLES)}), flush=True)
    return out / "plan.json"


def run(plan_path, expected_sha256):
    cpu_environment()
    plan_path = Path(plan_path).resolve(strict=True)
    require(sha(plan_path) == expected_sha256, "Paired source-view plan changed")
    plan = read(plan_path)
    require(plan["schema"] == SCHEMA and plan["roles"] == list(ROLES)
            and plan["track_indices"] == list(range(14)) and plan["workers"] == 2
            and not plan["audio_export"] and not plan["quality_selected"] and not plan["plugin_replaced"], "Diagnostic scope differs")
    out = Path(plan["output_directory"])
    require(out == plan_path.parent and out.is_relative_to(PHASE) and not (out / "result.json").exists(), "Preserve existing result")
    bindings = {**plan["source_bindings"], str(plan_path): expected_sha256}
    verify_inputs({"source_bindings": bindings})
    began = time.monotonic()
    observations, outputs = [budget_snapshot(plan["storage_budget"])], {}
    for role in ROLES:
        destination, qualification = out / role, out / role / "qualification"
        require(not destination.exists(), "Preserve earlier endpoint evaluation")
        qualification.mkdir(parents=True)
        model = plan["models"][role]
        functional = {**plan["checker_budget"], "schema": "latency58-branch-vocal-views-functional-plan-v1",
            "checkpoint": model["checkpoint"], "model_state_sha256": model["model_state_sha256"],
            "source_bindings": bindings, "output_directory": str(qualification)}
        write(qualification / "plan.json", functional)
        execute([PYTHON, "-u", "-m", "research.direct.check_latency58_branch_vocal_views", "--plan",
            str(qualification / "plan.json"), "--plan-sha256", sha(qualification / "plan.json")],
            qualification, "evaluation", plan["qualification_timeout_seconds"], bindings,
            {"plan_sha256": sha(qualification / "plan.json")})
        observations.append(budget_snapshot(plan["storage_budget"]))
        qualified = read(qualification / "result.json")
        require(qualified["status"] == "pass" and qualified["source_bindings_unchanged"]
                and qualified["plan_sha256"] == sha(qualification / "plan.json")
                and qualified["checkpoint"] == model["checkpoint"]
                and qualified["model_state_sha256"] == model["model_state_sha256"], "Wrong endpoint qualification")
        role_bindings = dict(bindings)
        for name in ("plan.json", "result.json", "execution.json"):
            path = qualification / name
            merge_bindings(role_bindings, {str(path): sha(path)})
        evaluation = {**plan["checker_budget"], **{k: plan[k] for k in
            ("workers", "track_indices", "track_intervals", "audio_export", "manifest", "config")},
            "schema": "latency58-branch-vocal-views-evaluation-plan-v1", "output_directory": str(destination),
            "model": model, "qualification": binding(qualification / "result.json"),
            "qualification_execution": binding(qualification / "execution.json"), "source_bindings": role_bindings,
            "protocol_template": plan["protocol_template"], "diagnostic_artifact_allowance_bytes": 10_000_000,
            "concurrent_training_reservation_bytes": 440_000_000}
        write(destination / "plan.json", evaluation)
        execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_branch_vocal_views", "--plan",
            str(destination / "plan.json"), "--plan-sha256", sha(destination / "plan.json")],
            destination, "evaluation", plan["evaluation_timeout_seconds"], role_bindings,
            {"plan_sha256": sha(destination / "plan.json")})
        observations.append(budget_snapshot(plan["storage_budget"]))
        result = read(destination / "result.json")
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and result["plan_sha256"] == sha(destination / "plan.json")
                and result["model"] == model and result["version"] == plan["protocol_version"]
                and [r["index"] for r in result["tracks"]] == list(range(14)), "Incomplete source-view endpoint")
        outputs[role] = result
        for path in [directory / name for directory in (destination, qualification)
                     for name in ("plan.json", "result.json", "execution.json")]:
            merge_bindings(bindings, {str(path): sha(path)})
        print(json.dumps({"event": "endpoint_source_views_complete", "role": role,
            "instrumental_vocal_output_dbfs": result["aggregate"]["instrumental"]["per_stem"]["vocals"]["output_rms_dbfs"]}), flush=True)
    from research.direct.compare_latency58_vocal_views import compare_reports
    from research.direct.report_latency58_branch_gru_int8 import compare_windows
    reference_plan = read(plan["released_reference"]["plan"]["path"])
    reference = read(plan["released_reference"]["result"]["path"])
    released = {"version": reference_plan["protocol_version"], "model": reference["checkpoint"],
                "tracks": reference["tracks"], "aggregate": reference["aggregate"]}
    reports = {"released": released, **outputs}
    comparisons = {}
    for name, left, right in (("raw_vs_released", "released", "raw"),
                              ("ema_vs_released", "released", "ema"), ("ema_vs_raw", "raw", "ema")):
        comparisons[name] = {"aggregate": compare_reports(reports[left], reports[right]),
                             "windows": compare_windows(reports[left]["tracks"], reports[right]["tracks"])}
    verify_inputs({"source_bindings": bindings})
    result = {"schema": "latency58-paired-vocal-views-result-v1", "status": "pass",
        "plan_sha256": expected_sha256, "source_bindings": bindings, "source_bindings_unchanged": True,
        "models": plan["models"], "track_count_per_endpoint": 14, "excerpt_count_per_view_per_endpoint": 28,
        "reports": {role: {name: binding(out / role / (name + ".json")) for name in ("plan", "result", "execution")}
                    for role in ROLES}, "comparisons": comparisons,
        "budget_observations": observations, "budget_after": budget_snapshot(plan["storage_budget"]),
        "elapsed_seconds": time.monotonic() - began, "gpu_used": False, "training_updates": 0,
        "quality_selected": False, "plugin_replaced": False, "overall_goal_complete": False,
        "limitations": ["The candidates are saved FP32 source models; the reference is the exact released quantized graph.",
            "A new deployment graph requires its own export, numerical, quality and callback qualification.",
            "These development source-remix views supplement the original full-mixture scores; they are not unseen tests.",
            "Desired vocal gain, real-vocal quality and Other accompany leakage; global attenuation is not successful separation.",
            "No M4 timing or human listening acceptance is established."]}
    encoded = json.dumps(result, indent=2, allow_nan=False).encode()
    require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) + len(encoded) + 1_000_000
            < plan["total_diagnostic_artifact_allowance_bytes"], "Diagnostic artifact reservation exceeded")
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "roles": list(ROLES), "comparisons": list(comparisons),
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--quality-root", type=Path)
    source.add_argument("--prepared-plan", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.quality_root is not None:
        require(args.prefix is not None and args.plan_sha256 is None, "Provide a new prefix when preparing")
        path = prepare(args.quality_root, args.prefix)
        if not args.prepare_only:
            run(path, sha(path))
    else:
        require(args.plan_sha256 is not None and args.prefix is None and not args.prepare_only,
                "Execute the already frozen plan using its hash")
        run(args.prepared_plan, args.plan_sha256)


if __name__ == "__main__":
    main()
