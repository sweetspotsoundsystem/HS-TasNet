"""Audit and score both packed roles with the unchanged full14 and vocal protocols."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.run_latency58_paired_vocal_serial import binding, merge_bindings
from research.direct.run_latency58_weighted_vocal import require_monitor_closed
from research.direct.latency58_weighted_storage import snapshot as storage_snapshot

ROLES = ("raw", "ema")
OUT = PHASE / "branch-weighted-vocal-014"
VIEWS = PHASE / "paired-vocal-weighted-014"
REVIEW = PHASE / "weighted-vocal-review-014"


def function(path, name):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(nodes) == 1, "Missing or ambiguous evaluation function")
    return nodes[0]


def transport_proof():
    root = ROOT / "research/direct"
    proof = []
    for previous, current, name in (
        ("evaluate_latency58_branch_memory.py", "evaluate_latency58_packed_branch_memory.py", "score"),
        ("evaluate_latency58_branch_vocal_views.py", "evaluate_latency58_packed_branch_vocal_views.py", "score_track")):
        left, right = root / previous, root / current
        require(ast.dump(function(left, name), include_attributes=False)
                == ast.dump(function(right, name), include_attributes=False), "Scoring function changed")
        proof.append({"function": name, "reference": binding(left), "packed": binding(right), "ast_identical": True})
    # Keep the independent rendering and metric fixtures, including the muted
    # output negative control, verbatim while changing checkpoint/file plumbing.
    left = function(root / "check_latency58_branch_vocal_views.py", "main").body
    right = function(root / "check_latency58_packed_branch_vocal_views.py", "main").body
    def scientific_slice(body):
        first = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "intervals" for t in node.targets))
        last = next(i for i, node in enumerate(body[first:], first) if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "require" and node.value.args
                    and "Qualification changed weights" in ast.unparse(node.value))
        return [ast.dump(node, include_attributes=False) for node in body[first:last + 1]]
    require(scientific_slice(left) == scientific_slice(right), "Source-view rendering or metric fixture changed")
    return {"status": "pass", "scoring_functions": proof, "source_view_independent_fixture_ast_identical": True,
            "changes": ["Authenticate explicit packed raw/EMA roles and state hashes.",
                        "Apply the allocated storage accounting.", "Remove only temporary synthetic fixture audio."]}


def require_execution(directory, *, actual_root=False):
    execution = read(directory / ("root-execution.json" if actual_root else "execution.json"))
    require(execution["actual_exit_code"] == 0 and not execution.get("timed_out", False)
            and execution["source_bindings_unchanged"], "Incomplete evaluation execution")
    return execution


def prepare(plan_path):
    from research.direct.train_latency58_weighted_vocal import validate_recipe
    source = read(plan_path)
    require(Path(source["output_directory"]) == OUT and plan_path == OUT / "plan.json", "Wrong training trial")
    validate_recipe(source)
    require(not (OUT / "quality-plan.json").exists() and not VIEWS.exists() and not REVIEW.exists(),
            "Preserve existing quality evidence")
    trained = read(OUT / "production-run/result.json")
    execution = read(OUT / "production-stage/execution.json")
    root_execution = read(OUT / "production-root-execution.json")
    require_monitor_closed(execution, read(execution["monitor_result"]), final_step=1000)
    require(root_execution["actual_exit_code"] == 0 and not root_execution["timed_out"]
            and root_execution["source_bindings_unchanged"]
            and root_execution["plan_sha256"] == sha(plan_path)
            and root_execution["stage_execution_sha256"] == sha(OUT / "production-stage/execution.json")
            and root_execution["result_sha256"] == sha(OUT / "production-run/result.json"),
            "Record actual production controller completion before quality evaluation")
    require(trained["status"] == "pass" and trained["source_bindings_unchanged"] and trained["checkpoint_written"]
            and trained["plan_sha256"] == sha(plan_path)
            and trained["updates"] == trained["ema_updates"] == trained["checkpoint"]["step"] == 1000
            and trained["checkpoint_roles"] == {"raw": trained["final_raw_model_state_sha256"],
                                                "ema": trained["final_model_state_sha256"]}
            and trained["packed_final_reuses_rolling_file"], "Saved weighted endpoint is incomplete")
    bindings = dict(source["source_bindings"])
    paths = [plan_path, Path(__file__).resolve(), OUT / "production-run/result.json", OUT / "production-stage/execution.json",
             OUT / "production-root-execution.json", OUT / "production-root-command.json", Path(execution["monitor_result"]),
             Path(trained["checkpoint"]["path"]), Path(trained["checkpoint"]["receipt"]), OUT / "production-run/metrics.jsonl"]
    paths.extend((OUT / "production-run/packed-recovery-receipts").glob("step-*.json"))
    paths.extend(ROOT / "research/direct" / name for name in ("latency58_packed_evaluation.py",
        "evaluate_latency58_packed_branch_memory.py", "check_latency58_packed_branch_vocal_views.py",
        "evaluate_latency58_packed_branch_vocal_views.py", "check_latency58_lossless_recovery_cpu_v3.py",
        "report_latency58_grouped_continuation.py", "report_latency58_grouped_vocal_parent.py"))
    template_path = PHASE / "paired-vocal-grouped-013/plan.json"
    template = read(template_path)
    merge_bindings(bindings, template["source_bindings"])
    paths.append(template_path)
    merge_bindings(bindings, {str(path): sha(path) for path in paths})
    models = {role: {"kind": "branch_memory", "label": source["name"] + "-" + role,
        "checkpoint": trained["checkpoint"], "checkpoint_role": role,
        "model_state_sha256": trained["checkpoint_roles"][role]} for role in ROLES}
    proof = transport_proof()
    plan = {"schema": "latency58-packed-weighted-quality-plan-v1", "packed_training_plan": binding(plan_path),
        "source_bindings": bindings, "models": models, "transport_proof": proof,
        "workers": 2, "track_indices": list(range(14)), "roles": list(ROLES),
        **{k: template[k] for k in ("track_intervals", "audio_export", "manifest", "config", "protocol_template",
                                  "protocol_version", "released_reference")},
        "reference_result": source["reference_result"], "source_training_root": str(OUT),
        "output_directory": str(OUT), "paired_output_directory": str(VIEWS), "review_output_directory": str(REVIEW),
        "full14_timeout_seconds": 1800, "source_view_qualification_timeout_seconds": 600,
        "source_view_evaluation_timeout_seconds": 3000, "budget_before": storage_snapshot(source),
        "quality_selected": False, "plugin_replaced": False, "overall_goal_complete": False}
    verify_inputs(plan)
    write(OUT / "quality-plan.json", plan)
    return plan


def audit_saved(plan):
    import torch
    from research.direct.latency58_packed_evaluation import load_model, training_plan
    from research.direct.latency58_lossless_recovery_files_v3 import read_snapshot, CURRENT, PENDING, FINAL
    from research.direct.check_latency58_lossless_recovery_cpu_v3 import render_parity
    source, plan_sha = training_plan(plan)
    trained = read(OUT / "production-run/result.json")
    checkpoint = trained["checkpoint"]
    generation = OUT / "production-run"
    require(Path(checkpoint["path"]) == generation / FINAL and not (generation / CURRENT).exists()
            and not (generation / PENDING).exists(), "Saved generation is not final")
    snapshot, audited = read_snapshot(checkpoint, source, plan_sha)
    raw, resume, averaged, ema = audited
    require(snapshot["journal"] == (generation / "metrics.jsonl").read_bytes()
            and snapshot["step"] == snapshot["planned_stop_step"] == 1000
            and state_sha256(raw.state_dict()) == trained["final_raw_model_state_sha256"]
            and state_sha256(averaged.state_dict()) == trained["final_model_state_sha256"]
            and len(resume["optimizer"]["state"]) == len(resume["parameter_names"]) == 40
            and ema.updates == 1000 and not torch.cuda.is_initialized(), "Saved optimizer or model endpoint differs")
    receipts = sorted((generation / "packed-recovery-receipts").glob("step-*.json"))
    require([path.name for path in receipts] == [f"step-{step:06d}.json" for step in range(50, 1001, 50)],
            "Packed recovery receipt schedule differs")
    previous = None
    for path, step in zip(receipts, range(50, 1001, 50), strict=True):
        receipt = read(path)
        require(receipt["step"] == step and receipt["plan_sha256"] == plan_sha and receipt["previous"] == previous
                and receipt["planned_stop_step"] == 1000 and 0 < receipt["bytes"] <= 380_000_000,
                "Packed generation receipt chain differs")
        previous = {key: receipt[key] for key in ("step", "sha256", "bytes")}
    require(len(trained["recovery_checkpoints"]) == 19 and trained["final_save_seconds"] < 60
            and all(row["complete_save_seconds"] < 60 for row in trained["recovery_checkpoints"]),
            "A packed save exceeded its prepared bound")
    parity = {}
    for role, expected in (("raw", raw), ("ema", averaged)):
        loaded, payload = load_model(plan, plan["models"][role])
        parity[role] = render_parity(expected, loaded)
        require(payload["provenance"]["training_updates"] == source["parent_training_updates"] + 1000,
                "Packed endpoint lineage differs")
        del loaded, payload
    verify_inputs(plan)
    result = {"schema": "latency58-packed-weighted-saved-audit-v1", "status": "pass",
        "plan_sha256": sha(OUT / "quality-plan.json"), "training_plan_sha256": plan_sha,
        "checkpoint": checkpoint, "checkpoint_roles": trained["checkpoint_roles"],
        "step": 1000, "saved_optimizer_tensor_count": 40, "optimizer_owner": "raw",
        "complete_journal_matches_disk": True, "receipt_chain_generations": 20,
        "raw_and_ema_native_parity": parity, "algorithmic_latency_samples": 256,
        "unpacked_model_files_written": False, "source_bindings_unchanged": True,
        "storage_after": storage_snapshot(source), "gpu_used": False}
    write(OUT / "checkpoint-audit.json", result)
    return result


def run_stage(module, directory, prepared, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory / "plan.json").exists(), "Preserve previous CPU stage")
    write(directory / "plan.json", prepared)
    bindings = {**prepared["source_bindings"], str(directory / "plan.json"): sha(directory / "plan.json")}
    execute([PYTHON, "-u", "-m", module, "--plan", str(directory / "plan.json"),
             "--plan-sha256", sha(directory / "plan.json")], directory, "evaluation", timeout, bindings,
            {"plan_sha256": sha(directory / "plan.json")})
    result = read(directory / "result.json")
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and result["plan_sha256"] == sha(directory / "plan.json"), "CPU stage result differs")
    return result


def evaluate(plan):
    from research.direct.latency58_packed_evaluation import training_plan
    source, _ = training_plan(plan)
    bindings = {**plan["source_bindings"], str(OUT / "quality-plan.json"): sha(OUT / "quality-plan.json"),
                str(OUT / "checkpoint-audit.json"): sha(OUT / "checkpoint-audit.json")}
    full, views = {}, {}
    VIEWS.mkdir()
    write(VIEWS / "plan.json", {**plan, "output_directory": str(VIEWS)})
    for role in ROLES:
        model = plan["models"][role]
        full_plan = {"schema": "latency58-packed-branch-memory-full14-plan-v1", **model,
            "reference_result": plan["reference_result"], "workers": 2, "track_indices": list(range(14)),
            "packed_training_plan": plan["packed_training_plan"], "source_bindings": dict(bindings),
            "output_directory": str(OUT / ("full14-" + role))}
        result = run_stage("research.direct.evaluate_latency58_packed_branch_memory", OUT / ("full14-" + role),
                           full_plan, plan["full14_timeout_seconds"])
        require(result["track_count"] == 14 and result["excerpt_count"] == 28
                and result["checkpoint_role"] == role and result["model_state_sha256"] == model["model_state_sha256"]
                and result["results"][0]["checkpoint"] == model["checkpoint"]
                and result["results"][0]["model"]["model_state_sha256_after"] == model["model_state_sha256"],
                "Full14 scored a different packed role")
        full[role] = result
        storage_snapshot(source)
        for name in ("plan.json", "result.json", "execution.json"):
            path = OUT / ("full14-" + role) / name
            bindings[str(path)] = sha(path)
        qualification = VIEWS / role / "qualification"
        qualified = run_stage("research.direct.check_latency58_packed_branch_vocal_views", qualification,
            {"schema": "latency58-packed-branch-vocal-views-functional-plan-v1", **model,
             "packed_training_plan": plan["packed_training_plan"], "source_bindings": dict(bindings),
             "output_directory": str(qualification)}, plan["source_view_qualification_timeout_seconds"])
        require(qualified["checkpoint_role"] == role and qualified["checkpoint"] == model["checkpoint"]
                and qualified["model_state_sha256"] == model["model_state_sha256"] and qualified["synthetic_audio_removed"],
                "Source views qualified a different role")
        for name in ("plan.json", "result.json", "execution.json"):
            path = qualification / name
            bindings[str(path)] = sha(path)
        view_plan = {"schema": "latency58-packed-branch-vocal-views-evaluation-plan-v1",
            **{k: plan[k] for k in ("workers", "track_indices", "track_intervals", "audio_export", "manifest", "config",
                                    "protocol_template", "packed_training_plan")},
            "model": model, "output_directory": str(VIEWS / role), "source_bindings": dict(bindings),
            "qualification": binding(qualification / "result.json"),
            "qualification_execution": binding(qualification / "execution.json"),
            "diagnostic_artifact_allowance_bytes": 10_000_000}
        views[role] = run_stage("research.direct.evaluate_latency58_packed_branch_vocal_views", VIEWS / role,
                               view_plan, plan["source_view_evaluation_timeout_seconds"])
        require(views[role]["model"] == model and views[role]["version"] == plan["protocol_version"]
                and [row["index"] for row in views[role]["tracks"]] == list(range(14))
                and all(track["excerpts"] == plan["track_intervals"][str(index)]["intervals"]
                        for index, track in enumerate(full[role]["results"][0]["tracks"])),
                "Full14 and source-view panels differ")
        for name in ("plan.json", "result.json", "execution.json"):
            path = VIEWS / role / name
            bindings[str(path)] = sha(path)
        storage_snapshot(source)
        print(json.dumps({"event": "packed_role_evaluated", "role": role,
                          "full_sdr_db": full[role]["results"][0]["aggregate"]["full_sdr_db"]}), flush=True)
    verify_inputs({"source_bindings": bindings})
    return full, views, bindings


def review(plan, full, views, bindings):
    from research import evaluate as legacy
    from research.direct.report_latency58_grouped_continuation import compare_endpoint
    from research.direct.report_latency58_grouped_vocal_parent import paired_root, full_report
    from research.direct.report_latency58_vocal_focus import load_views
    from research.direct.report_latency58_branch_output_int8 import require_close
    from research.direct.latency58_packed_evaluation import training_plan
    from research.direct.compare_latency58_vocal_views import compare_reports
    from research.direct.report_latency58_branch_gru_int8 import compare_windows
    source, _ = training_plan(plan)
    decision = read(source["scientific_decision"]["path"])
    references = {}
    for label, directory, role, decision_role in (
        ("starting_parent", "paired-vocal-grouped-012", "ema", "starting_parent"),
        ("retained_best", "paired-vocal-long-context-006", "ema", "retained_best"),
        ("unweighted_raw", "paired-vocal-grouped-013", "raw", "raw"),
        ("unweighted_ema", "paired-vocal-grouped-013", "ema", "ema")):
        directory = PHASE / directory
        ref_plan, ref_root = paired_root(directory, bindings)
        require(all(ref_plan[k] == plan[k] for k in
                    ("manifest", "config", "track_indices", "track_intervals", "protocol_version")),
                "Reference physical panel differs")
        ref_views = load_views(directory / role, bindings)
        expected = decision["preserved_models"][decision_role]
        require(ref_views["model"] == ref_root["models"][role] == expected, "Historical reference role differs")
        path = Path(expected["original_full_mixture_report"]["path"])
        ref_full, _ = full_report(path.parent, expected["checkpoint"], expected["model_state_sha256"], bindings)
        require_close(ref_full["aggregate"], legacy._aggregate_tracks(ref_full["tracks"]), label + " aggregate")
        references[label] = {"full": ref_full, "views": ref_views}
    candidates = {role: {"full": full[role]["results"][0], "views": views[role]} for role in ROLES}
    for role in ROLES:
        require_close(candidates[role]["full"]["aggregate"],
                      legacy._aggregate_tracks(candidates[role]["full"]["tracks"]), role + " aggregate")
    comparisons = {label: {role: compare_endpoint(reference, candidates[role]) for role in ROLES}
                   for label, reference in references.items()}
    comparisons["ema_vs_raw"] = compare_endpoint(candidates["raw"], candidates["ema"])
    for bound in plan["released_reference"].values():
        require(sha(bound["path"]) == bound["sha256"], "Released reference evidence changed")
        merge_bindings(bindings, {bound["path"]: bound["sha256"]})
    released_plan, released_result, released_execution = (read(plan["released_reference"][name]["path"])
                                                         for name in ("plan", "result", "execution"))
    require(released_result["status"] == "pass" and released_result["source_bindings_unchanged"]
            and released_execution["actual_exit_code"] == 0 and not released_execution["timed_out"]
            and released_execution["source_bindings_unchanged"]
            and released_result["plan_sha256"] == released_execution["plan_sha256"]
                == plan["released_reference"]["plan"]["sha256"]
            and all(released_plan[k] == plan[k] for k in
                    ("manifest", "config", "track_indices", "track_intervals", "protocol_version")),
            "Released source-view baseline is incomplete or uses another protocol")
    released = {"version": released_plan["protocol_version"], "model": released_result["checkpoint"],
                "tracks": released_result["tracks"], "aggregate": released_result["aggregate"]}
    released_comparisons = {role + "_vs_released": {"aggregate": compare_reports(released, views[role]),
        "windows": compare_windows(released["tracks"], views[role]["tracks"])} for role in ROLES}
    for role in ROLES:
        require_close(comparisons["starting_parent"][role]["full_mixture"], full[role]["comparison"], "stored full14 comparison")
        require_close(comparisons["starting_parent"][role]["all_track_stem_cells"], full[role]["all_track_stem_cells"], "stored full14 cells")
    verify_inputs({"source_bindings": bindings})
    REVIEW.mkdir()
    write(REVIEW / "plan.json", {"schema": "latency58-packed-weighted-review-plan-v1",
        "quality_plan": binding(OUT / "quality-plan.json"), "source_bindings": bindings, "quality_selected": False})
    review_result = {"schema": "latency58-packed-weighted-review-v1", "status": "pass",
        "plan_sha256": sha(REVIEW / "plan.json"), "source_bindings_unchanged": True,
        "candidate_models": plan["models"], "reference_models": {label: ref["views"]["model"] for label, ref in references.items()},
        "candidate_full_sdr_db": {role: full[role]["results"][0]["aggregate"]["full_sdr_db"] for role in ROLES},
        "comparisons": comparisons, "all_56_track_stem_cells_and_840_source_view_windows_per_comparison": True,
        "quality_selected": False, "plugin_replaced": False, "overall_goal_complete": False, "gpu_used": False}
    encoded = json.dumps(review_result, indent=2, allow_nan=False).encode()
    storage = storage_snapshot(source)
    require(len(encoded) + len(json.dumps(released_comparisons, indent=2, allow_nan=False).encode()) + 2_000_000
            < storage["remaining_reserved_bytes"]["diagnostics"], "Review exceeds allocated diagnostics")
    write(REVIEW / "result.json", review_result)
    write(VIEWS / "result.json", {"schema": "latency58-packed-paired-vocal-views-result-v1", "status": "pass",
        "plan_sha256": sha(VIEWS / "plan.json"), "source_bindings_unchanged": True, "models": plan["models"],
        "reports": {role: {name: binding(VIEWS / role / (name + ".json")) for name in ("plan", "result", "execution")}
                    for role in ROLES}, "track_count_per_endpoint": 14, "excerpt_count_per_view_per_endpoint": 28,
        "comparisons_against_working_released_graph": released_comparisons,
        "complete_comparisons": binding(REVIEW / "result.json"), "quality_selected": False, "plugin_replaced": False})
    return review_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1 quality controller")
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    began = time.monotonic()
    plan = prepare(args.training_plan.resolve(strict=True))
    audit_saved(plan)
    full, views, bindings = evaluate(plan)
    reviewed = review(plan, full, views, bindings)
    source = read(plan["packed_training_plan"]["path"])
    result = {"schema": "latency58-packed-weighted-quality-result-v1", "status": "saved_roles_full14_and_source_views_reviewed",
        "plan_sha256": sha(OUT / "quality-plan.json"), "source_bindings_unchanged": True,
        "models": plan["models"], "full_sdr_db": reviewed["candidate_full_sdr_db"],
        "target_reached": max(reviewed["candidate_full_sdr_db"].values()) >= 5.0,
        "target_full_sdr_db": 5.0, "checkpoint_audit": binding(OUT / "checkpoint-audit.json"),
        "quality_results": {role: binding(OUT / ("full14-" + role) / "result.json") for role in ROLES},
        "paired_source_views": binding(VIEWS / "result.json"), "complete_review": binding(REVIEW / "result.json"),
        "elapsed_seconds": time.monotonic() - began, "storage_after": storage_snapshot(source),
        "quality_selected": False, "plugin_replaced": False, "overall_goal_complete": False,
        "remaining_release_gates": ["Exact exported graph quality", "Sustained physical M4 playback", "Human listening acceptance"]}
    write(OUT / "result.json", result)
    print(json.dumps({"status": result["status"], "full_sdr_db": result["full_sdr_db"],
                      "target_reached": result["target_reached"], "overall_goal_complete": False}), flush=True)


if __name__ == "__main__":
    main()
