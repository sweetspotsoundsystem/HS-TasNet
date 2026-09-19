"""Supplement frozen historical comparisons with the authenticated v0.6.1 graph.

Reuse full14 and source-view measurements. No inference, training or model
selection is performed here. The original quality workflow remains immutable.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import snapshot
from research.direct.report_latency58_grouped_continuation import compare_endpoint

TRIAL = ROOT / "research/four_second_20260916/branch-four-second-015"
OUT = TRIAL / "current-release-review-001"
IDENTITY = TRIAL / "current-release-reference-001"
REFERENCE = ROOT / "research/m4_followup_20260916/attention-qkv-int8-quality-001"
TRAINING = TRIAL / "plan-recovery002.json"
PANEL_KEYS = ("manifest", "config", "track_indices", "track_intervals", "protocol_version")


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def require_panel(left, right):
    require(all(left[k] == right[k] for k in PANEL_KEYS)
            and left["track_indices"] == list(range(14)), "Comparison physical protocol changed")


def load_reference():
    from research import evaluate as legacy
    from research.direct.report_latency58_branch_output_int8 import require_close
    identity = read(IDENTITY / "identity.json")
    release = read(IDENTITY / "release-metadata.json")
    tree = {row["path"]: row for row in read(IDENTITY / "git-tree.json")["tree"]}
    require(identity["status"] == "pass" and release["tag_name"] == identity["release_tag"] == "v0.6.1"
            and release["target_commitish"] == identity["release_commit"]
                == "990df8ee5baa621f042d4534dacc65afee0a96ce", "Release identity changed")
    for name, blob in identity["source_blobs"].items():
        data = Path(blob["path"]).read_bytes()
        require(sha(blob["path"]) == blob["sha256"] and tree[name]["sha"] == blob["git_blob"]
                == hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
                "Release source blob changed")
    model = identity["model"]
    pointer = (IDENTITY / "model-lfs-pointer.txt").read_text()
    contract = (IDENTITY / "QualifiedModelContract.cmake").read_text()
    require(sha(model["path"]) == model["sha256"] == "08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748"
            and Path(model["path"]).stat().st_size == model["bytes"] == 37532574
            and "oid sha256:" + model["sha256"] in pointer
            and "size " + str(model["bytes"]) in pointer
            and 'set(STEMGENRT_QUALIFIED_MODEL_SHA256 "' + model["sha256"] + '")' in contract,
            "Retained graph differs from the released graph")
    plan, result, execution = (read(REFERENCE / (name + ".json")) for name in ("plan", "result", "execution"))
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and result["plan_sha256"] == execution["plan_sha256"] == sha(REFERENCE / "plan.json")
            and execution["result_sha256"] == sha(REFERENCE / "result.json")
            and result["checkpoint"] == plan["checkpoint"]
            and result["graph_sha256"] == plan["checkpoint"]["sha256"] == model["sha256"]
            and result["track_count"] == 14 and result["excerpt_count"] == 28
            and result["counterfactual_excerpt_count_per_view"] == 28
            and result["graph_delay_samples"] + result["host_queue_samples"] == 256,
            "Released graph lacks authenticated complete-panel measurements")
    verify_inputs(plan)
    full = result["results"][0]
    require_close(full["aggregate"], legacy._aggregate_tracks(full["tracks"]), "release aggregate")
    views = {"version": plan["protocol_version"], "model": result["checkpoint"], **result["vocal_views"]}
    return plan, result, {"full": full, "views": views}, identity


def prepare():
    require(not OUT.exists(), "Preserve release-comparison preparation")
    plan, result, reference, identity = load_reference()
    template_path = ROOT / "research/direct/runs/latency58/paired-vocal-grouped-013/plan.json"
    template = read(template_path)
    require_panel(plan, template)
    self_comparison = compare_endpoint(reference, reference)
    require(self_comparison["full_mixture"]["metrics"]["full_sdr_db"]["delta"] == 0,
            "Independent release self-comparison is not neutral")
    changed = copy.deepcopy(template)
    changed["track_intervals"]["0"]["intervals"][0]["reference_start"] += 1
    rejected = False
    try:
        require_panel(plan, changed)
    except RuntimeError:
        rejected = True
    require(rejected, "Mismatched physical interval was accepted")
    paths = [TRAINING, template_path, Path(__file__).resolve(), Path(identity["model"]["path"]),
             *(REFERENCE / (name + ".json") for name in ("plan", "result", "execution")), *IDENTITY.iterdir()]
    code = ("report_latency58_grouped_continuation.py", "report_latency58_grouped_vocal_parent.py",
            "compare.py", "compare_latency58_vocal_views.py", "report_latency58_vocal_focus.py",
            "report_latency58_branch_gru_int8.py", "report_latency58_branch_output_int8.py")
    paths.extend(ROOT / "research/direct" / name for name in code)
    paths.extend(ROOT / "research" / name for name in ("evaluate.py", "metrics.py"))
    bindings = {**plan["source_bindings"], **result["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    OUT.mkdir()
    write(OUT / "plan.json", {"source_bindings": bindings, "training_plan": binding(TRAINING),
        "release_identity": identity, "reference_plan": binding(REFERENCE / "plan.json"),
        "reference_result": binding(REFERENCE / "result.json"), "protocol": {k: plan[k] for k in PANEL_KEYS},
        "historical_comparisons_preserved": True, "quality_selection": False, "storage_before": snapshot()})
    write(OUT / "qualification.json", {"status": "pass", "plan_sha256": sha(OUT / "plan.json"),
        "release_graph_authenticated": True, "self_comparison_full_sdr_delta": 0,
        "track_stem_cells": 56, "paired_source_view_windows": 840,
        "physical_interval_negative_control_rejected": True, "gpu_used": False})
    print(json.dumps({"status": "prepared", "plan": binding(OUT / "plan.json"), "release": "v0.6.1"}), flush=True)


def compare_saved(expected):
    require(sha(OUT / "plan.json") == expected, "Comparison plan changed")
    plan = read(OUT / "plan.json")
    verify_inputs(plan)
    root_execution = read(TRIAL / "quality-root-execution-002.json")
    quality, quality_plan = read(TRIAL / "result.json"), read(TRIAL / "quality-plan.json")
    require(root_execution["actual_exit_code"] == 0 and root_execution["status"] == "pass"
            and root_execution["source_bindings_unchanged"]
            and root_execution["result_sha256"] == sha(TRIAL / "result.json")
            and quality["plan_sha256"] == sha(TRIAL / "quality-plan.json")
            and quality_plan["packed_training_plan"] == plan["training_plan"] == binding(TRAINING)
            and quality["models"] == quality_plan["models"], "Final saved quality workflow is incomplete")
    reference_plan, _, reference, identity = load_reference()
    require_panel(reference_plan, quality_plan)
    bindings = {**plan["source_bindings"], **quality_plan["source_bindings"]}
    paths = [OUT / "plan.json", OUT / "qualification.json", TRIAL / "quality-root-execution-002.json",
             TRIAL / "result.json", TRIAL / "quality-plan.json"]
    comparisons = {}
    for role in ("raw", "ema"):
        full_root = TRIAL / ("full14-" + role)
        views_root = TRIAL.parent / "paired-vocal-four-second-015" / role
        full, views = read(full_root / "result.json"), read(views_root / "result.json")
        for directory in (full_root, views_root):
            execution = read(directory / "execution.json")
            require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
                    and execution["source_bindings_unchanged"], "Incomplete candidate measurements")
            paths.extend(directory / (name + ".json") for name in ("plan", "result", "execution"))
        model = quality["models"][role]
        require(full["checkpoint_role"] == role and full["model_state_sha256"] == model["model_state_sha256"]
                and views["model"] == model, "Candidate model identity changed")
        comparisons[role] = compare_endpoint(reference, {"full": full["results"][0], "views": views})
    bindings.update({str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    write(OUT / "comparison-inputs.json", {"source_bindings": bindings, "plan_sha256": expected})
    write(OUT / "result.json", {"status": "pass", "plan_sha256": expected,
        "source_bindings_unchanged": True, "release_identity": identity, "candidate_models": quality["models"],
        "comparisons_against_v061": comparisons, "all_56_cells_and_840_windows_per_candidate": True,
        "historical_comparisons_preserved": True, "quality_selected": False, "plugin_replaced": False,
        "overall_goal_complete": False, "gpu_used": False, "storage_after": snapshot()})


def wait_then_compare(expected):
    require(sha(OUT / "plan.json") == expected, "Queued comparison plan changed")
    require(not (OUT / "wait.json").exists(), "Preserve comparison queue")
    write(OUT / "wait.json", {"pid": os.getpid(), "plan_sha256": expected, "maximum_wait_seconds": 27000})
    began = time.monotonic()
    receipt = TRIAL / "quality-root-execution-002.json"
    while not receipt.exists():
        require(time.monotonic() - began < 27000, "Saved quality completion receipt did not arrive")
        time.sleep(10)
    completed = read(receipt)
    require(completed["status"] == "pass" and completed["actual_exit_code"] == 0,
            "Saved quality workflow failed; no release comparison may be claimed")
    argv = [PYTHON, "-u", "-m", "research.direct.review_latency58_four_second_current_release",
            "--compare", "--plan-sha256", expected]
    write(OUT / "command.json", {"argv": argv})
    timed_out = False
    with (OUT / "console.log").open("x") as log:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write(OUT / "launch.json", {"pid": child.pid, "command_sha256": sha(OUT / "command.json")})
        try:
            code = child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait()
        log.flush(); os.fsync(log.fileno())
    write(OUT / "execution.json", {"actual_exit_code": code, "timed_out": timed_out,
          "plan_sha256": expected, "command_sha256": sha(OUT / "command.json"),
          "result_sha256": sha(OUT / "result.json") if code == 0 else None})
    require(code == 0 and not timed_out, "Current-release comparison failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--wait", action="store_true")
    modes.add_argument("--compare", action="store_true")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    prepare() if args.prepare else (wait_then_compare(args.plan_sha256) if args.wait else compare_saved(args.plan_sha256))
