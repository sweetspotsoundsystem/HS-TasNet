"""Freeze, qualify and run a metrics-only vocal-leakage baseline for v0.4.0."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, verify_inputs


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def require_cpu(plan=None):
    import onnxruntime as ort
    import torch
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0" and not torch.cuda.is_initialized(), "Require shipping ORT and CUDA-hidden CPU1")
    if plan is not None:
        require(str(Path(ort.__file__).resolve()) == plan["runtime"]["python_module"], "Wrong ORT installation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def budget_snapshot(plan):
    roots = [Path(p).resolve(strict=True) for p in plan["counted_roots"]]
    external = Path(plan["external_git_common_directory"]).resolve(strict=True)
    paths = [*roots, external]
    require(all(not a.is_relative_to(b) for i, a in enumerate(paths) for j, b in enumerate(paths) if i != j),
            "Budget roots overlap")
    counted = {str(p): disk_bytes(p) for p in roots}
    git_bytes = disk_bytes(external)
    total = (sum(counted.values()) + git_bytes + plan["other_outside_allowance_bytes"]
             + plan["live_training_save_reservation_bytes"] + plan["diagnostic_artifact_allowance_bytes"])
    require(plan["authorized_cap_bytes"] == 90_000_000_000 and total < plan["authorized_cap_bytes"],
            "Insufficient artifact space including external Git and live training reserves")
    return {"observed_utc": datetime.now(timezone.utc).isoformat(), "counted_roots": counted,
        "external_git_common_directory": str(external), "external_git_common_bytes": git_bytes,
        "other_outside_allowance_bytes": plan["other_outside_allowance_bytes"],
        "live_training_save_reservation_bytes": plan["live_training_save_reservation_bytes"],
        "diagnostic_artifact_allowance_bytes": plan["diagnostic_artifact_allowance_bytes"],
        "conservative_total_with_reservations": total,
        "authorized_cap_bytes": plan["authorized_cap_bytes"], "headroom_bytes": plan["authorized_cap_bytes"] - total}


def execute_stage(module, args, out, bindings, timeout):
    verify_inputs({"source_bindings": bindings})
    argv = [PYTHON, "-u", "-m", module, *args]
    began, timed_out = time.monotonic(), False
    # Preserve the explicitly selected shipping ORT path; the older generic
    # stage launcher overwrites PYTHONPATH with the project root.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with (out / "console.log").open("x") as log:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=15)
    unchanged = all(sha(p) == digest for p, digest in bindings.items())
    receipt = {"actual_exit_code": code, "timed_out": timed_out, "source_bindings_unchanged": unchanged,
        "elapsed_seconds": time.monotonic() - began, "argv": argv, "child_pid": child.pid}
    write(out / "execution.json", receipt)
    require(code == 0 and not timed_out and unchanged, "Stage failed; inspect preserved console and execution receipt")
    print({"stage": module, **receipt}, flush=True)


def prepare(prefix):
    import onnxruntime as ort
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.metrics import SOURCE_ORDER
    from research.direct.latency58_vocal_views import VIEWS, VERSION
    require(prefix and all(c.isalnum() or c in "-_" for c in prefix), "Invalid output prefix")
    out = PHASE / prefix
    require(not out.exists(), "Preserve existing diagnostics")
    quality_root = PHASE / "branch-plugin-full14-001"
    quality_plan_path, quality_path = quality_root / "plan.json", quality_root / "result.json"
    template_path = PHASE / "vocal-views-working-001/plan.json"
    template, old, quality = read(template_path), read(quality_plan_path), read(quality_path)
    require(quality["status"] == "pass" and quality["source_bindings_unchanged"]
            and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["plan_sha256"] == sha(quality_plan_path)
            and quality["results"][0]["aggregate"]["full_sdr_db"] == 4.4551880546632505,
            "Released graph full-mixture baseline changed")
    checkpoint = old["checkpoint"]
    require(checkpoint["sha256"] == quality["graph_sha256"]
            == "d2945742d27fe23469614aef4f5b79e46fb1a11696ee2c8e6055c494163bcffa"
            and sha(checkpoint["path"]) == checkpoint["sha256"]
            and Path(checkpoint["path"]).stat().st_size == checkpoint["bytes"] == 48_754_181,
            "Exact released graph changed")
    paths = [quality_plan_path, quality_path, Path(checkpoint["path"]), template_path]
    for path in (quality_root / "execution.json", Path(template["qualification_execution"]["path"])):
        value = read(path)
        require(value["actual_exit_code"] == 0 and value["source_bindings_unchanged"]
                and not value.get("timed_out", False), "Baseline or metric execution failed")
        paths.append(path)
    parity_paths = [PHASE / "branch-plugin-screen-001/result.json", PHASE / "branch-plugin-long-001/result.json"]
    for path in parity_paths:
        proof = read(path)
        require(proof["status"] == "pass" and proof["source_bindings_unchanged"]
                and proof["graph_sha256"] == checkpoint["sha256"], "Saved graph numerical qualification changed")
        paths.append(path)
    require(read(parity_paths[0])["strict_parity_passed"]
            and read(parity_paths[1])["reset_replay_all_outputs_and_states_bit_exact"], "Numerical checks incomplete")
    published_path = PHASE / "branch-plugin-release-040-published-verification.json"
    published = read(published_path)
    require(published["status"] == "PASS" and published["tag"] == "v0.4.0"
            and published["tag_commit"] == "ef09113c90b8259268308c1f11d4087f80ca88d9", "Release identity changed")
    paths.extend([published_path, PHASE / "branch-plugin-release-040-ci-proof.json"])
    for name in ("manifest", "config", "qualification", "qualification_execution"):
        item = template[name]
        require(sha(item["path"]) == item["sha256"], "Vocal-view protocol evidence changed")
        paths.append(Path(item["path"]))
    for name in ("research/evaluate.py", "research/metrics.py", "research/direct/evaluate.py",
                 "research/direct/latency58_evaluate.py", "research/direct/latency58_vocal_views.py"):
        path = ROOT / name
        require(template["source_bindings"][str(path)] == sha(path), "Qualified source-view scoring code changed")
        paths.append(path)
    for name in ("run_latency58_deployed_vocal_views.py", "check_latency58_deployed_vocal_views.py",
                 "evaluate_latency58_deployed_vocal_views.py", "latency58_onnx_vocal_views.py",
                 "run_latency58_quality.py", "train_latency58.py", "latency58_attention_int8_verify.py",
                 "latency58_best_onnx.py"):
        paths.append(ROOT / "research/direct" / name)
    module = Path(ort.__file__).resolve()
    require(str(module) == old["runtime"]["python_module"], "Shipping runtime path changed")
    paths.extend([module, *sorted((module.parent / "capi").glob("*.so*"))])
    manifest, config = read(template["manifest"]["path"]), read(template["config"]["path"])
    tracks, selected = shared.select_panel(manifest, config, panel="full", track_indices=list(range(14)),
        excerpt_starts=None, duration=15., alignment_samples=128)
    require(len(tracks) == 14 and list(VIEWS) == ["vocals_only", "instrumental"], "Fixed panel or views changed")
    for index, track in enumerate(tracks):
        require(template["track_intervals"][str(index)] == {"name": track["name"],
                "intervals": legacy._reference_intervals(track, selected)}, "Fixed physical intervals changed")
        for stem in SOURCE_ORDER:
            path = legacy._safe_dataset_path(Path(manifest["root"]), track["stems"][stem])
            require(template["source_bindings"][str(path)] == sha(path), "Fixed source audio changed")
            paths.append(path)
    storage_path = PHASE / "branch-plugin-release-040-storage-001.json"
    storage = read(storage_path)
    paths.append(storage_path)
    plan = {"schema": "latency58-deployed-vocal-views-plan-v1", "output_directory": str(out),
        **{k: template[k] for k in ("manifest", "config", "track_indices", "track_intervals", "workers", "audio_export")},
        "checkpoint": checkpoint, "interface": old["interface"], "runtime": old["runtime"],
        "release": {"tag": published["tag"], "commit": published["tag_commit"], "evidence": binding(published_path)},
        "original_full_mixture_report": binding(quality_path), "protocol_template": binding(template_path),
        "protocol_version": VERSION, "views": {k: list(v) for k, v in VIEWS.items()},
        "graph_numerical_parity_evidence": [binding(p) for p in parity_paths],
        "counted_roots": list(storage["counted_roots"]),
        "external_git_common_directory": storage["external_git_common_directory"],
        "other_outside_allowance_bytes": 800_000_000, "live_training_save_reservation_bytes": 600_000_000,
        "diagnostic_artifact_allowance_bytes": 10_000_000, "authorized_cap_bytes": 90_000_000_000,
        "source_bindings": {str(p): sha(p) for p in paths},
        "host_queue_samples": 128, "graph_delay_samples": 128, "new_audio_exports": False,
        "quality_selection": False, "purpose": "Baseline instrumental vocal leakage and isolated-vocal preservation for the released graph"}
    verify_inputs(plan)
    plan["budget_before"] = budget_snapshot(plan)
    out.mkdir()
    (out / "qualification").mkdir()
    write(out / "plan.json", plan)
    return out, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    require_cpu()
    out, plan = prepare(args.prefix)
    common = ["--plan", str(out / "plan.json"), "--plan-sha256", sha(out / "plan.json")]
    bindings = {**plan["source_bindings"], str(out / "plan.json"): sha(out / "plan.json")}
    execute_stage("research.direct.check_latency58_deployed_vocal_views", common,
                  out / "qualification", bindings, 600)
    qpath, epath = out / "qualification/result.json", out / "qualification/execution.json"
    bindings.update({str(p): sha(p) for p in (qpath, epath)})
    execute_stage("research.direct.evaluate_latency58_deployed_vocal_views", common + [
        "--qualification-sha256", sha(qpath), "--qualification-execution-sha256", sha(epath)], out, bindings, 10800)


if __name__ == "__main__":
    main()
