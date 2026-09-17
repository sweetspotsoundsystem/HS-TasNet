"""Unchanged full14 and vocal-view scoring for the fused integer QKV graph."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_m4_followup_budget import POLICY, snapshot

FOLLOWUP = ROOT / "research/m4_followup_20260916"
CANDIDATE = "08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748"
PARENT = "c7ea50ac67bf4bfddf1f5ff41c6eb419af00fe420ce1a0b0eaeef11a1861cd61"


def initialize_worker(plan, data):
    from research.direct import evaluate_latency58_attention_int8_memory as full
    from research.direct import evaluate_latency58_deployed_vocal_views as views
    full.initialize(plan, data)
    # One local session per process; both protocols explicitly reset their
    # states at each track/view origin. Parent diagnostic workers are separate
    # processes and do not share these module globals.
    views._SESSION, views._PLAN = full._SESSION, plan


def score(index):
    from research.direct import evaluate_latency58_attention_int8_memory as full
    from research.direct import evaluate_latency58_deployed_vocal_views as views
    returned, track, stream, closure = full.score(index)
    require(returned == index, "Wrong full-mixture track")
    stream["precision"] = "Seventeen U8S8 projections, FP64 surrounding recurrence, FP32 decoding and eight public states"
    returned, counterfactual = views.score_track(index)
    require(returned == index and track["name"] == counterfactual["name"], "Vocal-view track differs")
    return index, track, stream, closure, counterfactual


def main():
    import onnxruntime as ort
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.compare import compare
    from research.direct.report_latency58_vocal_focus import music_cells
    from research.direct.evaluate_latency58_deployed_vocal_views import aggregate_reports
    from research.direct.run_latency58_deployed_vocal_views import require_cpu
    baseline_path = PHASE / "deployed-vocal-views-001/plan.json"
    baseline = read(baseline_path)
    require_cpu()
    module = Path(ort.__file__).resolve()
    screen_root, long_root = FOLLOWUP / "attention-qkv-int8-screen-001", FOLLOWUP / "attention-qkv-int8-long-001"
    screen, long = read(screen_root / "result.json"), read(long_root / "result.json")
    paths = [Path(__file__).resolve(), baseline_path, POLICY, ROOT / "research/direct/latency58_m4_followup_budget.py", module, *sorted((module.parent / "capi").glob("*.so*"))]
    for directory, report, receipt in ((screen_root, screen, FOLLOWUP / "attention-qkv-int8-execution.json"),
                                      (long_root, long, FOLLOWUP / "attention-qkv-long-execution.json")):
        execution = read(receipt)
        require(report["status"] == "pass" and report["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "Complete numerical checks with actual exits first")
        verify_inputs(report)
        paths.extend(directory / n for n in ("plan.json", "result.json"))
        paths.append(receipt)
    require(screen["strict_parity_passed"] and screen["graph_sha256"] == long["graph_sha256"] == CANDIDATE,
            "Numerical checks used different candidate graphs")
    graph_path = Path(screen["saved_graph_path"])
    require(sha(graph_path) == screen["graph_sha256"] and graph_path.stat().st_size == screen["graph_bytes"],
            "Saved candidate graph changed")
    paths.append(graph_path)
    native_root = FOLLOWUP / "attention-qkv-native-001"
    native = read(native_root / "result.json")
    native_execution_path = FOLLOWUP / "attention-qkv-native-execution.json"
    native_execution = read(native_execution_path)
    require(native["status"] == "pass" and native["relative_speed_gate"]["passed"]
            and native["source_bindings_unchanged"] and native["plan_sha256"] == sha(native_root / "plan.json")
            and native_execution["actual_exit_code"] == 0 and not native_execution["timed_out"]
            and native_execution["source_bindings_unchanged"], "Complete native comparison first")
    verify_inputs(read(native_root / "plan.json"))
    paths.extend([native_root / "plan.json", native_root / "result.json", native_execution_path])
    parent_path = PHASE / "branch-output-int8-quality-001/result.json"
    parent_report = read(parent_path)
    parent = parent_report["results"][0]
    require(parent_report["status"] == "pass" and parent_report["graph_sha256"] == screen["parent_graph_sha256"] == PARENT, "Sixteen-projection parent identity changed")
    parent_plan_path, parent_execution_path = parent_path.parent / "plan.json", parent_path.parent / "execution.json"
    parent_plan, parent_execution = read(parent_plan_path), read(parent_execution_path)
    require(parent_report["source_bindings_unchanged"] and parent_report["plan_sha256"] == sha(parent_plan_path)
            and parent_execution["actual_exit_code"] == 0 and not parent_execution["timed_out"]
            and parent_execution["source_bindings_unchanged"]
            and parent_execution["result_sha256"] == sha(parent_path), "Parent quality evidence incomplete")
    verify_inputs(parent_plan)
    old_module = Path(parent_plan["runtime"]["python_module"])
    require(sha(module) == sha(old_module) and all(sha(p) == sha(old_module.parent / "capi" / p.name)
            for p in (module.parent / "capi").glob("*.so*")), "Candidate and parent runtime binaries differ")
    paths.extend([parent_path, parent_plan_path, parent_execution_path])
    previous_path = PHASE / "branch-plugin-full14-001/result.json"
    previous_plan_path = PHASE / "branch-plugin-full14-001/plan.json"
    source_path = PHASE / "branch-pitch-ema-002/full14-ema/result.json"
    c204_path = PHASE / "leader-cleanup-250-full14-001/result.json"
    previous_report, source_report, c204_report = (read(p) for p in (previous_path, source_path, c204_path))
    previous, source, c204 = (r["results"][0] for r in (previous_report, source_report, c204_report))
    require(previous["aggregate"]["full_sdr_db"] == 4.4551880546632505
            and source["aggregate"]["full_sdr_db"] == 4.46515742201644,
            "Comparison endpoint changed")
    paths.extend([previous_path, previous_plan_path, source_path, c204_path])
    # Bind only inputs to these saved-graph evaluations. Training-data and
    # optimizer lineage already belongs to the completed source/graph audits.
    bindings = {**baseline["source_bindings"], **screen["source_bindings"], **long["source_bindings"]}
    for name in ("evaluate_latency58_attention_int8_memory.py", "latency58_attention_int8_verify.py",
                 "evaluate_latency58_deployed_vocal_views.py", "latency58_onnx_vocal_views.py",
                 "run_latency58_deployed_vocal_views.py", "compare.py", "report_latency58_vocal_focus.py",
                 "latency58_m4_followup_budget.py", "latency58_weighted_storage.py"):
        paths.append(ROOT / "research/direct" / name)
    manifest, config = read(baseline["manifest"]["path"]), read(baseline["config"]["path"])
    tracks, selected = shared.select_panel(manifest, config, panel="full", track_indices=list(range(14)),
        excerpt_starts=None, duration=15., alignment_samples=128)
    previous_plan = read(previous_plan_path)
    for index, track in enumerate(tracks):
        require(baseline["track_intervals"][str(index)] == {"name": track["name"],
            "intervals": legacy._reference_intervals(track, selected)}, "Original physical panel changed")
        path = legacy._safe_dataset_path(Path(manifest["root"]), track["mixture"])
        require(sha(path) == previous_plan["source_bindings"][str(path)], "Original mixture bytes changed")
        paths.append(path)
    bindings.update({str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    out = FOLLOWUP / "attention-qkv-int8-quality-001"
    require(not out.exists(), "Preserve prior quality reports")
    checkpoint = {"kind": "saved_onnx", "path": str(graph_path), "sha256": screen["graph_sha256"],
        "bytes": screen["graph_bytes"], "saved": True, "source_checkpoint": screen["source_checkpoint"]}
    plan = {**baseline, "schema": "latency58-attention-qkv-int8-quality-plan-v1",
        "output_directory": str(out), "checkpoint": checkpoint, "source_bindings": bindings,
        "runtime": {"version": ort.__version__, "python_module": str(module)},
        "release": None, "comparison_release": baseline["release"],
        "graph_numerical_parity_evidence": [{"path": str(directory / "result.json"),
            "sha256": sha(directory / "result.json")} for directory in (screen_root, long_root)],
        "full_mixture_protocol_reference": {"path": str(previous_plan_path), "sha256": sha(previous_plan_path)},
        "evaluation_allowance_bytes": 20_000_000,
        "authorized_cap_bytes": 100_000_000_000, "combined_budget_policy": str(POLICY),
        "parent_sixteen_projection_report": str(parent_path),
        "previous_deployment_report": str(previous_path), "source_fp32_report": str(source_path),
        "c204_report": str(c204_path), "vocal_baseline_report_when_complete": str(PHASE / "deployed-vocal-views-001/result.json"),
        "quality_selection": False, "full_band_target_db": 5.,
        "purpose": "Measure quality and absence behavior after fusing and quantizing the three attention input products"}
    for obsolete in ("counted_roots", "external_git_common_directory", "other_outside_allowance_bytes",
                     "live_training_save_reservation_bytes", "diagnostic_artifact_allowance_bytes"):
        plan.pop(obsolete, None)
    plan["budget_before"] = snapshot()
    out.mkdir()
    write(out / "plan.json", plan)
    began, reports = time.monotonic(), {}
    with (out / "progress.jsonl").open("x", buffering=1) as log, ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker,
            initargs=(plan, graph_path.read_bytes())) as pool:
        futures = {pool.submit(score, index): index for index in range(14)}
        for future in as_completed(futures):
            index, track, stream, closure, views = future.result()
            require(index == futures[future] and index not in reports, "Wrong or repeated evaluation track")
            reports[index] = track, stream, closure, views
            write(out / ("track-%02d.json" % index), {"full_mixture": track, "stream": stream,
                "reconstruction_max_abs": closure, "vocal_views": views})
            summary = {"index": index, "track": track["name"], "full_sdr_db": track["full_sdr_db"],
                "instrumental_vocals": views["views"]["instrumental"]["native_output_levels"]["vocals"],
                "elapsed_seconds": time.monotonic() - began}
            log.write(json.dumps(summary, allow_nan=False) + "\n")
            print(json.dumps(summary, allow_nan=False), flush=True)
    require(set(reports) == set(range(14)), "Incomplete full14/counterfactual panel")
    ordered = [reports[i][0] for i in range(14)]
    require([t["name"] for t in ordered] == [t["name"] for t in previous["tracks"]], "Original track order changed")
    candidate = {"model": {"label": "Saved EMA with seventeen signed integer projections",
        "runtime_artifact": checkpoint, "source_model_state_sha256": long["source_model_state_sha256"],
        "training_updates": 39250, "additional_training_updates": 0, "runtime_precision_changed": True},
        "checkpoint": checkpoint, "tracks": ordered, "aggregate": legacy._aggregate_tracks(ordered),
        "stream_batches": [reports[i][1] for i in range(14)],
        "reconstruction_max_abs": max(r[2] for r in reports.values())}
    view_reports = [reports[i][3] for i in range(14)]
    verify_inputs(plan)
    result = {"status": "pass", "results": [candidate], "checkpoint": checkpoint,
        "parent_sixteen_projection_comparison": compare(parent, candidate),
        "parent_sixteen_projection_all_track_stem_cells": music_cells(parent, candidate),
        "previous_deployment_comparison": compare(previous, candidate),
        "source_fp32_comparison": compare(source, candidate), "c204_comparison": compare(c204, candidate),
        "previous_deployment_all_track_stem_cells": music_cells(previous, candidate),
        "source_fp32_all_track_stem_cells": music_cells(source, candidate),
        "vocal_views": {"tracks": view_reports, "aggregate": aggregate_reports(view_reports)},
        "source_bindings": bindings, "source_bindings_unchanged": True,
        "plan_sha256": sha(out / "plan.json"), "runtime": {"version": ort.__version__, **plan["runtime"]},
        "graph_sha256": sha(graph_path), "graph_bytes": graph_path.stat().st_size,
        "track_count": 14, "excerpt_count": 28, "counterfactual_excerpt_count_per_view": 28,
        "full_band_target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.,
        "quality_selected": False, "native_host_qualified": False, "gpu_used": False,
        "graph_delay_samples": 128, "host_queue_samples": 128,
        "budget_before": plan["budget_before"], "budget_after": snapshot(),
        "elapsed_seconds": time.monotonic() - began,
        "limitations": "Development-set scoring of a changed inference graph. Full-band quality, vocal/Other behavior and M4 timing require joint review; no automatic release or goal completion follows."}
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) + len(json.dumps(result, indent=2).encode())
    require(size + 500_000 < plan["evaluation_allowance_bytes"], "Quality artifact allowance exceeded")
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
        "delta_from_release_db": candidate["aggregate"]["full_sdr_db"] - previous["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
