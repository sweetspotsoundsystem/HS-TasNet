"""Original full14 quality plus fixed vocal views for the two-output-projection experiment."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


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
    stream["precision"] = "Sixteen U8S8 projections, FP64 surrounding recurrence, FP32 decoding and eight public states"
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
    from research.direct.run_latency58_deployed_vocal_views import require_cpu, budget_snapshot
    baseline_path = PHASE / "deployed-vocal-views-001/plan.json"
    baseline = read(baseline_path)
    require_cpu(baseline)
    screen_root, long_root = PHASE / "branch-output-int8-screen-003", PHASE / "branch-output-int8-long-001"
    screen, long = read(screen_root / "result.json"), read(long_root / "result.json")
    paths = [Path(__file__).resolve(), baseline_path]
    for directory, report in ((screen_root, screen), (long_root, long)):
        execution = read(directory / "execution.json")
        require(report["status"] == "pass" and report["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "Complete numerical checks with actual exits first")
        paths.extend(directory / n for n in ("plan.json", "result.json", "execution.json"))
    require(screen["strict_parity_passed"] and screen["graph_sha256"] == long["graph_sha256"],
            "Numerical checks used different candidate graphs")
    graph_path = Path(screen["saved_graph_path"])
    require(sha(graph_path) == screen["graph_sha256"] and graph_path.stat().st_size == screen["graph_bytes"],
            "Saved candidate graph changed")
    paths.append(graph_path)
    parent_path = PHASE / "branch-gru-int8-quality-001/result.json"
    parent_report = read(parent_path)
    parent = parent_report["results"][0]
    require(parent_report["status"] == "pass" and parent_report["graph_sha256"] == screen["parent_graph"]["sha256"], "Fourteen-projection parent identity changed")
    paths.append(parent_path)
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
    bindings = dict(baseline["source_bindings"])
    for name in ("evaluate_latency58_attention_int8_memory.py", "latency58_attention_int8_verify.py",
                 "evaluate_latency58_deployed_vocal_views.py", "latency58_onnx_vocal_views.py",
                 "run_latency58_deployed_vocal_views.py", "compare.py", "report_latency58_vocal_focus.py"):
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
    out = PHASE / "branch-output-int8-quality-001"
    require(not out.exists(), "Preserve prior quality reports")
    checkpoint = {"kind": "saved_onnx", "path": str(graph_path), "sha256": screen["graph_sha256"],
        "bytes": screen["graph_bytes"], "saved": True, "source_checkpoint": screen["source_checkpoint"]}
    plan = {**baseline, "schema": "latency58-branch-output-int8-quality-plan-v1",
        "output_directory": str(out), "checkpoint": checkpoint, "source_bindings": bindings,
        "release": None, "comparison_release": baseline["release"],
        "graph_numerical_parity_evidence": [{"path": str(directory / "result.json"),
            "sha256": sha(directory / "result.json")} for directory in (screen_root, long_root)],
        "full_mixture_protocol_reference": {"path": str(previous_plan_path), "sha256": sha(previous_plan_path)},
        "evaluation_allowance_bytes": 20_000_000, "concurrent_baseline_reservation_bytes": 10_000_000,
        "diagnostic_artifact_allowance_bytes": 250_000_000,
        "parent_fourteen_projection_report": str(parent_path),
        "previous_deployment_report": str(previous_path), "source_fp32_report": str(source_path),
        "c204_report": str(c204_path), "vocal_baseline_report_when_complete": str(PHASE / "deployed-vocal-views-001/result.json"),
        "quality_selection": False, "full_band_target_db": 5.,
        "purpose": "Measure quality and absence behavior after quantizing the two profiled branch output products"}
    plan["budget_before"] = budget_snapshot(plan)
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
    candidate = {"model": {"label": "Saved EMA with sixteen signed integer projections",
        "runtime_artifact": checkpoint, "source_model_state_sha256": long["source_model_state_sha256"],
        "training_updates": 39250, "additional_training_updates": 0, "runtime_precision_changed": True},
        "checkpoint": checkpoint, "tracks": ordered, "aggregate": legacy._aggregate_tracks(ordered),
        "stream_batches": [reports[i][1] for i in range(14)],
        "reconstruction_max_abs": max(r[2] for r in reports.values())}
    view_reports = [reports[i][3] for i in range(14)]
    verify_inputs(plan)
    result = {"status": "pass", "results": [candidate], "checkpoint": checkpoint,
        "parent_fourteen_projection_comparison": compare(parent, candidate),
        "parent_fourteen_projection_all_track_stem_cells": music_cells(parent, candidate),
        "previous_deployment_comparison": compare(previous, candidate),
        "source_fp32_comparison": compare(source, candidate), "c204_comparison": compare(c204, candidate),
        "previous_deployment_all_track_stem_cells": music_cells(previous, candidate),
        "source_fp32_all_track_stem_cells": music_cells(source, candidate),
        "vocal_views": {"tracks": view_reports, "aggregate": aggregate_reports(view_reports)},
        "source_bindings": bindings, "source_bindings_unchanged": True,
        "plan_sha256": sha(out / "plan.json"), "runtime": {"version": ort.__version__, **baseline["runtime"]},
        "graph_sha256": sha(graph_path), "graph_bytes": graph_path.stat().st_size,
        "track_count": 14, "excerpt_count": 28, "counterfactual_excerpt_count_per_view": 28,
        "full_band_target_reached": candidate["aggregate"]["full_sdr_db"] >= 5.,
        "quality_selected": False, "native_host_qualified": False, "gpu_used": False,
        "graph_delay_samples": 128, "host_queue_samples": 128,
        "budget_before": plan["budget_before"], "budget_after": budget_snapshot(plan),
        "elapsed_seconds": time.monotonic() - began,
        "limitations": "Development-set scoring of a changed inference graph. Full-band quality, vocal/Other behavior and M4 timing require joint review; no automatic release or goal completion follows."}
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) + len(json.dumps(result, indent=2).encode())
    require(size + 500_000 < plan["evaluation_allowance_bytes"], "Quality artifact allowance exceeded")
    write(out / "result.json", result)
    print(json.dumps({"status": "pass", "full_sdr_db": candidate["aggregate"]["full_sdr_db"],
        "delta_from_release_db": candidate["aggregate"]["full_sdr_db"] - previous["aggregate"]["full_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
