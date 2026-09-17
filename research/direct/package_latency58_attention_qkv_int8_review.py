"""Copy completed seventeen-projection evidence into the isolated Mac test checkout."""
from datetime import datetime, timezone
import json
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_m4_followup_budget import POLICY, snapshot

FOLLOWUP = ROOT / "research/m4_followup_20260916"


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items() if key != "source_bindings"}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
        return value[len(str(ROOT)) + 1:]
    return value


def main():
    out = FOLLOWUP / "attention-qkv-review-plugin-001"
    stage = FOLLOWUP / "attention-qkv-int8-package-review-001"
    destination = out / "model/quality-deployment.json"
    require(not stage.exists(), "Preserve completed package evidence")
    require(read(destination)["status"] == "pending", "Preserve an existing completed report")
    bindings = {str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    evidence = {}
    for key, name in (("quality", "attention-qkv-int8-quality-001"),
                      ("review", "attention-qkv-int8-review-001"),
                      ("native", "attention-qkv-native-tests-002")):
        paths = [FOLLOWUP / name / leaf for leaf in ("plan.json", "result.json", "execution.json")]
        plan, result, execution = map(read, paths)
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and result["plan_sha256"] == sha(paths[0])
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"], "Incomplete evidence: " + name)
        require(str(destination) not in plan["source_bindings"], "Destination is frozen")
        verify_inputs(plan)
        bindings.update(plan["source_bindings"])
        bindings.update({str(path): sha(path) for path in paths})
        evidence[key] = {"directory": str(FOLLOWUP / name), "result_sha256": sha(paths[1]),
                         "plan_sha256": sha(paths[0]), "execution": execution, "result": result}
    quality, review, native = (evidence[key]["result"] for key in ("quality", "review", "native"))
    require(native["runtime_session_setting"] == "mlas.disable_kleidiai=1", "Unexpected plugin backend setting")
    graph = out / "model/model.onnx"
    require(sha(graph) == quality["graph_sha256"] == native["graph_sha256"], "Graph differs")
    require(quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and len(review["paired_windows"]["all_windows"]) == 840
            and len(native["passed"]) == 164 and native["total"] == 172,
            "Incomplete panel or correctness suite")
    keys = ("status", "results", "parent_sixteen_projection_comparison",
            "parent_sixteen_projection_all_track_stem_cells", "previous_deployment_comparison", "source_fp32_comparison",
            "c204_comparison", "previous_deployment_all_track_stem_cells",
            "source_fp32_all_track_stem_cells", "vocal_views", "graph_sha256", "graph_bytes",
            "track_count", "excerpt_count", "counterfactual_excerpt_count_per_view",
            "full_band_target_reached", "quality_selected", "native_host_qualified",
            "graph_delay_samples", "host_queue_samples", "limitations")
    public = {"schema": "stemgenrt-seventeen-projection-deployment-quality-v1",
        "scope": "Completed exact-graph measurements on the unchanged development panel. "
                 "Pass describes evidence checks; quality, instrumental separation and M4 acceptance remain unmet.",
        "source_checkpoint_full14_sdr_db": quality["source_fp32_comparison"]["metrics"]["full_sdr_db"]["reference"],
        "evidence": {key: {k: v for k, v in value.items() if k != "result"}
                     for key, value in evidence.items()},
        **{key: quality[key] for key in keys},
        "music_regression_summary": review["music_regression_summary"],
        "source_view_comparison": review["vocal_views"],
        "paired_windows": review["paired_windows"],
        "local_native_cost": review["native_cost"],
        "plugin_runtime_session_setting": native["runtime_session_setting"],
        "native_platform": "Linux x86_64; M4 candidate parity and sustained playback pending",
        "human_listening_completed": False, "goal_complete": False}
    runtime_module = Path(quality["runtime"]["python_module"])
    bindings[str(runtime_module)] = sha(runtime_module)
    public["runtime"] = {"version": quality["runtime"]["version"],
        "python_module_sha256": sha(runtime_module), "execution_provider": "CPUExecutionProvider",
        "intra_op_threads": 1, "inter_op_threads": 1, "execution_mode": "sequential", "spinning": False}
    public["runtime"]["measurement_backend_scope"] = "Linux x86_64 CPU kernels; the plugin explicitly disables KleidiAI following retained parent M4 Pro parity evidence"
    encoded = json.dumps(portable(public), indent=2, allow_nan=False) + "\n"
    require(len(encoded.encode()) < 10_000_000, "Package exceeds diagnostic reserve")
    bindings.update({str(p): sha(p) for p in (POLICY, ROOT / "research/direct/latency58_m4_followup_budget.py")})
    before = snapshot()
    plan = {"schema": "latency58-seventeen-projection-package-plan-v1", "source_bindings": bindings,
            "destination": str(destination), "destination_previous_sha256": sha(destination), "budget_before": before,
            "maximum_output_bytes": 10_000_000, "quality_selected": False, "native_host_qualified": False}
    verify_inputs(plan)
    stage.mkdir()
    write(stage / "plan.json", plan)
    (stage / "previous-quality-deployment.json").write_bytes(destination.read_bytes())
    destination.write_text(encoded)
    require(read(destination) == portable(public), "Package round trip differs")
    verify_inputs(plan)
    result = {"schema": "latency58-seventeen-projection-package-v1", "status": "pass",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(stage / "plan.json"),
        "source_bindings_unchanged": True, "destination": str(destination),
        "destination_sha256": sha(destination), "destination_bytes": destination.stat().st_size,
        "graph_sha256": sha(graph), "full_sdr_db": quality["results"][0]["aggregate"]["full_sdr_db"],
        "quality_selected": False, "native_host_qualified": False, "goal_complete": False,
        "release_replaced": False, "gpu_used": False, "budget_after": snapshot()}
    write(stage / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
