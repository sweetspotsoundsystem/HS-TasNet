"""Qualify and score fixed vocal views from an audited branch-memory endpoint."""
import argparse
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.prefix and all(c.isalnum() or c in "-_" for c in args.prefix)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require a bounded output name and CUDA-hidden CPU1 launcher")
    endpoint = args.quality_root.resolve(strict=True)
    require(endpoint.is_relative_to(PHASE), "Use a retained branch-memory endpoint")
    source_path, quality_path = endpoint / "plan.json", endpoint / "full14/result.json"
    source, quality, audit, terminal = (read(p) for p in
        (source_path, quality_path, endpoint / "checkpoint-audit.json", endpoint / "result.json"))
    verify_inputs(source)
    verify_inputs(quality)
    require(source["schema"] == "latency58-branch-memory-training-plan-v1"
            and terminal["status"] == "training_audit_and_full14_complete"
            and quality["status"] == audit["status"] == "pass"
            and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and audit["saved_optimizer_tensor_count"] == 40 and audit["algorithmic_latency_samples"] == 256
            and terminal["checkpoint"] == audit["checkpoint"] == quality["results"][0]["checkpoint"]
            and audit["model_state_sha256"] == quality["results"][0]["model"]["model_state_sha256_after"],
            "Require an audited saved checkpoint and completed original full-mixture score")
    paths = [source_path, quality_path, endpoint / "checkpoint-audit.json", endpoint / "result.json"]
    for path in (endpoint / "root-execution.json", endpoint / "production-stage/execution.json",
                 endpoint / "full14/execution.json"):
        execution = read(path)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution.get("timed_out", False), "An endpoint execution remains incomplete")
        paths.append(path)
    template_path = PHASE / "vocal-views-working-001/plan.json"
    template = read(template_path)
    paths.append(template_path)
    for name in ("run_latency58_branch_vocal_views.py", "check_latency58_branch_vocal_views.py",
                 "evaluate_latency58_branch_vocal_views.py", "latency58_vocal_views.py"):
        paths.append(ROOT / "research/direct" / name)
    for name in ("manifest", "config"):
        expected = template[name]
        require(sha(expected["path"]) == expected["sha256"], "Fixed vocal-view protocol changed")
        paths.append(Path(expected["path"]))
    # The old template fixes physical intervals and source definitions. Its
    # historical optimizer reservations are not inputs to this new diagnostic.
    require(template["workers"] == 2 and template["track_indices"] == list(range(14))
            and not template["audio_export"]
            and sha(ROOT / "research/direct/latency58_vocal_views.py")
                == template["source_bindings"][str(ROOT / "research/direct/latency58_vocal_views.py")],
            "Preserve the qualified continuous source-view implementation and full panel")
    bindings = {**source["source_bindings"], **quality["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    budget = {key: source[key] for key in ("counted_roots", "stop_counted_bytes", "outside_roots_reservation_bytes")}
    require(budget["stop_counted_bytes"] + budget["outside_roots_reservation_bytes"] == 90_000_000_000,
            "Use the current 90 GB artifact cap")
    require_space(budget, 450_000_000)
    out = PHASE / args.prefix
    require(not out.exists(), "Preserve prior diagnostics")
    out.mkdir()
    qualification = out / "qualification"
    qualification.mkdir()
    checkpoint = terminal["checkpoint"]
    functional = {**budget, "schema": "latency58-branch-vocal-views-functional-plan-v1",
                  "checkpoint": checkpoint, "model_state_sha256": audit["model_state_sha256"],
                  "source_bindings": bindings, "output_directory": str(qualification)}
    write(qualification / "plan.json", functional)
    execute([PYTHON, "-u", "-m", "research.direct.check_latency58_branch_vocal_views", "--plan",
             str(qualification / "plan.json"), "--plan-sha256", sha(qualification / "plan.json")],
            qualification, "evaluation", 600, bindings, {"plan_sha256": sha(qualification / "plan.json")})
    for name in ("plan.json", "result.json", "execution.json"):
        paths.append(qualification / name)
    bindings = {**bindings, **{str(p): sha(p) for p in paths}}
    plan = {**budget, **{key: template[key] for key in ("workers", "track_indices", "track_intervals", "audio_export", "manifest", "config")},
            "schema": "latency58-branch-vocal-views-evaluation-plan-v1", "output_directory": str(out),
            "model": {"kind": "branch_memory", "label": endpoint.name, "checkpoint": checkpoint,
                      "model_state_sha256": audit["model_state_sha256"],
                      "original_full_mixture_report": binding(quality_path)},
            "qualification": binding(qualification / "result.json"),
            "qualification_execution": binding(qualification / "execution.json"),
            "source_bindings": bindings, "protocol_template": binding(template_path),
            "diagnostic_artifact_allowance_bytes": 10_000_000, "concurrent_training_reservation_bytes": 440_000_000}
    verify_inputs(plan)
    write(out / "plan.json", plan)
    execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_branch_vocal_views", "--plan",
             str(out / "plan.json"), "--plan-sha256", sha(out / "plan.json")],
            out, "evaluation", 3000, bindings, {"plan_sha256": sha(out / "plan.json")})


if __name__ == "__main__":
    main()
