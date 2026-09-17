"""Preserve quiet-window and vocal-gain arithmetic while adding the new checkpoint family."""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use frozen CPU1 supplement qualification")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-cleanup-supplements-check-plan-v1", "Different supplement check")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve check")
    root = ROOT / "research/direct"
    old = functions(root / "evaluate_latency58_quiet_wanted.py")
    new = functions(root / "evaluate_latency58_quiet_leader_cleanup.py")
    require(all(old[name] == new[name] for name in ("window_metrics", "check_window_metrics", "score_track", "main")),
            "Quiet supplement changes more than model dispatch and allowed prefix")
    require(functions(root / "report_latency58_vocal_gain_error.py")["compare_gain"]
                == functions(root / "report_latency58_leader_cleanup_gain_error.py")["compare_gain"],
            "Gain supplement changes comparison arithmetic")
    from research.direct.evaluate_latency58_quiet_leader_cleanup import check_window_metrics, PREFIXES
    from research.direct.evaluate_latency58_quiet_wanted import check_window_metrics as old_check
    checked = check_window_metrics()
    require(checked == old_check() and PREFIXES["leader_cleanup"] == "leader-cleanup-250"
            and PREFIXES["leader"] == "sdr-drum-accum-500",
            "Analytic quiet fidelity or new checkpoint prefix differs")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-leader-cleanup-supplements-check-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "quiet_metrics_scoring_and_primary_replay_checks_unchanged": True,
          "gain_arithmetic_and_track_bootstrap_unchanged": True, "analytic_quiet_check": checked,
          "model_instances": 0, "training_updates_executed": 0, "quality_selected": False,
          "limitations": ["New endpoint dispatch will be exercised after the full primary score; this check scores no model."]})
    print({"status": "pass", "quiet_and_gain_arithmetic_unchanged": True}, flush=True)


if __name__ == "__main__":
    main()
