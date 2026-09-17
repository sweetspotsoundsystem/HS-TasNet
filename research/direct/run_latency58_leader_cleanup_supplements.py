"""Score leader and candidate quiet fidelity, then compare three references and vocal gain."""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import select
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.report_latency58_sdr import load_completed as load_primary
from research.direct.compare_latency58_quiet_wanted import load_completed as load_quiet
from research.direct.report_latency58_leader_cleanup import load_views, reference_views
from research.direct.latency58_sdr_checkpoint import require_space


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Supplement plan changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-cleanup-supplements-plan-v1"
            and plan["maximum_wait_seconds"] == 21000 and plan["evaluation_timeout_seconds"] == 2400,
            "Different bounded supplement work")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve supplements")
    training = read(plan["training_plan"]["path"])
    from research.direct.latency58_leader_cleanup_checkpoint_v2 import validate_recipe
    validate_recipe(training)
    require(not training["resource_only"], "Production candidate required")
    for key in ("training_plan", "qualification", "qualification_execution", "geometry_plan", "quality_queue_plan"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound supplement prerequisite")
    checked, checked_execution = read(plan["qualification"]["path"]), read(plan["qualification_execution"]["path"])
    require(checked["schema"] == "latency58-leader-cleanup-supplements-check-v1" and checked["status"] == "pass"
            and checked["quiet_metrics_scoring_and_primary_replay_checks_unchanged"]
            and checked["gain_arithmetic_and_track_bootstrap_unchanged"] and checked["source_bindings_unchanged"]
            and checked_execution["actual_exit_code"] == 0 and not checked_execution["timed_out"]
            and checked_execution["source_bindings_unchanged"] and checked_execution["plan_sha256"] == checked["plan_sha256"],
            "Supplement arithmetic qualification failed")
    evidence = {**plan["source_bindings"], str(args.plan.resolve()): args.plan_sha256}
    geometry = read(plan["geometry_plan"]["path"])
    for key in ("manifest", "config", "reference_inventory", "inventory_execution"):
        item = geometry[key]
        require(evidence.get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound quiet source geometry")
    queue_path = Path(plan["quality_queue_plan"]["path"])
    queue = read(queue_path)
    require(queue["schema"] == "latency58-leader-cleanup-queued-quality-plan-v1"
            and queue["training_plan"] == plan["training_plan"], "Different quality owner")
    queue_terminal = queue_path.parent / "queue-execution.json"
    owner = plan["quality_execution_owner"]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes, libc.pidfd_open.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    descriptor = None
    if not queue_terminal.is_file():
        descriptor = libc.pidfd_open(owner["pid"], 0)
        if descriptor < 0:
            raise OSError(ctypes.get_errno(), "Quality execution owner unavailable")
        proc = Path(f"/proc/{owner['pid']}")
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        argv = [s.decode() for s in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
        require(int(stat[19]) == owner["start_ticks"] and argv == owner["argv"], "Different live quality owner")
    began = time.monotonic()
    write(out / "start.json", {"quality_execution_owner": owner, "pidfd_opened": descriptor is not None,
                               "plan_sha256": args.plan_sha256, "wall_time": time.time()})

    def wait_for(path):
        while not path.is_file():
            require(time.monotonic() - began < plan["maximum_wait_seconds"], "Bounded supplement wait expired")
            require(not queue_terminal.is_file() or path.is_file(), "Quality owner ended without the required result")
            if descriptor is not None:
                poller = select.poll()
                poller.register(descriptor, select.POLLIN)
                require(not poller.poll(0) or path.is_file(), "Quality owner exited without a required result")
            time.sleep(5)

    def score_quiet(kind, prefix, destination):
        require(destination.parent == PHASE and not destination.exists(), "Preserve quiet evaluation")
        primary = PHASE / (prefix + "-full14-001")
        quality, report = load_primary(primary, evidence)
        fingerprint = report["results"][0]["model"]["model_state_sha256"]
        if kind == "leader":
            require(binding(primary / "plan.json") == training["parent"]["quality_plan"]
                    and fingerprint == training["parent"]["model_state_sha256"], "Different SDR parent")
        else:
            from research.direct.audit_latency58_leader_cleanup_training_match import load_completed_match
            matched = load_completed_match(quality["training_match"], evidence)
            require(kind == "leader_cleanup" and quality["training_plan"] == plan["training_plan"]
                    and matched["training_plans"]["candidate"] == plan["training_plan"]
                    and matched["model_states"]["candidate"] == fingerprint, "Different candidate or production comparison")
        evidence.update(quality["source_bindings"])
        quiet_plan = {"schema": "latency58-quiet-wanted-evaluation-plan-v1", "workers": 2, "track_indices": list(range(14)),
            "model": {"kind": kind, "prefix": prefix, "model_state_sha256": fingerprint},
            "quality_plan": binding(primary / "plan.json"), "quality_result": binding(primary / "result.json"),
            **{k: geometry[k] for k in ("manifest", "config", "reference_inventory", "inventory_execution")},
            "output_directory": str(destination), "source_bindings": dict(evidence),
            **{k: training[k] for k in ("counted_roots", "stop_counted_bytes")}}
        verify_inputs(quiet_plan)
        # The leader runs alongside training and preserves room for its only
        # final checkpoint. Candidate quiet scoring starts after that is saved.
        require_space(quiet_plan, 355_000_000 if kind == "leader" else 5_000_000)
        destination.mkdir()
        write(destination / "plan.json", quiet_plan)
        execute([PYTHON, "-u", "-m", "research.direct.evaluate_latency58_quiet_leader_cleanup",
                 "--plan", str(destination / "plan.json"), "--plan-sha256", sha(destination / "plan.json")],
                destination, "evaluation", plan["evaluation_timeout_seconds"], quiet_plan["source_bindings"],
                {"plan_sha256": sha(destination / "plan.json")})
        return load_quiet(destination, evidence)

    try:
        leader_dir = PHASE / "quiet-wanted-drum500-001"
        candidate_dir = PHASE / "quiet-wanted-leader-cleanup-001"
        score_quiet("leader", "sdr-drum-accum-500", leader_dir)
        wait_for(PHASE / "leader-cleanup-250-full14-001/execution.json")
        candidate = score_quiet("leader_cleanup", "leader-cleanup-250", candidate_dir)
        references = {"working": PHASE / "quiet-wanted-working-001",
                      "working_cleanup": PHASE / "quiet-wanted-controlled-deployed-half-001", "leader": leader_dir}
        comparisons = {}
        for label, directory in references.items():
            reference = load_quiet(directory, evidence)
            require(reference["reference_inventory"] == candidate["reference_inventory"], "Different quiet source support")
            destination = PHASE / ("quiet-wanted-leader-cleanup-versus-" + label.replace("_", "-") + "-001")
            require(not destination.exists(), "Preserve quiet comparison")
            comparison = {"schema": "latency58-quiet-wanted-comparison-plan-v1", "candidate_directory": str(candidate_dir),
                "reference_directory": str(directory), "output_directory": str(destination), "source_bindings": dict(evidence),
                **{k: training[k] for k in ("counted_roots", "stop_counted_bytes")}}
            require_space(comparison, 3_000_000)
            destination.mkdir()
            write(destination / "plan.json", comparison)
            execute([PYTHON, "-u", "-m", "research.direct.compare_latency58_quiet_wanted",
                     "--plan", str(destination / "plan.json"), "--plan-sha256", sha(destination / "plan.json")],
                    destination, "comparison", 240, comparison["source_bindings"], {"plan_sha256": sha(destination / "plan.json")})
            comparisons[label] = {"plan": binding(destination / "plan.json"), "result": binding(destination / "result.json"),
                                  "execution": binding(destination / "comparison-execution.json")}
        wait_for(queue_terminal)
        execution, completed = read(queue_terminal), read(queue_path.parent / "result.json")
        require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"] and completed["status"] == "pass"
                and completed["source_bindings_unchanged"]
                and execution["plan_sha256"] == completed["plan_sha256"] == plan["quality_queue_plan"]["sha256"],
                "Main quality bundle did not complete")
        for path in (queue_terminal, queue_path.parent / "result.json"):
            evidence[str(path)] = sha(path)
        views_dir = PHASE / "leader-cleanup-250-views-001"
        views = load_views(views_dir, evidence)
        require(views["model"]["model_state_sha256"] == candidate["model"]["model_state_sha256"], "Quiet and gain models differ")
        gain_references = {label: str(reference_views(label)) for label in references}
        for directory in gain_references.values():
            load_views(Path(directory), evidence)
        destination = PHASE / "leader-cleanup-250-gain-error-001"
        require(not destination.exists(), "Preserve gain comparison")
        gain_plan = {"schema": "latency58-leader-cleanup-gain-error-plan-v1", "candidate_directory": str(views_dir),
            "references": gain_references, "output_directory": str(destination), "source_bindings": dict(evidence),
            **{k: training[k] for k in ("counted_roots", "stop_counted_bytes")}}
        require_space(gain_plan, 3_000_000)
        destination.mkdir()
        write(destination / "plan.json", gain_plan)
        execute([PYTHON, "-u", "-m", "research.direct.report_latency58_leader_cleanup_gain_error",
                 "--plan", str(destination / "plan.json"), "--plan-sha256", sha(destination / "plan.json")],
                destination, "gain-error", 240, gain_plan["source_bindings"], {"plan_sha256": sha(destination / "plan.json")})
        verify_inputs(plan)
        write(out / "result.json", {"schema": "latency58-leader-cleanup-supplements-v1", "status": "pass",
            "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
            "leader_quiet": binding(leader_dir / "result.json"), "candidate_quiet": binding(candidate_dir / "result.json"),
            "quiet_comparisons": comparisons, "gain_comparison": {"plan": binding(destination / "plan.json"),
                "result": binding(destination / "result.json"), "execution": binding(destination / "gain-error-execution.json")},
            "model_state_sha256": candidate["model"]["model_state_sha256"], "training_updates_executed": 0,
            "quality_selected": False, "human_listening_completed": False})
        print({"status": "pass", "quiet_comparisons": 3, "gain_comparisons": 3}, flush=True)
    finally:
        if descriptor is not None:
            os.close(descriptor)


if __name__ == "__main__":
    main()
