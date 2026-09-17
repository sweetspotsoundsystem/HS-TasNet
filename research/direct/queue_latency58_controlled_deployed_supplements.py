"""Complete quiet comparisons and vocal gain checks after authenticated live jobs finish."""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import select
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs
from research.direct.compare_latency58_quiet_wanted import load_completed as load_quiet
from research.direct.report_latency58_vocal_focus import load_views
from research.direct.latency58_sdr_checkpoint import require_space


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Queue plan changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-controlled-deployed-supplements-queue-plan-v1"
            and plan["maximum_wait_seconds"] == 21000 and set(plan["owners"]) == {"quality", "quiet"}
            and set(plan["quiet_references"]) == {"working", "focused", "ordinary_only"}
            and set(plan["gain_references"]) == {"working", "focused_control", "ordinary_control", "drum500", "drum1000"},
            "Different fixed supplement scope")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve queue")
    destinations = [Path(plan["gain_output_directory"]),
                    *(Path(p) for p in plan["quiet_comparison_directories"].values())]
    require(set(plan["quiet_comparison_directories"]) == set(plan["quiet_references"])
            and all(p.parent == PHASE and not p.exists() for p in destinations), "Existing or unexpected comparison output")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes, libc.pidfd_open.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    descriptors = {}
    began = time.monotonic()
    evidence = {**plan["source_bindings"], str(args.plan): args.plan_sha256}
    terminals = {label: Path(spec["queue_directory"]) / "queue-execution.json" for label, spec in plan["owners"].items()}
    try:
        for label, spec in plan["owners"].items():
            if terminals[label].is_file():
                continue
            owner = spec["execution_owner"]
            fd = libc.pidfd_open(owner["pid"], 0)
            if fd < 0:
                if terminals[label].is_file():
                    continue
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            descriptors[label] = fd
            proc = Path(f"/proc/{owner['pid']}")
            try:
                stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                argv = [p.decode() for p in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
            except FileNotFoundError:
                require(terminals[label].is_file(), "Execution owner ended without receipt")
                continue
            require(int(stat[19]) == owner["start_ticks"] and argv == owner["argv"], "Different live execution owner")
        write(out / "wait-start.json", {"owners": plan["owners"], "pidfds_opened": list(descriptors),
                                       "plan_sha256": args.plan_sha256, "wall_time": time.time()})

        def wait_completed(label):
            while not terminals[label].is_file():
                require(time.monotonic() - began < plan["maximum_wait_seconds"], "Bounded supplement wait expired")
                if label in descriptors:
                    poller = select.poll(); poller.register(descriptors[label], select.POLLIN)
                    require(not poller.poll(0) or terminals[label].is_file(), "Execution owner ended without terminal receipt")
                time.sleep(5)
            spec = plan["owners"][label]
            directory = Path(spec["queue_directory"])
            execution, result = read(terminals[label]), read(directory / "result.json")
            require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
                    and execution["source_bindings_unchanged"] and result["status"] == "pass"
                    and result["source_bindings_unchanged"] and result["plan_sha256"] == execution["plan_sha256"]
                    == sha(directory / "plan.json") == spec["queue_plan"]["sha256"]
                    and execution["argv"][execution["argv"].index("-m") + 1] == spec["module"],
                    "Prerequisite queue did not complete cleanly")
            for p in (terminals[label], directory / "result.json", directory / "plan.json"):
                evidence[str(p)] = sha(p)
            verify_inputs(result)
            evidence.update(result["source_bindings"])
            return result

        wait_completed("quiet")
        candidate = Path(plan["quiet_candidate_directory"])
        quiet_result = load_quiet(candidate, evidence)
        require(quiet_result["model"]["kind"] == "controlled_deployed", "Different quiet candidate")
        comparisons = {}
        for label, directory in plan["quiet_references"].items():
            load_quiet(Path(directory), evidence)
            destination = Path(plan["quiet_comparison_directories"][label])
            comparison = {"schema": "latency58-quiet-wanted-comparison-plan-v1",
                          "candidate_directory": str(candidate), "reference_directory": directory,
                          "output_directory": str(destination), "source_bindings": dict(evidence),
                          **{k: plan[k] for k in ("counted_roots", "stop_counted_bytes")}}
            require_space(comparison, 3_000_000)
            destination.mkdir(); write(destination / "plan.json", comparison)
            execute([PYTHON, "-u", "-m", "research.direct.compare_latency58_quiet_wanted",
                     "--plan", str(destination / "plan.json"), "--plan-sha256", sha(destination / "plan.json")],
                    destination, "comparison", 240, evidence, {"plan_sha256": sha(destination / "plan.json")})
            require(read(destination / "result.json")["status"] == "pass", "Quiet comparison failed")
            comparisons[label] = {"plan": binding(destination / "plan.json"), "result": binding(destination / "result.json"),
                                  "execution": binding(destination / "comparison-execution.json")}
            print({"event": "quiet_comparison_complete", "reference": label}, flush=True)
        wait_completed("quality")
        candidate_views = load_views(Path(plan["gain_candidate_directory"]), evidence)
        require(candidate_views["model"]["model_state_sha256"] == quiet_result["model"]["model_state_sha256"],
                "Gain and quiet scores use different endpoints")
        for directory in plan["gain_references"].values():
            load_views(Path(directory), evidence)
        destination = Path(plan["gain_output_directory"])
        gain_plan = {"schema": "latency58-controlled-deployed-gain-error-plan-v1",
                     "candidate_directory": plan["gain_candidate_directory"], "references": plan["gain_references"],
                     "output_directory": str(destination), "source_bindings": dict(evidence),
                     **{k: plan[k] for k in ("counted_roots", "stop_counted_bytes")}}
        require_space(gain_plan, 3_000_000)
        destination.mkdir(); write(destination / "plan.json", gain_plan)
        execute([PYTHON, "-u", "-m", "research.direct.report_latency58_controlled_deployed_gain_error",
                 "--plan", str(destination / "plan.json"), "--plan-sha256", sha(destination / "plan.json")],
                destination, "gain-error", 240, evidence, {"plan_sha256": sha(destination / "plan.json")})
        require(read(destination / "result.json")["status"] == "pass", "Gain comparison failed")
        verify_inputs(plan)
        write(out / "result.json", {"schema": "latency58-controlled-deployed-supplements-queue-v1", "status": "pass",
              "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
              "quiet_comparisons": comparisons, "gain_comparison": {"plan": binding(destination / "plan.json"),
                  "result": binding(destination / "result.json"), "execution": binding(destination / "gain-error-execution.json")},
              "model_state_sha256": quiet_result["model"]["model_state_sha256"], "training_updates_executed": 0,
              "quality_selected": False, "human_listening_completed": False})
        print({"status": "pass", "quiet_comparisons": 3, "gain_comparisons": 5}, flush=True)
    finally:
        for fd in descriptors.values():
            os.close(fd)


if __name__ == "__main__":
    main()
