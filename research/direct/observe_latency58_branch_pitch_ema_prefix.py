"""Record the live continuation launcher and independently compare its first two updates."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def process(pid):
    entry = Path("/proc") / str(pid)
    argv = [value.decode() for value in (entry / "cmdline").read_bytes().split(b"\0") if value]
    ticks = (entry / "stat").read_text().rsplit(") ", 1)[1].split()[19]
    return argv, ticks, (entry / "cwd").resolve(strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-root-session", type=int, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.actual_root_session > 0, "Use the active continuation workspace and tool handle")
    root = PHASE / "branch-pitch-ema-001"
    plan_path = root / "plan.json"
    plan = read(plan_path)
    verify_inputs(plan)
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv, ticks, cwd = process(int(entry.name))
        except (OSError, ValueError):
            continue
        if len(argv) > 3 and argv[:4] == [PYTHON, "-u", "-m", "research.direct.run_latency58_branch_pitch_ema"]:
            matches.append((int(entry.name), argv, ticks, cwd))
    require(len(matches) == 1, "Require exactly one live continuation launcher")
    pid, argv, ticks, cwd = matches[0]
    require(cwd == ROOT and "--after-resource" in argv
            and (ROOT / argv[argv.index("--prepared-plan") + 1]).resolve(strict=True) == plan_path,
            "Live launcher has the wrong command or plan")
    paths = [plan_path, Path(__file__).resolve(), ROOT / "research/direct/report_latency58_branch_pitch_ema.py",
             root / "resource-stage/root-execution.json", root / "resource-stage/execution.json",
             root / "resource-run/result.json", PHASE / "branch-pitch-ema-preparation-stage-001/execution.json",
             PHASE / "branch-pitch-ema-preparation-stage-001/root-execution.json"]
    bindings = {**plan["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    require(process(pid) == (argv, ticks, cwd), "Launcher identity changed before recording")
    write(root / "root-command.json", {"record_kind": "observed_live_root_command",
          "actual_root_session": args.actual_root_session, "actual_root_pid": pid,
          "process_start_ticks": ticks, "observed_utc": datetime.now(timezone.utc).isoformat(),
          "argv": argv, "cwd": str(cwd), "source_bindings": bindings, "plan_sha256": sha(plan_path)})
    resource = read(root / "resource-run/result.json")
    deadline = time.monotonic() + 300
    while True:
        require(process(pid) == (argv, ticks, cwd), "Launcher ended or changed before observing the prefix")
        journal = root / "production-run/metrics.jsonl"
        lines = journal.read_text().splitlines() if journal.exists() else []
        if len(lines) >= 2:
            rows = [json.loads(line) for line in lines[:2]]
            break
        require(time.monotonic() < deadline, "No complete production prefix before the observation deadline")
        time.sleep(1)
    normalized = [{k: v for k, v in row.items() if k not in ("elapsed_seconds", "peak_vram_gib", "data_wait_seconds", "compute_and_audit_seconds")} for row in rows]
    require(normalized == resource["matching_production_updates"]
            and all(len(row["branch_memory_gradient_norms"]) == 10
                    and all(value > 0 for value in row["branch_memory_gradient_norms"].values()) for row in rows),
            "Production prefix or trained branch gradients differ from rehearsal")
    bindings[str(root / "root-command.json")] = sha(root / "root-command.json")
    verify_inputs({"source_bindings": bindings})
    write(root / "production-prefix-independent-check.json", {"status": "pass", "normalized_fields_equal": True,
          "excluded_observation_fields": ["elapsed_seconds", "peak_vram_gib", "data_wait_seconds", "compute_and_audit_seconds"], "production_first_two_rows": rows,
          "resource_first_two_updates": resource["matching_production_updates"],
          "resource_step2_raw_model_state_sha256": resource["final_raw_model_state_sha256"],
          "resource_step2_ema_parameters_sha256": resource["final_ema_parameters_sha256"],
          "model_state_evidence_scope": "The frozen trainer asserts step-2 model-state equality before writing step 2. This observer compares journal fields and does not read live model memory.",
          "actual_root_session": args.actual_root_session, "root_pid": pid, "process_start_ticks": ticks,
          "observed_utc": datetime.now(timezone.utc).isoformat(), "all_ten_trained_branch_gradients_nonzero": True,
          "source_bindings": bindings})
    print(json.dumps({"status": "pass", "actual_root_session": args.actual_root_session, "root_pid": pid,
                      "first_two_updates_match_rehearsal": True, "all_ten_trained_branch_gradients_nonzero": True}), flush=True)


if __name__ == "__main__":
    main()
