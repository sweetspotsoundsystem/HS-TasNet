"""Build and check the native Actions comparison after the complete quality queue exits."""
from __future__ import annotations

import argparse
import copy
import ctypes
import os
from pathlib import Path
import select
import time

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write, execute
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_sdr_checkpoint import require_space


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Changed listening workflow")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-controlled-deployed-listening-workflow-plan-v1"
            and plan["maximum_wait_seconds"] == 21000, "Different bounded workflow")
    out, queue = Path(plan["output_directory"]), Path(plan["quality_queue_directory"])
    player, browser = Path(plan["player_directory"]), Path(plan["browser_directory"])
    require(out.parent == queue.parent == player.parent == browser.parent == PHASE and out.is_dir()
            and not (out / "result.json").exists() and not player.exists() and not browser.exists(),
            "Preserve listening workflow and output")
    for key in ("quality_queue_plan", "prior_player_plan", "prior_browser_execution"):
        item = plan[key]
        require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound prerequisite")
    terminal = queue / "queue-execution.json"
    owner, fd = plan["execution_owner"], None
    began = time.monotonic()
    if not terminal.exists():
        libc = ctypes.CDLL(None, use_errno=True)
        libc.pidfd_open.argtypes, libc.pidfd_open.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
        fd = libc.pidfd_open(owner["pid"], 0)
        if fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        try:
            proc = Path(f"/proc/{owner['pid']}")
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            argv = [x.decode() for x in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
            require(int(stat[19]) == owner["start_ticks"] and argv == owner["argv"], "Different quality execution owner")
            write(out / "wait-start.json", {"execution_owner": owner, "pidfd_opened": True,
                                           "plan_sha256": args.plan_sha256, "wall_time": time.time()})
            poller = select.poll(); poller.register(fd, select.POLLIN)
            while not terminal.exists():
                require(time.monotonic() - began < plan["maximum_wait_seconds"], "Bounded listening wait expired")
                require(not poller.poll(5000) or terminal.exists(), "Quality execution ended without a receipt")
        finally:
            os.close(fd)
    execution, completed = read(terminal), read(queue / "result.json")
    require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and completed["status"] == "pass"
            and completed["source_bindings_unchanged"] and completed["plan_sha256"] == execution["plan_sha256"]
            == plan["quality_queue_plan"]["sha256"] == sha(queue / "plan.json"), "Complete quality queue failed")
    verify_inputs(plan)
    evidence = {**plan["source_bindings"], str(args.plan): args.plan_sha256}
    evidence.update(completed["source_bindings"])
    for path in (terminal, queue / "result.json"):
        evidence[str(path)] = sha(path)
    template = read(plan["prior_player_plan"]["path"])
    verify_inputs(template)
    evidence.update(template["source_bindings"])
    models = []
    match = None
    choices = (("working", "teacher-half250", "Working model"),
               ("ordinary_control", "counterfactual-teacher-ordinary-only-250", "Previous vocal cleanup"),
               ("controlled_deployed", "controlled-deployed-half-250", "New vocal cleanup"))
    for model_id, prefix, label in choices:
        directory = PHASE / (prefix + "-actions60-001")
        quality, report = load_completed(directory, evidence, canonical_baseline=model_id == "working")
        current = report["results"][0]["model"]
        row = {"id": model_id, "prefix": prefix, "label": label,
               "model_state_sha256": current["model_state_sha256"],
               "checkpoint_sha256": read(PHASE / "teacher-half-canonical-001/receipt.json")["output"]["sha256"]
                    if model_id == "working" else current["checkpoint"]["sha256"]}
        if model_id != "working":
            summary = PHASE / (prefix + "-summary-001")
            row.update(summary=binding(summary / "result.json"),
                       summary_execution=binding(summary / "summary-execution.json"), training_plan=quality["training_plan"])
            for path in (summary / "plan.json", summary / "result.json", summary / "summary-execution.json"):
                evidence[str(path)] = sha(path)
            if model_id == "controlled_deployed":
                match = quality["training_match"]
                require(completed["training_match"] == match and completed["summary"] == row["summary"], "Different queue endpoint")
        for path in (directory / "audio").glob("estimate-*.wav"):
            evidence[str(path)] = sha(path)
        models.append(row)
    require(match is not None, "Missing full training pairing")
    prepared = copy.deepcopy(template)
    prepared.update(schema="latency58-controlled-deployed-listening-plan-v1", output_directory=str(player),
                    models=models, training_match=match, source_bindings=evidence)
    require_space(prepared, 5_000_000)
    player.mkdir(); write(player / "plan.json", prepared)
    execute([PYTHON, "-u", "-m", "research.direct.prepare_latency58_controlled_deployed_listening",
             "--plan", str(player / "plan.json"), "--plan-sha256", sha(player / "plan.json")],
            player, "preparation", 240, evidence, {"plan_sha256": sha(player / "plan.json")})
    captured = read(player / "result.json")
    require(captured["status"] == "pass" and len(captured["audio_files"]) == 17
            and captured["all_decoded_track_and_aggregate_scores_exact"] and captured["audio_files_copied"] == 0,
            "Native player preparation failed")
    for path in (player / "plan.json", player / "result.json", player / "preparation-execution.json", player / "index.html"):
        evidence[str(path)] = sha(path)
    evidence.update({path: item["sha256"] for path, item in captured["audio_files"].items()})
    page = "http://127.0.0.1:8766/latency58/" + player.name + "/index.html"
    browser_plan = {"schema": "latency58-controlled-deployed-listening-browser-execution-v1", "page_url": page,
                    "last_arm": "controlled_deployed", "step": 250, "timeout_seconds": 150,
                    "expected_audio_files": 17, "muted": True, "human_listening_completed": False,
                    "source_bindings": dict(evidence)}
    browser.mkdir(); write(browser / "plan.json", browser_plan)
    prior = read(plan["prior_browser_execution"]["path"])
    require(prior["actual_exit_code"] == 0 and not prior["timed_out"] and prior["source_bindings_unchanged"],
            "Browser invocation template did not pass")
    argv = list(prior["argv"])
    require(argv[argv.index("-File") + 1].endswith("check_latency58_counterfactual_listening_browser.ps1")
            and argv[argv.index("-LastArm") + 1] == "ordinary_only"
            and Path(plan["prior_browser_execution"]["path"]).parent.name
                in argv[argv.index("-ReceiptPath") + 1], "Unexpected browser invocation template")
    argv[argv.index("-File") + 1] = argv[argv.index("-File") + 1].replace(
        "check_latency58_counterfactual_listening_browser.ps1", "check_latency58_controlled_deployed_listening_browser.ps1")
    argv[argv.index("-PageUrl") + 1] = page
    argv[argv.index("-ReceiptPath") + 1] = argv[argv.index("-ReceiptPath") + 1].replace(
        Path(plan["prior_browser_execution"]["path"]).parent.name, browser.name)
    argv[argv.index("-LastArm") + 1] = "controlled_deployed"
    write(browser / "command.json", {"argv": argv, "plan_sha256": sha(browser / "plan.json")})
    execute(argv, browser, "browser", 150, {**evidence, str(browser / "plan.json"): sha(browser / "plan.json")},
            {"plan_sha256": sha(browser / "plan.json")})
    for path in (browser / "plan.json", browser / "result.json", browser / "browser-execution.json"):
        evidence[str(path)] = sha(path)
    inventory = {"schema": "latency58-controlled-deployed-browser-inventory-plan-v1", "last_arm": "controlled_deployed",
                 "browser_directory": str(browser), "player_directory": str(player), "page_url": page,
                 "output": str(browser / "inventory-check.json"), "source_bindings": dict(evidence)}
    write(browser / "inventory-plan.json", inventory)
    execute([PYTHON, "-u", "-m", "research.direct.audit_latency58_controlled_deployed_browser",
             "--plan", str(browser / "inventory-plan.json"), "--plan-sha256", sha(browser / "inventory-plan.json")],
            browser, "inventory", 240, evidence, {"plan_sha256": sha(browser / "inventory-plan.json")})
    checked = read(browser / "inventory-check.json")
    require(checked["status"] == "pass" and len(checked["files"]) == 17
            and checked["all_served_bytes_exact"] and checked["all_byte_ranges_exact"], "Native playback inventory failed")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-controlled-deployed-listening-workflow-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "player": binding(player / "result.json"), "browser": binding(browser / "result.json"),
          "inventory": binding(browser / "inventory-check.json"), "page_url": page,
          "training_updates_executed": 0, "audio_files_copied": 0, "new_inference": False,
          "quality_selected": False, "human_listening_completed": False})
    print({"status": "pass", "page_url": page, "verified_native_files": 17}, flush=True)


if __name__ == "__main__":
    main()
