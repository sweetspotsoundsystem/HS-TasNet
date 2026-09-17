"""Run the reserved native Skelpolu comparison after all three pilots close."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from research.direct.run_latency58_quality import PHASE, ROOT, PYTHON, read, require, sha, write, execute
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_vocal_focus_checkpoint import require_space
from research.direct.capture_latency58_vocal_focus_skelpolu import ARMS, completed


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def windows_path(path):
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True, timeout=10).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-result", type=Path, required=True)
    parser.add_argument("--match-execution", type=Path, required=True)
    parser.add_argument("--suffix", default="001")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and len(args.suffix) == 3 and args.suffix.isdecimal(), "Unexpected cwd or suffix")
    player = PHASE / ("vocal-focus-skelpolu-audio-" + args.suffix)
    browser = PHASE / ("vocal-focus-skelpolu-browser-" + args.suffix)
    require(not player.exists() and not browser.exists(), "Preserve previous capture and browser evidence")
    reservation_path = PHASE / "vocal-focus-skelpolu-reservation-001/reservation.json"
    reservation = read(reservation_path)
    verify_inputs(reservation)
    sources = {**reservation["source_bindings"], str(reservation_path): sha(reservation_path)}
    for name in ("run_latency58_vocal_focus_skelpolu.py", "capture_latency58_vocal_focus_skelpolu.py",
                 "check_latency58_vocal_focus_skelpolu_browser.ps1", "audit_latency58_vocal_focus_skelpolu_browser.py",
                 "run_latency58_quality.py", "report_latency58_sdr.py", "record_latency58_vocal_focus_review.py",
                 "retire_latency58_vocal_focus_optimizer.py"):
        path = ROOT / "research/direct" / name
        sources[str(path)] = sha(path)
    execution_path = args.match_execution.resolve(strict=True)
    result_path = args.match_result.resolve(strict=True)
    require(execution_path.is_relative_to(PHASE) and result_path.is_relative_to(PHASE), "Use this phase's matching audit")
    command = read(execution_path)["argv"]
    match_plan_path = Path(command[command.index("--plan") + 1])
    match_plan = read(match_plan_path)
    sources.update(match_plan["source_bindings"])
    sources.update(read(result_path)["source_bindings"])
    for path in (execution_path, result_path, match_plan_path):
        sources[str(path)] = sha(path)
    old = Path(reservation["reuse_capture_directory"])
    prior_plan = read(old / "plan.json")
    prior = read(old / "result.json")
    sources.update(prior["source_bindings"])
    reuse_models, models = [], []
    for arm, row in reservation["reused_models"].items():
        _, report = load_completed(PHASE / (row["prefix"] + "-full14-001"), sources, canonical_baseline=arm == "working")
        captured = prior["models"][row["capture_model_id"]]
        require(captured["model_state_sha256"] == row["model_state_sha256"]
                == report["results"][0]["model"]["model_state_sha256"], "Different reused baseline")
        reuse_models.append({"id": arm, "capture_id": row["capture_model_id"], "prefix": row["prefix"],
                             "model_state_sha256": row["model_state_sha256"],
                             "checkpoint_sha256": captured["checkpoint"]["sha256"],
                             "label": "Working" if arm == "working" else "Drum 500"})
    for arm in ARMS:
        prefix = "vocal-focus-" + arm.replace("_", "-") + "-250"
        quality, report = load_completed(PHASE / (prefix + "-full14-001"), sources)
        model = report["results"][0]["model"]
        row = {"id": arm, "prefix": prefix, "training_plan": quality["training_plan"],
               "model_state_sha256": model["model_state_sha256"], "checkpoint_sha256": model["checkpoint"]["sha256"],
               "label": {"original": "Original augmentation", "focused": "Focused augmentation",
                         "focused_mixer": "Focused + mask correction"}[arm]}
        for kind, directory, filenames in (
            ("review", PHASE / (prefix + "-review-001"), ("plan.json", "result.json", "review-execution.json")),
            ("retirement", PHASE / (prefix + "-optimizer-retirement-001"), ("plan.json", "receipt.json", "retirement-execution.json")),
        ):
            for key, name in zip((kind + "_plan", kind, kind + "_execution"), filenames, strict=True):
                path = directory / name
                row[key] = binding(path)
                sources[str(path)] = sha(path)
            sources.update(read(row[kind + "_plan"]["path"])["source_bindings"])
        models.append(row)
    plan = {"schema": "latency58-vocal-focus-skelpolu-audio-plan-v1", "step": 250,
            "track_index": 10, "track_name": reservation["track_name"], "reservation": binding(reservation_path),
            "models": models, "reuse_models": reuse_models, "output_directory": str(player),
            "reuse_capture_directory": str(old), "working_model_state_sha256": reuse_models[0]["model_state_sha256"],
            "match_audit": binding(result_path), "match_execution": binding(execution_path),
            "match_plan": binding(match_plan_path), "source_bindings": sources,
            "new_audio_reserve_bytes": 135_000_000, "concurrent_checkpoint_reserve_bytes": 0,
            **{key: reservation[key] for key in ("counted_roots", "stop_counted_bytes")},
            **{key: prior_plan[key] for key in ("manifest", "evaluation_config", "template", "torch_version")}}
    verify_inputs(plan)
    completed(plan, plan, "match_audit", "match_execution", "match_plan",
              "research.direct.audit_latency58_vocal_focus_training_match")
    for row in models:
        completed(plan, row, "review", "review_execution", "review_plan", "research.direct.record_latency58_vocal_focus_review")
        completed(plan, row, "retirement", "retirement_execution", "retirement_plan",
                  "research.direct.retire_latency58_vocal_focus_optimizer")
    require_space(plan, 135_000_000)
    player.mkdir()
    write(player / "plan.json", plan)
    execute([PYTHON, "-u", "-m", "research.direct.capture_latency58_vocal_focus_skelpolu",
             "--plan", str(player / "plan.json"), "--plan-sha256", sha(player / "plan.json")],
            player, "capture", 900, sources, {"plan_sha256": sha(player / "plan.json")})
    captured = read(player / "result.json")
    sources = dict(sources)
    for path in (player / name for name in ("plan.json", "result.json", "capture-execution.json", "index.html")):
        sources[str(path)] = sha(path)
    sources.update({path: row["sha256"] for path, row in captured["audio_files"].items()})
    page = "http://127.0.0.1:8766/latency58/" + player.name + "/index.html"
    browser_plan = {"schema": "latency58-vocal-focus-skelpolu-browser-execution-v1", "last_arm": "focused_mixer",
                    "page_url": page, "timeout_seconds": 150, "expected_audio_files": 50,
                    "source_bindings": dict(sources), "muted": True, "human_listening_completed": False}
    browser.mkdir()
    write(browser / "plan.json", browser_plan)
    execute(["/mnt/c/WINDOWS/system32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", windows_path(ROOT / "research/direct/check_latency58_vocal_focus_skelpolu_browser.ps1"),
             "-PageUrl", page, "-ReceiptPath", windows_path(browser / "result.json"), "-LastArm", "focused_mixer"],
            browser, "browser", 150, sources, {"plan_sha256": sha(browser / "plan.json")})
    for name in ("plan.json", "result.json", "browser-execution.json"):
        path = browser / name
        sources[str(path)] = sha(path)
    inventory = {"schema": "latency58-vocal-focus-skelpolu-browser-inventory-plan-v1", "last_arm": "focused_mixer",
                 "player_directory": str(player), "browser_directory": str(browser), "page_url": page,
                 "output": str(browser / "inventory-check.json"), "source_bindings": dict(sources)}
    write(browser / "inventory-plan.json", inventory)
    execute([PYTHON, "-u", "-m", "research.direct.audit_latency58_vocal_focus_skelpolu_browser",
             "--plan", str(browser / "inventory-plan.json"), "--plan-sha256", sha(browser / "inventory-plan.json")],
            browser, "inventory", 240, sources, {"plan_sha256": sha(browser / "inventory-plan.json")})
    print({"event": "native_skelpolu_player_verified", "page_url": page, "files": 50,
           "human_listening_completed": False}, flush=True)


if __name__ == "__main__":
    main()
