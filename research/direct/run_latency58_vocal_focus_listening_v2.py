"""Prepare and verify a native Actions player for a completed vocal-pilot prefix."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from research.direct.run_latency58_quality import PHASE, ROOT, PYTHON, execute, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed
from research.direct.prepare_latency58_vocal_focus_listening_v2 import ORDER
from research.direct.train_latency58 import verify_inputs


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def windows_path(path):
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True, timeout=10).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last-arm", required=True, choices=ORDER[2:])
    parser.add_argument("--suffix", default="001")
    args = parser.parse_args()
    require(Path.cwd() == ROOT and len(args.suffix) == 3 and args.suffix.isdecimal(), "Unexpected cwd or output suffix")
    prefix = "vocal-focus-" + args.last_arm.replace("_", "-") + "-250"
    player = PHASE / (prefix + "-listening-" + args.suffix)
    browser = PHASE / (prefix + "-actions-browser-" + args.suffix)
    require(not player.exists() and not browser.exists(), "Preserve previous player and browser outputs")
    sources = {str(path): sha(path) for path in (
        Path(__file__).resolve(), ROOT / "research/direct/prepare_latency58_vocal_focus_listening_v2.py",
        ROOT / "research/direct/check_latency58_vocal_focus_listening_browser.ps1",
        ROOT / "research/direct/audit_latency58_vocal_focus_browser.py",
        ROOT / "research/direct/run_latency58_quality.py", ROOT / "research/direct/report_latency58_sdr.py")}
    models = []
    training = None
    for arm in ORDER[:ORDER.index(args.last_arm) + 1]:
        current_prefix = {"working": "teacher-half250", "drum500": "sdr-drum-accum-500"}.get(
            arm, "vocal-focus-" + arm.replace("_", "-") + "-250")
        directory = PHASE / (current_prefix + "-actions60-001")
        quality, report = load_completed(directory, sources, canonical_baseline=arm == "working")
        model = report["results"][0]["model"]
        checkpoint_sha = read(PHASE / "teacher-half-canonical-001/receipt.json")["output"]["sha256"] \
            if arm == "working" else model["checkpoint"]["sha256"]
        row = {"id": arm, "prefix": current_prefix,
               "label": {"working": "Working", "drum500": "Drum 500", "original": "Original augmentation",
                         "focused": "Focused augmentation", "focused_mixer": "Focused + mask correction"}[arm],
               "model_state_sha256": model["model_state_sha256"], "checkpoint_sha256": checkpoint_sha}
        for path in (directory / "audio").glob("estimate-*.wav"):
            sources[str(path)] = sha(path)
        if arm in ORDER[2:]:
            summary_dir = PHASE / (current_prefix + "-summary-001")
            summary = read(summary_dir / "result.json")
            summary_execution = read(summary_dir / "summary-execution.json")
            require(summary["schema"] == "latency58-vocal-focus-quality-summary-v1" and summary["status"] == "pass"
                    and summary["source_bindings_unchanged"] and summary["all_metrics_compared"]
                    and summary["arm"] == arm and summary["step"] == 250
                    and summary["model_state_sha256"] == row["model_state_sha256"]
                    and summary_execution["actual_exit_code"] == 0 and not summary_execution["timed_out"]
                    and summary_execution["source_bindings_unchanged"]
                    and summary_execution["plan_sha256"] == summary["plan_sha256"] == sha(summary_dir / "plan.json"),
                    "Pilot quality bundle is not complete")
            verify_inputs(summary)
            sources.update(summary["source_bindings"])
            row.update(summary=binding(summary_dir / "result.json"),
                       summary_execution=binding(summary_dir / "summary-execution.json"),
                       training_plan=quality["training_plan"])
            for item in (row["summary"], row["summary_execution"], row["training_plan"], binding(summary_dir / "plan.json")):
                sources[item["path"]] = item["sha256"]
            training = read(row["training_plan"]["path"])
        models.append(row)
    require(training is not None and training["arm"] == args.last_arm, "Different final pilot")
    template = ROOT / "research/direct/runs/latency11/listening/actions60-tradeoff/index.html"
    shared = ROOT / "research/direct/runs/latency11/listening/recovered-actions60-c191/00-c191-native/Actions_-_One_Minute_Smile/excerpt-0"
    source = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile")
    config = ROOT / "research/eval_config.json"
    for path in (template, config, *(source / (s + ".wav") for s in ("mixture", "drums", "bass", "vocals", "other")),
                 *(shared / name for name in ("mixture.wav", "reference-drums.wav", "reference-bass.wav",
                                               "reference-vocals.wav", "reference-other.wav"))):
        sources[str(path)] = sha(path)
    plan = {"schema": "latency58-vocal-focus-listening-plan-v1", "step": 250, "models": models,
            "output_directory": str(player), "shared_reference_directory": str(shared),
            "source_track_directory": str(source), "template": str(template), "evaluation_config": str(config),
            "source_bindings": sources, "counted_roots": training["counted_roots"],
            "stop_counted_bytes": training["stop_counted_bytes"]}
    verify_inputs(plan)
    from research.direct.latency58_vocal_focus_checkpoint import require_space
    require_space(plan, 2_000_000)
    player.mkdir()
    write(player / "plan.json", plan)
    execute([PYTHON, "-u", "-m", "research.direct.prepare_latency58_vocal_focus_listening_v2",
             "--plan", str(player / "plan.json"), "--plan-sha256", sha(player / "plan.json")],
            player, "preparation", 180, sources, {"plan_sha256": sha(player / "plan.json")})
    for name in ("plan.json", "result.json", "preparation-execution.json", "index.html"):
        path = player / name
        sources[str(path)] = sha(path)
    page_url = "http://127.0.0.1:8766/latency58/" + player.name + "/index.html"
    browser_plan = {"schema": "latency58-vocal-focus-listening-browser-execution-v1",
                    "page_url": page_url, "last_arm": args.last_arm, "step": 250, "timeout_seconds": 150,
                    "expected_audio_files": 4 * len(models) + 5, "source_bindings": dict(sources),
                    "muted": True, "human_listening_completed": False}
    browser.mkdir()
    write(browser / "plan.json", browser_plan)
    execute(["/mnt/c/WINDOWS/system32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", windows_path(ROOT / "research/direct/check_latency58_vocal_focus_listening_browser.ps1"),
             "-PageUrl", page_url, "-ReceiptPath", windows_path(browser / "result.json"), "-LastArm", args.last_arm],
            browser, "browser", 150, sources, {"plan_sha256": sha(browser / "plan.json")})
    for name in ("plan.json", "result.json", "browser-execution.json"):
        path = browser / name
        sources[str(path)] = sha(path)
    inventory_plan = {"schema": "latency58-vocal-focus-browser-inventory-plan-v1", "last_arm": args.last_arm,
                      "browser_directory": str(browser), "player_directory": str(player), "page_url": page_url,
                      "output": str(browser / "inventory-check.json"), "source_bindings": sources}
    write(browser / "inventory-plan.json", inventory_plan)
    execute([PYTHON, "-u", "-m", "research.direct.audit_latency58_vocal_focus_browser",
             "--plan", str(browser / "inventory-plan.json"), "--plan-sha256", sha(browser / "inventory-plan.json")],
            browser, "inventory", 180, sources, {"plan_sha256": sha(browser / "inventory-plan.json")})
    print({"event": "native_actions_player_verified", "page_url": page_url, "last_arm": args.last_arm,
           "files": 4 * len(models) + 5, "human_listening_completed": False}, flush=True)


if __name__ == "__main__":
    main()
