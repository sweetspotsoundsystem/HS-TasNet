"""Check real retained summary formats and parse the expanded native browser test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.report_latency58_sdr import load_completed
from research.direct.prepare_latency58_leader_cleanup_listening import validate_summary, ORDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Listening preflight changed")
    plan = read(args.plan)
    verify_inputs(plan)
    require(plan["schema"] == "latency58-leader-cleanup-listening-check-plan-v1", "Different check")
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists(), "Preserve preflight")
    evidence, models, rejected = {}, {}, 0
    for label, prefix in (("leader", "sdr-drum-accum-500"), ("working_cleanup", "controlled-deployed-half-250")):
        quality, report = load_completed(PHASE / (prefix + "-actions60-001"), evidence)
        summary = PHASE / (prefix + "-summary-001")
        row = {"id": label, "prefix": prefix, "training_plan": quality["training_plan"]}
        for key, name in (("summary", "result.json"), ("summary_plan", "plan.json"), ("summary_execution", "summary-execution.json")):
            row[key] = {"path": str(summary / name), "sha256": sha(summary / name)}
        fingerprint = report["results"][0]["model"]["model_state_sha256"]
        validate_summary(row, quality, fingerprint, plan["source_bindings"])
        models[label] = {"model_state_sha256": fingerprint, "summary_schema": read(summary / "result.json")["schema"]}
        for changed_quality, changed_fingerprint in ((quality, "0" * 64), ({**quality, "step": quality["step"] + 1}, fingerprint)):
            try:
                validate_summary(row, changed_quality, changed_fingerprint, plan["source_bindings"])
            except RuntimeError:
                rejected += 1
    require(rejected == 4 and all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Bad summary accepted or unbound score")
    prior = read(plan["prior_browser_execution"]["path"])
    require(prior["actual_exit_code"] == 0 and not prior["timed_out"] and prior["source_bindings_unchanged"], "Different native shell reference")
    script = ROOT / "research/direct/check_latency58_leader_cleanup_listening_browser.ps1"
    require(plan["source_bindings"].get(str(script)) == sha(script), "Unbound browser script")
    windows_path = subprocess.run(["wslpath", "-w", str(script)], check=True, text=True, capture_output=True, timeout=10).stdout.strip()
    literal = "'" + windows_path.replace("'", "''") + "'"
    code = "$leaderParseTokens=$null; $leaderParseErrors=$null; " + \
        "[System.Management.Automation.Language.Parser]::ParseFile(" + literal + ", [ref]$leaderParseTokens, [ref]$leaderParseErrors) | Out-Null; " + \
        "if ($leaderParseErrors.Count -ne 0) { $leaderParseErrors | Out-String | Write-Output; exit 1 }; " + \
        "$leaderParseHash=[System.Security.Cryptography.SHA256]::Create().ComputeHash([System.IO.File]::ReadAllBytes(" + literal + ")); " + \
        "@{status='pass'; source_sha256=[BitConverter]::ToString($leaderParseHash).Replace('-', '').ToLowerInvariant(); parse_errors=0} | ConvertTo-Json -Compress"
    argv = [prior["argv"][0], "-NoProfile", "-NonInteractive", "-Command", code]
    parsed = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=30, check=False)
    write(out / "native-parser-execution.json", {"argv": argv, "actual_exit_code": parsed.returncode,
                                               "stdout": parsed.stdout, "stderr": parsed.stderr})
    require(parsed.returncode == 0, "Native PowerShell parsing failed")
    native = json.loads(parsed.stdout)
    require(native["status"] == "pass" and native["parse_errors"] == 0 and native["source_sha256"] == sha(script),
            "Native parser read a different script")
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-leader-cleanup-listening-check-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
          "retained_summaries_authenticated": models, "incorrect_summary_identities_or_steps_rejected": rejected,
          "model_order": list(ORDER), "expected_audio_files": 4 * len(ORDER) + 5, "native_browser_script_parse": native,
          "new_candidate_summary_dispatch_exercised": False, "browser_playback_tested": False,
          "inference_executed": False, "audio_files_copied": 0, "training_updates_executed": 0,
          "quality_selected": False, "human_listening_completed": False,
          "limitations": ["The candidate summary and all 21 decoded audio files remain to be checked after the live quality queue finishes.",
                          "Native script parsing does not establish browser playback, human listening quality or M4 runtime performance."]})
    print({"status": "pass", "retained_summary_models": list(models), "native_script_parse_pass": True}, flush=True)


if __name__ == "__main__":
    main()
