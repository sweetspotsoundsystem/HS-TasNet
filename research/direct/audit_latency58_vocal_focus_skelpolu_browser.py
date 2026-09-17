"""Verify playback and exact HTTP bytes for all fifty reserved Skelpolu WAVs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.capture_latency58_vocal_focus_skelpolu import ARMS

ORDER = ("working", "drum500", *ARMS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Plan or cwd differs")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-focus-skelpolu-browser-inventory-plan-v1"
            and plan["last_arm"] == "focused_mixer", "Unexpected audit scope")
    verify_inputs(plan)
    binding = plan["source_bindings"]
    browser_path, player_path = Path(plan["browser_directory"]), Path(plan["player_directory"])
    require(browser_path.parent == player_path.parent == PHASE, "Use the vocal pilot phase")
    require(Path(plan["output"]) == browser_path / "inventory-check.json", "Unexpected audit output")
    for path in (browser_path / "plan.json", browser_path / "result.json",
                 browser_path / "browser-execution.json", player_path / "plan.json",
                 player_path / "result.json", player_path / "index.html"):
        require(binding.get(str(path)) == sha(path), "Unbound player or browser evidence")
    browser, execution = read(browser_path / "result.json"), read(browser_path / "browser-execution.json")
    saved, source_plan = read(player_path / "result.json"), read(player_path / "plan.json")
    source_execution_path = player_path / "capture-execution.json"
    require(binding.get(str(source_execution_path)) == sha(source_execution_path), "Unbound player execution")
    source_execution = read(source_execution_path)
    expected_schema = "latency58-vocal-focus-skelpolu-audio-v1"
    require(saved["status"] == browser["status"] == "pass"
            and saved["schema"] == expected_schema
            and browser["schema"] == "latency58-vocal-focus-skelpolu-browser-check-v1"
            and saved["exact_player_inventory"] and len(saved["audio_files"]) == 50
            and len(saved["reused_audio_files"]) == 26 and len(saved["new_audio_files"]) == 24
            and all(m["exact_stored_track_score"] and m["exact_stored_stream_metadata"]
                    and m["model_and_rng_unchanged"] for m in saved["models"].values())
            and all(saved["models"][arm]["decoded_audio_track_and_excerpt_scores_exact"]
                    for arm in ("working", "drum500"))
            and source_execution["actual_exit_code"] == execution["actual_exit_code"] == 0
            and not source_execution["timed_out"] and not execution["timed_out"]
            and saved["source_bindings_unchanged"] and source_execution["source_bindings_unchanged"]
            and execution["source_bindings_unchanged"] and browser["isolated_profile"]
            and browser["temporary_profile_removed"] and not browser["human_listening_completed"]
            and browser["human_listening_verdict"] is None
            and source_execution["plan_sha256"] == saved["plan_sha256"] == sha(player_path / "plan.json")
            and execution["plan_sha256"] == sha(browser_path / "plan.json"), "Incomplete executions")
    verify_inputs(source_plan)
    source_argv = source_execution["argv"]
    require(source_argv[source_argv.index("-m") + 1] == "research.direct.capture_latency58_vocal_focus_skelpolu"
            and source_argv[source_argv.index("--plan") + 1] == str(player_path / "plan.json")
            and source_plan["output_directory"] == str(player_path)
            and all(binding.get(p) == s for p, s in source_plan["source_bindings"].items()),
            "Different player execution or omitted inputs")
    argv = execution["argv"]
    page = plan["page_url"]
    require(argv[argv.index("-LastArm") + 1] == plan["last_arm"]
            and argv[argv.index("-PageUrl") + 1] == browser["page_url"] == page
            and urlparse(page).hostname == "127.0.0.1", "Wrong browser invocation")
    checks = browser["checks"]
    models = list(ORDER[:ORDER.index(plan["last_arm"]) + 1])
    clips = 2
    require(checks["status"] == "pass" and checks["panel"] == "skelpolu"
            and checks["last_arm"] == plan["last_arm"]
            and checks["verified_clip_count"] == clips
            and all(checks[k] for k in ("strict_nonzero_seek_verified", "expected_physical_intervals_verified",
                                       "keyboard_source_switch", "loop_toggle", "browser_audio_muted"))
            and not checks["human_listening_completed"] and checks["human_listening_verdict"] is None,
            "Incomplete playback checks")
    for key in ("source_switch", "arm_switch", "stem_switch", "rapid_source_switch"):
        require(checks[key]["before"] >= 4.24 and abs(checks[key]["after"] - checks[key]["before"]) < .25,
                "Playback position differs")
    with urlopen(page, timeout=10) as response:
        html = response.read()
    require(hashlib.sha256(html).hexdigest() == saved["player_sha256"] == sha(player_path / "index.html"),
            "Served player bytes differ")
    data = json.loads(re.search(r'<script id="listening-data" type="application/json">(.*?)</script>',
                               html.decode(), re.DOTALL)[1])
    require([s["id"] for s in data["sources"]] == models + ["reference", "mixture"]
            and list(saved["models"]) == models
            and [row["id"] for row in source_plan["reuse_models"] + source_plan["models"]] == models
            and len(data["tracks"]) == 1 and data["tracks"][0]["name"] == "Skelpolu - Human Mistakes"
            and len(data["tracks"][0]["clips"]) == 2
            and [[row["reference_start"], row["reference_end"]] for row in data["tracks"][0]["clips"]]
                == [[1323000, 1984500], [3307500, 3969000]]
            and data["human_listening_verdict"] is None, "Player identities or physical interval differ")
    expected = {}
    for index, clip in enumerate(data["tracks"][0]["clips"]):
        for source in data["sources"]:
            base = clip.get("source_bases", {}).get(source["id"], source["base"])
            for stem in ([None] if source["kind"] == "mixture" else ("drums", "bass", "vocals", "other")):
                name = "mixture.wav" if stem is None else source["kind"] + "-" + stem + ".wav"
                url = urljoin(page, base + "/" + name)
                expected[url] = (index, source["id"], stem)
    require(len(expected) == len(checks["files"]) == (4 * len(models) + 5) * clips, "Incomplete playback inventory")
    inventory = {}
    for row in checks["files"]:
        url = row["url"]
        require(url in expected and expected.pop(url) == (row["clip_index"], row["source"], row["stem"])
                and row["decoded"] and row["playback_advanced"] and row["played_to_seconds"] > 3.05
                and abs(row["duration"] - 15) < 1e-6, "Unexpected decoded file or playback")
        path = (PHASE.parent / unquote(urlparse(url).path).lstrip("/")).resolve(strict=True)
        require(path.is_relative_to(PHASE.parent) and str(path) in saved["audio_files"], "Unknown served file")
        actual = saved["audio_files"][str(path)]
        require(binding.get(str(path)) == actual["sha256"] == sha(path), "Captured audio changed")
        with urlopen(url, timeout=10) as response:
            payload = response.read()
        require(hashlib.sha256(payload).hexdigest() == actual["sha256"]
                and len(payload) == path.stat().st_size, "HTTP audio bytes differ")
        with urlopen(Request(url, headers={"Range": "bytes=1000000-1004095"}), timeout=10) as response:
            require(response.status == 206 and response.headers["Content-Range"] == f"bytes 1000000-1004095/{len(payload)}"
                    and response.read() == payload[1000000:1004096], "WAV seeking bytes differ")
        inventory[str(path)] = {**actual, "browser_url": url, "http_get_sha256": actual["sha256"],
                               "played_to_seconds": row["played_to_seconds"], "byte_range_exact": True}
    require(not expected and set(inventory) == set(saved["audio_files"]), "Playback skipped captured files")
    verify_inputs(plan)
    write(Path(plan["output"]), {"schema": "latency58-vocal-focus-skelpolu-browser-inventory-v1",
        "status": "pass", "last_arm": plan["last_arm"], "plan_sha256": args.plan_sha256,
        "source_bindings": binding, "source_bindings_unchanged": True,
        "browser_result_sha256": sha(browser_path / "result.json"), "capture_result_sha256": sha(player_path / "result.json"),
        "files": inventory, "all_served_bytes_exact": True, "all_byte_ranges_exact": True,
        "human_listening_completed": False, "human_listening_verdict": None})
    print({"status": "pass", "last_arm": plan["last_arm"], "files": len(inventory)}, flush=True)


if __name__ == "__main__":
    main()
