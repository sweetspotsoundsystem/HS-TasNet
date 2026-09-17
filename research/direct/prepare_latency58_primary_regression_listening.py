"""Build a two-excerpt player over the authenticated primary regression audio.

The reused HTML expects an explicit source_bases map for each excerpt.
This preparation retains all existing audio and the first failed browser check.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Player plan or cwd differs")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-primary-regression-listening-plan-v1", "Unknown player plan")
    verify_inputs(plan)
    capture = Path(plan["capture_directory"])
    out = Path(plan["output_directory"])
    require(capture.is_relative_to(PHASE) and out.is_relative_to(PHASE)
            and out.is_dir() and not (out / "result.json").exists()
            and not (out / "index.html").exists(), "Preserve existing player")
    result, execution = read(capture / "result.json"), read(capture / "capture-execution.json")
    require(result["schema"] == "latency58-primary-regression-audio-v1" and result["status"] == "pass"
            and result["source_bindings_unchanged"] and execution["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and result["plan_sha256"] == execution["plan_sha256"] == sha(capture / "plan.json")
            and result["player_sha256"] == sha(capture / "index.html")
            and result["readme_sha256"] == sha(capture / "README.md")
            and len(result["audio_files"]) == 34
            and all(m["exact_stored_track_score"] and m["exact_stored_stream_metadata"]
                    for m in result["models"].values()), "Incomplete or changed audio capture")
    for path, row in result["audio_files"].items():
        require(plan["source_bindings"].get(path) == sha(path) == row["sha256"]
                and Path(path).stat().st_size == row["bytes"], "Audio inventory changed")
    html = (capture / "index.html").read_text()
    pattern = r'(<script id="listening-data" type="application/json">)(.*?)(</script>)'
    matches = list(re.finditer(pattern, html, flags=re.DOTALL))
    require(len(matches) == 1, "Missing or ambiguous player data")
    data = json.loads(matches[0][2])
    require([s["id"] for s in data["sources"]] == ["working", "parent", "candidate", "reference", "mixture"]
            and len(data["tracks"]) == 1 and len(data["tracks"][0]["clips"]) == 2
            and data["human_listening_verdict"] is None, "Unexpected primary comparison")
    resolved = set()
    for i, clip in enumerate(data["tracks"][0]["clips"]):
        bases = {}
        for source in data["sources"]:
            directory = capture / (source["id"] if source["kind"] == "estimate" else "shared") / f"excerpt-{i}"
            bases[source["id"]] = os.path.relpath(directory, out)
            stems = [None] if source["kind"] == "mixture" else ["drums", "bass", "vocals", "other"]
            for stem in stems:
                file = "mixture.wav" if stem is None else source["kind"] + "-" + stem + ".wav"
                path = (out / bases[source["id"]] / file).resolve(strict=True)
                row = result["audio_files"].get(str(path))
                require(path.is_relative_to(capture) and row is not None
                        and row["source_id"] == source["id"] and row["stem"] == stem
                        and row["clip_index"] == i, "Player maps to the wrong source, stem or passage")
                resolved.add(str(path))
        clip["source_bases"] = bases
    require(resolved == set(result["audio_files"]), "Player does not cover the exact audio inventory")
    html = re.sub(pattern, lambda m: m[1] + json.dumps(data, indent=2) + m[3], html, flags=re.DOTALL)
    with (out / "index.html").open("x") as stream:
        stream.write(html)
    with (out / "README.md").open("x") as stream:
        stream.write((capture / "README.md").read_text().replace(
            "result.json records their hashes and per-excerpt diagnostics.",
            f"[Capture details]({os.path.relpath(capture / 'result.json', out)}) "
            "record their hashes and per-excerpt diagnostics."))
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-primary-regression-listening-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
        "source_bindings_unchanged": True, "capture_result_sha256": sha(capture / "result.json"),
        "audio_files": result["audio_files"], "exact_player_inventory": True,
        "player_sha256": sha(out / "index.html"), "readme_sha256": sha(out / "README.md"),
        "audio_files_copied": 0, "new_inference_executed": False, "normalization": None,
        "confirmation_excerpts_used": False, "human_listening_verdict": None,
        "browser_playback_tested": False, "quality_selected": False})
    print(json.dumps({"status": "pass", "audio_files": len(resolved), "player": str(out / "index.html")}), flush=True)


if __name__ == "__main__":
    main()
