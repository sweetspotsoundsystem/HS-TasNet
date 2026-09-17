"""Build an SDR comparison player from authenticated existing Actions audio.

Reuse the established player controls and all WAVs. No model inference,
normalization, source-audio export or listening judgment occurs here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Listening plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-sdr-listening-plan-v1"
            and plan["step"] in (250, 500, 1000) and Path.cwd() == ROOT,
            "Unsupported listening plan")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CUDA-hidden CPU1")
    bindings = plan["source_bindings"]
    require(all(sha(p) == s for p, s in bindings.items()), "Listening inputs changed")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "index.html").exists()
            and not (out / "result.json").exists(), "Preserve existing player")
    ids = [x["id"] for x in plan["models"]]
    require(ids in (["working", "parent", "candidate"], ["working", "parent", "previous", "candidate"]),
            "Require the working model, training parent, optional prior endpoint and candidate")
    import numpy as np
    import soundfile as sf

    stems = ("drums", "bass", "vocals", "other")
    shared = Path(plan["shared_reference_directory"])
    source = Path(plan["source_track_directory"])
    files, evidence = {}, {}

    def audio(path):
        require(bindings.get(str(path)) == sha(path), "Unbound listening audio: " + str(path))
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 44100 and values.shape == (661500, 2) and np.isfinite(values).all(),
                "Listening audio differs in physical geometry or finiteness")
        files[str(path)] = {"sha256": sha(path), "frames": len(values), "sample_rate": rate,
                            "peak_abs": float(np.abs(values).max())}
        return values

    for stem in ("mixture", *stems):
        original = source / (stem + ".wav")
        require(bindings.get(str(original)) == sha(original), "Unbound original audio")
        reference, rate = sf.read(original, start=2646000, stop=3307500,
                                  dtype="float32", always_2d=True)
        path = shared / ("mixture.wav" if stem == "mixture" else "reference-" + stem + ".wav")
        reused = audio(path)
        require(rate == 44100 and np.array_equal(reference, reused), "Shared audio is not the exact source interval")
    mixture = audio(shared / "mixture.wav")
    sources, scores = [], {}
    for row in plan["models"]:
        directory = PHASE / (row["prefix"] + "-actions60-001")
        evaluation_plan, report = load_completed(directory, evidence, canonical_baseline=row["id"] == "working")
        if row["id"] == "candidate":
            require(evaluation_plan["step"] == plan["step"], "Candidate listening endpoint differs")
        require(report["inputs_unchanged"] and report["track_names"] == ["Actions - One Minute Smile"],
                "Listening capture is not the completed Actions render")
        model = report["results"][0]["model"]
        require(model["model_state_sha256"] == row["model_state_sha256"], "Listening model differs")
        expected_checkpoint = read(PHASE / "teacher-half-canonical-001/receipt.json")["output"]["sha256"] \
            if row["id"] == "working" else model["checkpoint"]["sha256"]
        require(row["checkpoint_sha256"] == expected_checkpoint, "Player checkpoint identity differs")
        estimates = np.stack([audio(directory / "audio" / ("estimate-" + s + ".wav")) for s in stems])
        closure = float(np.max(np.abs(estimates.sum(axis=0, dtype=np.float32) - mixture)))
        require(closure <= 1e-6, "Native stems do not reconstruct the aligned mixture")
        scores[row["id"]] = {"model_state_sha256": row["model_state_sha256"],
                              "aggregate": report["results"][0]["aggregate"],
                              "mixture_closure_max_abs": closure}
        sources.append({"id": row["id"], "label": row["label"], "kind": "estimate",
                        "base": Path(os.path.relpath(directory / "audio", out)).as_posix(),
                        "algorithmic_latency_samples": 256, "sample_rate": 44100,
                        "status": "Working baseline" if row["id"] == "working" else "Development comparison",
                        "checkpoint_sha256": row["checkpoint_sha256"]})
    require(all(bindings.get(p) == s for p, s in evidence.items()), "Missing completed-render evidence")
    require(all(sha(p) == s for p, s in bindings.items()), "Listening input changed during validation")
    base = Path(os.path.relpath(shared, out)).as_posix()
    sources += [{"id": "reference", "label": "Original stem", "kind": "reference", "base": base},
                {"id": "mixture", "label": "Mixture", "kind": "mixture", "base": base}]
    data = {"schema_version": 1, "default_source": "working", "default_stem": "bass",
            "sources": sources, "tracks": [{"name": "Actions - One Minute Smile", "folder": "",
            "clips": [{"label": "60–75 seconds", "folder": "", "reference_start": 2646000,
                       "reference_end": 3307500}]}], "human_listening_verdict": None}
    template = Path(plan["template"])
    require(bindings.get(str(template)) == sha(template), "Player template changed")
    html = template.read_text()
    html = html.replace("Actions · training comparison", "5.8 ms · SDR comparison")
    html = html.replace(
        "Compare the C91 reference, native C191 and two completed training trials on Actions – One Minute Smile, 60–75 seconds. No new candidate has been retained.",
        "Compare the working model and separation candidates on Actions – One Minute Smile, 60–75 seconds.")
    html = html.replace('<a href="../index.html">More listening excerpts</a> · ', "")
    html = html.replace(
        "Latency values describe model buffering. No human listening verdict has been recorded for these comparisons.",
        "All estimates retain their native output levels. Candidate models are development comparisons. No listening verdict has been recorded.")
    html, count = re.subn(r'(<script id="listening-data" type="application/json">).*?(</script>)',
                          lambda match: match[1] + json.dumps(data, indent=2) + match[2],
                          html, flags=re.DOTALL)
    require(count == 1 and "C191" not in html, "Player template replacement failed")
    with (out / "index.html").open("x") as stream:
        stream.write(html)
    write(out / "result.json", {
        "schema": "latency58-sdr-listening-preparation-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
        "audio_files": files, "models": scores, "player_sha256": sha(out / "index.html"),
        "shared_reference_samples_exact": True, "normalization": None, "audio_files_copied": 0,
        "new_inference": False, "model_selected": False, "human_listening_verdict": None,
        "browser_playback_tested": False,
    })
    with (out / "README.md").open("x") as stream:
        stream.write("# SDR comparison at " + str(plan["step"]) + " updates\n\n")
        stream.write("[Open the player](index.html). All clips cover the same physical 60–75 seconds of Actions – One Minute Smile. "
                     "Each model was streamed continuously from the song’s start. Source and stem changes preserve playback position; "
                     "native audio levels are unchanged.\n\n")
        stream.write("The working 5.8 ms model, training parent, candidate and any prior checkpoint are labelled. "
                     "The original stems and mixture are shared exact source excerpts. "
                     "All WAVs are reused in place; no model or source audio is copied.\n\n")
        stream.write("This player supplies a comparison, with no human listening preference or deployment selection recorded. "
                     "The full 14-track panel governs the numerical comparison. The new models retain their runtime qualification requirements.\n\n")
        stream.write("[Preparation evidence](result.json) · [Frozen inputs](plan.json).\n")
    print(json.dumps({"status": "pass", "player": str(out / "index.html"), "reused_audio_files": len(files),
                      "new_audio_files": 0, "human_listening_verdict": None}), flush=True)


if __name__ == "__main__":
    main()
