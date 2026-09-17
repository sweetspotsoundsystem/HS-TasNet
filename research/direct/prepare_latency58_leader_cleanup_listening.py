"""Reuse authenticated Actions WAVs in a vocal-pilot comparison player."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs

ORDER = ("working", "leader", "working_cleanup", "leader_cleanup")


def validate_summary(row, quality, fingerprint, bindings):
    summary, execution = read(row["summary"]["path"]), read(row["summary_execution"]["path"])
    for item in (row["summary"], row["summary_execution"], row["summary_plan"], row["training_plan"]):
        require(bindings.get(item["path"]) == item["sha256"] == sha(item["path"]), "Unbound model review input")
    require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and execution["plan_sha256"] == row["summary_plan"]["sha256"]
            and summary["model_state_sha256"] == fingerprint
            and quality["training_plan"] == row["training_plan"], "Model lacks its completed quality summary")
    if row["id"] == "leader":
        summary_plan = read(row["summary_plan"]["path"])
        command = execution["argv"]
        require(summary["schema"] == "latency58-sdr-drum-accum-quality-summary-v1"
                and summary["step"] == quality["step"] == 500 and summary["prefix"] == row["prefix"]
                and summary_plan["schema"] == "latency58-sdr-drum-accum-summary-execution-v1"
                and summary_plan["prefix"] == row["prefix"] and not summary["quality_selected"]
                and command[command.index("-m") + 1] == "research.direct.report_latency58_sdr_drum_accum"
                and command[command.index("--prefix") + 1] == row["prefix"]
                and command[command.index("--output") + 1] == row["summary"]["path"],
                "Different retained SDR-leader summary")
    else:
        expected = "latency58-controlled-deployed-quality-summary-v1" if row["id"] == "working_cleanup" \
            else "latency58-leader-cleanup-quality-summary-v1"
        require(summary["schema"] == expected and summary["status"] == "pass"
                and summary["source_bindings_unchanged"] and summary["arm"] == "focused"
                and summary["step"] == quality["step"] == 250 and summary["training_plan"] == quality["training_plan"]
                and summary["plan_sha256"] == row["summary_plan"]["sha256"], "Different completed cleanup summary")
    verify_inputs(summary)
    require(all(bindings.get(p) == s for p, s in summary["source_bindings"].items()), "Unbound completed summary inputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256 and Path.cwd() == ROOT, "Listening plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-leader-cleanup-listening-plan-v1"
            and plan["step"] == 250 and len(plan["models"]) == 4
            and [row["id"] for row in plan["models"]] == list(ORDER)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the completed pilot prefix and CUDA-hidden CPU1")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists()
            and not (out / "index.html").exists(), "Preserve listening output")
    from research.direct.latency58_sdr_checkpoint import require_space
    require_space(plan, 2_000_000)
    import numpy as np
    import soundfile as sf
    import torch
    from research import evaluate as legacy
    from research.metrics import MetricConfig
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    stems = ("drums", "bass", "vocals", "other")
    shared = Path(plan["shared_reference_directory"])
    source = Path(plan["source_track_directory"])
    bindings, evidence, files = plan["source_bindings"], {}, {}
    metric_config = MetricConfig.from_mapping(read(plan["evaluation_config"])["metrics"])
    require(bindings.get(plan["evaluation_config"]) == sha(plan["evaluation_config"]), "Unbound original metric configuration")

    def audio(path):
        require(bindings.get(str(path)) == sha(path), "Unbound native audio")
        info = sf.info(path)
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 44100 and values.shape == (661500, 2) and np.isfinite(values).all()
                and info.subtype == "FLOAT", "Native audio geometry, precision or finiteness differs")
        files[str(path)] = {"sha256": sha(path), "frames": len(values), "sample_rate": rate,
                            "channels": 2, "subtype": info.subtype, "peak_abs": float(np.abs(values).max())}
        return np.ascontiguousarray(values.T)

    originals = {}
    for stem in ("mixture", *stems):
        path = source / (stem + ".wav")
        require(bindings.get(str(path)) == sha(path), "Unbound original source")
        values, rate = sf.read(path, start=2646000, stop=3307500, dtype="float64", always_2d=True)
        require(rate == 44100 and values.shape == (661500, 2), "Original physical interval differs")
        originals[stem] = np.ascontiguousarray(values.T)
        stored = audio(shared / ("mixture.wav" if stem == "mixture" else "reference-" + stem + ".wav"))
        require(np.array_equal(stored, values.T.astype(np.float32)), "Shared reference is not the exact source interval")
    sources, models = [], {}
    for row in plan["models"]:
        directory = PHASE / (row["prefix"] + "-actions60-001")
        quality, report = load_completed(directory, evidence, canonical_baseline=row["id"] == "working")
        require(report["inputs_unchanged"] and report["track_names"] == ["Actions - One Minute Smile"], "Different audition track")
        current = report["results"][0]
        fingerprint = current["model"]["model_state_sha256"]
        require(fingerprint == row["model_state_sha256"] and len(current["tracks"]) == 1, "Audition model differs")
        intervals = current["tracks"][0]["excerpts"]
        require(len(intervals) == 1 and intervals[0]["reference_start"] == 2646000
                and intervals[0]["reference_end"] == 3307500
                and intervals[0]["estimate_start"] == 2646128 and intervals[0]["estimate_end"] == 3307628,
                "Audition physical alignment differs")
        if row["id"] in ORDER[1:]:
            validate_summary(row, quality, fingerprint, bindings)
        checkpoint_sha = read(PHASE / "teacher-half-canonical-001/receipt.json")["output"]["sha256"] \
            if row["id"] == "working" else current["model"]["checkpoint"]["sha256"]
        require(checkpoint_sha == row["checkpoint_sha256"], "Player checkpoint identity differs")
        estimates = np.stack([audio(directory / "audio" / ("estimate-" + stem + ".wav")) for stem in stems])
        score = legacy._score_track("Actions - One Minute Smile", intervals, [originals["mixture"]],
                                    [np.stack([originals[stem] for stem in stems])], [estimates], metric_config)
        require(score == current["tracks"][0] and legacy._aggregate_tracks([score]) == current["aggregate"],
                "Decoded native WAVs do not reproduce stored Actions metrics exactly")
        closure = float(np.max(np.abs(estimates.sum(axis=0, dtype=np.float32) - originals["mixture"].astype(np.float32))))
        require(closure <= 1e-6, "Native stems do not reconstruct their physical mixture")
        models[row["id"]] = {"model_state_sha256": fingerprint, "aggregate": current["aggregate"],
                              "decoded_track_and_aggregate_exact": True, "mixture_closure_max_abs": closure}
        sources.append({"id": row["id"], "label": row["label"], "kind": "estimate",
                        "base": Path(os.path.relpath(directory / "audio", out)).as_posix(),
                        "algorithmic_latency_samples": 256, "sample_rate": 44100,
                        "status": "Working baseline" if row["id"] == "working" else "Development comparison",
                        "checkpoint_sha256": checkpoint_sha})
    require(all(bindings.get(p) == s for p, s in evidence.items()), "Unbound completed render evidence")
    from research.direct.audit_latency58_leader_cleanup_training_match import load_completed_match
    match = load_completed_match(plan["training_match"], evidence)
    require(match["model_states"] == {"reference": models["working_cleanup"]["model_state_sha256"],
                                       "candidate": models["leader_cleanup"]["model_state_sha256"]}
            and all(bindings.get(p) == s for p, s in evidence.items()), "Player pair does not match the authenticated training comparison")
    candidate_plan = read(next(row["training_plan"]["path"] for row in plan["models"] if row["id"] == "leader_cleanup"))
    require(candidate_plan["parent"]["model_state_sha256"] == models["leader"]["model_state_sha256"],
            "The audible SDR leader is not the candidate's actual trained parent")
    require(len(files) == 4 * len(models) + 5, "Incomplete listening file inventory")
    base = Path(os.path.relpath(shared, out)).as_posix()
    sources += [{"id": "reference", "label": "Original stem", "kind": "reference", "base": base},
                {"id": "mixture", "label": "Mixture", "kind": "mixture", "base": base}]
    data = {"schema_version": 1, "default_source": "working", "default_stem": "vocals", "sources": sources,
            "tracks": [{"name": "Actions - One Minute Smile", "folder": "", "clips": [{
                "label": "60–75 seconds", "folder": "", "reference_start": 2646000, "reference_end": 3307500}]}],
            "human_listening_verdict": None}
    template = Path(plan["template"])
    require(bindings.get(str(template)) == sha(template), "Player template changed")
    html = template.read_text()
    replacements = {
        "Actions · training comparison": "SDR leader and vocal cleanup · Actions",
        "Compare the C91 reference, native C191 and two completed training trials on Actions – One Minute Smile, 60–75 seconds. No new candidate has been retained.":
            "Compare the working model, SDR leader and cleanup from each model on Actions – One Minute Smile, 60–75 seconds. Listen for vocals in Other, unwanted sound in Vocals and changes to instruments.",
        '<a href="../index.html">More listening excerpts</a> · ': "",
        "Keys 1–6 select the sources in order.": "Keys 1–" + str(len(sources)) + " select the sources in order.",
        "Latency values describe model buffering. No human listening verdict has been recorded for these comparisons.":
            "All models use 256 samples of intended buffering. New pilot models require runtime qualification. No human listening verdict has been recorded.",
        '<option value="bass" selected>': '<option value="bass">',
        '<option value="vocals">': '<option value="vocals" selected>',
    }
    for old, new in replacements.items():
        require(old in html, "Player template text differs")
        html = html.replace(old, new)
    html, count = re.subn(r'(<script id="listening-data" type="application/json">).*?(</script>)',
                          lambda m: m[1] + json.dumps(data, indent=2) + m[2], html, flags=re.DOTALL)
    require(count == 1 and "C191" not in html and not torch.cuda.is_initialized(), "Player replacement or CPU scope differs")
    verify_inputs(plan)
    with (out / "index.html").open("x") as stream:
        stream.write(html)
    write(out / "result.json", {"schema": "latency58-leader-cleanup-listening-preparation-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": bindings, "source_bindings_unchanged": True,
        "audio_files": files, "models": models, "model_order": [row["id"] for row in plan["models"]],
        "player_sha256": sha(out / "index.html"), "shared_reference_samples_exact": True,
        "all_decoded_track_and_aggregate_scores_exact": True, "decoded_arrays_contiguous": True, "audio_files_copied": 0, "new_inference": False,
        "normalization": None, "human_listening_completed": False, "human_listening_verdict": None,
        "browser_playback_tested": False, "quality_selected": False})
    with (out / "README.md").open("x") as stream:
        stream.write("# Vocal cleanup comparison\n\n[Open the player](index.html). "
            "Every clip covers Actions 60–75 seconds after continuous streaming from the track origin. "
            "Native levels are unchanged; no WAV is copied. Source and stem switches preserve playback position.\n\n"
            "All decoded files reproduce their stored scores exactly. This one excerpt is a diagnostic comparison; "
            "the complete primary panel and controlled vocal views remain part of review. No human listening "
            "verdict or deployment selection has been recorded.\n\n[Preparation evidence](result.json) · [Frozen inputs](plan.json).\n")
    print({"status": "pass", "models": list(models), "files": len(files), "decoded_scores_exact": True}, flush=True)


if __name__ == "__main__":
    main()
