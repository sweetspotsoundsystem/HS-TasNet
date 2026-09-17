"""Capture the accumulated drum candidate with three authenticated reused models.

Every new model replays the original primary score and stream metadata before
export. Existing working audio and source stems are authenticated and reused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Capture plan or cwd differs")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-primary-regression-audio-plan-v5"
            and plan["track_index"] == 10 and plan["track_name"] == "Skelpolu - Human Mistakes"
            and [m["id"] for m in plan["models"]] == ["candidate"]
            and [m["kind"] for m in plan["models"]] == ["drum_accum_candidate"]
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the bounded CPU1 capture")
    verify_inputs(plan)
    for key in ("manifest", "evaluation_config", "template"):
        require(plan["source_bindings"].get(plan[key]) == sha(plan[key]), "Unbound capture configuration")
    out, old = Path(plan["output_directory"]), Path(plan["reuse_capture_directory"])
    require(out.parent == PHASE and old.parent == PHASE and out.is_dir()
            and all(not (out / name).exists() for name in ("result.json", "index.html", "candidate")),
            "Preserve existing output")
    for name in ("plan.json", "result.json", "capture-execution.json"):
        require(plan["source_bindings"].get(str(old / name)) == sha(old / name), "Unbound prior capture")
    saved, old_execution = read(old / "result.json"), read(old / "capture-execution.json")
    require(saved["schema"] == "latency58-primary-regression-audio-v4" and saved["status"] == "pass"
            and saved["source_bindings_unchanged"] and old_execution["actual_exit_code"] == 0
            and not old_execution["timed_out"] and old_execution["source_bindings_unchanged"]
            and old_execution["plan_sha256"] == saved["plan_sha256"] == sha(old / "plan.json"),
            "Prior capture is incomplete")
    verify_inputs(saved)
    require(set(saved["models"]) == {"working", "parent", "previous", "candidate"}
            and len(saved["audio_files"]) == 42
            and plan["omitted_capture_model_id"] == "previous"
            and all(plan["source_bindings"].get(p) == h for p, h in saved["source_bindings"].items()),
            "Prior capture inventory or provenance differs")
    from research.direct.latency58_sdr_checkpoint import require_space
    space_plan = read(plan["storage_plan"]["path"])
    require(sha(plan["storage_plan"]["path"]) == plan["storage_plan"]["sha256"], "Storage plan changed")
    require(plan["new_audio_reserve_bytes"] == 45_000_000
            and plan["concurrent_checkpoint_reserve_bytes"] == 350_000_000, "Reserve the active training save")
    require_space(space_plan, 395_000_000)
    import numpy as np
    import soundfile as sf
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_evaluate import (
        HOP, SOURCE_ORDER, model_state_sha256, plan_latency58_stream, stream_latency58_track,
    )
    from research.metrics import MetricConfig

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    require(torch.__version__ == plan["torch_version"] and not torch.cuda.is_initialized(), "CPU runtime differs")
    began = time.monotonic()
    manifest, source_config = read(plan["manifest"]), read(plan["evaluation_config"])
    tracks, config = shared.select_panel(manifest, source_config, panel="full", track_indices=[10],
                                        excerpt_starts=None, duration=15.0, alignment_samples=HOP)
    require(len(tracks) == 1 and tracks[0]["name"] == plan["track_name"], "Primary track differs")
    track = tracks[0]
    rows = legacy._reference_intervals(track, config)
    require([(r["reference_start"], r["reference_end"]) for r in rows]
            == [(1323000, 1984500), (3307500, 3969000)], "Use both original primary intervals")
    source_root = Path(manifest["root"])
    stream_plan = plan_latency58_stream(rows, int(track["frames"]))
    metric_config = MetricConfig.from_mapping(config["metrics"])
    evidence, inventory, reused, new_files = {}, {}, [], []

    def excerpt(relative, row):
        path = legacy._safe_dataset_path(source_root, relative)
        require(plan["source_bindings"].get(str(path)) == sha(path), "Unbound source audio")
        return legacy._read_excerpt(path, int(row["reference_start"]), int(row["reference_end"]),
                                    expected_frames=int(track["frames"]))

    mixtures = [excerpt(track["mixture"], row) for row in rows]
    references = [np.stack([excerpt(track["stems"][stem], row) for stem in SOURCE_ORDER]) for row in rows]
    require([m["id"] for m in plan["reuse_models"]] == ["working", "parent", "previous"]
            and [m["capture_id"] for m in plan["reuse_models"]] == ["working", "parent", "candidate"],
            "Reused comparison identities differ")
    models, sources, replay = {}, [], {}
    for item in plan["reuse_models"]:
        _, stored = load_completed(PHASE / (item["prefix"] + "-full14-001"), evidence,
                                   canonical_baseline=item["id"] == "working")
        prior = saved["models"][item["capture_id"]]
        require(prior["model_state_sha256"] == item["model_state_sha256"]
                == stored["results"][0]["model"]["model_state_sha256"]
                and prior["checkpoint"]["sha256"] == item["checkpoint_sha256"]
                and prior["exact_stored_track_score"] and prior["exact_stored_stream_metadata"]
                and prior["model_and_rng_unchanged"]
                and prior["track_score"] == stored["results"][0]["tracks"][10]
                and prior["stream_metadata"] == stored["results"][0]["stream_batches"][10],
                "Reused model is not the exact primary capture")
        models[item["id"]] = {**prior, "reused_from_capture_result": str(old / "result.json"),
                              "reused_capture_model_id": item["capture_id"]}
        sources.append({"id": item["id"], "label": item["label"], "kind": "estimate", "base": "",
                        "checkpoint_sha256": item["checkpoint_sha256"], "algorithmic_latency_samples": 256,
                        "sample_rate": 44100, "status": "Working baseline" if item["id"] == "working"
                        else "Development comparison"})
        replay[item["id"]] = {}
    require(models["working"]["model_state_sha256"] == plan["working_model_state_sha256"],
            "Working model identity differs")
    mapping = {m["capture_id"]: m["id"] for m in plan["reuse_models"]}
    omitted = []
    for path, old_row in saved["audio_files"].items():
        row = {**old_row, "source_id": mapping.get(old_row["source_id"], old_row["source_id"])}
        file = Path(path)
        require(file.is_relative_to(PHASE) and plan["source_bindings"].get(path) == sha(file) == row["sha256"]
                and file.stat().st_size == row["bytes"], "Reused WAV differs")
        values, rate = sf.read(file, dtype="float32", always_2d=True)
        require(rate == 44100 and values.shape == (661500, 2) and np.isfinite(values).all()
                and sf.info(file).subtype == "FLOAT", "Reused WAV geometry differs")
        index = row["clip_index"]
        require(index in (0, 1), "Unexpected reused passage")
        if old_row["source_id"] == plan["omitted_capture_model_id"]:
            require(old_row["stem"] in SOURCE_ORDER, "Unexpected omitted estimate stem")
            omitted.append(path)
            continue
        if row["source_id"] in replay:
            require(row["stem"] in SOURCE_ORDER and (index, row["stem"]) not in replay[row["source_id"]],
                    "Unexpected or duplicate estimate stem")
            replay[row["source_id"]][index, row["stem"]] = values.T
        else:
            require(row["source_id"] in ("mixture", "reference"), "Unknown audio source")
            expected = mixtures[index] if row["source_id"] == "mixture" else references[index][SOURCE_ORDER.index(row["stem"])]
            require(np.array_equal(values.T, expected.astype(np.float32)), "Reused source samples differ")
        inventory[path] = row
        reused.append(path)
    require(len(reused) == 34 and len(omitted) == 8
            and {(saved["audio_files"][p]["clip_index"], saved["audio_files"][p]["stem"]) for p in omitted}
            == {(i, stem) for i in (0, 1) for stem in SOURCE_ORDER}
            and all(len(v) == 8 for v in replay.values()), "Incomplete reused inventory")
    for source_id, audio in replay.items():
        # Restore the original prediction memory order before exact scoring.
        # Interleaved WAV decoding changes strides, not samples; reduction order
        # otherwise changes the last bits of a few stored metric values.
        estimates = [np.ascontiguousarray(np.stack([audio[i, stem] for stem in SOURCE_ORDER])) for i in range(2)]
        require(all(float(np.abs(values.sum(axis=0, dtype=np.float32) - mixtures[i].astype(np.float32)).max()) <= 1e-6
                    for i, values in enumerate(estimates)), "Reused stems do not reconstruct the mixture")
        require(legacy._score_track(track["name"], rows, mixtures, references, estimates, metric_config)
                == models[source_id]["track_score"], "Decoded reused WAV track score differs")
        per_excerpt = [legacy._score_track(track["name"], [r], [m], [ref], [est], metric_config)
                       for r, m, ref, est in zip(rows, mixtures, references, estimates, strict=True)]
        require(per_excerpt == models[source_id]["per_excerpt_scores"], "Reused per-excerpt score differs")
        models[source_id]["decoded_audio_track_and_excerpt_scores_exact"] = True
    del replay, audio, estimates

    def export_audio(path, values, source_id, stem, clip_index):
        values = values.astype(np.float32)
        require(values.shape == (2, 661500) and np.isfinite(values).all() and not path.exists(), "Malformed new WAV")
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, values.T, 44100, subtype="FLOAT")
        decoded, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 44100 and np.array_equal(decoded.T, values), "WAV changed native samples")
        inventory[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size, "frames": 661500,
                               "sample_rate": 44100, "peak_abs": float(np.abs(values).max()),
                               "source_id": source_id, "stem": stem, "clip_index": clip_index}
        new_files.append(str(path))

    with (out / "progress.jsonl").open("x", buffering=1) as progress:
        for item in plan["models"]:
            directory = PHASE / (item["prefix"] + "-full14-001")
            full_plan, stored = load_completed(directory, evidence)
            require(stored["track_names"][10] == track["name"] and stored["torch_version"] == torch.__version__,
                    "Stored scope differs")
            from research.direct.evaluate_latency58_sdr_drum_accum import load_evaluation_model
            model, _ = load_evaluation_model(full_plan)
            fingerprint, checkpoint = model_state_sha256(model), full_plan["checkpoint"]
            require(fingerprint == item["model_state_sha256"] == stored["results"][0]["model"]["model_state_sha256"]
                    and checkpoint["sha256"] == item["checkpoint_sha256"], "Different capture weights")
            rng = torch.get_rng_state().clone()
            raw, delayed, metadata = stream_latency58_track(
                model, legacy._safe_dataset_path(source_root, track["mixture"]), stream_plan)
            require(all(np.array_equal(a, b.astype(np.float32)) for a, b in zip(delayed, mixtures, strict=True)),
                    "Physical mixture differs")
            estimates = [shared.shipping_residual(a, b) for a, b in zip(raw, mixtures, strict=True)]
            score = legacy._score_track(track["name"], rows, mixtures, references, estimates, metric_config)
            metadata.update(track=track["name"], physical_alignment_verified_by_delayed_mixture=True)
            require(score == stored["results"][0]["tracks"][10]
                    and metadata == stored["results"][0]["stream_batches"][10], "Original primary replay differs")
            closure = max(float(np.abs(e.sum(axis=0, dtype=np.float32) - m.astype(np.float32)).max())
                          for e, m in zip(estimates, mixtures, strict=True))
            require(closure <= 1e-6 and model_state_sha256(model) == fingerprint
                    and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
                    "Model, RNG, reconstruction or CPU scope changed")
            per_excerpt = []
            for i, (row, mixture, reference, estimate) in enumerate(zip(rows, mixtures, references, estimates, strict=True)):
                per_excerpt.append(legacy._score_track(track["name"], [row], [mixture], [reference], [estimate], metric_config))
                for stem, values in zip(SOURCE_ORDER, estimate, strict=True):
                    export_audio(out / item["id"] / f"excerpt-{i}" / ("estimate-" + stem + ".wav"),
                                 values, item["id"], stem, i)
            models[item["id"]] = {"model_state_sha256": fingerprint, "checkpoint": checkpoint,
                "stored_full14_result": {"path": str(directory / "result.json"), "sha256": sha(directory / "result.json")},
                "exact_stored_track_score": True, "exact_stored_stream_metadata": True,
                "track_score": score, "per_excerpt_scores": per_excerpt, "stream_metadata": metadata,
                "closure_max_abs": closure, "model_and_rng_unchanged": True}
            sources.append({"id": item["id"], "label": item["label"], "kind": "estimate", "base": "",
                            "checkpoint_sha256": checkpoint["sha256"], "algorithmic_latency_samples": 256,
                            "sample_rate": 44100, "status": "Development comparison"})
            event = {"event": "captured", "model": item["id"], "exact_primary_replay": True,
                     "elapsed_seconds": time.monotonic() - began}
            progress.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            del model, raw, delayed, estimates
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound completed evidence")
    require(len(inventory) == 42 and len(new_files) == 8
            and sum(inventory[p]["bytes"] for p in new_files) < 45_000_000, "Wrong audio inventory or allowance")
    sources += [{"id": "reference", "label": "Original stem", "kind": "reference", "base": ""},
                {"id": "mixture", "label": "Mixture", "kind": "mixture", "base": ""}]
    clips, resolved = [], set()
    for i, row in enumerate(rows):
        bases = {}
        for source in sources:
            selected = {p: v for p, v in inventory.items() if v["source_id"] == source["id"] and v["clip_index"] == i}
            directories = {Path(p).parent for p in selected}
            require(len(selected) == (1 if source["kind"] == "mixture" else 4) and len(directories) == 1,
                    "Ambiguous source, stem or passage")
            bases[source["id"]] = Path(os.path.relpath(next(iter(directories)), out)).as_posix()
            for stem in ([None] if source["kind"] == "mixture" else SOURCE_ORDER):
                name = "mixture.wav" if stem is None else source["kind"] + "-" + stem + ".wav"
                path = str((out / bases[source["id"]] / name).resolve(strict=True))
                require(path in selected and selected[path]["stem"] == stem, "Player maps to the wrong WAV")
                resolved.add(path)
        clips.append({"label": "30–45 seconds" if i == 0 else "75–90 seconds", "folder": "",
                      "reference_start": row["reference_start"], "reference_end": row["reference_end"],
                      "source_bases": bases})
    require(resolved == set(inventory), "Player does not cover the audio inventory")
    data = {"schema_version": 1, "default_source": "working", "default_stem": "other", "sources": sources,
            "tracks": [{"name": track["name"], "folder": "", "clips": clips}], "human_listening_verdict": None}
    html = Path(plan["template"]).read_text()
    html = html.replace("Actions · training comparison", "Skelpolu · regression comparison")
    html = html.replace(
        "Compare the C91 reference, native C191 and two completed training trials on Actions – One Minute Smile, 60–75 seconds. No new candidate has been retained.",
        "Compare the working model, training parent, previous checkpoint and current candidate on the two primary excerpts from Skelpolu – Human Mistakes.")
    html = html.replace('<a href="../index.html">More listening excerpts</a> · ', "")
    html = html.replace(
        "Latency values describe model buffering. No human listening verdict has been recorded for these comparisons.",
        "All clips retain native levels. This track was selected to inspect a measured regression. No listening verdict has been recorded.")
    html, count = re.subn(r'(<script id="listening-data" type="application/json">).*?(</script>)',
                         lambda m: m[1] + json.dumps(data, indent=2) + m[2], html, flags=re.DOTALL)
    require(count == 1 and "C191" not in html, "Player replacement failed")
    with (out / "index.html").open("x") as stream:
        stream.write(html)
    with (out / "README.md").open("x") as stream:
        stream.write("# Skelpolu primary-excerpt comparison\n\n[Open the player](index.html). "
                     "Both passages retain the original 30–45 and 75–90 second primary intervals. "
                     "The working 5.8 ms baseline and original stems are reused at exact native levels. "
                     "Only the current candidate needs a new capture; the other models reuse authenticated audio.\n\n"
                     "Each new model streams from sample zero and reproduces its stored complete track score "
                     "and stream metadata exactly before WAV export. All 42 files are stereo FLOAT at 44.1 kHz; "
                     "34 are reused and 8 are new. [Capture details](result.json) record hashes and per-excerpt diagnostics.\n\n"
                     "This is an observed primary-panel regression, not independent confirmation. "
                     "No human verdict, model selection or deployment qualification is recorded.\n")
    verify_inputs(plan)
    require_space(space_plan, 0)
    write(out / "result.json", {"schema": "latency58-primary-regression-audio-v5", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "track": track["name"], "primary_intervals": rows, "models": models, "audio_files": inventory,
        "reused_capture_result_sha256": sha(old / "result.json"), "reused_audio_files": reused,
        "omitted_prior_capture_files_retained": omitted,
        "new_audio_files": new_files, "audio_bytes": sum(v["bytes"] for v in inventory.values()),
        "new_audio_bytes": sum(inventory[p]["bytes"] for p in new_files),
        "player_sha256": sha(out / "index.html"), "readme_sha256": sha(out / "README.md"),
        "exact_player_inventory": True, "normalization": None, "source_audio_changed": False,
        "training_updates_executed": 0, "checkpoint_written": False, "cuda_initialized": False,
        "quality_selected": False, "confirmation_excerpts_used": False, "human_listening_verdict": None,
        "browser_playback_tested": False, "elapsed_seconds": time.monotonic() - began,
        "limitations": ["Track chosen after its primary regression was observed; no independent validation.",
                        "Per-excerpt diagnostics do not replace the unchanged full-panel aggregation.",
                        "Audio capture establishes no human listening verdict or runtime qualification."]})
    print({"status": "pass", "audio_files": len(inventory), "new_audio_files": len(new_files)}, flush=True)


if __name__ == "__main__":
    main()
