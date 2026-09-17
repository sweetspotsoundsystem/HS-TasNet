"""Replay one primary track exactly, then retain its two listening excerpts.

The working model, context parent and accumulated-gradient candidate each
stream once from sample zero. No model, output gain or score rule changes.
The original two-excerpt track report must match before any estimate export.
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
    require(plan["schema"] == "latency58-primary-regression-audio-plan-v1"
            and plan["track_index"] == 10 and plan["track_name"] == "Skelpolu - Human Mistakes"
            and [row["id"] for row in plan["models"]] == ["working", "parent", "candidate"]
            and [row["kind"] for row in plan["models"]] == ["working_baseline", "context_candidate", "accum_candidate"]
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the bounded CPU1 primary-regression capture")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_relative_to(PHASE) and out.is_dir() and not (out / "result.json").exists()
            and not (out / "shared").exists() and not (out / "index.html").exists(), "Preserve previous audio")
    from research.direct.latency58_sdr_checkpoint import require_space
    space_plan = read(plan["storage_plan"]["path"])
    require(sha(plan["storage_plan"]["path"]) == plan["storage_plan"]["sha256"], "Storage plan changed")
    require_space(space_plan, 200_000_000)
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
            == [(1323000, 1984500), (3307500, 3969000)], "Use both original primary intervals only")
    source_root = Path(manifest["root"])
    stream_plan = plan_latency58_stream(rows, int(track["frames"]))
    metric_config = MetricConfig.from_mapping(config["metrics"])
    evidence, inventory, models, sources = {}, {}, {}, []

    def excerpt(relative, row):
        path = legacy._safe_dataset_path(source_root, relative)
        require(plan["source_bindings"].get(str(path)) == sha(path), "Unbound source audio")
        return legacy._read_excerpt(path, int(row["reference_start"]), int(row["reference_end"]),
                                    expected_frames=int(track["frames"]))

    mixtures = [excerpt(track["mixture"], row) for row in rows]
    references = [np.stack([excerpt(track["stems"][stem], row) for stem in SOURCE_ORDER]) for row in rows]

    def export_audio(path, values, *, source_id, stem, clip_index):
        values = values.astype(np.float32)
        require(values.shape == (2, 661500) and bool(np.isfinite(values).all()) and not path.exists(),
                "Malformed or existing audition audio")
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), values.T, 44100, subtype="FLOAT")
        decoded, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 44100 and np.array_equal(decoded.T, values), "WAV changed exact native samples")
        inventory[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size, "frames": 661500,
                                "sample_rate": 44100, "peak_abs": float(np.abs(values).max()),
                                "source_id": source_id, "stem": stem, "clip_index": clip_index}

    with (out / "progress.jsonl").open("x", buffering=1) as progress:
        for model_row in plan["models"]:
            full_directory = PHASE / (model_row["prefix"] + "-full14-001")
            full_plan, stored = load_completed(full_directory, evidence,
                                               canonical_baseline=model_row["id"] == "working")
            require(stored["track_names"][10] == track["name"]
                    and stored["torch_version"] == torch.__version__, "Stored primary scope differs")
            if model_row["kind"] == "working_baseline":
                from research.direct.latency58_sdr_teacher import load_initial_student, STUDENT_SHA256
                model = load_initial_student().eval().requires_grad_(False)
                checkpoint = {"path": str(PHASE / "teacher-half-canonical-001/model.pt"), "sha256": STUDENT_SHA256}
            elif model_row["kind"] == "context_candidate":
                from research.direct.evaluate_latency58_context import load_evaluation_model
                model, _ = load_evaluation_model(full_plan)
                checkpoint = full_plan["checkpoint"]
            else:
                from research.direct.evaluate_latency58_sdr_accum import load_evaluation_model
                model, _ = load_evaluation_model(full_plan)
                checkpoint = full_plan["checkpoint"]
            fingerprint = model_state_sha256(model)
            require(fingerprint == model_row["model_state_sha256"]
                    == stored["results"][0]["model"]["model_state_sha256"]
                    and checkpoint["sha256"] == model_row["checkpoint_sha256"], "Different listening weights")
            rng = torch.get_rng_state().clone()
            raw, delayed, metadata = stream_latency58_track(
                model, legacy._safe_dataset_path(source_root, track["mixture"]), stream_plan)
            require(all(np.array_equal(a, b.astype(np.float32)) for a, b in zip(delayed, mixtures, strict=True)),
                    "Captured physical mixture differs")
            estimates = [shared.shipping_residual(a, b) for a, b in zip(raw, mixtures, strict=True)]
            score = legacy._score_track(track["name"], rows, mixtures, references, estimates, metric_config)
            metadata.update(track=track["name"], physical_alignment_verified_by_delayed_mixture=True)
            require(score == stored["results"][0]["tracks"][10]
                    and metadata == stored["results"][0]["stream_batches"][10], "Stored primary replay differs")
            closure = max(float(np.abs(e.sum(axis=0, dtype=np.float32) - m.astype(np.float32)).max())
                          for e, m in zip(estimates, mixtures, strict=True))
            require(closure <= 1e-6 and model_state_sha256(model) == fingerprint
                    and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
                    "Capture changed model, RNG, reconstruction or CPU scope")
            per_excerpt = []
            for i, (row, mixture, reference, estimate) in enumerate(zip(rows, mixtures, references, estimates, strict=True)):
                per_excerpt.append(legacy._score_track(track["name"], [row], [mixture], [reference], [estimate], metric_config))
                for stem, values in zip(SOURCE_ORDER, estimate, strict=True):
                    export_audio(out / model_row["id"] / f"excerpt-{i}" / ("estimate-" + stem + ".wav"),
                                 values, source_id=model_row["id"], stem=stem, clip_index=i)
            models[model_row["id"]] = {"model_state_sha256": fingerprint, "checkpoint": checkpoint,
                "stored_full14_result": {"path": str(full_directory / "result.json"), "sha256": sha(full_directory / "result.json")},
                "exact_stored_track_score": True, "exact_stored_stream_metadata": True,
                "track_score": score, "per_excerpt_scores": per_excerpt, "stream_metadata": metadata,
                "closure_max_abs": closure, "model_and_rng_unchanged": True}
            sources.append({"id": model_row["id"], "label": model_row["label"], "kind": "estimate",
                            "base": model_row["id"], "checkpoint_sha256": checkpoint["sha256"],
                            "algorithmic_latency_samples": 256, "sample_rate": 44100,
                            "status": "Working baseline" if model_row["id"] == "working" else "Development comparison"})
            event = {"event": "captured", "model": model_row["id"], "exact_primary_replay": True,
                     "elapsed_seconds": time.monotonic() - began}
            progress.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            del model, raw, delayed, estimates
    require(all(plan["source_bindings"].get(p) == s for p, s in evidence.items()), "Unbound completed evidence")
    for i, (mixture, reference) in enumerate(zip(mixtures, references, strict=True)):
        export_audio(out / "shared" / f"excerpt-{i}" / "mixture.wav", mixture,
                     source_id="mixture", stem=None, clip_index=i)
        for stem, values in zip(SOURCE_ORDER, reference, strict=True):
            export_audio(out / "shared" / f"excerpt-{i}" / ("reference-" + stem + ".wav"), values,
                         source_id="reference", stem=stem, clip_index=i)
    require(len(inventory) == 34 and sum(f["bytes"] for f in inventory.values()) < 200_000_000,
            "Audition inventory or allowance differs")
    sources += [{"id": "reference", "label": "Original stem", "kind": "reference", "base": "shared"},
                {"id": "mixture", "label": "Mixture", "kind": "mixture", "base": "shared"}]
    data = {"schema_version": 1, "default_source": "working", "default_stem": "drums", "sources": sources,
            "tracks": [{"name": track["name"], "folder": "", "clips": [
                {"label": "30–45 seconds", "folder": "excerpt-0", "reference_start": 1323000, "reference_end": 1984500},
                {"label": "75–90 seconds", "folder": "excerpt-1", "reference_start": 3307500, "reference_end": 3969000},
            ]}], "human_listening_verdict": None}
    html = Path(plan["template"]).read_text()
    html = html.replace("Actions · training comparison", "Skelpolu · regression comparison")
    html = html.replace(
        "Compare the C91 reference, native C191 and two completed training trials on Actions – One Minute Smile, 60–75 seconds. No new candidate has been retained.",
        "Compare the working model, training parent and candidate on the two primary excerpts from Skelpolu – Human Mistakes.")
    html = html.replace('<a href="../index.html">More listening excerpts</a> · ', "")
    html = html.replace(
        "Latency values describe model buffering. No human listening verdict has been recorded for these comparisons.",
        "All clips retain native levels. This track was selected to inspect a measured regression. No listening verdict has been recorded.")
    html, count = re.subn(r'(<script id="listening-data" type="application/json">).*?(</script>)',
                         lambda m: m[1] + json.dumps(data, indent=2) + m[2], html, flags=re.DOTALL)
    require(count == 1 and "C191" not in html, "Player template replacement failed")
    with (out / "index.html").open("x") as stream:
        stream.write(html)
    with (out / "README.md").open("x") as stream:
        stream.write(
            "# Skelpolu primary-excerpt comparison\n\n"
            "Compare Skelpolu – Human Mistakes at 30–45 and 75–90 seconds. "
            "Select a passage, stem and model in index.html. Source and stem switches preserve playback position.\n\n"
            "Working is the accepted 5.8 ms model; parent is context step 500; "
            "candidate is accumulated-gradient step 500. All estimates use the same physical "
            "alignment and shipping residual policy, at their native levels. Original stems "
            "and mixture come from the unchanged validation audio.\n\n"
            "Each model streamed continuously from sample zero. Its complete two-excerpt "
            "track score and stream metadata matched the earlier full-panel evaluation exactly "
            "before export. The 34 stereo, float WAVs are 15 seconds at 44.1 kHz; "
            "result.json records their hashes and per-excerpt diagnostics.\n\n"
            "This track was chosen to inspect a measured regression. It is existing primary "
            "validation material, not independent confirmation. No listening verdict, model "
            "selection or deployment qualification is recorded by this capture.\n")
    verify_inputs(plan)
    require_space(space_plan, 0)
    write(out / "result.json", {"schema": "latency58-primary-regression-audio-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "track": track["name"], "primary_intervals": rows, "models": models, "audio_files": inventory,
        "player_sha256": sha(out / "index.html"), "readme_sha256": sha(out / "README.md"),
        "audio_bytes": sum(f["bytes"] for f in inventory.values()),
        "normalization": None, "source_audio_changed": False, "training_updates_executed": 0,
        "checkpoint_written": False, "cuda_initialized": False, "quality_selected": False,
        "confirmation_excerpts_used": False, "human_listening_verdict": None,
        "browser_playback_tested": False, "elapsed_seconds": time.monotonic() - began,
        "limitations": ["Track chosen after primary-panel regression was observed; no independent validation is claimed.",
                        "Per-excerpt metrics describe the existing track; the unchanged full panel governs selection.",
                        "No human listening verdict or deployment qualification is inferred from audio capture."]})
    print(json.dumps({"status": "pass", "audio_files": len(inventory), "player": str(out / "index.html")}), flush=True)


if __name__ == "__main__":
    main()
