"""Capture the reserved Skelpolu passages after all three vocal pilots close.

Reuse the authenticated working/drum-500 WAVs. Each new model must reproduce
its original primary score and stream metadata exactly before native export.
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
from research.direct.record_latency58_vocal_focus_review import completed_endpoint

ARMS = ("original", "focused", "focused_mixer")


def bound(plan, item):
    require(plan["source_bindings"].get(item["path"]) == item["sha256"] == sha(item["path"]),
            "Unbound capture prerequisite")
    return read(item["path"])


def completed(plan, item, result_key, execution_key, plan_key, module):
    result, execution, source_plan = (bound(plan, item[k]) for k in (result_key, execution_key, plan_key))
    argv = execution["argv"]
    require(execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and result["source_bindings_unchanged"]
            and execution["plan_sha256"] == result["plan_sha256"] == item[plan_key]["sha256"]
            and argv[argv.index("-m") + 1] == module
            and argv[argv.index("--plan") + 1] == item[plan_key]["path"]
            and Path(source_plan["output_directory"]) == Path(item[result_key]["path"]).parent,
            "Prerequisite execution is incomplete or belongs to another plan")
    verify_inputs(source_plan)
    require(all(plan["source_bindings"].get(p) == s for p, s in source_plan["source_bindings"].items()),
            "Capture omitted prerequisite sources")
    return result, source_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256, "Capture plan or cwd differs")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-vocal-focus-skelpolu-audio-plan-v1"
            and plan["step"] == 250 and plan["track_index"] == 10
            and plan["track_name"] == "Skelpolu - Human Mistakes"
            and [m["id"] for m in plan["models"]] == list(ARMS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the reserved CPU1 capture")
    verify_inputs(plan)
    require(plan["source_bindings"].get(str(Path(__file__).resolve())) == sha(Path(__file__).resolve()),
            "Unbound capture implementation")
    reservation = bound(plan, plan["reservation"])
    require(reservation["schema"] == "latency58-vocal-focus-skelpolu-reservation-v1"
            and reservation["status"] == "prepared_not_executed"
            and reservation["new_models"] == list(ARMS)
            and reservation["reference_sample_intervals"] == [[1323000, 1984500], [3307500, 3969000]]
            and reservation["run_only_after_all_three_reviews_and_terminal_optimizer_retirements"]
            and reservation["require_completed_250_update_training_match"]
            and not reservation["independent_confirmation"] and reservation["normalization"] is None
            and all(plan[k] == reservation[k] for k in ("track_index", "track_name", "reuse_capture_directory",
                                                       "counted_roots", "stop_counted_bytes", "new_audio_reserve_bytes"))
            and plan["new_audio_reserve_bytes"] == 135_000_000
            and plan["concurrent_checkpoint_reserve_bytes"] == 0,
            "Capture differs from its prospective reservation")
    verify_inputs(reservation)
    require(all(plan["source_bindings"].get(p) == s for p, s in reservation["source_bindings"].items()),
            "Capture omitted reserved sources")
    for key in ("manifest", "evaluation_config", "template"):
        require(plan["source_bindings"].get(plan[key]) == sha(plan[key]), "Unbound capture configuration")
    out, old = Path(plan["output_directory"]), Path(plan["reuse_capture_directory"])
    require(out.parent == old.parent == PHASE and out.is_dir()
            and all(not (out / name).exists() for name in ("result.json", "index.html", *ARMS)),
            "Preserve existing capture output")
    for name in ("plan.json", "result.json", "capture-execution.json"):
        require(plan["source_bindings"].get(str(old / name)) == sha(old / name), "Unbound prior capture")
    saved, old_execution = read(old / "result.json"), read(old / "capture-execution.json")
    require(saved["schema"] == "latency58-primary-regression-audio-v6" and saved["status"] == "pass"
            and saved["source_bindings_unchanged"] and old_execution["actual_exit_code"] == 0
            and not old_execution["timed_out"] and old_execution["source_bindings_unchanged"]
            and old_execution["plan_sha256"] == saved["plan_sha256"] == sha(old / "plan.json"),
            "Prior capture is incomplete")
    verify_inputs(saved)
    match, _ = completed(plan, plan, "match_audit", "match_execution", "match_plan",
                         "research.direct.audit_latency58_vocal_focus_training_match")
    require(match["schema"] == "latency58-vocal-focus-training-match-v1" and match["status"] == "pass"
            and match["all_three_pristine_and_original_draws_exact"]
            and match["focused_inputs_and_teacher_targets_exact"] and match["initial_inherited_gradients_exact"]
            and match["final_rng_states_exact"] and match["updates_per_arm"] == 250
            and match["microbatches_per_arm"] == 1000 and match["examples_per_arm"] == 4000,
            "Three completed pilots were not authenticated as matched")
    verify_inputs(match)
    for item in plan["models"]:
        arm = item["id"]
        require(item["training_plan"] == reservation["training_plans"][arm] == match["training_plans"][arm]
                and item["model_state_sha256"] == match["trained_model_states"][arm]
                and item["prefix"] == "vocal-focus-" + arm.replace("_", "-") + "-250",
                "Different reserved endpoint")
        review, review_plan = completed(plan, item, "review", "review_execution", "review_plan",
                                       "research.direct.record_latency58_vocal_focus_review")
        require(review["schema"] == "latency58-vocal-focus-review-v1" and review["status"] == "pass"
                and review["arm"] == arm and review["training_closed"] and review["all_quality_metrics_reviewed"]
                and review["completed_step"] == review["original_maximum_step"] == 250
                and review["further_optimizer_updates"] == 0 and not review["quality_selected"]
                and review["review"] == review_plan["review"] and review["training_plan"] == item["training_plan"],
                "Complete quality review has not closed this pilot")
        verify_inputs(review)
        training, generation, receipt, _ = completed_endpoint(review_plan)
        require(training["arm"] == arm and receipt["model_state_sha256"] == item["model_state_sha256"]
                and receipt["files"]["model.pt"]["sha256"] == item["checkpoint_sha256"], "Different reviewed model")
        retired, retirement_plan = completed(plan, item, "retirement", "retirement_execution", "retirement_plan",
                                            "research.direct.retire_latency58_vocal_focus_optimizer")
        require(retired["schema"] == "latency58-vocal-focus-optimizer-retirement-v1"
                and retired["status"] == "complete" and retired["arm"] == arm
                and retired["review"] == retirement_plan["review"] == item["review"]
                and retired["retired_path"] == str(generation / "optimizer.pt")
                and not (generation / "optimizer.pt").exists()
                and retired["freed_bytes"] == receipt["files"]["optimizer.pt"]["bytes"]
                and all(plan["source_bindings"].get(p) == s == sha(p) for p, s in retired["protected_files"].items()),
                "Terminal optimizer retirement is incomplete or retained weights differ")
    from research.direct.latency58_vocal_focus_checkpoint import require_space
    space_plan = plan
    require_space(space_plan, 135_000_000)
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
    require([m["id"] for m in plan["reuse_models"]] == ["working", "drum500"]
            and [m["capture_id"] for m in plan["reuse_models"]] == ["working", "previous"],
            "Reused comparison identities differ")
    models, sources, replay = {}, [], {}
    for item in plan["reuse_models"]:
        _, stored = load_completed(PHASE / (item["prefix"] + "-full14-001"), evidence,
                                   canonical_baseline=item["id"] == "working")
        reserved = reservation["reused_models"][item["id"]]
        require(reserved["capture_model_id"] == item["capture_id"] and reserved["prefix"] == item["prefix"]
                and reserved["model_state_sha256"] == item["model_state_sha256"], "Different reserved baseline")
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
    for path, old_row in reservation["reused_audio_files"].items():
        require(saved["audio_files"].get(path) == old_row, "Reserved audio differs from prior capture")
        row = {**old_row, "source_id": mapping.get(old_row["source_id"], old_row["source_id"])}
        file = Path(path)
        require(file.is_relative_to(PHASE) and plan["source_bindings"].get(path) == sha(file) == row["sha256"]
                and file.stat().st_size == row["bytes"], "Reused WAV differs")
        values, rate = sf.read(file, dtype="float32", always_2d=True)
        require(rate == 44100 and values.shape == (661500, 2) and np.isfinite(values).all()
                and sf.info(file).subtype == "FLOAT", "Reused WAV geometry differs")
        index = row["clip_index"]
        require(index in (0, 1), "Unexpected reused passage")
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
    require(len(reused) == 26 and all(len(v) == 8 for v in replay.values()), "Incomplete reused inventory")
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
            from research.direct.evaluate_latency58_vocal_focus import load_evaluation_model
            require(full_plan["step"] == plan["step"] and full_plan["training_plan"] == item["training_plan"],
                    "Capture generation is not the matched endpoint")
            training = read(item["training_plan"]["path"])
            require(training["warmup_samples"] == training["scored_samples"] == 88064
                    and training["arm"] == item["id"] and training["teacher_kind"] == "c91"
                    and training["parent"]["model_state_sha256"] == models["working"]["model_state_sha256"],
                    "Vocal pilot geometry or parent differs")
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
    require(len(inventory) == 50 and len(new_files) == 24
            and sum(inventory[p]["bytes"] for p in new_files) < 135_000_000, "Wrong audio inventory or allowance")
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
    data = {"schema_version": 1, "default_source": "working", "default_stem": "vocals", "sources": sources,
            "tracks": [{"name": track["name"], "folder": "", "clips": clips}], "human_listening_verdict": None}
    html = Path(plan["template"]).read_text()
    html = html.replace("Actions · training comparison", "Vocal-cleanliness pilot · Skelpolu")
    html = html.replace(
        "Compare the C91 reference, native C191 and two completed training trials on Actions – One Minute Smile, 60–75 seconds. No new candidate has been retained.",
        "Compare the working model, drum checkpoint and three vocal-cleanliness pilots on Skelpolu – Human Mistakes, 30–45 and 75–90 seconds. Check Vocals and Other for vocal assignment, and Drums and Bass for instrument fidelity.")
    html = html.replace('<a href="../index.html">More listening excerpts</a> · ', "")
    html = html.replace(
        "Latency values describe model buffering. No human listening verdict has been recorded for these comparisons.",
        "All clips retain native levels. These primary passages were selected to inspect measured vocal-assignment failures. No listening verdict has been recorded.")
    html = html.replace("Keys 1–6 select the sources in order.", "Keys 1–7 select the sources in order.")
    html = html.replace('<option value="bass" selected>', '<option value="bass">')
    html = html.replace('<option value="vocals">', '<option value="vocals" selected>')
    html, count = re.subn(r'(<script id="listening-data" type="application/json">).*?(</script>)',
                         lambda m: m[1] + json.dumps(data, indent=2) + m[2], html, flags=re.DOTALL)
    require(count == 1 and "C191" not in html, "Player replacement failed")
    with (out / "index.html").open("x") as stream:
        stream.write(html)
    with (out / "README.md").open("x") as stream:
        stream.write("# Skelpolu primary-excerpt comparison\n\n[Open the player](index.html). "
                     "Both passages retain the original 30–45 and 75–90 second primary intervals. "
                     "The working 5.8 ms baseline and original stems are reused at exact native levels. "
                     "All three vocal pilots receive new captures; working and drum 500 reuse authenticated audio.\n\n"
                     "Each new model streams from sample zero and reproduces its stored complete track score "
                     "and stream metadata exactly before WAV export. All 50 files are stereo FLOAT at 44.1 kHz; "
                     "26 are reused and 24 are new. [Capture details](result.json) record hashes and per-excerpt diagnostics.\n\n"
                     "This is an observed primary-panel regression, not independent confirmation. "
                     "No human verdict, model selection or deployment qualification is recorded.\n")
    verify_inputs(plan)
    require_space(space_plan, 0)
    write(out / "result.json", {"schema": "latency58-vocal-focus-skelpolu-audio-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
        "track": track["name"], "primary_intervals": rows, "models": models, "audio_files": inventory,
        "step": plan["step"], "match_audit": plan["match_audit"], "reservation": plan["reservation"],
        "reused_capture_result_sha256": sha(old / "result.json"), "reused_audio_files": reused,
        "new_audio_files": new_files, "audio_bytes": sum(v["bytes"] for v in inventory.values()),
        "new_audio_bytes": sum(inventory[p]["bytes"] for p in new_files),
        "player_sha256": sha(out / "index.html"), "readme_sha256": sha(out / "README.md"),
        "exact_player_inventory": True, "normalization": None, "source_audio_changed": False,
        "training_updates_executed": 0, "checkpoint_written": False, "cuda_initialized": False,
        "quality_selected": False, "confirmation_excerpts_used": False, "human_listening_verdict": None,
        "browser_playback_tested": False, "human_listening_completed": False,
        "elapsed_seconds": time.monotonic() - began,
        "limitations": ["Track chosen after its primary regression was observed; no independent validation.",
                        "Per-excerpt diagnostics do not replace the unchanged full-panel aggregation.",
                        "Audio capture establishes no human listening verdict or runtime qualification."]})
    print({"status": "pass", "audio_files": len(inventory), "new_audio_files": len(new_files)}, flush=True)


if __name__ == "__main__":
    main()
