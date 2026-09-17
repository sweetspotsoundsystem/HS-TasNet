"""Complete fixed literal-hop warmup cases after a recorded bounded timeout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, read, require, sha, write


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Diagnostic plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-context-horizon-continuation-v1"
            and plan["track_indices"] == [0, 27, 55]
            and plan["start_samples"] == [880640, 1761280]
            and plan["warmup_samples"] == [0, 88064, 176128, 352256, 704512]
            and plan["scored_samples"] == 88064
            and all(sha(p) == s for p, s in plan["source_bindings"].items()), "Diagnostic scope or inputs changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous diagnostic")
    previous = plan["previous_incomplete_run"]
    prior_plan, prior_execution = read(previous["plan"]), read(previous["execution"])
    require(prior_plan["schema"] == "latency58-context-horizon-diagnostic-v1"
            and prior_execution["actual_exit_code"] in (-15, -9)
            and prior_execution["timed_out"] and prior_execution["source_bindings_unchanged"]
            and prior_execution["plan_sha256"] == sha(previous["plan"])
            and not (Path(prior_plan["output_directory"]) / "result.json").exists()
            and Path(previous["execution"]) == Path(prior_plan["output_directory"]) / "diagnostic-execution.json"
            and Path(previous["progress"]) == Path(prior_plan["output_directory"]) / "progress.jsonl"
            and prior_execution["argv"][prior_execution["argv"].index("--plan") + 1] == previous["plan"]
            and prior_execution["argv"][-1] == sha(previous["plan"])
            and all(prior_plan[k] == plan[k] for k in
                    ("track_indices", "start_samples", "warmup_samples", "scored_samples",
                     "parent_quality_plan", "model_state_sha256"))
            and all(plan["source_bindings"].get(p) == s for p, s in prior_plan["source_bindings"].items())
            and all(plan["source_bindings"].get(previous[k]) == sha(previous[k])
                    for k in ("plan", "execution", "progress")), "Not a bound incomplete diagnostic")
    prior_rows = [json.loads(line) for line in Path(previous["progress"]).read_text().splitlines()]
    key = lambda r: (r["track_index"], r["start_sample"], r["warmup_samples"])
    expected = [(t, s, w) for t in plan["track_indices"] for s in plan["start_samples"]
                for w in plan["warmup_samples"]]
    require(5 <= len(prior_rows) <= 30 and [key(r) for r in prior_rows] == expected[:len(prior_rows)],
            "Require a complete, correctly ordered prior case and a valid journal prefix")
    # Keep whole completed cases; recompute an interrupted case from audio sample zero.
    retained_count = min(len(prior_rows) // 5 * 5, 25)
    retained = {key(r): r for r in prior_rows[:retained_count]}
    prior_lookup = {key(r): r for r in prior_rows}
    recomputed_prior_rows = 0
    import numpy as np
    import soundfile as sf
    import torch
    from research.direct.evaluate_latency58_sdr_accum import load_evaluation_model
    from research.direct.latency58_evaluate import model_state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260915)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    parent_plan = read(plan["parent_quality_plan"]["path"])
    require(sha(plan["parent_quality_plan"]["path"]) == plan["parent_quality_plan"]["sha256"],
            "Parent quality plan changed")
    model, receipt = load_evaluation_model(parent_plan)
    before_state = model_state_sha256(model)
    require(before_state == plan["model_state_sha256"] == receipt["model_state_sha256"]
            and model.hop_samples == model.graph_alignment_samples == 128
            and all(t.device.type == "cpu" and t.dtype == torch.float32 for t in model.state_dict().values()),
            "Wrong parent or inference geometry")
    train, valid = (read(ROOT / ("research/manifests/" + split + ".json")) for split in ("train", "valid"))
    require(train["split"] == "train" and valid["split"] == "valid"
            and not ({t["name"] for t in train["tracks"]} & {t["name"] for t in valid["tracks"]}),
            "Training and primary validation tracks overlap")
    rng = torch.get_rng_state().clone()
    count, hop = plan["scored_samples"], 128
    rows = []

    def render(audio, start, stop, state=None, *, capture=False):
        require(0 <= start < stop <= audio.shape[-1] and start % hop == stop % hop == 0,
                "Use literal aligned hops and real audio")
        if state is None:
            state = model.initial_state(1)
        pieces = []
        for cursor in range(start, stop, hop):
            output = model.render(audio[..., cursor:cursor + hop], state)
            state = output.state
            require(all(bool(torch.isfinite(t).all()) for t in (*state, output.raw, output.deployed)),
                    "Non-finite continuous inference")
            if capture:
                pieces.append((output.raw.clone(), output.deployed.clone(), output.delayed_mixture.clone()))
        captured = tuple(torch.cat([p[k] for p in pieces], dim=-1) for k in range(3)) if capture else None
        return state, captured

    def summary(reference, candidate):
        difference = candidate.double() - reference.double()
        rms = difference.square().mean(dim=(0, 2, 3)).sqrt()
        ref_rms = reference.double().square().mean(dim=(0, 2, 3)).sqrt()
        return {"difference_rms": rms.tolist(),
                "difference_rms_dbfs": (20 * rms.clamp_min(1e-12).log10()).tolist(),
                "reference_rms": ref_rms.tolist(),
                "difference_to_reference_db": (20 * (rms.clamp_min(1e-12) / ref_rms.clamp_min(1e-12)).log10()).tolist(),
                "difference_max_abs": difference.abs().amax(dim=(0, 2, 3)).tolist()}

    with torch.inference_mode(), (out / "progress.jsonl").open("x", buffering=1) as log:
        for track_index in plan["track_indices"]:
            track = train["tracks"][track_index]
            path = Path(train["root"]) / track["mixture"]
            require(str(path) in plan["source_bindings"], "Unbound diagnostic audio")
            frames = max(plan["start_samples"]) + count + hop
            info = sf.info(path)
            require(info.frames == track["frames"] and info.frames >= frames
                    and info.samplerate == 44100 and info.channels == 2, "Audio geometry changed")
            decoded, rate = sf.read(path, start=0, stop=frames, dtype="float32", always_2d=True)
            require(rate == 44100 and decoded.shape == (frames, 2) and np.isfinite(decoded).all(), "Invalid audio")
            audio = torch.from_numpy(decoded.T.copy())[None]
            input_hash = tensor_hash(audio)
            for start in plan["start_samples"]:
                if (track_index, start, plan["warmup_samples"][0]) in retained:
                    rows.extend(retained[(track_index, start, w)] for w in plan["warmup_samples"])
                    continue
                continuous_state, _ = render(audio, 0, start)
                initial_hashes = [tensor_hash(t) for t in continuous_state]
                _, full = render(audio, start, start + count + hop, continuous_state, capture=True)
                full = tuple(t[..., hop:] for t in full)
                require(torch.equal(full[2], audio[..., start:start + count])
                        and float((full[1].sum(1) - full[2]).abs().max()) <= 1e-6
                        and initial_hashes == [tensor_hash(t) for t in continuous_state],
                        "Continuous alignment, closure or state immutability failed")
                # A same-input replay checks deterministic state and capture handling.
                if track_index == plan["track_indices"][0] and start == plan["start_samples"][0]:
                    replay_state, _ = render(audio, 0, start)
                    _, replay = render(audio, start, start + count + hop, replay_state, capture=True)
                    require(all(torch.equal(a, b) for a, b in zip(continuous_state, replay_state))
                            and all(torch.equal(a, b[..., hop:]) for a, b in zip(full, replay)), "Replay differs")
                for warmup in plan["warmup_samples"]:
                    state = model.initial_state(1) if warmup == 0 else render(audio, start - warmup, start)[0]
                    hidden_delta = (state.fusion_hidden.double() - continuous_state.fusion_hidden.double()) / model.public_fusion_state_scale
                    history_equal = torch.equal(state.audio_history, continuous_state.audio_history)
                    state_differences = {name: float((a.double() - b.double()).abs().max())
                                         for name, a, b in zip(state._fields, state, continuous_state)}
                    before_hashes = [tensor_hash(t) for t in state]
                    _, captured = render(audio, start, start + count + hop, state, capture=True)
                    captured = tuple(t[..., hop:] for t in captured)
                    require(torch.equal(captured[2], full[2])
                            and float((captured[1].sum(1) - captured[2]).abs().max()) <= 1e-6
                            and (warmup == 0 or history_equal)
                            and before_hashes == [tensor_hash(t) for t in state], "Warm alignment or state changed")
                    regions = {"whole_score": (0, count), "first_hop": (0, 128),
                               "first_quarter_second": (0, 11008), "second_half": (count // 2, count)}
                    row = {"track_index": track_index, "track_name": track["name"], "start_sample": start,
                           "warmup_samples": warmup, "warmup_seconds": warmup / 44100,
                           "physical_hidden_difference_max_abs": float(hidden_delta.abs().max()),
                           "physical_hidden_difference_rms": float(hidden_delta.square().mean().sqrt()),
                           "audio_history_exact": history_equal, "state_difference_max_abs": state_differences,
                           "regions": {name: {kind: summary(full[k][..., left:right], captured[k][..., left:right])
                                              for k, kind in enumerate(("raw", "deployed"))}
                                       for name, (left, right) in regions.items()}}
                    if key(row) in prior_lookup:
                        require(row == prior_lookup[key(row)], "Recomputed partial case differs from its original journal")
                        recomputed_prior_rows += 1
                    rows.append(row)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps({"event": "warmup", "track": track_index, "start": start,
                                      "warmup": warmup, "hidden_max": row["physical_hidden_difference_max_abs"]}), flush=True)
                require(input_hash == tensor_hash(audio), "Decoded input changed")
    require(len(rows) == 30 and model_state_sha256(model) == before_state
            and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and all(p.grad is None for p in model.parameters())
            and all(sha(p) == s for p, s in plan["source_bindings"].items()), "Diagnostic changed protected inputs")
    write(out / "result.json", {"schema": "latency58-context-horizon-continuation-result-v1", "status": "pass",
          "previous_incomplete_run": previous, "previous_actual_exit_code": prior_execution["actual_exit_code"],
          "previous_timed_out": True, "retained_prior_rows": retained_count,
          "prior_rows_recomputed_and_matched": recomputed_prior_rows, "new_execution_rows": 30 - retained_count,
          "plan_sha256": args.plan_sha256, "model_state_sha256": before_state, "rows": rows,
          "source_order": ["drums", "bass", "vocals", "other"], "precision": "native_fp32_cpu1",
          "unroll_hops": 1, "flush_hops": 0, "real_audio_after_scored_interval_samples": hop,
          "reset_replay_exact": True, "reset_replay_case_count": 1,
          "scored_regions": {"whole_score": [0, count], "first_hop": [0, 128],
                             "first_quarter_second": [0, 11008], "second_half": [count // 2, count]},
          "source_bindings_unchanged": True, "cuda_initialized": False,
          "optimizer_updates": 0, "saved_weights": False, "saved_audio": False,
          "elapsed_seconds": time.monotonic() - began,
          "limitations": ["Original run timed out; whole completed cases are retained and the unfinished case is rerun from sample zero.",
                          "Three fixed training tracks; no primary or reserved confirmation quality scoring.",
                          "Difference from continuous predictions measures context mismatch, not separation accuracy.",
                          "Native FP32 inference; BF16 training and augmented crops may behave differently.",
                          "No longer-warmup training result or improved quality is established."]})
    print(json.dumps({"status": "pass", "rows": len(rows), "elapsed_seconds": time.monotonic() - began}), flush=True)


if __name__ == "__main__":
    main()
