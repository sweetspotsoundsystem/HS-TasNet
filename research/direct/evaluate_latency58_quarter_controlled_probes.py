"""Run preserved bass probes through an audited quarter_controlled generation.

The stimuli and frequency-fit scorer are unchanged from the accepted model.
Each probe has independent state and uses literal hops with one final flush.
Only metrics are retained. Synthetic tones have no prescribed stem routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from research.direct.latency58_checkpoint import require, sha

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Probe plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-quarter-controlled-probe-plan-v1"
            and all(sha(path) == digest for path, digest in plan["source_bindings"].items()),
            "Probe source or checkpoint identity differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(name) == "1" for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    reference_binding = plan["accepted_probe_report"]
    require(sha(reference_binding["path"]) == reference_binding["sha256"], "Accepted probe report changed")
    reference = json.loads(Path(reference_binding["path"]).read_text())
    require(reference["source_bindings_before"][str(ROOT / "research/direct/bass.py")]
            == sha(ROOT / "research/direct/bass.py"), "Accepted probe generator or scorer differs")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Use a prepared directory without an existing result")
    import numpy as np
    import torch
    from research.direct import bass
    from research.direct.evaluate import shipping_residual
    from research.direct.latency58 import HOP, SOURCE_ORDER
    from research.direct.evaluate_latency58_quarter_controlled import load_evaluation_model
    from research.direct.latency58_evaluate import model_state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    model, receipt = load_evaluation_model(plan)
    step = receipt["step"]
    fingerprint = model_state_sha256(model)
    rng = torch.get_rng_state().clone()
    rows, streams = [], []
    started = time.monotonic()
    with torch.inference_mode(), (out / "progress.jsonl").open("x", buffering=1) as progress:
        for probe in bass.make_probes():
            mixture = np.asarray(probe["mixture"], dtype=np.float32)
            count = mixture.shape[-1]
            padding = (-count) % HOP
            require(count == 352800 and padding == 96, "Preserved eight-second probe geometry differs")
            received = np.pad(mixture, ((0, 0), (0, padding + HOP)))
            state = model.initial_state(1)
            raw = np.empty((4, 2, count), dtype=np.float32)
            data_hops = (count + padding) // HOP
            for index in range(data_hops + 1):
                previous = state.audio_history[..., -HOP:].clone()
                if index == data_hops:
                    output, state = model.flush(state, return_raw=True)
                else:
                    chunk = torch.from_numpy(np.ascontiguousarray(received[:, index * HOP:(index + 1) * HOP][None]))
                    output, state = model.forward_chunk(chunk, state, return_raw=True)
                require(tuple(output.shape) == (1, 4, 2, HOP)
                        and all(torch.isfinite(value).all().item() for value in (output, *state)),
                        "Probe output or state is malformed or non-finite")
                if index:
                    start, end = (index - 1) * HOP, min(index * HOP, count)
                    require(np.array_equal(previous[0, :, :end-start].numpy(), mixture[:, start:end]),
                            "Probe physical history does not match the real target samples")
                    raw[..., start:end] = output[0, ..., :end-start].numpy()
            estimates = shipping_residual(raw, mixture)
            score = bass.score_tone_outputs(estimates, probe)
            require(score["reconstruction_max_abs"] <= 1e-6, "Probe reconstruction failed")
            rows.append(score)
            streams.append({"id": probe["id"], "decoded_input_sha256": hashlib.sha256(mixture.tobytes()).hexdigest(),
                            "real_samples": count, "partial_hop_padding": padding, "flush_hops": 1,
                            "callback_hop_samples": HOP, "callback_count": data_hops + 1,
                            "global_output_cut_samples": HOP, "physical_target_samples": count,
                            "all_finite": True, "physical_alignment_verified": True,
                            "absolute_peak": float(np.max(np.abs(estimates))),
                            "state_scope": "reset once per probe; final flushed state discarded"})
            event = {"event": "probe", "id": probe["id"], "step": step}
            progress.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            del raw, estimates, output, state
    geometry = lambda values: [(r["id"], [(s["start"], s["end"], s["frequencies_hz"])
                               for s in r["segments"]]) for r in values]
    require(geometry(reference["results"]) == geometry(rows), "Probe stimuli or physical scoring intervals differ")
    def means(values):
        return {row["id"]: {stem: {metric: None if any(s["per_stem"][stem][metric] is None for s in row["segments"])
                else statistics.mean(s["per_stem"][stem][metric] for s in row["segments"])
                for metric in row["segments"][0]["per_stem"][stem]} for stem in SOURCE_ORDER} for row in values}
    original_means, candidate_means = means(reference["results"]), means(rows)
    comparisons = {}
    for probe_id, stem_rows in candidate_means.items():
        comparisons[probe_id] = {}
        for stem, metrics in stem_rows.items():
            comparisons[probe_id][stem] = {}
            for key, value in metrics.items():
                old = original_means[probe_id][stem][key]
                comparisons[probe_id][stem][key] = {"reference": old, "candidate": value,
                                                   "delta": None if value is None or old is None else value - old}
    require(torch.equal(rng, torch.get_rng_state()) and fingerprint == model_state_sha256(model)
            and not torch.cuda.is_initialized()
            and all(sha(path) == digest for path, digest in plan["source_bindings"].items()),
            "Probe evaluation changed inputs, model tensors, RNG or CPU scope")
    report = {"schema": "latency58-quarter-controlled-probe-result-v1", "status": "pass", "checkpoint": plan["checkpoint"],
              "step": step, "model_state_sha256": fingerprint, "results": rows, "streaming": streams,
              "segment_means": candidate_means, "paired_accepted_segment_means": comparisons,
              "accepted_probe_report": reference_binding, "source_bindings": plan["source_bindings"],
              "source_bindings_unchanged": True, "plan_sha256": args.plan_sha256,
              "elapsed_seconds": time.monotonic() - started, "cuda_initialized": False,
              "native_output_source_scales": model.output_source_scales.tolist(),
              "model_provenance": model.provenance, "metrics_sha256": sha(ROOT / "research/direct/bass.py"),
              "graph_alignment_samples": HOP, "intended_total_latency_samples": 2 * HOP,
              "host_queue_implemented": False,
              "retained_audio": False, "normalization": "none", "host_qualified": False,
              "quality_retention_decision": None,
              "stimulus_policy": "Unchanged bass.make_probes; hop_tones are 86.13 and 172.27 Hz for both models",
              "interpretation": "Each stem fits input frequencies independently; unexplained energy, level and DC are separate diagnostics, not automatic audible-quality gates"}
    with (out / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "pass", "step": step, "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
