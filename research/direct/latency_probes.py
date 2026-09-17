#!/usr/bin/env python3
"""Run the same short continuous bass probes through C91/C126/C191 in FP32.

C91 output is shifted by its known 512-sample graph delay, with enough zero
callbacks to flush the tail. Current-chunk adapters use zero graph delay.
The host's additional scheduling latency is reported separately. Probe streams
have independent state from sample zero; no state resets occur between notes.

Examples::
    python -m research.direct.latency_probes --kind c91 --output v1-probes.json
    python -m research.direct.latency_probes --kind c191 --checkpoint candidate.pt \
        --output candidate-probes.json --audio-dir candidate-probes

Synthetic source routing is unspecified. The score fits the input frequencies
to each output, recording unexplained energy, output level and DC separately.
These are diagnostics, not separation scores or automatic perceptual gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch

from research.direct import bass, evaluate
from research.direct.latency11 import load_model

DEFAULT_C91 = evaluate.ROOT / "research/direct/runs/c91-refined-v1/model.pt"


@torch.inference_mode()
def render_probes(model: Any, kind: str, probes: Sequence[dict[str, Any]],
                  device: torch.device) -> tuple[list[np.ndarray], dict[str, Any]]:
    if kind not in ("c91", "c126", "c191") or not probes:
        raise ValueError("Provide C91/C126/C191 and at least one probe")
    audio = [np.asarray(probe["mixture"], dtype=np.float32) for probe in probes]
    if any(values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0
           or not np.isfinite(values).all() for values in audio):
        raise ValueError("Every probe needs finite, nonempty stereo audio")
    alignment = 512 if kind == "c91" else 0
    hop, batch_size = 512, len(audio)
    lengths = [values.shape[-1] for values in audio]
    end = max(lengths) + alignment
    raw = np.empty((batch_size, 4, 2, ((end + hop - 1) // hop) * hop), dtype=np.float32)
    if kind == "c91":
        transform = evaluate.legacy._init_batched_stateful_transform(model,
            batch_size=batch_size, device=device)
    else:
        state = model.initial_state(batch_size, device=device)

        def transform(chunk):
            nonlocal state
            output, state = model.forward_chunk(chunk, state)
            return output

    staging = np.zeros((batch_size, 2, hop), dtype=np.float32)
    for position in range(0, end, hop):
        staging.fill(0)
        for index, values in enumerate(audio):
            count = max(0, min(hop, values.shape[-1] - position))
            if count:
                staging[index, :, :count] = values[:, position:position + count]
        output = transform(torch.from_numpy(staging).to(device=device, dtype=torch.float32))
        if tuple(output.shape) != (batch_size, 4, 2, hop):
            raise ValueError(f"Unexpected callback output: {tuple(output.shape)}")
        cpu = output.float().cpu().numpy()
        if not np.isfinite(cpu).all():
            raise FloatingPointError(f"Non-finite probe output at input sample {position}")
        raw[..., position:position + hop] = cpu
    aligned = [evaluate.shipping_residual(raw[i, ..., alignment:alignment + length], values)
               for i, (length, values) in enumerate(zip(lengths, audio))]
    return aligned, {"callback_hop_samples": hop, "callback_count": raw.shape[-1] // hop,
                     "graph_alignment_samples": alignment,
                     "algorithmic_latency_samples": 1024 if kind == "c91" else 512,
                     "algorithmic_latency_ms": 1000 * (1024 if kind == "c91" else 512) / bass.SAMPLE_RATE,
                     "state": "independent per probe, continuous from sample zero",
                     "output_policy": "float32 Other = aligned mixture - sum(Drums,Bass,Vocals)",
                     "zero_padding": "partial final callback, plus C91 delayed-tail flush"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("c91", "c126", "c191"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--vocal-gain", type=float)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.kind == "c91" and args.vocal_gain is not None:
        parser.error("C91 uses its checkpoint's deployment gains")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    started = time.monotonic()
    if args.kind == "c91":
        checkpoint = args.checkpoint or DEFAULT_C91
        model, identity = evaluate.load_checkpoint(checkpoint, device, allow_custom_gains=True)
    else:
        model = load_model(args.kind, args.checkpoint, device=device, vocal_gain=args.vocal_gain)
        identity = dict(model.provenance)
        identity["output_source_scales"] = model.output_source_scales.detach().cpu().tolist()
        identity["vocal_gain_override"] = args.vocal_gain
    probes = bass.make_probes()
    outputs, streaming = render_probes(model, args.kind, probes, device)
    results = []
    for probe, output in zip(probes, outputs):
        result = bass.score_tone_outputs(output, probe)
        result["mixture_peak"] = float(np.max(np.abs(probe["mixture"])))
        results.append(result)
        if args.audio_dir is not None:
            folder = args.audio_dir / probe["id"]
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / "mixture.wav", probe["mixture"].T, bass.SAMPLE_RATE, subtype="FLOAT")
            for source, samples in zip(bass.SOURCES, output):
                sf.write(folder / f"estimate-{source}.wav", samples.T, bass.SAMPLE_RATE, subtype="FLOAT")
    report = {"schema_version": 1, "kind": args.kind, "checkpoint": identity,
              "device": str(device), "precision": "float32", "streaming": streaming,
              "interpretation": "No prescribed stem routing; fit input frequencies per stem. Unexplained energy includes generated tones and time-varying allocation. Read output level and DC alongside residual ratios.",
              "display_floor_dbfs": -120, "probe_input_seconds": 32,
              "runner_sha256": evaluate.legacy._sha256_file(Path(__file__)),
              "metrics_sha256": evaluate.legacy._sha256_file(Path(bass.__file__)),
              "elapsed_seconds": time.monotonic() - started,
              "audio_dir": str(args.audio_dir.resolve()) if args.audio_dir is not None else None,
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "kind": args.kind,
                      "elapsed_seconds": report["elapsed_seconds"],
                      "maximum_reconstruction_error": max(row["reconstruction_max_abs"] for row in results)}))


if __name__ == "__main__":
    main()
