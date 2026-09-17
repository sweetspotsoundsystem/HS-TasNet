"""Separate export-copy arithmetic from ORT optimization on a full prefix.

Independent recurrent trajectories preserve weights, inputs and tolerances.
This produces diagnostic evidence and never declares native qualification.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import time

from research.direct.latency58_checkpoint import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Diagnostic plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-asymmetric-backends-plan-v1"
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Diagnostic inputs differ")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous diagnostic")
    import numpy as np
    import torch
    import soundfile as sf
    import onnxruntime as ort
    from research.direct.latency58_asymmetric_checkpoint import load_model_state, make_model
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_asymmetric_onnx import (
        INPUT_NAMES, OUTPUT_NAMES, TOLERANCES, make_export_copy, _validate_values,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    model = make_model(plan["parent_checkpoint"])
    require(load_model_state(model, plan["checkpoint"]) == plan["step"]
            and model_state_sha256(model) == plan["model_state_sha256"], "Diagnostic model differs")
    wrapper = make_export_copy(model)
    rng = torch.get_rng_state().clone()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(plan["graph"]["path"], sess_options=options, providers=["CPUExecutionProvider"])
    require(session.get_providers() == ["CPUExecutionProvider"], "Use only CPU execution")
    frames, start, end = (plan[k] for k in ("prefix_frames", "capture_start", "capture_end"))
    require(0 <= start < end <= frames and frames % 128 == 0, "Invalid physical geometry")
    mixture, rate = sf.read(plan["mixture"]["path"], frames=frames, dtype="float32", always_2d=True)
    require(rate == 44100 and mixture.shape == (frames, 2), "Mixture shape/rate differs")
    references = {}
    for name, files in plan["reference_groups"].items():
        audio_rows = []
        for stem in ("drums", "bass", "vocals", "other"):
            audio, rate = sf.read(files[stem]["path"], dtype="float32", always_2d=True)
            require(rate == 44100 and audio.shape == (end - start, 2), "Reference geometry differs")
            audio_rows.append(audio.T)
        references[name] = np.stack(audio_rows)
    native_state = model.initial_state(1)
    copy_state = tuple(v.clone() for v in native_state)
    ort_state = [v.numpy().copy() for v in native_state]
    live_names = ("native", "export_copy", "ort_disabled")
    pairs = list(itertools.combinations(live_names, 2))
    capture_pairs = pairs + [(name, ref) for name in live_names for ref in references]
    comparisons = {f"{a}_vs_{b}": {"max_abs": np.zeros(4), "max_callback_rms": np.zeros(4),
                    "worst_callback_physical_start": [None] * 4, "squared_error": np.zeros(4)}
                   for a, b in capture_pairs}
    full = {f"{a}_vs_{b}": {"waveform_max_abs": 0., "state_max_abs_decoded_units": np.zeros(4),
             "worst_hidden_input_start": None} for a, b in pairs}
    witnesses = {}
    closure_max = 0.
    recovered = 0
    began = time.monotonic()
    with torch.inference_mode():
        for index in range(frames // 128 + 1):
            chunk = np.ascontiguousarray(mixture[index * 128:(index + 1) * 128].T[None]) \
                if index < frames // 128 else np.zeros((1, 2, 128), np.float32)
            prior = {"native": [v.numpy() for v in native_state],
                     "export_copy": [v.numpy() for v in copy_state], "ort_disabled": ort_state}
            x = torch.from_numpy(chunk)
            prediction, native_state = model.forward_chunk(x, native_state) \
                if index < frames // 128 else model.flush(native_state)
            copied = wrapper(x, *copy_state)
            copy_state = copied[1:]
            values = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *ort_state], strict=True)))
            ort_state = values[1:]
            current = {"native": [prediction.numpy(), *(v.numpy() for v in native_state)],
                       "export_copy": [v.numpy() for v in copied], "ort_disabled": values}
            for name, row in current.items():
                _validate_values(row, f"{name}/{index}")
                require(np.array_equal(row[1], current["native"][1]), "Audio history differs")
            for a, b in pairs:
                name = f"{a}_vs_{b}"
                row = full[name]
                row["waveform_max_abs"] = max(row["waveform_max_abs"], float(np.max(np.abs(
                    current[a][0].astype(np.float64) - current[b][0]))))
                errors = np.array([float(np.max(np.abs(l.astype(np.float64) - r)))
                                   * (2. ** 18 if i == 1 else 1.)
                                   for i, (l, r) in enumerate(zip(current[a][1:], current[b][1:], strict=True))])
                if errors[1] > row["state_max_abs_decoded_units"][1]:
                    row["worst_hidden_input_start"] = index * 128
                    witness = {"audio_chunk": chunk.copy(), "input_start": np.array(index * 128)}
                    for backend in (a, b):
                        for i, v in enumerate(prior[backend]):
                            witness[f"{backend}_prior_{i}"] = v.copy()
                        for i, v in enumerate(current[backend]):
                            witness[f"{backend}_output_{i}"] = v.copy()
                    witnesses[name] = witness
                row["state_max_abs_decoded_units"] = np.maximum(row["state_max_abs_decoded_units"], errors)
            physical = (index - 1) * 128
            expected = mixture[physical:physical + 128].T if physical >= 0 else np.zeros((2, 128))
            for values in current.values():
                closure_max = max(closure_max, float(np.max(np.abs(values[0][0].astype(np.float64).sum(0) - expected))))
            left, right = max(start, physical), min(end, physical + 128)
            if left < right:
                target = slice(left - start, right - start)
                source = slice(left - physical, right - physical)
                rows = {name: values[0][0, ..., source] for name, values in current.items()}
                rows.update({name: values[..., target] for name, values in references.items()})
                for a, b in capture_pairs:
                    row = comparisons[f"{a}_vs_{b}"]
                    difference = rows[a].astype(np.float64) - rows[b].astype(np.float64)
                    rms = np.sqrt(np.mean(difference ** 2, axis=(1, 2)))
                    for i in range(4):
                        if rms[i] > row["max_callback_rms"][i]:
                            row["worst_callback_physical_start"][i] = physical
                    row["max_callback_rms"] = np.maximum(row["max_callback_rms"], rms)
                    row["max_abs"] = np.maximum(row["max_abs"], np.max(np.abs(difference), axis=(1, 2)))
                    row["squared_error"] += np.sum(difference ** 2, axis=(1, 2))
                recovered += right - left
            if index and index % 2048 == 0:
                print(json.dumps({"physical_input_frames": min(frames, index * 128),
                                  "elapsed_seconds": time.monotonic() - began}), flush=True)
    require(recovered == end - start and torch.equal(rng, torch.get_rng_state())
            and not torch.cuda.is_initialized() and model_state_sha256(model) == plan["model_state_sha256"]
            and model_state_sha256(wrapper.model) == plan["model_state_sha256"], "Recovery, model or CPU state differs")
    witness_files = {}
    for name, witness in witnesses.items():
        p = out / (name + "-worst-hidden.npz")
        require(not p.exists(), "Preserve prior witness")
        np.savez(p, **witness)
        witness_files[name] = {"path": str(p), "sha256": sha(p)}
    for row in comparisons.values():
        row["overall_rms"] = np.sqrt(row.pop("squared_error") / (2 * recovered)).tolist()
        row["within_unchanged_music_tolerance"] = bool(np.max(row["max_abs"]) <= TOLERANCES["waveform_max_abs"]
            and np.max(row["max_callback_rms"]) <= TOLERANCES["stem_callback_rms"])
        row["max_abs"] = row["max_abs"].tolist()
        row["max_callback_rms"] = row["max_callback_rms"].tolist()
    for row in full.values():
        row["within_unchanged_state_tolerance"] = bool(np.max(row["state_max_abs_decoded_units"])
                                                        <= TOLERANCES["state_max_abs_decoded_units"])
        row["state_max_abs_decoded_units"] = row["state_max_abs_decoded_units"].tolist()
    require(all(sha(p) == h for p, h in plan["source_bindings"].items()), "Diagnostic inputs changed")
    result = {"status": "diagnostic_completed", "quality_or_native_qualified": False,
              "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
              "runtime_versions": {"torch": torch.__version__, "onnxruntime": ort.__version__},
              "ort_optimization": "ORT_DISABLE_ALL", "comparisons": comparisons,
              "full_prefix_comparisons": full, "full_prefix_mixture_closure_max_abs": closure_max,
              "captured_samples": recovered, "native_output_gains_unchanged": True,
              "tolerances": TOLERANCES, "witness_files": witness_files,
              "elapsed_seconds": time.monotonic() - began}
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
