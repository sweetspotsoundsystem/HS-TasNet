"""CPU check of audited-resume loading and the independent hop128 export copy.

This writes only a small report. It does not export or execute ONNX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from research.direct.latency58_checkpoint import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-checkpoint-loader-and-export-copy-check-v1"
            and all(sha(path) == digest for path, digest in plan["source_bindings"].items()), "Input identity differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(name) == "1" for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    output = Path(plan["output_directory"]) / "result.json"
    require(not output.exists(), "Preserve existing result")
    import numpy as np
    import torch
    from research.direct.latency58 import HOP, PUBLIC_FUSION_SCALE, Latency58Model, Latency58State
    from research.direct.latency58_checkpoint import load_model_state
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_onnx import make_export_copy, verification_cases, OUTPUT_SHAPES

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    model = Latency58Model.from_accepted().eval().requires_grad_(False)
    rng = torch.get_rng_state().clone()
    require(load_model_state(model, plan["inference"]) == plan["expected_step"], "Inference step differs")
    state_hash = model_state_sha256(model)
    audio = torch.linspace(-0.1, 0.1, 2 * HOP).reshape(1, 2, HOP)
    with torch.inference_mode():
        first = model.forward_chunk(audio)
        require(load_model_state(model, plan["resume"]) == plan["expected_step"], "Audited resume step differs")
        second = model.forward_chunk(audio)
    require(state_hash == plan["expected_model_state_sha256"] == model_state_sha256(model)
            and torch.equal(first[0], second[0])
            and all(torch.equal(x, y) for x, y in zip(first[1], second[1], strict=True)),
            "Inference and resume do not produce identical model tensors and outputs")
    wrapper = make_export_copy(model)
    cases = []
    with torch.inference_mode():
        for case in verification_cases(8, ()):
            actual_audio = case["audio"]
            received = np.pad(actual_audio, ((0, 0), (0, (-actual_audio.shape[-1]) % HOP + HOP)))
            trace_hashes = []
            maximum = 0.0
            state_maximum = 0.0
            closure_maximum = 0.0
            for _ in range(2):
                native = Latency58State(*(torch.from_numpy(v.copy()) for v in case["initial_states"]))
                copied = tuple(v.clone() for v in native)
                incoming = case["initial_states"][0].copy()
                trace = hashlib.sha256()
                for index in range(received.shape[-1] // HOP):
                    chunk = torch.from_numpy(np.ascontiguousarray(received[:, index * HOP:(index + 1) * HOP][None]))
                    expected = incoming[..., -HOP:].copy()
                    incoming = np.concatenate((incoming[..., HOP:], chunk.numpy()), axis=-1)
                    if index == received.shape[-1] // HOP - 1:
                        audio_out, native = model.flush(native)
                    else:
                        audio_out, native = model.forward_chunk(chunk, native)
                    exported = wrapper(chunk, *copied)
                    copied = exported[1:]
                    for values in ((audio_out, *native), exported):
                        for value, shape in zip(values, OUTPUT_SHAPES, strict=True):
                            require(tuple(value.shape) == shape and value.dtype == torch.float32
                                    and torch.isfinite(value).all().item(), "Output ABI or finite values differ")
                            trace.update(value.numpy().tobytes())
                        require(np.array_equal(values[1].numpy(), incoming), "Physical history shift differs")
                        closure_maximum = max(closure_maximum, float(np.abs(values[0].numpy().sum(1) - expected).max()))
                    maximum = max(maximum, float((audio_out - exported[0]).abs().max()))
                    for state_index, (left, right) in enumerate(zip(native, copied, strict=True)):
                        scale = PUBLIC_FUSION_SCALE if state_index == 1 else 1.0
                        state_maximum = max(state_maximum, float((left - right).abs().max()) / scale)
                trace_hashes.append(trace.hexdigest())
            require(maximum <= 1e-5 and state_maximum <= 5e-4 and closure_maximum <= 1e-6
                    and trace_hashes[0] == trace_hashes[1], "Export copy parity, closure or exact reset replay failed")
            cases.append({"name": case["name"], "waveform_max_abs": maximum,
                          "state_max_abs_physical_units": state_maximum, "closure_max_abs": closure_maximum,
                          "exact_reset_replay": True, "trace_sha256": trace_hashes[0]})
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and model_state_sha256(model) == model_state_sha256(wrapper.model) == state_hash
            and all(sha(path) == digest for path, digest in plan["source_bindings"].items()),
            "Inputs, model bytes, RNG or CPU scope changed")
    result = {"status": "pass", "plan_sha256": args.plan_sha256, "step": plan["expected_step"],
              "model_state_sha256": state_hash, "inference_resume_outputs_and_states_bit_exact": True,
              "cases": cases, "source_bindings_unchanged": True, "cuda_initialized": False,
              "optimizer_instances": 0, "training_updates_executed": 0, "onnx_exported": False,
              "onnx_parity_qualified": False, "native_host_qualified": False}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
