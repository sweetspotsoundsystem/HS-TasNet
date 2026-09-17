"""Locate long-stream differences for an audited quality-recovery candidate.

All trajectories start at physical sample zero. This diagnostic preserves the
failed native result and its tolerance; it does not grant qualification.
"""
from __future__ import annotations

import argparse
import importlib
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
    require(plan["schema"] == "latency58-sdr-candidate-long-parity-plan-v1"
            and all(sha(p) == h for p, h in plan["source_bindings"].items()), "Diagnostic inputs differ")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1 with CUDA hidden")
    bindings = plan["source_bindings"]
    for item in (plan["checkpoint"], plan["training_plan"], plan["graph"], plan["mixture"],
                 plan["failed_native_result"], plan["failed_native_execution"], *plan["references"].values()):
        require(bindings.get(item["path"]) == item["sha256"], "Unbound diagnostic input")
    failed = json.loads(Path(plan["failed_native_execution"]["path"]).read_text())
    require(failed["actual_exit_code"] == 3 and not failed["timed_out"]
            and failed["source_bindings_unchanged"], "Preserve the actual native qualification failure")
    from research.direct.train_latency58 import disk_bytes
    require(sum(disk_bytes(Path(p)) for p in plan["counted_roots"]) + 60_000_000
            < plan["stop_counted_bytes"], "Insufficient allowance for eight diagnostic WAVs")
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / "result.json").exists(), "Preserve previous diagnostic")
    import numpy as np
    import torch
    import soundfile as sf
    import onnxruntime as ort
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, TOLERANCES

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    from research.direct.export_latency58_sdr_candidate_v5 import CANDIDATE_FAMILIES
    require(plan["model_kind"] in CANDIDATE_FAMILIES, "Unknown model kind")
    family = CANDIDATE_FAMILIES[plan["model_kind"]]
    root = Path(__file__).resolve().parents[2]
    evaluator_name = "evaluate_latency58_" + family
    for name in ("export_latency58_sdr_candidate_v5.py", evaluator_name + ".py",
                 "latency58_" + family + "_checkpoint.py", Path(__file__).name):
        source = root / "research/direct" / name
        require(bindings.get(str(source)) == sha(source), "Unbound diagnostic loader source")
    load_evaluation_model = importlib.import_module("research.direct." + evaluator_name).load_evaluation_model
    model, receipt = load_evaluation_model(plan)
    require(receipt["step"] == plan["step"]
            and model_state_sha256(model) == plan["model_state_sha256"], "Diagnostic model differs")
    rng = torch.get_rng_state().clone()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(plan["graph"]["path"], sess_options=options, providers=["CPUExecutionProvider"])
    require(session.get_providers() == ["CPUExecutionProvider"], "Use only CPU execution")
    frames, start, end = (plan[k] for k in ("prefix_frames", "capture_start", "capture_end"))
    require(0 <= start < end <= frames and frames % 128 == 0, "Invalid physical geometry")
    mixture, rate = sf.read(plan["mixture"]["path"], frames=frames, dtype="float32", always_2d=True)
    require(rate == 44100 and mixture.shape == (frames, 2), "Mixture shape/rate differs")
    grouped = []
    for stem in ("drums", "bass", "vocals", "other"):
        audio, rate = sf.read(plan["references"][stem]["path"], dtype="float32", always_2d=True)
        require(rate == 44100 and audio.shape == (end - start, 2), "Grouped reference geometry differs")
        grouped.append(audio.T)
    grouped = np.stack(grouped)
    captured = {name: np.empty_like(grouped) for name in ("literal_torch", "ort")}
    state = model.initial_state(1)
    ort_state = [value.numpy().copy() for value in state]
    pairs = (("literal_torch", "ort"), ("literal_torch", "grouped"), ("ort", "grouped"))
    comparisons = {f"{a}_vs_{b}": {"max_abs": np.zeros(4), "max_callback_rms": np.zeros(4),
                    "worst_callback_physical_start": [None] * 4, "squared_error": np.zeros(4)} for a, b in pairs}
    trajectory_max, closure_max, state_max = 0.0, 0.0, np.zeros(4)
    recovered = 0
    began = time.monotonic()
    with torch.inference_mode():
        for index in range(frames // 128 + 1):
            chunk = np.ascontiguousarray(mixture[index * 128:(index + 1) * 128].T[None]) \
                if index < frames // 128 else np.zeros((1, 2, 128), np.float32)
            prediction, state = model.forward_chunk(torch.from_numpy(chunk), state) \
                if index < frames // 128 else model.flush(state)
            values = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *ort_state], strict=True)))
            ort_state = values[1:]
            native = prediction.numpy()
            require(np.isfinite(native).all() and all(np.isfinite(v).all() for v in values), "Nonfinite trajectory")
            require(np.array_equal(state.audio_history.numpy(), ort_state[0]), "History differs")
            trajectory_max = max(trajectory_max, float(np.max(np.abs(native.astype(np.float64) - values[0]))))
            for i, (left, right) in enumerate(zip(state, ort_state, strict=True)):
                scale = 2.0 ** 18 if i == 1 else 1.0
                state_max[i] = max(state_max[i], float(np.max(np.abs(left.numpy().astype(np.float64) - right))) * scale)
            physical = (index - 1) * 128
            expected = mixture[physical:physical + 128].T if physical >= 0 else np.zeros((2, 128))
            for value in (native[0], values[0][0]):
                closure_max = max(closure_max, float(np.max(np.abs(value.astype(np.float64).sum(0) - expected))))
            left, right = max(start, physical), min(end, physical + 128)
            if left < right:
                target = slice(left - start, right - start)
                source = slice(left - physical, right - physical)
                rows = {"literal_torch": native[0, ..., source], "ort": values[0][0, ..., source],
                        "grouped": grouped[..., target]}
                for name in captured:
                    captured[name][..., target] = rows[name]
                for a, b in pairs:
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
            if index and index % 4096 == 0:
                print(json.dumps({"physical_input_frames": min(frames, index * 128),
                                  "elapsed_seconds": time.monotonic() - began}), flush=True)
    require(recovered == end - start and torch.equal(rng, torch.get_rng_state())
            and not torch.cuda.is_initialized() and model_state_sha256(model) == plan["model_state_sha256"],
            "Physical recovery, model or CPU state differs")
    audio_files = {}
    for name, values in captured.items():
        directory = out / name
        directory.mkdir()
        audio_files[name] = {}
        for i, stem in enumerate(("drums", "bass", "vocals", "other")):
            p = directory / f"estimate-{stem}.wav"
            sf.write(p, values[i].T, 44100, subtype="FLOAT")
            audio_files[name][stem] = {"path": str(p), "sha256": sha(p)}
    for row in comparisons.values():
        row["overall_rms"] = np.sqrt(row.pop("squared_error") / (2 * recovered)).tolist()
        row["within_unchanged_music_tolerance"] = bool(np.max(row["max_abs"]) <= TOLERANCES["waveform_max_abs"]
            and np.max(row["max_callback_rms"]) <= TOLERANCES["stem_callback_rms"])
        row["max_abs"] = row["max_abs"].tolist()
        row["max_callback_rms"] = row["max_callback_rms"].tolist()
    require(all(sha(p) == h for p, h in plan["source_bindings"].items()), "Diagnostic inputs changed")
    result = {"status": "diagnostic_completed", "quality_or_native_qualified": False,
              "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
              "runtime_versions": {"torch": torch.__version__, "onnxruntime": ort.__version__},
              "checkpoint": plan["checkpoint"], "graph": plan["graph"], "comparisons": comparisons,
              "failed_native_result": plan["failed_native_result"],
              "failed_native_execution": plan["failed_native_execution"],
              "full_prefix_literal_torch_vs_ort_max_abs": trajectory_max,
              "full_prefix_state_max_abs_decoded_units": state_max.tolist(),
              "full_prefix_mixture_closure_max_abs": closure_max,
              "captured_samples": recovered, "native_output_gains_unchanged": True,
              "tolerances": TOLERANCES, "audio_files": audio_files, "elapsed_seconds": time.monotonic() - began}
    with (out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
