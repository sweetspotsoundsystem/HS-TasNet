"""Screen dynamic per-channel U8U8 inference on the preserved C204 graph."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, verify_inputs


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1")
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import torch
    from research.direct.check_latency58_fused_gru import session_for, benchmark
    from research.direct.latency58_asymmetric_onnx import INPUT_NAMES, OUTPUT_NAMES, _initial_states
    from research.direct.latency58_conv_gemm import convert
    from research.direct.latency58_direct_sdr import objective
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_path = PHASE / "full-magnitude-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    counted_before = require_space(source, 405_000_000)
    original_path = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/model/model.onnx")
    original_sha = sha(original_path)
    require(original_sha == "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3", "C204 graph changed")
    out = PHASE / "m4-int8-screen-002"
    require(not out.exists(), "Preserve quantization screens")
    bindings = {**source["source_bindings"], str(source_path): sha(source_path), str(original_path): original_sha,
                str(Path(__file__).resolve()): sha(__file__),
                str(ROOT / "research/direct/latency58_conv_gemm.py"): sha(ROOT / "research/direct/latency58_conv_gemm.py"),
                str(ROOT / "research/direct/check_latency58_fused_gru.py"): sha(ROOT / "research/direct/check_latency58_fused_gru.py")}
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "quantization": "dynamic U8 activations/U8 per-output-channel weights",
          "weight_symmetric": False, "reduce_range": False, "quantized_ops": ["MatMul"],
          "integer_projection_count": 9, "final_waveform_decoder_kept_fp32": True,
          "why_u8u8": "Avoid U8S8 saturation on this non-VNNI x86 screening host; Arm execution still needs measurement.",
          "cpu_only": True, "training_crop_indices": [source["config"]["data_start"], source["config"]["data_start"] + 1],
          "validation_used": False, "native_plugin_replaced": False, "counted_bytes_before": counted_before})
    began = time.monotonic()
    graph = onnx.load(original_path, load_external_data=False)
    require(not any(t.external_data for t in graph.graph.initializer), "Require a self-contained source graph")
    conversion = convert(graph, ["/conv_encode/Conv", "/basis_to_embed/Conv"])
    quantized_path = out / "quantized.onnx"
    quantize_dynamic(graph, quantized_path, op_types_to_quantize=["MatMul"], per_channel=True,
        reduce_range=False, weight_type=QuantType.QUInt8, use_external_data_format=False,
        extra_options={"WeightSymmetric": False, "MatMulConstBOnly": True})
    quantized = onnx.load(quantized_path, load_external_data=False)
    # Inherited qualification/export identities describe the float graph.
    # Retain them only as source metadata; this approximate artifact has its
    # own unqualified identity and cannot satisfy the plugin's metadata lock.
    metadata = {"source." + item.key: item.value for item in quantized.metadata_props}
    metadata.update({"hs_tasnet.runtime_variant": "c204-hop128-dynamic-u8u8-v1",
                     "hs_tasnet.parent_fp32_graph_sha256": original_sha,
                     "hs_tasnet.native_host_qualified": "false",
                     "hs_tasnet.quantized_full14_quality_qualified": "false"})
    onnx.helper.set_model_props(quantized, metadata)
    onnx.checker.check_model(quantized, full_check=True)
    node_counts = {kind: sum(n.op_type == kind for n in quantized.graph.node)
                   for kind in sorted({n.op_type for n in quantized.graph.node})}
    remaining = [n for n in quantized.graph.node if n.op_type in ("MatMul", "Gemm", "Conv")]
    require(node_counts.get("MatMulInteger", 0) == 9 and len(remaining) == 1
            and remaining[0].op_type == "MatMul" and remaining[0].name == "/MatMul"
            and list(remaining[0].input) == ["/Transpose_6_output_0", "/Flatten_output_0"],
            "Require nine integer projections and the unchanged float waveform decoder")
    data = {"original": original_path.read_bytes(), "u8u8": quantized.SerializeToString()}
    require(len(data["u8u8"]) < 35_000_000, "Quantized graph exceeds its reservation")
    with quantized_path.open("wb") as stream:
        stream.write(data["u8u8"])
        stream.flush()
        os.fsync(stream.fileno())
    del graph, quantized
    print(json.dumps({"event": "quantized", "bytes": len(data["u8u8"]), "matrix_count": node_counts["MatMulInteger"]}), flush=True)

    def stream_audio(session, audio):
        require(audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] % 128 == 0, "Audio shape differs")
        states, output, closure = _initial_states(), [], 0.
        count = audio.shape[1]
        padded = np.pad(audio, ((0, 0), (0, 128)))
        for offset in range(0, count + 128, 128):
            values = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES,
                [np.ascontiguousarray(padded[None, :, offset:offset + 128]), *states], strict=True)))
            require(all(np.isfinite(value).all() for value in values), "Nonfinite quantized output/state")
            states = values[1:]
            output.append(values[0])
        result = np.concatenate(output, axis=-1)[..., 128:128 + count]
        closure = float(np.abs(result.sum(axis=1) - audio[None]).max())
        require(closure < 2e-6, "Quantized mixture closure failed")
        return result, closure

    # A zero input must execute finite recurrent states, including the flush.
    for label in data:
        session = session_for(data[label])
        stream_audio(session, np.zeros((2, 16 * 128), dtype=np.float32))
        del session
    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    config = source["config"]
    _, tracks, _, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=read(PRODUCTION / "full_config.json"))
    tracks = select_tracks(tracks, source["training_selection"])
    dataset = production.CounterAddressedCropDataset(tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"], vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=config["data_start"] + 2)
    examples = [dataset[index] for index in range(config["data_start"], config["data_start"] + 2)]
    references = torch.stack([truth[..., source["warmup_samples"]:] for _, truth in examples])
    mixtures = torch.stack([audio[..., source["warmup_samples"]:] for audio, _ in examples])
    predictions, diagnostics, closures = {}, {}, {}
    for label in data:
        session = session_for(data[label])
        rendered = [stream_audio(session, audio.numpy()) for audio, _ in examples]
        del session
        closures[label] = max(row[1] for row in rendered)
        predictions[label] = np.concatenate([row[0][..., source["warmup_samples"]:] for row in rendered])
        estimate = torch.from_numpy(predictions[label])
        terms = objective(estimate, estimate, references, mixtures)
        diagnostics[label] = {"training_sdr_db": -float(terms.negative_sdr_db),
                              "per_stem_training_sdr_db": (-terms.per_stem_negative_sdr_db).tolist()}
        print(json.dumps({"event": "training_crop_diagnostic", "variant": label, **diagnostics[label]}), flush=True)
    delta = diagnostics["u8u8"]["training_sdr_db"] - diagnostics["original"]["training_sdr_db"]
    per_stem_delta = [a - b for a, b in zip(diagnostics["u8u8"]["per_stem_training_sdr_db"],
                                          diagnostics["original"]["per_stem_training_sdr_db"], strict=True)]
    timing_audio = examples[0][0][:, :128 * 320].numpy()
    timings = []
    for cycle, order in enumerate((("original", "u8u8"), ("u8u8", "original"), ("u8u8", "original"), ("original", "u8u8"))):
        for label in order:
            session = session_for(data[label])
            row = {"cycle": cycle + 1, "variant": label, **benchmark(session, timing_audio, 64, 256)}
            del session
            timings.append(row)
            print(json.dumps({k: row[k] for k in ("cycle", "variant", "p50_ms", "p95_ms", "p99_ms")}), flush=True)
    pooled = {label: np.asarray([t for row in timings if row["variant"] == label for t in row["times_ms"]]) for label in data}
    summary = {label: {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                      "p99_ms": float(np.percentile(values, 99)), "p50_ratio_to_original": float(np.percentile(values, 50) / np.percentile(pooled["original"], 50))}
               for label, values in pooled.items()}
    verify_inputs({"source_bindings": bindings})
    require(not torch.cuda.is_initialized(), "CPU screen initialized CUDA")
    result = {"status": "screen_complete", "original": {"path": str(original_path), "sha256": original_sha},
          "quantized": {"path": str(quantized_path), "sha256": sha(quantized_path), "bytes": quantized_path.stat().st_size},
          "node_counts": node_counts, "conv_rewrites": conversion, "training_diagnostics": diagnostics,
          "quantized_minus_original_training_sdr_db": delta, "per_stem_training_sdr_delta_db": per_stem_delta,
          "closure_max_abs": closures, "summary": summary, "timing_cycles": timings,
          "native_plugin_unchanged": True, "source_bindings_unchanged": True, "gpu_used": False,
          "target_m4_qualified": False, "full14_quality_measured": False,
          "elapsed_seconds": time.monotonic() - began, "counted_bytes_after": require_space(source, 370_000_000),
          "limitations": "Only two training crops; approximate integer inference needs complete held-out scoring and actual M4 CPU1 plugin timing. Linux timing was recorded during GPU training."}
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "quantized", "quantized_minus_original_training_sdr_db", "summary")}), flush=True)


if __name__ == "__main__":
    main()
