"""CPU ONNX export for one identified hop128 checkpoint and its quality evidence.

Requires a bound execution plan. Import is stdlib-only. Actual ONNX parity
does not establish a plugin queue, deadline performance or retained quality.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from research.direct.latency58_checkpoint import require, sha

ROOT = Path(__file__).resolve().parents[2]


def metadata(plan, model, step, fingerprint, shapes):
    from research.direct.latency58_asymmetric import VERSION, AsymmetricState

    return {
        "hs_tasnet.kind": "cropped1024_asymmetric_hop128", "hs_tasnet.mode": "streaming",
        "hs_tasnet.architecture_version": VERSION, "hs_tasnet.state_family": VERSION,
        "hs_tasnet.state_interchangeable_with_old_ola512": "false",
        "hs_tasnet.sample_rate": "44100", "hs_tasnet.hop_samples": "128",
        "hs_tasnet.analysis_fft_samples": "1024", "hs_tasnet.carrier_fft_samples": "1024",
        "hs_tasnet.synthesis_fft_samples": "1024", "hs_tasnet.spectral_mask_bins": "513",
        "hs_tasnet.spectral_output_crop": "[768,1024]", "hs_tasnet.synthesis_frame_samples": "256",
        "hs_tasnet.waveform_decoder_samples": "256", "hs_tasnet.analysis_history_samples": "896",
        "hs_tasnet.graph_output_delay_samples": "128", "hs_tasnet.alignment_samples": "128",
        "hs_tasnet.future_context_samples": "128", "hs_tasnet.future_callbacks_beyond_received_input": "0",
        "hs_tasnet.flush_required": "true", "hs_tasnet.flush_hops": "1",
        "hs_tasnet.initial_state": "all_zeros", "hs_tasnet.preroll": "discard_first_output_hop_after_reset",
        "hs_tasnet.external_host_queue_implemented": "false",
        "hs_tasnet.intended_external_host_queue_samples": "128", "hs_tasnet.intended_total_latency_samples": "256",
        "hs_tasnet.graph_qualified": "false", "hs_tasnet.native_host_timing_qualified": "false",
        "hs_tasnet.output_policy": "complete deployed four stems; Other = previous physical mixture - sum(unchanged DBV), once",
        "hs_tasnet.source_order": "drums,bass,vocals,other",
        "hs_tasnet.state_names": json.dumps(AsymmetricState._fields), "hs_tasnet.state_shapes": json.dumps(shapes),
        "hs_tasnet.public_fusion_state_scale": str(2.0 ** -18),
        "hs_tasnet.output_source_scales": json.dumps(model.output_source_scales.tolist()),
        "hs_tasnet.checkpoint_sha256": plan["checkpoint"]["sha256"],
        "hs_tasnet.model_state_sha256": fingerprint,
        "hs_tasnet.training_plan_sha256": model.provenance["training_plan_sha256"],
        "hs_tasnet.snapshot_step": str(step), "hs_tasnet.training_updates": str(model.provenance["training_updates"]),
        "hs_tasnet.exporter_sha256": sha(__file__),
        "hs_tasnet.model_source_sha256": sha(ROOT / "research/direct/latency58_asymmetric.py"),
        "hs_tasnet.export_copy_source_sha256": sha(ROOT / "research/direct/latency58_asymmetric_onnx.py"),
        "hs_tasnet.export_helpers_sha256": sha(ROOT / "export_onnx.py"),
        "hs_tasnet.fft_implementation": "DFT1024; explicit513 Hermitian endpoints/interior; inverse1024 then crop768:1024",
        "hs_tasnet.external_data": "false",
        "hs_tasnet.analysis_window": "Wang2021 K1024 M128 d0",
        "hs_tasnet.spectral_synthesis_window": "matched asymmetric pair",
        "hs_tasnet.waveform_synthesis_window": "periodic Hann256",
        "hs_tasnet.analysis_synthesis_product": "periodic Hann256; hop128 overlap unity",
        "hs_tasnet.asymmetric_training_updates": str(model.provenance["asymmetric_training_updates"]),
    }


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Export plan changed")
    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "latency58-asymmetric-onnx-export-plan-v1" and Path.cwd() == ROOT,
            "Export schema or working directory differs")
    bindings = plan["source_bindings"]
    for relative in ("research/direct/export_latency58_asymmetric.py", "research/direct/latency58_asymmetric.py",
                     "research/direct/latency58_asymmetric_checkpoint.py", "research/direct/latency58_evaluate.py",
                     "research/direct/latency58_asymmetric_onnx.py", "research/direct/latency58.py",
                     "research/direct/latency58_gpu.py", "research/direct/latency58_encoder_window.py",
                     "research/direct/latency58_checkpoint.py", "export_onnx.py"):
        require(str(ROOT / relative) in bindings, "Missing export source binding: " + relative)
    require(all(sha(path) == digest for path, digest in bindings.items())
            and bindings.get(plan["checkpoint"]["path"]) == plan["checkpoint"]["sha256"]
            and bindings.get(plan["parent_checkpoint"]["path"]) == plan["parent_checkpoint"]["sha256"],
            "Export input or checkpoint identity differs")
    require(plan["quality_evidence"] and all(sha(row["path"]) == row["sha256"]
            and bindings.get(row["path"]) == row["sha256"] for row in plan["quality_evidence"]),
            "Bind the completed quality evidence for this checkpoint")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(name) == "1" for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CUDA-hidden CPU1")
    require(type(plan["verify_hops"]) is int and plan["verify_hops"] >= 8
            and all(bindings.get(row["path"]) == row["sha256"] for row in plan["verify_audio"]),
            "Verification geometry or audio binding differs")
    output = Path(plan["output"])
    require(output.is_absolute() and output.suffix == ".onnx" and output.parent.is_dir(), "Output directory must already exist")
    result_path, failed_path = output.with_suffix(".verification.json"), output.with_suffix(".failed-verification.json")
    failed_graph = output.with_suffix(".failed.onnx")
    require(all(not path.exists() and not path.is_symlink() for path in (output, result_path, failed_path, failed_graph)),
            "Preserve existing outputs")
    from research.direct.train_latency58 import disk_bytes
    phase = ROOT / "research/direct/runs/latency58"
    require(output.is_relative_to(phase), "Keep the graph inside the measured artifact allowance")
    before_bytes = disk_bytes(phase)
    require(before_bytes + 240_000_000 < plan["artifact_allowance_bytes"], "No room for temporary and verified graph")
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    from research.direct.latency58_asymmetric_checkpoint import load_model_state, make_model
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_asymmetric_onnx import (
        INPUT_NAMES, INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES, STATE_SHAPES, make_export_copy, verify_onnx,
    )
    versions = {"torch": torch.__version__, "numpy": np.__version__, "onnx": onnx.__version__,
                "onnxruntime": ort.__version__, "soundfile": sf.__version__}
    require(versions == plan["runtime_versions"] and not torch.cuda.is_initialized(), "CPU runtime versions differ")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    model = make_model(plan["parent_checkpoint"])
    step = load_model_state(model, plan["checkpoint"])
    fingerprint = model_state_sha256(model)
    require(step == plan["step"] and fingerprint == plan["model_state_sha256"], "Export checkpoint state differs")
    rng = torch.get_rng_state().clone()
    wrapper = make_export_copy(model)
    props = metadata(plan, model, step, fingerprint, STATE_SHAPES)
    descriptor, name = tempfile.mkstemp(prefix=output.stem + ".", suffix=".pending.onnx", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    started = time.monotonic()
    report = {"schema": "latency58-asymmetric-onnx-verification-v1", "plan_sha256": args.plan_sha256,
              "checkpoint": plan["checkpoint"], "model_state_sha256": fingerprint, "step": step,
              "model_provenance": model.provenance,
              "metadata": props, "runtime_versions": versions, "quality_evidence": plan["quality_evidence"],
              "source_bindings": bindings, "native_host_qualified": False, "host_queue_implemented": False,
              "quality_retention_decision": None, "training_updates_executed": 0}
    try:
        inputs = (torch.zeros(INPUT_SHAPES[0]), *model.initial_state(1))
        with torch.inference_mode():
            example = wrapper(*inputs)
            torch.onnx.export(wrapper, inputs, str(temporary), export_params=True, opset_version=17,
                              do_constant_folding=True, input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                              dynamo=False, external_data=False)
        graph = onnx.load(str(temporary), load_external_data=False)
        require(tuple(value.name for value in graph.graph.output) == OUTPUT_NAMES
                and not any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer),
                "Graph output order or self-contained storage differs")
        for value, tensor, shape in zip(graph.graph.output, example, OUTPUT_SHAPES, strict=True):
            require(tuple(tensor.shape) == shape, "Export-copy shape differs")
            dims = value.type.tensor_type.shape
            dims.ClearField("dim")
            for size in shape:
                dims.dim.add().dim_value = size
        onnx.helper.set_model_props(graph, props)
        onnx.save(graph, str(temporary))
        onnx.checker.check_model(str(temporary), full_check=True)
        report["verification"] = verify_onnx(model, wrapper, temporary, hops=plan["verify_hops"],
                                             audio_paths=tuple(row["path"] for row in plan["verify_audio"]), threads=1)
        require(report["verification"]["passed"]
                and model_state_sha256(model) == model_state_sha256(wrapper.model) == fingerprint
                and torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
                and all(sha(path) == digest for path, digest in bindings.items()),
                "ONNX parity, unchanged source/tensors/RNG or CPU scope failed")
        report.update(status="passed_cpu_numerical_verification_only", onnx_sha256=sha(temporary),
                      onnx_bytes=temporary.stat().st_size, elapsed_seconds=time.monotonic() - started,
                      source_bindings_unchanged=True)
        os.link(temporary, output)
        write_new(result_path, report)
        print(json.dumps({"status": report["status"], "output": str(output), "onnx_sha256": report["onnx_sha256"]}))
    except Exception as error:
        report.update(status="failed_not_qualified", error=repr(error), elapsed_seconds=time.monotonic() - started)
        if temporary.stat().st_size:
            os.link(temporary, failed_graph)
        write_new(failed_path, report)
        raise
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
