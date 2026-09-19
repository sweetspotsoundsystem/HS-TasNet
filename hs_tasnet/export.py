"""Verified, fixed-geometry ONNX export for the current eight-state model.

FP32 is the default. The optional integer variant retains the current
seventeen-product deployment transform and verifies against independently
reconstructed integer arithmetic. Newly exported graphs do not inherit the
released graph's quality or host-timing measurements.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from ._export.fp32 import TOLERANCES, interface, make_export_copy
from ._export.helpers import require, sha, state_sha256

__all__ = ["export_model", "export_streaming_model", "build_fp32", "verify_onnx", "interface"]


def build_fp32(model, *, checkpoint_sha256=None):
    """Return a private PyTorch wrapper and a self-contained fixed-shape graph.

    Model parameters, buffers, training flags and RNG state remain unchanged.
    No analysis-window, synthesis-window or hop override is accepted.
    """
    import onnx

    contract = interface(model)
    fingerprint = state_sha256(model.state_dict())
    wrapper = make_export_copy(model)
    inputs = (torch.zeros(1, 2, 128), *wrapper.model.initial_state(1))
    with io.BytesIO() as stream, torch.inference_mode():
        outputs = wrapper(*inputs)
        require(tuple(tuple(value.shape) for value in outputs) == contract["output_shapes"],
                "Export wrapper output dimensions differ")
        torch.onnx.export(
            wrapper, inputs, stream, export_params=True, opset_version=17,
            do_constant_folding=True, input_names=list(contract["input_names"]),
            output_names=list(contract["output_names"]), dynamo=False, external_data=False,
        )
        graph = onnx.load_model_from_string(stream.getvalue())
    # The legacy exporter leaves some FFT-derived output dimensions symbolic.
    # These are fixed by the one-hop interface, not configurable dimensions.
    for value, shape in zip(graph.graph.output, contract["output_shapes"], strict=True):
        dims = value.type.tensor_type.shape
        dims.ClearField("dim")
        for size in shape:
            dims.dim.add().dim_value = size
    architecture = model.architecture_metadata
    metadata = {
        "runtime_variant": "branch-memory-fp32-v1",
        "state_family": architecture["state_family"],
        "architecture": json.dumps(architecture, sort_keys=True),
        "model_state_sha256": fingerprint,
        "sample_rate": "44100", "hop_samples": "128", "analysis_fft_samples": "1024",
        "synthesis_frame_samples": "256", "graph_output_delay_samples": "128",
        "source_order": "drums,bass,vocals,other", "external_data": "false",
        "state_names": json.dumps(contract["state_names"]),
        "state_shapes": json.dumps(contract["state_shapes"]),
        "public_fusion_state_scale": str(2.0 ** -18),
        "public_branch_memory_state_scale": str(2.0 ** -18),
        "initial_state": "all_zeros", "preroll": "discard_first_output_hop_after_reset",
        "flush_hops": "1", "native_host_qualified": "false", "quality_measured": "false",
        "output_policy": "DBV += (delayed mixture - sum(native raw4))/16; Other = delayed mixture - sum(corrected DBV)",
        "fixed_residual_share": "0.0625",
    }
    if checkpoint_sha256 is not None:
        require(isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64
                and all(c in "0123456789abcdef" for c in checkpoint_sha256),
                "checkpoint_sha256 must be a lowercase SHA-256 digest")
        metadata["checkpoint_sha256"] = checkpoint_sha256
    onnx.helper.set_model_props(graph, {"hs_tasnet." + key: value for key, value in metadata.items()})
    onnx.checker.check_model(graph, full_check=True)
    require(not any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer),
            "Export unexpectedly uses external tensor data")
    require(state_sha256(model.state_dict()) == fingerprint
            and state_sha256(wrapper.model.state_dict()) == fingerprint,
            "Export changed model tensor bytes")
    return wrapper, graph


def verify_onnx(model, path, *, hops=48, reference=None):
    """Compare ONNX and native PyTorch trajectories with independent state carry.

    Every public state is compared. Hidden-state errors are measured in decoded
    recurrent units, accounting for the public 2**-18 scale. The sequence includes
    silence, an impulse, quiet audio and a final zero hop to verify alignment.
    """
    import copy
    import onnxruntime as ort
    from .model import StreamingState

    require(isinstance(hops, int) and not isinstance(hops, bool) and hops >= 4,
            "Verification requires at least four recurrent hops")
    contract = interface(model)
    before = state_sha256(model.state_dict())
    # Evaluate a separate copy so the caller can export a model during training.
    integer_reference = reference is not None
    if reference is None:
        reference = copy.deepcopy(model).cpu().eval()
        reference.training_precision = "fp32"
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    options.add_session_config_entry("mlas.disable_kleidiai", "1")
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    require(session.get_providers() == ["CPUExecutionProvider"], "Verification requires CPU execution")
    for nodes, names, shapes in ((session.get_inputs(), contract["input_names"], contract["input_shapes"]),
                                  (session.get_outputs(), contract["output_names"], contract["output_shapes"])):
        require(tuple(node.name for node in nodes) == names
                and tuple(tuple(node.shape) for node in nodes) == shapes
                and all(node.type == "tensor(float)" for node in nodes),
                "ONNX public interface differs from the current model")
    generator = np.random.default_rng(617)
    sequence = generator.normal(0, .03, (hops, 1, 2, 128)).astype(np.float32)
    sequence[0] = 0
    sequence[1] = 0
    sequence[1, 0, :, -1] = .5
    sequence[-2] *= 1e-3
    sequence[-1] = 0
    # Verification states belong to the CPU oracle/runtime, not the caller's
    # device-bound training model. Allocate directly from the checked contract.
    native_state = tuple(torch.zeros(shape, dtype=torch.float32)
                         for shape in contract["state_shapes"])
    runtime_state = [value.numpy().copy() for value in native_state]
    maxima = {key: 0.0 for key in TOLERANCES}
    state_errors = dict.fromkeys(contract["state_names"], 0.0)
    delayed = np.zeros((1, 2, 128), np.float32)
    first = None
    with torch.inference_mode():
        for audio in sequence:
            if integer_reference:
                reference_outputs = reference(torch.from_numpy(audio), *native_state)
                native_state = reference_outputs[1:]
            else:
                native = reference.render(torch.from_numpy(audio), StreamingState(*native_state))
                reference_outputs = (native.deployed, *native.state)
                native_state = native.state
            runtime = session.run(list(contract["output_names"]),
                                  dict(zip(contract["input_names"], (audio, *runtime_state), strict=True)))
            require(all(np.isfinite(value).all() for value in runtime), "ONNX returned non-finite values")
            expected = tuple(value.numpy() for value in reference_outputs)
            difference = runtime[0].astype(np.float64) - expected[0]
            maxima["waveform_max_abs"] = max(maxima["waveform_max_abs"], float(np.abs(difference).max()))
            maxima["stem_callback_rms"] = max(maxima["stem_callback_rms"],
                float(np.sqrt(np.mean(difference ** 2, axis=(-2, -1))).max()))
            closure = float(np.abs(runtime[0].sum(axis=1) - delayed).max())
            maxima["reconstruction_max_abs"] = max(maxima["reconstruction_max_abs"], closure)
            for name, actual, wanted in zip(contract["state_names"], runtime[1:], expected[1:], strict=True):
                error = float(np.abs(actual.astype(np.float64) - wanted).max())
                if name in {"fusion_hidden", "spec_memory_hidden", "waveform_memory_hidden"}:
                    error /= 2.0 ** -18
                state_errors[name] = max(state_errors[name], error)
                maxima["state_max_abs_decoded_units"] = max(maxima["state_max_abs_decoded_units"], error)
            if first is None:
                first = [value.copy() for value in runtime]
            runtime_state = runtime[1:]
            delayed = audio
        # A reset must discard every state and reproduce the initial callback.
        zero_state = [np.zeros(shape, dtype=np.float32) for shape in contract["state_shapes"]]
        replay = session.run(list(contract["output_names"]),
                             dict(zip(contract["input_names"], (sequence[0], *zero_state), strict=True)))
        require(all(np.array_equal(a, b) for a, b in zip(first, replay, strict=True)),
                "Reset did not reproduce the initial output and all states")
    require(state_sha256(model.state_dict()) == before, "Verification changed source model tensors")
    for name, bound in TOLERANCES.items():
        require(maxima[name] <= bound,
                f"ONNX parity failed for {name}: {maxima[name]} exceeds {bound}")
    return {"passed": True, "hops": hops, "metrics": maxima, "tolerances": dict(TOLERANCES),
            "state_errors_decoded_units": state_errors, "reset_replay_exact": True,
            "reference": ("independently reconstructed integer weights and PyTorch integer arithmetic"
                          if integer_reference else "independently carried native PyTorch states"),
            "provider": "CPUExecutionProvider"}


def export_model(model, path, *, verify_hops=48, checkpoint_sha256=None, variant="fp32"):
    """Export and verify bytes before publishing a new ONNX file and report.

    Existing output files are preserved. Returns the verification report, also
    written beside the model as ``<name>.verification.json``.
    """
    import onnx

    require(variant in {"fp32", "integer"}, "variant must be 'fp32' or 'integer'")
    require(isinstance(verify_hops, int) and not isinstance(verify_hops, bool) and verify_hops >= 4,
            "Verification requires at least four recurrent hops")
    path = Path(path)
    report_path = path.with_suffix(".verification.json")
    for destination in (path, report_path):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference, conversion = None, None
    if variant == "integer":
        from ._export.deployment import build
        reference, graph, conversion = build(model, checkpoint_sha256=checkpoint_sha256)
    else:
        wrapper, graph = build_fp32(model, checkpoint_sha256=checkpoint_sha256)
        del wrapper
    temporary_paths = []
    published = []
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".export-", suffix=".onnx", delete=False) as stream:
            temporary = Path(stream.name)
        temporary_paths.append(temporary)
        onnx.save_model(graph, temporary)
        del graph
        verification = verify_onnx(model, temporary, hops=verify_hops, reference=reference)
        report = {"status": "pass", "onnx_sha256": sha(temporary),
                  "model_state_sha256": state_sha256(model.state_dict()),
                  "variant": variant, "verification": verification,
                  "quality_measured": False, "native_host_qualified": False}
        if checkpoint_sha256 is not None:
            report["checkpoint_sha256"] = checkpoint_sha256
        if conversion is not None:
            report["integer_conversion"] = conversion
        report = json.loads(json.dumps(report, allow_nan=False))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".verification-", delete=False) as stream:
            report_temporary = Path(stream.name)
            temporary_paths.append(report_temporary)
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        for source, destination in ((temporary, path), (report_temporary, report_path)):
            os.link(source, destination)
            published.append(destination)
        return report
    except BaseException:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


# Preserve the descriptive entry point used by the export command.
export_streaming_model = export_model
