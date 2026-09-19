"""Numerical and publication checks for the maintained eight-state exporter."""
from __future__ import annotations

import json
from types import MethodType

import numpy as np
import pytest
import torch

from hs_tasnet.model import StreamingHSTasNet
from hs_tasnet.export import export_model, interface
from hs_tasnet._export.fp32 import make_export_copy
from hs_tasnet._export.helpers import state_sha256


def active_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(617)
        model = StreamingHSTasNet()
        # Exercise trained paths whose residual projection starts at zero.
        with torch.no_grad():
            for parameter in model.parameters():
                if torch.count_nonzero(parameter) == 0:
                    parameter.normal_(std=.001)
    return model


@pytest.mark.parametrize("variant", ("fp32", "integer"))
def test_export_matches_reference_trajectory_without_mutating_source(tmp_path, variant):
    import onnx
    torch.set_num_threads(1)
    model = active_model().train()
    # Emulate a device-bound caller on CPU-only CI: its state allocator may be
    # used on its own device, but must not be asked for the verifier's CPU state.
    # A real CUDA model enforces the same boundary for device="cpu" requests.
    original_initial_state = type(model).initial_state
    def device_bound_state(self, batch_size, *, device=None):
        if device is not None:
            raise ValueError("Caller model cannot allocate cross-device verification state")
        return original_initial_state(self, batch_size)
    model.initial_state = MethodType(device_bound_state, model)
    model.spec_memory.eval()
    flags = [module.training for module in model.modules()]
    before = state_sha256(model.state_dict())
    rng = torch.get_rng_state().clone()
    path = tmp_path / "current.onnx"
    report = export_model(model, path, verify_hops=12, variant=variant)
    assert report["status"] == "pass"
    assert report["verification"]["passed"]
    assert len(report["verification"]["state_errors_decoded_units"]) == 8
    assert report["verification"]["reset_replay_exact"]
    assert state_sha256(model.state_dict()) == before
    assert [module.training for module in model.modules()] == flags
    assert torch.equal(torch.get_rng_state(), rng)
    graph = onnx.load(path)
    if variant == "integer":
        assert len(report["integer_conversion"]["projections"]) == 17
        assert sum(node.op_type == "MatMulInteger" for node in graph.graph.node) == 17
        assert "independently reconstructed" in report["verification"]["reference"]
    contract = interface(model)
    assert tuple(value.name for value in graph.graph.input) == contract["input_names"]
    assert tuple(value.name for value in graph.graph.output) == contract["output_names"]
    assert tuple(tuple(d.dim_value for d in value.type.tensor_type.shape.dim)
                 for value in graph.graph.output) == contract["output_shapes"]
    assert json.loads(path.with_suffix(".verification.json").read_text()) == report
    with pytest.raises(FileExistsError):
        export_model(model, path, verify_hops=4)


def test_wrapper_carries_all_states_for_nonzero_initial_context():
    torch.set_num_threads(1)
    model = active_model().eval()
    wrapper = make_export_copy(model)
    generator = torch.Generator().manual_seed(11)
    values = []
    for name, value in zip(interface(model)["state_names"], model.initial_state(1), strict=True):
        scale = 2**-21 if name in {"fusion_hidden", "spec_memory_hidden", "waveform_memory_hidden"} else .001
        values.append(torch.randn(value.shape, generator=generator) * scale)
    state = type(model.initial_state(1))(*values)
    chunk = torch.randn(1, 2, 128, generator=generator) * .03
    with torch.inference_mode():
        expected = model.render(chunk, state)
        actual = wrapper(chunk, *state)
    for a, b in zip(actual, (expected.deployed, *expected.state), strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


def test_export_rejects_geometry_override_and_nonfinite_weights():
    torch.set_num_threads(1)
    model = active_model()
    model.conv_encode.stride = (64,)
    with pytest.raises(ValueError, match="geometry"):
        make_export_copy(model)
    model.conv_encode.stride = (128,)
    with torch.no_grad():
        model.conv_encode.weight[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite FP32"):
        make_export_copy(model)


def test_failed_verification_does_not_publish_partial_files(tmp_path, monkeypatch):
    import hs_tasnet.export as exporting
    torch.set_num_threads(1)
    model = active_model()
    def fail(*args, **kwargs):
        raise ValueError("intentional numerical mismatch")
    monkeypatch.setattr(exporting, "verify_onnx", fail)
    with pytest.raises(ValueError, match="numerical mismatch"):
        export_model(model, tmp_path / "rejected.onnx", verify_hops=4)
    assert list(tmp_path.iterdir()) == []
