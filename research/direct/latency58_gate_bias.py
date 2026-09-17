"""Copy an asymmetric model with one offset per GRU update gate.

This is a parameter transform, not an equivalent recurrent time conversion.
No model, checkpoint or device is created on import.
"""
from __future__ import annotations

import copy
import math

VERSION = "latency58-gru-update-bias-transform-v1"
DEFAULT_OFFSET = math.log(2.0)


def with_update_gate_bias(parent, *, offset=DEFAULT_OFFSET):
    """Return an isolated CPU copy, changing only bias_ih's middle third.

    For a fixed input and hidden state, adding log(2) to the update preactivation
    maps z to 2z/(1+z). Nonlinear recurrent trajectories are not equivalent.
    Offset addition is computed in FP64 and rounded once to the stored FP32.
    The existing hidden-to-hidden bias is preserved; adding to both would
    inadvertently double the requested offset.
    """
    import torch
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_evaluate import model_state_sha256

    if (type(parent) is not Latency58AsymmetricModel or type(offset) not in (int, float)
            or not math.isfinite(offset) or not 0 <= offset <= DEFAULT_OFFSET
            or any(m.training for m in parent.modules())
            or any(t.device.type != "cpu" or t.dtype != torch.float32 or not bool(torch.isfinite(t).all())
                   for t in parent.state_dict().values())):
        raise ValueError("Use an evaluated CPU FP32 asymmetric parent and an offset in [0, log(2)]")
    recurrent = parent.fusion_branch
    if (type(recurrent) is not torch.nn.GRU or recurrent.num_layers != 2 or not recurrent.bias
            or recurrent.bidirectional or recurrent.hidden_size != 1000
            or len(list(parent.parameters())) != 21 or len(list(parent.buffers())) != 6):
        raise ValueError("The authenticated two-layer GRU inventory differs")
    before = model_state_sha256(parent)
    result = copy.deepcopy(parent)
    expected_names = {f"fusion_branch.bias_ih_l{layer}" for layer in range(2)}
    changed = []
    with torch.no_grad():
        for layer in range(2):
            name = f"fusion_branch.bias_ih_l{layer}"
            bias = dict(result.named_parameters())[name]
            hidden = recurrent.hidden_size
            bias[hidden:2 * hidden].copy_((bias[hidden:2 * hidden].double() + offset).float())
        original, transformed = parent.state_dict(), result.state_dict()
        if set(original) != set(transformed):
            raise RuntimeError("Parameter transform changed the tensor inventory")
        for name, old in original.items():
            new = transformed[name]
            expected = old.clone()
            if name in expected_names:
                hidden = recurrent.hidden_size
                expected[hidden:2 * hidden] = (old[hidden:2 * hidden].double() + offset).float()
            if not torch.equal(new, expected):
                raise RuntimeError("Unexpected transformed tensor: " + name)
            if not torch.equal(new, old):
                changed.append(name)
    if (set(changed) != (expected_names if offset else set())
            or model_state_sha256(parent) != before
            or result.architecture_metadata != parent.architecture_metadata):
        raise RuntimeError("The transform modified its parent, inventory or inference geometry")
    result.provenance = {**copy.deepcopy(parent.provenance),
                         "parameter_transform_version": VERSION,
                         "parameter_transform_parent_provenance": copy.deepcopy(parent.provenance),
                         "parameter_transform_parent_state_sha256": before,
                         "gru_update_bias_offset": float(offset),
                         "parameter_transform_training_updates": 0,
                         "parameter_transform_changed_tensors": sorted(changed)}
    return result
