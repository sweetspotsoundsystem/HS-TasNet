"""In-memory rank-128 spectral-mask factorization of the preserved C204 model."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.direct.run_latency58_quality import require

RANK = 128
WEIGHT = "onnx::MatMul_363"
NODE = "/to_spec_masks/MatMul"


class FactoredMask(nn.Module):
    def __init__(self, left, right, bias):
        super().__init__()
        self.register_buffer("left", left.detach().clone())
        self.register_buffer("right", right.detach().clone())
        self.register_buffer("bias", bias.detach().clone())

    def forward(self, values):
        return F.linear(F.linear(values, self.right), self.left, self.bias)


def convert(native, graph):
    """Change the mask projection only; preserve every other node and tensor."""
    from onnx import helper, numpy_helper

    original = native.to_spec_masks
    require(type(original) is nn.Linear and original.weight.shape == (8208, 500)
            and original.bias.shape == (8208,), "Unexpected spectral head")
    initializers = {t.name: t for t in graph.graph.initializer}
    require(WEIGHT in initializers and np.array_equal(numpy_helper.to_array(initializers[WEIGHT]),
            original.weight.detach().numpy().T), "Native and ONNX spectral weights differ")
    nodes = [n for n in graph.graph.node if n.name == NODE]
    require(len(nodes) == 1 and nodes[0].op_type == "MatMul" and list(nodes[0].input)[1:] == [WEIGHT]
            and sum(WEIGHT in n.input for n in graph.graph.node) == 1, "Spectral projection boundary changed")
    original_nodes = {n.name: n.SerializeToString() for n in graph.graph.node if n.name != NODE}
    original_tensors = {t.name: t.SerializeToString() for t in graph.graph.initializer if t.name != WEIGHT}
    with torch.inference_mode():
        matrix = original.weight.double()
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        left = (u[:, :RANK] * singular[:RANK]).float().contiguous()
        right = vh[:RANK].float().contiguous()
        relative = float(torch.linalg.vector_norm(left.double() @ right.double() - matrix)
                         / torch.linalg.vector_norm(matrix))
        tail = float(torch.sqrt(singular[RANK:].square().sum() / singular.square().sum()))
        require(abs(relative - tail) < 1e-7, "Rounded factors disagree with singular-energy prediction")
    native.to_spec_masks = FactoredMask(left, right, original.bias)
    prefix = "/rank128_spec_mask/"
    replacements = [helper.make_node("MatMul", [nodes[0].input[0], prefix + "right"],
                                    [prefix + "latent"], name=prefix + "reduce"),
                    helper.make_node("MatMul", [prefix + "latent", prefix + "left"],
                                    list(nodes[0].output), name=prefix + "expand")]
    rewritten = [replacement for n in graph.graph.node
                 for replacement in (replacements if n.name == NODE else [n])]
    del graph.graph.node[:]
    graph.graph.node.extend(rewritten)
    retained = [t for t in graph.graph.initializer if t.name != WEIGHT]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend([*retained,
        numpy_helper.from_array(np.ascontiguousarray(right.numpy().T), prefix + "right"),
        numpy_helper.from_array(np.ascontiguousarray(left.numpy().T), prefix + "left")])
    require({n.name: n.SerializeToString() for n in graph.graph.node if n.name in original_nodes} == original_nodes
            and all(t.SerializeToString() == original_tensors[t.name] for t in graph.graph.initializer
                    if t.name in original_tensors), "Unrelated graph bytes changed")
    metadata = {"source." + item.key: item.value for item in graph.metadata_props}
    metadata.update({"hs_tasnet.runtime_variant": "c204-hop128-rank128-spectral-mask-v1",
                     "hs_tasnet.native_host_qualified": "false"})
    helper.set_model_props(graph, metadata)
    return {"rank": RANK, "relative_weight_error": relative,
            "original_weight_elements": original.weight.numel(),
            "factor_weight_elements": left.numel() + right.numel(),
            "factor_sha256": {name: hashlib.sha256(t.numpy().tobytes()).hexdigest()
                              for name, t in (("left", left), ("right", right))},
            "unrelated_nodes_and_initializers_byte_exact": True,
            "additional_buffering_samples": 0, "recurrent_weights_unchanged": True}
