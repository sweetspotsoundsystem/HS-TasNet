"""Build the fixed seventeen-product integer variant and its independent oracle.

Each rewrite authenticates the immediately preceding in-memory graph and checks
its supported topology. Weight quantization is independently reconstructed from
the native FP32 tensors; ONNX Runtime output is never used as the oracle.
"""
from __future__ import annotations
import hashlib
from . import integer, integer_branch_gru, integer_branch_output, integer_qkv
from .integer_qkv_reference import make_reference
from .helpers import require, state_sha256


def build(model, *, checkpoint_sha256=None):
    from ..export import build_fp32
    before = state_sha256(model.state_dict())
    wrapper, floating = build_fp32(model, checkpoint_sha256=checkpoint_sha256)
    del wrapper
    raw, ten, ten_proof = integer.build(model, floating)
    del raw, floating
    def digest(graph):
        return hashlib.sha256(graph.SerializeToString()).hexdigest()
    fourteen, fourteen_proof = integer_branch_gru.build(ten, expected_parent_sha256=digest(ten))
    source_matrices = {name: getattr(model, name).weight.detach().cpu().numpy().T
                       for name in integer_branch_output.TARGETS.values()}
    sixteen, sixteen_proof = integer_branch_output.build(
        fourteen, expected_parent_sha256=digest(fourteen), expected_weights=source_matrices)
    attention_matrices = {f"/{name}/MatMul": getattr(model, name).weight.detach().cpu().numpy().T
                          for name in ("temporal_query", "temporal_key", "temporal_value")}
    graph, qkv_proof = integer_qkv.build(sixteen, expected_parent_sha256=digest(sixteen),
                                       expected_weights=attention_matrices)
    reference, independent = make_reference(model, ten, ten_proof, fourteen, fourteen_proof,
                                            sixteen, sixteen_proof, graph, qkv_proof)
    require(state_sha256(model.state_dict()) == before, "Integer export changed source model tensors")
    require(len(independent) == 17, "Integer reference must independently reconstruct all seventeen products")
    proof = {"version": integer_qkv.VERSION, "source_model_state_sha256": before,
             "projections": independent, "stages": {"base": ten_proof,
             "branch_recurrence": fourteen_proof, "branch_output": sixteen_proof,
             "fused_qkv": qkv_proof}, "quality_measured": False, "native_host_qualified": False}
    return reference, graph, proof
