"""CPU integration checks for the actual eight-state research implementation."""
from __future__ import annotations

import copy
import ast
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from research.direct.latency58_branch_memory import Latency58BranchMemoryModel
from research.direct.latency58_branch_memory_context import render_scored_context
from research.direct.train_latency58 import state_sha256

ROOT = Path(__file__).resolve().parents[1]


def new_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(617)
        model = Latency58BranchMemoryModel()
        # Exercise all learned paths, including initially zero residual heads.
        with torch.no_grad():
            for parameter in model.parameters():
                if torch.count_nonzero(parameter) == 0:
                    parameter.normal_(std=.001)
    model.training_precision = "fp32"
    return model


def test_complete_source_snapshot():
    subprocess.run([sys.executable, "scripts/sync_research.py"], cwd=ROOT, check=True)
    manifest = json.loads((ROOT / "research/source-manifest.json").read_text())
    for relative in manifest["files"]:
        if relative.endswith(".py"):
            compile((ROOT / relative).read_bytes(), relative, "exec")


def test_recovery_evaluation_changes_only_bound_artifact_paths():
    directory = ROOT / "research/direct"
    original = ast.parse((directory / "run_latency58_four_second_shared_quality.py").read_text())
    recovered = ast.parse((directory / "run_latency58_four_second_recovery_quality.py").read_text())
    paths = {"plan-recovery001.json": "plan-shared001.json",
             "recovery-stage-001/execution.json": "production-stage/execution.json",
             "recovery-root-execution-001.json": "production-root-execution.json",
             "recovery-stage-001/command.json": "production-root-command.json"}
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for current, previous in paths.items():
                    node.value = node.value.replace(current, previous)
            return node
    assert ast.dump(original, include_attributes=False) == ast.dump(Normalize().visit(recovered), include_attributes=False)


def test_current_native_and_released_state_contract():
    import onnx
    from research.direct.latency58_branch_onnx import interface
    from research.direct.train_latency58_four_second_shared import (
        applied_policy, reduction_policy, accumulation_policy, data_policy,
        packed_policy, recovery_policy, runtime_policy)
    from research.direct.latency58_branch_ema_checkpoint import policy as ema_policy

    model = new_model().eval()
    release = json.loads((ROOT / "hs_tasnet/streaming_models.json").read_text())["current"]
    contract = interface(model)
    assert contract["state_names"] == tuple(release["states"])
    assert contract["state_shapes"] == tuple(tuple(v) for v in release["states"].values())
    assert len(list(model.parameters())) == 40
    assert model.sample_rate == 44100
    assert model.hop_samples == model.graph_alignment_samples == model.host_queue_samples == 128
    assert model.algorithmic_latency_samples == 256
    plan = json.loads((ROOT / "research/recipes/active-plan.json").read_text())
    # Scientific policies are portable. The full recipe validator additionally
    # authenticates the original machine's paths, monitor and artifact budget.
    for name, expected in (("grouped_vocal_loss", applied_policy()),
                           ("logical_batch_loss", reduction_policy()),
                           ("accumulation_policy", accumulation_policy()),
                           ("packed_recovery", packed_policy()),
                           ("recovery_checkpoint", recovery_policy()),
                           ("runtime_allowance", runtime_policy()),
                           ("ema", ema_policy(.995))):
        assert plan[name] == expected
    assert plan["config"]["augmentation"] == data_policy()
    assert model.architecture_metadata == plan["inference_architecture"]
    assert [name for name, _ in model.named_parameters()] == plan["parameter_names"]
    path = ROOT / "models" / release["filename"]
    assert path.is_file(), "Run python scripts/download_streaming_model.py before this integration suite"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == release["sha256"]
    graph = onnx.load(path)
    for values, names, shapes in ((graph.graph.input, contract["input_names"], contract["input_shapes"]),
                                 (graph.graph.output, contract["output_names"], contract["output_shapes"])):
        assert tuple(v.name for v in values) == names
        assert tuple(tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in values) == shapes
    metadata = {row.key: row.value for row in graph.metadata_props}
    assert metadata["hs_tasnet.state_family"] == model.architecture_metadata["state_family"]
    assert not torch.cuda.is_initialized()


def test_streaming_state_carry_closure_and_detached_warmup():
    model = new_model().eval()
    audio = torch.randn(1, 2, 6 * 128) * .03
    with torch.no_grad():
        block = model.render(audio)
        state, chunks = model.initial_state(1), []
        for hop in audio.split(128, dim=-1):
            output = model.render(hop, state)
            state = output.state
            chunks.append(output.deployed)
        torch.testing.assert_close(torch.cat(chunks, dim=-1), block.deployed, atol=1e-6, rtol=1e-5)
        for actual, expected in zip(state, block.state, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(block.deployed.sum(1), block.delayed_mixture, atol=1e-6, rtol=0)
        assert torch.equal(model.render(audio).deployed, block.deployed)
    model.train()
    audio = audio.detach().requires_grad_()
    scored = render_scored_context(model, audio, warmup_samples=256, carry_state=True)
    assert scored.flush_hops == 1 and scored.initial_state_detached
    assert torch.equal(scored.physical_mixture, audio[..., 256:])
    (scored.raw.square().mean() + scored.deployed.square().mean()).backward()
    assert torch.count_nonzero(audio.grad[..., :256]) == 0
    assert torch.count_nonzero(audio.grad[..., 256:]) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_current_weighted_loss_against_independent_scalar_and_gradients():
    from research.direct.check_latency58_weighted_vocal_cpu_v2 import scalar_checks
    result = scalar_checks(lambda *args: None)
    assert result["status"] == "pass"
    assert len(result["cases"]) == 6


def assert_tree_equal(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        assert first.dtype == second.dtype and first.shape == second.shape
        assert torch.equal(first.reshape(-1).view(torch.uint8), second.reshape(-1).view(torch.uint8))
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_tree_equal(a, b)
    else:
        assert first == second


def test_exact_native_adam_ema_rng_and_lossless_recovery(tmp_path):
    from research.direct.latency58_branch_ema import BranchParameterEMA
    from research.direct.latency58_branch_ema_checkpoint import policy as ema_policy
    from research.direct.latency58_branch_memory_checkpoint import make_payloads, load_model
    from research.direct.latency58_grouped_vocal_recovery import (
        policy, make_snapshot, audit_snapshot, restore_training)
    from research.direct.latency58_lossless_recovery_codec_v3 import (
        pack_snapshot, unpack_snapshot, policy as packed_policy)

    model = new_model().train()
    initial_hash = state_sha256(model.state_dict())
    model.provenance["branch_memory_parent_model_state_sha256"] = initial_hash
    plan = {"config": {"steps": 2, "batch_size": 16, "data_start": 0, "lr": 3e-5},
            "parent_model_state_sha256": initial_hash,
            "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
            "parent_checkpoint": {"path": "synthetic-parent", "sha256": "0" * 64},
            "parent_training_updates": 0, "objective_version": "cpu-recovery-fixture",
            "precision_policy": "fp32", "ema": ema_policy(.995),
            "recovery_checkpoint": policy(), "packed_recovery": packed_policy()}
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, foreach=False)
    audio = torch.randn(1, 2, 384) * .03
    def advance(m, opt, ema=None, step=None):
        opt.zero_grad(set_to_none=True)
        output = render_scored_context(m, audio, warmup_samples=128, carry_state=True)
        output.raw.square().mean().backward()
        opt.step()
        if ema is not None:
            ema.update(m, step=step)

    # Create an authenticated synthetic parent so the real XOR loader runs.
    advance(model, optimizer)
    parent, _ = make_payloads(model, optimizer, 1, {**plan, "config": {**plan["config"], "steps": 1}}, "0" * 64)
    parent_path = tmp_path / "parent.pt"
    torch.save(parent, parent_path)
    plan["parent_checkpoint"] = {"path": str(parent_path), "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest()}
    plan["parent_model_state_sha256"] = parent["model_state_sha256"]
    plan["parent_training_updates"] = 1
    load_model(plan["parent_checkpoint"])
    del parent
    model.provenance["branch_memory_parent_model_state_sha256"] = plan["parent_model_state_sha256"]
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, foreach=False)
    ema = BranchParameterEMA(model, decay=.995, base_state_sha256=plan["parent_model_state_sha256"])
    advance(model, optimizer, ema, 1)
    ema_state = ema.state_dict(model)
    journal = (json.dumps({"step": 1, "first_sample_index": 0, "next_sample_index": 16,
                          "raw_model_state_sha256": state_sha256(model.state_dict()),
                          "ema_parameters_sha256": ema_state["ema_parameters_sha256"]}) + "\n").encode()
    del ema_state
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    saved = make_snapshot(model, optimizer, ema, 1, plan, plan_sha, journal)
    packed, _ = pack_snapshot(saved, plan, plan_sha)
    recovered, _ = unpack_snapshot(packed, plan, plan_sha)
    # The decoder returns the complete audited snapshot, preserving raw bits.
    assert_tree_equal(saved, recovered)
    with pytest.raises(RuntimeError):
        unpack_snapshot(packed, plan, "f" * 64)
    del packed
    expected_random = (random.random(), np.random.random(), torch.rand(3))
    advance(model, optimizer, ema, 2)
    expected_hash = state_sha256(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_ema = ema.state_dict(model)
    del model, optimizer, ema, saved
    restored, optimizer, ema = restore_training(recovered, audit_snapshot(recovered, plan, plan_sha),
                                               plan, device="cpu", precision="fp32")
    actual_random = (random.random(), np.random.random(), torch.rand(3))
    assert_tree_equal(expected_random, actual_random)
    advance(restored, optimizer, ema, 2)
    assert state_sha256(restored.state_dict()) == expected_hash
    assert_tree_equal(optimizer.state_dict(), expected_optimizer)
    assert_tree_equal(ema.state_dict(restored), expected_ema)
    assert not torch.cuda.is_initialized()


def test_eight_state_fp32_onnx_export_trajectory(tmp_path):
    import onnxruntime as ort
    from research.direct.latency58_branch_onnx import interface, make_export_copy
    model = new_model().eval()
    fingerprint = state_sha256(model.state_dict())
    wrapper, contract = make_export_copy(model), interface(model)
    state = model.initial_state(1)
    path = tmp_path / "current-native.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, (torch.zeros(1, 2, 128), *state), str(path),
                          input_names=list(contract["input_names"]), output_names=list(contract["output_names"]),
                          opset_version=18, dynamo=False, external_data=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        runtime_state = [value.numpy().copy() for value in state]
        # Carry independent native/ORT states through sound, silence and flush.
        for index in range(8):
            audio = torch.randn(1, 2, 128) * .02 if index < 5 else torch.zeros(1, 2, 128)
            native = model.render(audio, state)
            outputs = session.run(None, dict(zip(contract["input_names"], [audio.numpy(), *runtime_state], strict=True)))
            for actual, expected in zip(outputs, (native.deployed, *native.state), strict=True):
                np.testing.assert_allclose(actual, expected.numpy(), atol=1e-5, rtol=1e-4)
            state, runtime_state = native.state, outputs[1:]
    assert state_sha256(model.state_dict()) == fingerprint
    assert not torch.cuda.is_initialized()
