"""Check GRU gate layout and an isolated parent transform without scoring music."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Functional plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-gate-bias-functional-v1"
            and all(sha(p) == s for p, s in plan["source_bindings"].items())
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the bound functional plan with CUDA hidden and CPU1")
    import torch
    from research.direct.latency58_gate_bias import DEFAULT_OFFSET, VERSION, with_update_gate_bias
    from research.direct.latency58_log_relative_checkpoint import load_parent
    from research.direct.latency58_evaluate import model_state_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = torch.get_rng_state().clone()

    def manual_gru(gru, inputs, hidden):
        sequence = inputs
        final = []
        for layer in range(gru.num_layers):
            current = hidden[layer].clone()
            outputs = []
            for frame in sequence.unbind(dim=1):
                affine_input = frame @ getattr(gru, f"weight_ih_l{layer}").t() + getattr(gru, f"bias_ih_l{layer}")
                affine_hidden = current @ getattr(gru, f"weight_hh_l{layer}").t() + getattr(gru, f"bias_hh_l{layer}")
                ir, iz, candidate_input = affine_input.chunk(3, dim=-1)
                hr, hz, candidate_hidden = affine_hidden.chunk(3, dim=-1)
                reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
                candidate = (candidate_input + reset * candidate_hidden).tanh()
                current = (1 - update) * candidate + update * current
                outputs.append(current)
            sequence = torch.stack(outputs, dim=1)
            final.append(current)
        return sequence, torch.stack(final)

    with torch.random.fork_rng(devices=[]), torch.no_grad():
        toy = torch.nn.GRU(3, 3, num_layers=2, batch_first=True, dtype=torch.float64).eval()
        for index, parameter in enumerate(toy.parameters()):
            parameter.copy_(torch.linspace(-.3, .4, parameter.numel(), dtype=torch.float64).reshape_as(parameter)
                            * (index + 1) / 8)
        inputs = torch.linspace(-.2, .3, 24, dtype=torch.float64).reshape(2, 4, 3)
        hidden = torch.linspace(-.1, .2, 12, dtype=torch.float64).reshape(2, 2, 3)
        shifted = copy.deepcopy(toy)
        for layer in range(2):
            getattr(shifted, f"bias_ih_l{layer}")[3:6].add_(DEFAULT_OFFSET)
        manual_errors = []
        for model in (toy, shifted):
            actual, independent = model(inputs, hidden), manual_gru(model, inputs, hidden)
            manual_errors.append(max(float((a - b).abs().max()) for a, b in zip(actual, independent, strict=True)))
        require(max(manual_errors) < 1e-14, "Installed GRU behavior differs from the r/z/n layout or reset convention")
        preactivation = torch.linspace(-15, 15, 61, dtype=torch.float64)
        gate = preactivation.sigmoid()
        identity_error = float(((preactivation + DEFAULT_OFFSET).sigmoid() - 2 * gate / (1 + gate)).abs().max())
        require(identity_error < 1e-14, "The additive gate-bias identity differs")

    training_binding = plan["parent_training_plan"]
    require(sha(training_binding["path"]) == training_binding["sha256"], "Parent loader plan changed")
    parent = load_parent(read(training_binding["path"])).eval().requires_grad_(False)
    require(model_state_sha256(parent) == plan["parent_model_state_sha256"], "Wrong functional parent")
    transform_rng = torch.get_rng_state().clone()
    identity = with_update_gate_bias(parent, offset=0.0)
    shifted = with_update_gate_bias(parent)
    require(torch.equal(transform_rng, torch.get_rng_state()), "The transform changed RNG state")
    old, new = parent.state_dict(), shifted.state_dict()
    changed = sorted(name for name in old if not torch.equal(old[name], new[name]))
    require(changed == ["fusion_branch.bias_ih_l0", "fusion_branch.bias_ih_l1"]
            and all(old[name].data_ptr() != new[name].data_ptr() for name in old)
            and model_state_sha256(identity) == model_state_sha256(parent)
            and all(torch.equal(value, dict(shifted.named_buffers())[name]) for name, value in parent.named_buffers()),
            "Parent isolation, zero transform or fixed buffers differ")
    time = torch.arange(512, dtype=torch.float32) / 44100
    audio = torch.stack((.02 * torch.sin(2 * torch.pi * 110 * time),
                         .015 * torch.cos(2 * torch.pi * 173 * time))).unsqueeze(0)
    with torch.inference_mode():
        reference = parent.render(audio, parent.initial_state(1))
        identical = identity.render(audio, identity.initial_state(1))
        transformed = shifted.render(audio, shifted.initial_state(1))
        repeated = shifted.render(audio, shifted.initial_state(1))
    require(torch.equal(reference.raw, identical.raw) and torch.equal(reference.deployed, identical.deployed)
            and all(torch.equal(a, b) for a, b in zip(reference.state, identical.state, strict=True))
            and torch.equal(transformed.raw, repeated.raw) and torch.equal(transformed.deployed, repeated.deployed)
            and all(torch.equal(a, b) for a, b in zip(transformed.state, repeated.state, strict=True)),
            "Zero-offset audio/state identity or transformed reset replay differs")
    closure = float((transformed.deployed.sum(dim=1) - transformed.delayed_mixture).abs().max())
    require(closure < 1e-6 and all(bool(torch.isfinite(t).all()) for t in
                                 (transformed.raw, transformed.deployed, *transformed.state)),
            "Transformed output is not finite or lost mixture closure")
    rejected = []
    for value in (-.1, float("nan"), DEFAULT_OFFSET + .1, True):
        try:
            with_update_gate_bias(parent, offset=value)
        except ValueError:
            rejected.append(repr(value))
        else:
            raise RuntimeError("Invalid gate offset was accepted")
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
            and model_state_sha256(parent) == plan["parent_model_state_sha256"]
            and all(sha(p) == s for p, s in plan["source_bindings"].items()),
            "Functional check changed RNG, parent, sources or CPU scope")
    write(Path(plan["output_directory"]) / "result.json", {
        "schema": "latency58-gate-bias-functional-result-v1", "status": "pass",
        "plan_sha256": args.plan_sha256, "source_bindings": plan["source_bindings"],
        "source_bindings_unchanged": True, "torch_version": torch.__version__, "version": VERSION,
        "parent_model_state_sha256": plan["parent_model_state_sha256"],
        "transformed_model_state_sha256": model_state_sha256(shifted),
        "gru_manual_reference_max_abs_errors": manual_errors, "gate_identity_max_abs_error": identity_error,
        "changed_tensors": changed, "changed_values_per_tensor": 1000, "offset": DEFAULT_OFFSET,
        "fp64_addition_rounded_once_to_fp32": True, "hidden_biases_unchanged": True,
        "zero_offset_audio_and_states_bit_exact": True, "parent_unchanged": True,
        "tensor_storage_disjoint": True, "fixed_buffers_unchanged": True,
        "transformed_reset_replay_bit_exact": True, "mixture_closure_max_abs": closure,
        "synthetic_audio_max_abs_change": float((transformed.deployed - reference.deployed).abs().max()),
        "rejected_offsets": rejected, "checkpoint_written": False, "music_quality_evaluated": False,
        "training_updates_executed": 0, "quality_selected": False,
        "limitation": "A fixed-preactivation gate identity is not equivalence of nonlinear recurrent trajectories or a quality claim.",
    })
    print("GRU gate layout, isolated bias transform, zero identity and deterministic synthetic render passed.", flush=True)


if __name__ == "__main__":
    main()
