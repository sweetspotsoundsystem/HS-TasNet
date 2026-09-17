"""CPU functional checks for the untrained branch-memory architecture."""
from __future__ import annotations

import io
import argparse
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import state_sha256, verify_inputs


def compare_context(model, mixture, warmup):
    import torch
    from research.direct.latency58_sdr_context import render_scored_context as full
    from research.direct.latency58_branch_memory_context import render_scored_context as fast
    fingerprint, rows = state_sha256(model.state_dict()), []
    for render in (full, fast):
        model.zero_grad(set_to_none=True)
        audio = mixture.detach().clone().requires_grad_(True)
        result = render(model, audio, warmup_samples=warmup, carry_state=True)
        loss = result.raw.square().mean() + result.deployed.square().mean()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                "Missing or nonfinite model gradient")
        require(torch.count_nonzero(audio.grad[..., :warmup]) == 0
                and torch.count_nonzero(audio.grad[..., warmup:]) > 0, "Detached warmup boundary changed")
        rows.append({"raw": result.raw.detach().cpu(), "deployed": result.deployed.detach().cpu(),
                     "physical": result.physical_mixture.detach().cpu(), "input_gradient": audio.grad.detach().cpu(),
                     "gradients": {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()},
                     "loss": float(loss.detach())})
    first, second = rows
    errors = {k: float((first[k] - second[k]).abs().max()) for k in ("raw", "deployed", "physical", "input_gradient")}
    gradients = {name: {"maximum_error": float((value - second["gradients"][name]).abs().max()),
                        "reference_norm": float(value.norm())} for name, value in first["gradients"].items()}
    require(all(v == 0 for v in errors.values()) and first["loss"] == second["loss"]
            and len(gradients) == 40 and all(v["maximum_error"] == 0 and v["reference_norm"] > 0 for v in gradients.values()),
            "Warmup optimization changed scored output/gradients or left a parameter unexercised")
    require(state_sha256(model.state_dict()) == fingerprint, "Context comparison changed the model")
    model.zero_grad(set_to_none=True)
    return {"status": "pass", "output_and_input_gradient_errors": errors, "all_40_gradients": gradients,
            "warmup_input_gradient_zero": True, "scored_samples": mixture.shape[-1] - warmup,
            "loss": first["loss"], "model_unchanged": True}


def check(parent):
    import numpy as np
    import torch
    from research.direct.latency58 import PUBLIC_FUSION_SCALE
    from research.direct.latency58_temporal_attention import TemporalAttentionState
    from research.direct.latency58_branch_memory import (
        ADAPTERS, WINDOW, KEY, VALUE, Latency58BranchMemoryModel, BranchMemoryState,
    )
    generator = torch.Generator().manual_seed(202609141)
    parent_sha = state_sha256(parent.state_dict())
    model = Latency58BranchMemoryModel.from_parent(parent)
    audio = .03 * torch.randn(2, 2, 19 * 128, generator=generator)
    zero_cases = []
    def bits_equal(a, b):
        return a.shape == b.shape and a.dtype == b.dtype == torch.float32 and torch.equal(a.view(torch.int32), b.view(torch.int32))
    with torch.inference_mode():
        for signal, samples in (("noise", audio), ("silence", torch.zeros_like(audio))):
            for nonzero in (False, True):
                state = parent.initial_state(2)
                if nonzero:
                    state = TemporalAttentionState(*(torch.randn(v.shape, generator=generator) *
                        (.01 * PUBLIC_FUSION_SCALE if i == 1 else .001) for i, v in enumerate(state)))
                extra = model.initial_state(2)[6:]
                if nonzero:
                    extra = tuple(.01 * PUBLIC_FUSION_SCALE * torch.randn(v.shape, generator=generator) for v in extra)
                first, second = parent.render(samples, state), model.render(samples, BranchMemoryState(*state, *extra))
                require(all(bits_equal(getattr(first, k), getattr(second, k)) for k in
                            ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                        and all(bits_equal(a, b) for a, b in zip(first.state, second.state[:6], strict=True)),
                        "Zero correction differs from preserved parent")
                zero_cases.append({"signal": signal, "nonzero_state": nonzero, "all_outputs_and_states_bit_exact": True})
        for projection in (model.spec_memory_output, model.waveform_memory_output):
            projection.weight.copy_(.002 * torch.randn(projection.weight.shape, generator=generator))
        # Independent NumPy FP64 gate recurrence for each new GRU.
        features = .1 * torch.randn(2, 7, 500, generator=generator)
        initial = .01 * torch.randn(1, 2, 500, generator=generator)
        numpy_errors = {}
        for name in ("spec_memory", "waveform_memory"):
            cell = getattr(model, name)
            weights = {key: value.numpy().astype(np.float64) for key, value in cell.state_dict().items()}
            hidden = initial[0].numpy().astype(np.float64)
            expected = []
            for frame in range(features.shape[1]):
                incoming = features[:, frame].numpy().astype(np.float64) @ weights["weight_ih_l0"].T + weights["bias_ih_l0"]
                recurrent = hidden @ weights["weight_hh_l0"].T + weights["bias_hh_l0"]
                ir, iz, inn = np.split(incoming, 3, axis=-1)
                hr, hz, hn = np.split(recurrent, 3, axis=-1)
                reset, update = 1 / (1 + np.exp(-ir - hr)), 1 / (1 + np.exp(-iz - hz))
                candidate = np.tanh(inn + reset * hn)
                hidden = (1 - update) * candidate + update * hidden
                expected.append(hidden.copy())
            actual, terminal = cell(features, initial)
            numpy_errors[name] = max(float(np.max(np.abs(actual.numpy() - np.stack(expected, 1)))),
                                     float(np.max(np.abs(terminal[0].numpy() - hidden))))
            require(numpy_errors[name] < 2e-6, "Private GRU differs from independent gate recurrence")
        whole = model.render(audio)
        require(not torch.equal(whole.deployed, parent.render(audio).deployed), "Nonzero private memory has no waveform effect")
        pieces, state, offset = [], None, 0
        for hops in (1, 3, 7, 8):
            output = model.render(audio[..., offset:offset + hops * 128], state)
            pieces.append(output.deployed)
            state, offset = output.state, offset + hops * 128
        partition_error = float((whole.deployed - torch.cat(pieces, -1)).abs().max())
        state_errors = [float((a - b).abs().max()) / (PUBLIC_FUSION_SCALE if i in (1, 6, 7) else 1.)
                        for i, (a, b) in enumerate(zip(whole.state, state, strict=True))]
        require(partition_error < 1e-4 and max(state_errors) < 5e-4, "Partitioned stream differs")
        future = audio.clone()
        future[..., 8 * 128:] *= -2
        require(torch.equal(whole.deployed[..., :8 * 128], model.render(future).deployed[..., :8 * 128]),
                "Unreceived future input changed emitted output")
        require(torch.equal(whole.deployed, model.render(audio).deployed), "Reset replay changed")
        warm = model.warm_state(audio)
        warm_errors = [float((a - b).abs().max()) / (PUBLIC_FUSION_SCALE if i in (1, 6, 7) else 1.)
                       for i, (a, b) in enumerate(zip(whole.state, warm, strict=True))]
        require(max(warm_errors) < 5e-4, "Final-frame warmup state differs")
        eof = []
        for length in (1, 127, 128, 129, 1023, 1024, 1025):
            real = audio[:1, :, :length]
            padded = torch.nn.functional.pad(real, (0, (-length) % 128))
            state, outputs = None, []
            for chunk in padded.split(128, -1):
                output, state = model.forward_chunk(chunk, state)
                outputs.append(output)
            flushed, _ = model.flush(state)
            host = torch.cat([torch.zeros_like(outputs[0]), *outputs, flushed], -1)
            recovered = host[..., 256:256 + length]
            closure = float((recovered.sum(1) - real).abs().max())
            require(recovered.shape == (1, 4, 2, length) and bool(torch.isfinite(host).all())
                    and closure < 1e-6, "Partial EOF or physical host delay changed")
            eof.append({"samples": length, "data_hops": len(outputs), "flush_hops": 1,
                        "host_delay_samples": 256, "closure_max_abs": closure})
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = Latency58BranchMemoryModel().eval().requires_grad_(False)
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
        replay = restored.render(audio)
        require(bits_equal(replay.deployed, whole.deployed)
                and all(bits_equal(a, b) for a, b in zip(replay.state, whole.state, strict=True)),
                "In-memory serialization changed inference")
        del restored, buffer
    try:
        model.render(audio, parent.initial_state(2))
    except ValueError as error:
        require(str(error) == "Use the distinct branch-memory state family", "Unexpected state rejection reason")
    else:
        raise RuntimeError("Cross-family state was accepted")
    model.train().requires_grad_(True)
    model.training_precision = "fp32"
    crop = .03 * torch.randn(2, 2, 23 * 128 + 17 * 128 + 37, generator=generator)
    context = compare_context(model, crop, 23 * 128)
    # Two discarded CPU fixture updates verify optimizer coverage, not quality.
    with torch.no_grad():
        model.spec_memory_output.weight.zero_()
        model.waveform_memory_output.weight.zero_()
    frozen = {k: v.clone() for k, v in model.named_buffers()}
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    branch_memory_gradient_norms = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model.render(audio[..., :7 * 128])
        loss = output.raw.square().mean() + output.deployed.square().mean()
        loss.backward()
        branch_memory_gradient_norms.append({name: float(dict(model.named_parameters())[name].grad.norm()) for name in ADAPTERS})
        output_names = {"spec_memory_output.weight", "waveform_memory_output.weight"}
        require(all(branch_memory_gradient_norms[-1][name] > 0 for name in output_names)
                and all((branch_memory_gradient_norms[-1][name] == 0 if step == 0 else
                         branch_memory_gradient_norms[-1][name] > 0) for name in ADAPTERS if name not in output_names),
                "Zero initialization did not activate both complete private memories")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True, foreach=False)
        optimizer.step()
    updated = [k for k, v in model.named_parameters() if not torch.equal(v, before[k])]
    require(len(updated) == len(optimizer.state) == 40 and all(torch.equal(v, frozen[k]) for k, v in model.named_buffers()),
            "Fixture missed a learned tensor or changed fixed buffers")
    for parameter in model.parameters():
        adam = optimizer.state[parameter]
        require(adam["step"].item() == 2 and bool(torch.isfinite(parameter).all())
                and all(bool(torch.isfinite(adam[k]).all()) for k in ("exp_avg", "exp_avg_sq")),
                "Invalid fixture optimizer state")
    require(state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized(), "Parent or CPU-only scope changed")
    return {"status": "pass", "parent_model_state_sha256": parent_sha, "zero_correction_cases": zero_cases,
            "private_gru_numpy_max_abs": numpy_errors,
            "partition_max_abs": partition_error, "state_partition_errors_in_physical_units": state_errors,
            "warm_state_errors_in_physical_units": warm_errors, "future_input_independence": True,
            "reset_replay_bit_exact": True, "in_memory_serialization_bit_exact": True,
            "cross_family_state_rejected": True, "partial_eof_cases": eof, "training_context": context,
            "all_40_neural_tensors_updated": updated, "discarded_functional_adam_updates": 2,
            "branch_memory_gradient_norms_from_zero_output": branch_memory_gradient_norms,
            "fixed_buffers_unchanged": True, "additional_parameters": model.architecture_metadata["branch_memory_parameters"],
            "inference_architecture": model.architecture_metadata, "quality_measured": False,
            "gpu_used": False, "native_timing_measured": False, "checkpoint_written": False}


def main():
    import torch
    from research.direct.latency58_temporal_attention_checkpoint import load_model
    from research.direct.latency58_sdr_checkpoint import require_space
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="branch-memory-functional-001")
    args = parser.parse_args()
    require(bool(args.name) and all(c.isalnum() or c in "-_" for c in args.name), "Invalid evidence name")
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CPU1 with CUDA hidden")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "attention-continuation-001/plan.json"
    source = read(source_path)
    review_path = PHASE / "attention-continuation-001/selection-review.json"
    review = read(review_path)
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and review["best_full_sdr_db"] == 4.330200584072513,
            "Require the reviewed selected attention parent")
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(source_path.parent / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"], "Previous run has not closed")
    verify_inputs(review)
    counted = require_space(source, 385_000_000)
    parent_binding = review["best_research_checkpoint"]
    parent, _ = load_model(parent_binding)
    require(state_sha256(parent.state_dict()) == "94bf99892cfdabb134d76cf4238528129ab1427dbfc879337c9212ae6756e875",
            "Wrong preserved attention parent")
    out = PHASE / args.name
    require(not out.exists(), "Preserve previous functional evidence")
    paths = [source_path, review_path, Path(parent_binding["path"]), Path(__file__).resolve(),
             ROOT / "research/direct/latency58_temporal_attention.py"]
    paths.extend(ROOT / "research/direct" / name for name in
                 ("latency58_branch_memory.py", "latency58_branch_memory_context.py", "latency58_sdr_context.py"))
    bindings = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "parent_checkpoint": parent_binding,
          "scope": "Synthetic CPU functional checks; no production training, GPU run, validation audio or timing",
          "counted_bytes_before": counted, "gpu_training_active": True, "concurrent_gpu_trial": "attention-sdr-blend-001", "independent_cpu_quality_scoring_may_be_active": True,
          "candidate_weights_written": False, "zero_output_initialization_seed": 20261021,
          "hypothesis": "Separate spectral and waveform GRUs read pre-attention normalized refined features; zero output projections preserve the selected parent. Two extra recurrent states add no future input or audio buffering. This CPU prototype is not a selected production experiment.",
          "production_recipe_selected": False, "architecture_reference": "https://arxiv.org/html/2402.17701",
          "reference_scope": "The original HS-TasNet uses recurrent processing in separate branches; its 23-ms test scores do not establish this prototype at 256 samples or our validation protocol.", "completed_selected_parent_unchanged": True})
    began = time.monotonic()
    result = check(parent)
    verify_inputs({"source_bindings": bindings})
    result.update(source_bindings_unchanged=True, plan_sha256=sha(out / "plan.json"),
                  elapsed_seconds=time.monotonic() - began, counted_bytes_after=require_space(source, 380_000_000))
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "additional_parameters", "partition_max_abs", "quality_measured", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
