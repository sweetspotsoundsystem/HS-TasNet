"""CPU checks for all-layer magnitude training and self-contained saved inference."""
from __future__ import annotations

import tempfile
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require
from research.direct.train_latency58 import state_sha256


def check():
    import torch
    from research.direct.latency58_direct_sdr_checkpoint import load_model as load_parent
    from research.direct.latency58_full_magnitude_checkpoint import load_model, audit_live, save_generation
    from research.direct.latency58_magnitude import ADAPTER, Latency58MagnitudeModel
    torch.set_num_threads(1)
    binding = read(PHASE / "c204-residual-model-001/checkpoint.json")
    parent, _ = load_parent(binding)
    parent_sha = state_sha256(parent.state_dict())
    model = Latency58MagnitudeModel.from_parent(parent)
    initial_sha = state_sha256(model.state_dict())
    generator = torch.Generator().manual_seed(20261007)
    audio = torch.randn(1, 2, 8 * 128, generator=generator) * .03
    with torch.inference_mode():
        original, initial = parent.render(audio), model.render(audio)
        require(torch.equal(original.raw, initial.raw) and torch.equal(original.deployed, initial.deployed)
                and all(torch.equal(a, b) for a, b in zip(original.state, initial.state, strict=True)),
                "Full-model initialization differs from parent")
    model.train().requires_grad_(True)
    frozen = {n: v.clone() for n, v in model.named_buffers()}
    before = {n: v.detach().clone() for n, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    audit_live(model, optimizer, 0, frozen)
    output = model.render(audio)
    target = torch.randn(output.deployed.shape, generator=generator) * .02
    loss = (output.deployed - target).square().mean() + .25 * (output.raw - target).square().mean()
    loss.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and p.grad.abs().max() > 0
                for p in model.parameters()), "Every neural parameter must receive a nonzero finite gradient")
    optimizer.step()
    audit_live(model, optimizer, 1, frozen)
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    require(len(changed) == 22 and ADAPTER in changed, "Every neural tensor must update")
    budget = read(PHASE / "wave-spectral-001/plan.json")
    plan = {**budget, "parent_checkpoint": binding, "parent_model_state_sha256": initial_sha,
            "parent_training_updates": parent.provenance["training_updates"]}
    with tempfile.TemporaryDirectory(prefix="full-magnitude-cpu-", dir=PHASE) as temporary:
        run = Path(temporary)
        (run / "metrics.jsonl").write_text('{"cpu_test_step":1}\n')
        checkpoint = save_generation(model, optimizer, 1, plan, "cpu-functional-check", run)
        loaded, payload = load_model(checkpoint)
        resume = torch.load(run / "checkpoint/optimizer.pt", map_location="cpu", weights_only=True)
        require(len(resume["optimizer"]["state"]) == 22 and resume["model_state_sha256"] == payload["model_state_sha256"],
                "Saved full-model optimizer inventory differs")
        with torch.inference_mode():
            live, replay = model.eval().render(audio), loaded.render(audio)
            require(torch.equal(live.raw, replay.raw) and torch.equal(live.deployed, replay.deployed)
                    and all(torch.equal(a, b) for a, b in zip(live.state, replay.state, strict=True)),
                    "Saved full-model replay differs")
        sizes = {p.name: p.stat().st_size for p in (run / "checkpoint").iterdir()}
    require(state_sha256(parent.state_dict()) == parent_sha and not torch.cuda.is_initialized(),
            "CPU check changed the parent or initialized CUDA")
    return {"status": "pass", "gpu_used": False, "quality_measured": False,
            "zero_initialization_parent_bit_exact": True, "all_neural_tensors_updated": changed,
            "all_adam_steps": 1, "fixed_buffers_unchanged": True, "saved_inference_replay_bit_exact": True,
            "saved_optimizer_tensor_count": 22, "temporary_checkpoint_bytes": sizes,
            "parent_state_unchanged": True}
