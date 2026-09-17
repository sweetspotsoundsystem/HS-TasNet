"""Bounded BF16 initialization and nonzero-adapter warmup-gradient rehearsal."""
import torch

from research.direct.check_latency58_long_magnitude_context import compare
from research.direct.latency58_full_magnitude_checkpoint import load_model as load_parent
from research.direct.run_latency58_quality import PHASE, read, require
from research.direct.train_latency58 import state_sha256


def check_gpu(model):
    require(next(model.parameters()).device.type == "cuda" and model.training_precision == "bf16",
            "Require the proposed BF16 CUDA model")
    fingerprint = state_sha256(model.state_dict())
    parent, _ = load_parent(read(PHASE / "full-magnitude-001/result.json")["checkpoint"])
    parent.cuda().train().requires_grad_(False)
    parent.training_precision = "bf16"
    generator = torch.Generator(device="cuda").manual_seed(202610111)
    with torch.no_grad():
        short = .03 * torch.randn(16, 2, 8 * 128, device="cuda", generator=generator)
        old, new = parent.render(short), model.render(short)
        require(torch.equal(old.raw, new.raw) and torch.equal(old.deployed, new.deployed)
                and torch.equal(old.state.audio_history, new.state.audio_history[..., -896:])
                and all(torch.equal(a,b) for a,b in zip(old.state[1:],new.state[1:],strict=True)),
                "BF16 zero projection changed inherited execution")
    del parent, old, new, short
    original = model.long_projection.weight.detach().clone()
    try:
        with torch.no_grad():
            model.long_projection.weight.fill_(.0001)
        audio = .03 * torch.randn(16, 2, 88064 + 8 * 128, device="cuda", generator=generator)
        result = compare(model, audio, 88064)
    finally:
        with torch.no_grad():
            model.long_projection.weight.copy_(original)
        model.zero_grad(set_to_none=True)
    require(state_sha256(model.state_dict()) == fingerprint, "GPU rehearsal changed initial model tensors")
    torch.cuda.synchronize()
    return {**result, "precision": "BF16 learned operations; FP32 FFT, state, loss and parameters",
            "bf16_zero_projection_parent_bit_exact": True, "nonzero_long_projection_exercised": True,
            "initial_model_state_restored_bit_exact": True, "training_optimizer_updates": 0,
            "initial_model_state_sha256": fingerprint,
            "peak_vram_gib_including_check": torch.cuda.max_memory_allocated() / 2**30}
