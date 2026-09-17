"""Check detached fast context on CPU and the exact proposed GPU microbatch."""
import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require
from research.direct.train_latency58 import state_sha256


def compare(model, mixture, warmup):
    from research.direct.latency58_sdr_context import render_scored_context as original
    from research.direct.latency58_fast_context import render_scored_context as fast
    fingerprint = state_sha256(model.state_dict())
    values = []
    for function in (original, fast):
        model.zero_grad(set_to_none=True)
        audio = mixture.detach().clone().requires_grad_(True)
        result = function(model, audio, warmup_samples=warmup, carry_state=True)
        loss = result.raw.square().mean() + result.deployed.square().mean()
        loss.backward()
        require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()), "Missing or nonfinite scored gradient")
        require(torch.count_nonzero(audio.grad[..., :warmup]).item() == 0
                and torch.count_nonzero(audio.grad[..., warmup:]).item() > 0,
                "Warmup detach or scored gradient boundary changed")
        values.append({"raw": result.raw.detach().cpu(), "deployed": result.deployed.detach().cpu(),
                       "physical": result.physical_mixture.detach().cpu(),
                       "gradients": {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()},
                       "input_gradient": audio.grad.detach().cpu(), "loss": float(loss.detach())})
        del result, loss, audio
    first, second = values
    errors = {name: float((first[name] - second[name]).abs().max()) for name in ("raw", "deployed", "physical", "input_gradient")}
    gradients = {name: {"maximum_error": float((value - second["gradients"][name]).abs().max()),
                        "relative_l2_error": float((value - second["gradients"][name]).norm() / value.norm().clamp_min(1e-30)),
                        "reference_norm": float(value.norm())} for name, value in first["gradients"].items()}
    require(all(error == 0 for error in errors.values()) and
            all(row["maximum_error"] == 0 for row in gradients.values()) and first["loss"] == second["loss"],
            "Fast context must preserve the scored outputs and all gradients bit for bit")
    require(len(gradients) == 22 and all(row["reference_norm"] > 0 for row in gradients.values()), "Unexercised neural tensor")
    require(state_sha256(model.state_dict()) == fingerprint, "Parity check changed weights or buffers")
    model.zero_grad(set_to_none=True)
    return {"status": "pass", "microbatch": mixture.shape[0], "warmup_samples": warmup,
            "scored_samples": mixture.shape[-1] - warmup, "output_and_input_gradient_errors": errors,
            "all_22_gradient_errors": gradients, "loss": first["loss"], "warmup_input_gradient_zero": True,
            "model_state_sha256": fingerprint, "model_unchanged": True}


def check_gpu(model):
    require(next(model.parameters()).device.type == "cuda" and model.training_precision == "bf16", "Require training BF16 CUDA model")
    generator = torch.Generator(device="cuda").manual_seed(202609121)
    audio = .03 * torch.randn(8, 2, 88064 + 8 * 128, device="cuda", generator=generator)
    result = compare(model, audio, 88064)
    torch.cuda.synchronize()
    return {**result, "precision": "bf16 learned operations, FP32 FFT/state/loss",
            "training_optimizer_updates": 0, "peak_vram_gib_including_check": torch.cuda.max_memory_allocated() / 2**30}


def check_cpu():
    from research.direct.latency58_full_magnitude_checkpoint import load_model
    checkpoint = read(PHASE / "full-magnitude-001/result.json")["checkpoint"]
    model, _ = load_model(checkpoint)
    model.train().requires_grad_(True)
    model.training_precision = "fp32"
    generator = torch.Generator().manual_seed(202609121)
    audio = .03 * torch.randn(2, 2, 23 * 128 + 17 * 128 + 37, generator=generator)
    result = compare(model, audio, 23 * 128)
    require(not torch.cuda.is_initialized(), "CPU check initialized CUDA")
    return {**result, "checkpoint": checkpoint, "precision": "CPU FP32", "partial_final_hop": True}


if __name__ == "__main__":
    import json
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps(check_cpu()), flush=True)
