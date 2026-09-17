"""Use the model's accepted fp32 label; keep the failed first wrapper intact."""
from research.direct.check_latency58_four_second_model import *

def main():
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.check_latency58_branch_long_context_gpu_b4 import compare_context
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.plan.resolve().is_relative_to(PHASE)
            and sha(args.plan) == args.plan_sha256 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require frozen CPU1 qualification with CUDA hidden")
    plan = read(args.plan)
    require(plan["torch_version"] == torch.__version__ and plan["ordinary_microbatch"] == 2
            and plan["warmup_samples"] == WARMUP_SAMPLES and plan["scored_samples"] == SCORED_SAMPLES
            and plan["new_training_recipe_selected"] is False, "Qualification geometry changed")
    verify_inputs(plan)
    require(not (args.plan.parent / "result.json").exists(), "Preserve completed qualification")
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    before = snapshot()
    model, _ = load_model(plan["fixture_checkpoint"])
    model.train().requires_grad_(True); model.training_precision = "fp32"
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint == plan["fixture_model_state_sha256"], "Fixture weights changed")
    rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(plan["synthetic_seed"])
    def progress(phase, group, offset):
        print(json.dumps({"event": "four_second_model_progress", "phase": phase, "group": group,
            "offset": offset, "elapsed_seconds": time.monotonic() - began,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}), flush=True)
    audio = .03 * torch.randn(2, 2, CROP_SAMPLES, generator=generator)
    progress("context_comparison_start", "ordinary", 0)
    context = compare_context(model, audio, WARMUP_SAMPLES)
    del audio
    gc.collect()
    progress("context_comparison_pass", "ordinary", 0)
    truth = .02 * torch.randn(16, 4, 2, CROP_SAMPLES, generator=generator)
    truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
    gradients = compare_group_gradients(model, truth.sum(1), truth,
                                        warmup_samples=WARMUP_SAMPLES, progress=progress)
    del truth
    gc.collect()
    require(state_sha256(model.state_dict()) == fingerprint and torch.equal(rng, torch.get_rng_state())
            and not torch.cuda.is_initialized() and all(p.grad is None for p in model.parameters()),
            "Qualification changed weights or RNG, left gradients or initialized CUDA")
    verify_inputs(plan)
    result = {"status": "pass", "plan_sha256": args.plan_sha256, "source_bindings_unchanged": True,
        "fixture_model_state_sha256": fingerprint, "context_comparison": context,
        "whole_group_neural_gradient_comparison": gradients,
        "ordinary_microbatch": 2, "auxiliary_microbatch": 2, "logical_batch_size": 16,
        "warmup_samples": WARMUP_SAMPLES, "scored_samples": SCORED_SAMPLES,
        "gpu_used": False, "optimizer_updates": 0, "model_weights_unchanged": True,
        "rng_unchanged": True, "quality_measured": False, "new_training_recipe_selected": False,
        "elapsed_seconds": time.monotonic() - began,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "budget_before": before, "budget_after": snapshot(),
        "limits": ["Synthetic CPU FP32 fixture on retained 012 EMA; no quality measurement.",
                   "Full training update/restart, CUDA BF16 and GPU resource qualification remain separate requirements."]}
    write(args.plan.parent / "result.json", result)
    print(json.dumps({"status": "pass", "elapsed_seconds": result["elapsed_seconds"],
        "peak_rss_bytes": result["peak_rss_bytes"]}), flush=True)

if __name__ == "__main__":
    main()
