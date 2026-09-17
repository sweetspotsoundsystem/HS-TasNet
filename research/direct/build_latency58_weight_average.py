"""Construct or independently audit one frozen CPU-only checkpoint average."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.latency58_weight_average import VERSION, expected_provenance, load_components, load_average


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=("build", "audit"), required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use the frozen CPU1 construction plan")
    plan = read(args.plan)
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.is_dir() and not (out / (args.mode + "-result.json")).exists(), "Preserve prior evidence")
    from research.direct.latency58_sdr_checkpoint import require_space
    require(plan["history_trial_reserve_bytes"] == 1_400_000_000
            and plan["average_trial_reserve_bytes"] == 180_000_000, "Trial reservation changed")
    counted = require_space(plan, 1_580_000_000 if args.mode == "build" else 1_400_000_000)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260918)
    torch.use_deterministic_algorithms(True)
    require(torch.__version__ == plan["torch_version"] and not torch.cuda.is_initialized(), "Different CPU runtime")
    began = time.monotonic()
    left, right = load_components(plan)
    if args.mode == "build":
        require(all(not (out / name).exists() for name in ("model.pt", ".model.pt.tmp", "receipt.json")),
                "Do not replace an averaged checkpoint")
        values = {name: value.clone() for name, value in left.state_dict().items()}
        for name, parameter in left.named_parameters():
            values[name] = ((parameter.double() + dict(right.named_parameters())[name].double()) * .5).float()
        left.load_state_dict(values, strict=True)
        model = left
        model.provenance = expected_provenance(plan, args.plan_sha256)
    else:
        build, execution = read(out / "build-result.json"), read(out / "build-execution.json")
        require(build["status"] == "pass" and execution["actual_exit_code"] == 0
                and not execution["timed_out"] and execution["source_bindings_unchanged"]
                and execution["plan_sha256"] == build["plan_sha256"] == args.plan_sha256,
                "Construction did not complete")
        model, receipt = load_average(plan, expected_plan_sha=args.plan_sha256)
        require(receipt["model_state_sha256"] == build["model_state_sha256"], "Construction fingerprint differs")
        # Independently recompute every learned tensor by interpolation in a
        # separate process. The builder uses a pair sum followed by scaling.
        for name, parameter in model.named_parameters():
            expected = torch.lerp(dict(left.named_parameters())[name].double(),
                                  dict(right.named_parameters())[name].double(), .5).float()
            require(torch.equal(parameter, expected), "Saved tensor is not the exact midpoint: " + name)
    fingerprint = state_sha256(model.state_dict())
    require(fingerprint not in [c["model_state_sha256"] for c in plan["components"]]
            and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
            "Average is nonfinite or identical to a component")
    rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(20260918)
    audio = torch.randn(2, 2, 4096, generator=generator) * .03
    with torch.inference_mode():
        full, replay = model.render(audio), model.render(audio)
        require(all(torch.equal(getattr(full, k), getattr(replay, k))
                    for k in ("raw", "deployed", "delayed_mixture"))
                and all(torch.equal(a, b) for a, b in zip(full.state, replay.state, strict=True)),
                "Reset replay differs")
        physical = torch.cat((torch.zeros_like(audio[..., :128]), audio[..., :-128]), dim=-1)
        require(torch.equal(full.delayed_mixture, physical), "Graph alignment is not exactly one hop")
        future = audio.clone()
        future[..., 2048:] += .1
        changed = model.render(future)
        require(torch.equal(full.raw[..., :2048], changed.raw[..., :2048])
                and torch.equal(full.deployed[..., :2048], changed.deployed[..., :2048]),
                "Future input changed an emitted prefix")
        state, raw, deployed = None, [], []
        for chunk in audio.split(128, dim=-1):
            piece = model.render(chunk, state)
            state = piece.state
            raw.append(piece.raw)
            deployed.append(piece.deployed)
        raw_error = float((torch.cat(raw, dim=-1) - full.raw).abs().max())
        deployed_error = float((torch.cat(deployed, dim=-1) - full.deployed).abs().max())
        state_errors = [float((a - b).abs().max()) for a, b in zip(state, full.state, strict=True)]
        closure = float((full.deployed.sum(dim=1) - full.delayed_mixture).abs().max())
        flush, flush_state = model.flush(state, return_raw=True)
        final = model.render(torch.zeros_like(audio[..., :128]), state)
        require(raw_error <= 1e-6 and deployed_error <= 1e-6 and max(state_errors) <= 1e-6 and closure <= 1e-6
                and torch.equal(flush, final.raw)
                and torch.equal(final.delayed_mixture, audio[..., -128:])
                and all(torch.equal(a, b) for a, b in zip(flush_state, final.state, strict=True)),
                "Literal streaming, reconstruction or final flush differs")
    require(fingerprint == state_sha256(model.state_dict()) and torch.equal(rng, torch.get_rng_state())
            and not torch.cuda.is_initialized(), "Fixture changed the model, RNG or device scope")
    if args.mode == "build":
        payload = {"schema": "latency58-weight-average-inference-v1", "plan_sha256": args.plan_sha256,
                   "architecture": model.architecture_metadata, "provenance": model.provenance,
                   "model_state_sha256": fingerprint, "model": model.state_dict()}
        with (out / ".model.pt.tmp").open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        (out / ".model.pt.tmp").rename(out / "model.pt")
        write(out / "receipt.json", {"schema": "latency58-weight-average-generation-v1", "version": VERSION,
              "plan_sha256": args.plan_sha256, "step": 0, "component_steps": [500, 1000],
              "model_state_sha256": fingerprint, "files": {"model.pt": {
                  "sha256": sha(out / "model.pt"), "bytes": (out / "model.pt").stat().st_size}},
              "training_updates_executed": 0, "optimizer_instances": 0})
    verify_inputs(plan)
    final_counted = require_space(plan, 1_400_000_000)
    bindings = {**plan["source_bindings"], str(args.plan.resolve()): args.plan_sha256,
                **{str(out / n): sha(out / n) for n in ("model.pt", "receipt.json")}}
    if args.mode == "audit":
        bindings.update({str(out / n): sha(out / n) for n in ("build-result.json", "build-execution.json")})
    write(out / (args.mode + "-result.json"), {
        "schema": "latency58-weight-average-" + args.mode + "-v1", "status": "pass", "version": VERSION,
        "plan_sha256": args.plan_sha256, "source_bindings": bindings, "source_bindings_unchanged": True,
        "model_state_sha256": fingerprint, "component_steps": [500, 1000], "component_weights": [.5, .5],
        "parameter_tensors": 21, "buffer_tensors": 6, "fixed_buffers_unchanged": True,
        "independent_saved_tensor_average_verified": args.mode == "audit", "exact_reset_replay": True,
        "literal_raw_max_abs": raw_error, "literal_deployed_max_abs": deployed_error,
        "literal_state_max_abs": state_errors, "closure_max_abs": closure, "future_prefix_invariant": True,
        "exact_flush_and_physical_alignment": True, "model_and_rng_unchanged_by_fixture": True,
        "counted_bytes_before": counted, "counted_bytes_after": final_counted,
        "training_updates_executed": 0, "optimizer_instances": 0, "cuda_initialized": False,
        "validation_material_used": False, "quality_selected": False,
        "elapsed_seconds": time.monotonic() - began,
        "limitations": ["One predeclared arithmetic midpoint; no quality result or human listening verdict.",
                        "Unchanged inference architecture; native export and M4 qualification remain pending."]})
    print({"status": "pass", "mode": args.mode, "model_state_sha256": fingerprint,
           "elapsed_seconds": time.monotonic() - began}, flush=True)


if __name__ == "__main__":
    main()
