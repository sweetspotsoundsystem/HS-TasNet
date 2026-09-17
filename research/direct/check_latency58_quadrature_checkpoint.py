"""CPU save/load audit using two disposable quadrature fixture updates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import state_sha256, verify_inputs


def check(source, evidence):
    import torch
    from research.direct.latency58_full_magnitude_checkpoint import load_model as load_parent
    from research.direct.latency58_quadrature import Latency58QuadratureModel
    from research.direct.latency58_quadrature_checkpoint import load_model, audit_live, save_generation, audit_saved
    from research.direct.latency58_sdr_checkpoint import require_space
    # Reserve this disposable generation and the still-running magnitude endpoint.
    require_space(source, 750_000_000)
    parent, _ = load_parent(source["parent_checkpoint"])
    fingerprint = state_sha256(parent.state_dict())
    model = Latency58QuadratureModel.from_parent(parent).train().requires_grad_(True)
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    generator = torch.Generator().manual_seed(202609127)
    audio = .03 * torch.randn(1, 2, 8 * 128, generator=generator)
    target = .02 * torch.randn(1, 4, 2, 8 * 128, generator=generator)
    audit_live(model, optimizer, 0, frozen)
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        output = model.render(audio)
        loss = (output.deployed - target).square().mean() + .25 * (output.raw - target).square().mean()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                "A fixture parameter lacks a finite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True, foreach=False)
        optimizer.step()
        audit_live(model, optimizer, step, frozen)
    changed = [name for name, value in model.named_parameters() if not torch.equal(value, initial[name])]
    require(len(changed) == 24, "Disposable fixture did not change every learned tensor")
    plan = {**source, "config": {**source["config"], "steps": 2},
            "fixed_buffers_sha256": state_sha256(frozen), "parent_model_state_sha256": fingerprint,
            "parent_training_updates": parent.provenance["training_updates"]}
    write(evidence / "fixture-plan.json", plan)
    plan_sha = sha(evidence / "fixture-plan.json")
    with tempfile.TemporaryDirectory(prefix="quadrature-checkpoint-fixture-", dir=PHASE) as temporary:
        run = Path(temporary)
        (run / "metrics.jsonl").write_text('{"discarded_cpu_fixture_step":1}\n{"discarded_cpu_fixture_step":2}\n')
        checkpoint = save_generation(model, optimizer, 2, plan, plan_sha, run)
        audit = audit_saved(checkpoint, plan, plan_sha)
        loaded, payload = load_model(checkpoint)
        require(state_sha256(model.state_dict()) == payload["model_state_sha256"], "Saved state changed")
        with torch.inference_mode():
            live, replay = model.eval().render(audio), loaded.render(audio)
            require(all(torch.equal(getattr(live, k).view(torch.int32), getattr(replay, k).view(torch.int32))
                        for k in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                    and all(torch.equal(a.view(torch.int32), b.view(torch.int32))
                            for a, b in zip(live.state, replay.state, strict=True)),
                    "Saved quadrature output or state differs at the bit level")
        rejected = []
        for name, operation, message in (
            ("wrong_file_binding", lambda: load_model({**checkpoint, "sha256": "0" * 64}),
             "Quadrature checkpoint bytes changed"),
            ("wrong_training_plan", lambda: audit_saved(checkpoint, plan, "0" * 64),
             "Saved quadrature endpoint or parent differs from its plan"),
        ):
            try:
                operation()
            except RuntimeError as error:
                require(str(error) == message, "Unexpected checkpoint rejection reason")
                rejected.append(name)
            else:
                raise RuntimeError("Invalid fixture checkpoint was accepted: " + name)
        sizes = {p.name: p.stat().st_size for p in (run / "checkpoint").iterdir()}
        # The generation is deliberately removed by TemporaryDirectory, including
        # both model and optimizer; retain measurements, never a broken file link.
        audit.pop("checkpoint")
    require(not Path(temporary).exists() and state_sha256(parent.state_dict()) == fingerprint
            and not torch.cuda.is_initialized(), "Fixture cleanup, parent preservation or CPU scope failed")
    return {"status": "pass", "saved_audit": audit, "all_24_neural_tensors_updated": changed,
            "saved_output_and_state_replay_bit_exact": True, "rejected_invalid_cases": rejected,
            "temporary_checkpoint_bytes": sizes, "temporary_checkpoint_removed": True,
            "parent_state_unchanged": True, "gpu_used": False, "quality_measured": False,
            "production_checkpoint_written": False, "discarded_fixture_updates": 2}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="quadrature-checkpoint-functional-001")
    args = parser.parse_args()
    require(bool(args.name) and all(c.isalnum() or c in "-_" for c in args.name)
            and Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require a new CPU1 evidence directory with CUDA hidden")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "remix-magnitude-001/plan.json"
    source = read(source_path)
    paths = [source_path, Path(__file__).resolve()]
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_quadrature.py", "latency58_quadrature_context.py", "latency58_quadrature_checkpoint.py",
        "check_latency58_quadrature_gpu.py", "evaluate_latency58_quadrature.py"))
    bindings = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    verify_inputs({"source_bindings": bindings})
    out = PHASE / args.name
    require(not out.exists(), "Preserve completed checkpoint checks")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "parent_checkpoint": source["parent_checkpoint"],
          "scope": "CPU checkpoint round trip; no GPU, quality scoring or production training"})
    began = time.monotonic()
    result = check(source, out)
    verify_inputs({"source_bindings": bindings})
    result.update(source_bindings_unchanged=True, plan_sha256=sha(out / "plan.json"),
                  elapsed_seconds=time.monotonic() - began)
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "saved_output_and_state_replay_bit_exact",
                     "temporary_checkpoint_removed", "quality_measured", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
