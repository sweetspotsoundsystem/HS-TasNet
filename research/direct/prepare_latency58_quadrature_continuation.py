"""Freeze the reviewed 4,000-update continuation from the saved quadrature pilot."""
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def main():
    import torch
    from research.direct.latency58_quadrature_checkpoint import load_model, audit_saved
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require single-threaded CPU preparation")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    parent_root = PHASE / "quadrature-001"
    source_path = parent_root / "plan.json"
    source, review = read(source_path), read(parent_root / "selection-review.json")
    verify_inputs(source)
    verify_inputs(review)
    terminal, quality = read(parent_root / "result.json"), read(parent_root / "full14/result.json")
    recipe = review["next_direction"]
    require(review["status"] == "selected_for_research" and review["actual_root_exit_code"] == 0
            and terminal["status"] == "training_audit_and_full14_complete"
            and terminal["full_sdr_db"] == 4.227699923177355 < 5.0
            and quality["status"] == "pass" and quality["track_count"] == 14 and quality["excerpt_count"] == 28
            and quality["results"][0]["checkpoint"] == review["best_research_checkpoint"] == terminal["checkpoint"]
            and recipe["steps"] == 4000 and recipe["lr"] == 6e-5 and recipe["min_lr"] == 6e-6
            and recipe["data_start"] == 2_900_000 and recipe["seed"] == 20261014
            and recipe["optimizer_initialization"] == "fresh_adam", "Reviewed continuation changed")
    for name in ("production-stage/execution.json", "full14/execution.json"):
        execution = read(parent_root / name)
        require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"], "Parent is incomplete")
    audit = audit_saved(terminal["checkpoint"], source, sha(source_path))
    model, payload = load_model(terminal["checkpoint"])
    fingerprint = state_sha256(model.state_dict())
    require(audit["status"] == "pass" and audit["saved_optimizer_tensor_count"] == 24
            and payload["step"] == 1000 and model.provenance["training_updates"] == 15250,
            "Saved parent audit or lineage failed")
    with torch.inference_mode():
        audio = .03 * torch.randn(2, 2, 19 * 128, generator=torch.Generator().manual_seed(202609129))
        first, second = model.render(audio), model.render(audio)
        require(all(torch.equal(getattr(first, key).view(torch.int32), getattr(second, key).view(torch.int32))
                    for key in ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
                and all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(first.state, second.state, strict=True)),
                "Saved parent reset replay changed")
    out = PHASE / "quadrature-continuation-001"
    require(not out.exists(), "Preserve prior plans")
    paths = [Path(__file__).resolve(), source_path]
    paths.extend(parent_root / name for name in (
        "selection-review.json", "result.json", "full14/result.json", "full14/execution.json",
        "production-stage/execution.json", "checkpoint-audit.json", "production-run/checkpoint/receipt.json",
        "production-run/checkpoint/model.pt", "production-run/checkpoint/optimizer.pt"))
    paths.extend(ROOT / "research/direct" / name for name in (
        "train_latency58_quadrature_continuation.py", "run_latency58_quadrature_continuation.py"))
    retirement_root = PHASE / "long-magnitude-optimizer-retirement-001"
    require(read(retirement_root / "receipt.json")["status"] == "complete", "Space recovery incomplete")
    paths.extend(retirement_root / name for name in ("intent.json", "receipt.json"))
    config = {**source["config"], **{k: recipe[k] for k in ("steps", "lr", "min_lr", "warmup", "data_start", "seed")},
              "checkpoint_every": recipe["steps"]}
    plan = {**source, "name": out.name, "output_directory": str(out), "config": config,
            "source_bindings": {**source["source_bindings"], **{str(path): sha(path) for path in paths}},
            "parent_checkpoint": terminal["checkpoint"], "parent_kind": "saved_quadrature",
            "optimizer_initialization": "fresh_adam", "reference_result": str(parent_root / "full14/result.json"),
            "parent_model_state_sha256": fingerprint, "parent_training_updates": 15250,
            "initialized_model_state_sha256": fingerprint, "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
            "quality_endpoints": [4000], "inference_architecture_changed": False,
            "continuation_uses_trained_nonzero_phase_head": True, "native_cost_measured": False}
    # Reserve the new Mac comparison archive in addition to this training run.
    counted = require_space(plan, 452_000_000)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized() and state_sha256(model.state_dict()) == fingerprint, "Preparation changed the parent")
    out.mkdir()
    write(out / "preparation.json", {"status": "pass", "parent_full_sdr_db": terminal["full_sdr_db"],
          "parent_audit": audit, "saved_parent_reset_replay_bit_exact": True,
          "all_24_tensors_trainable_in_production": True, "optimizer_initialization": "fresh_adam",
          "target_full_sdr_db": 5.0, "graph_plus_host_samples": 256, "recipe": recipe,
          "counted_bytes_before": counted, "reserved_training_bytes": 390_000_000,
          "reserved_mac_archive_bytes": 62_000_000,
          "forecast_including_outside_and_run_and_archive": counted + 800_000_000 + 452_000_000,
          "cuda_initialized": False, "limitation": "A new training schedule from the saved pilot, not an exact optimizer continuation."})
    plan["source_bindings"][str(out / "preparation.json")] = sha(out / "preparation.json")
    write(out / "plan.json", plan)
    print(json.dumps({"event": "prepared", "plan": str(out / "plan.json"), "sha256": sha(out / "plan.json")}), flush=True)


if __name__ == "__main__":
    main()
