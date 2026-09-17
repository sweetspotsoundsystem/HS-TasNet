"""Read and audit the first published packed generation while training continues."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.latency58_lossless_recovery_files_v3 import read_snapshot, load_inference, CURRENT, PENDING
from research.direct.latency58_weighted_storage import snapshot as storage_snapshot

MAIN = PHASE / "branch-weighted-vocal-014"
OUT = MAIN / "first-save-audit"
EXPECTED_STEP = 50


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1 checkpoint inspection")
    require(not OUT.exists(), "Preserve previous first-save audit")
    began = time.monotonic()
    plan_path = MAIN / "plan.json"
    plan, plan_sha = read(plan_path), sha(plan_path)
    require(plan["name"] == MAIN.name and plan["config"]["steps"] == 1000, "Wrong training trial")
    run = MAIN / "production-run"
    current = run / CURRENT
    receipt_path = run / "packed-recovery-receipts" / f"step-{EXPECTED_STEP:06d}.json"
    receipt = read(receipt_path)
    require(receipt["step"] == EXPECTED_STEP and receipt["planned_stop_step"] == 1000
            and receipt["plan_sha256"] == plan_sha and receipt["previous"] is None
            and current.is_file() and not current.is_symlink() and not (run / PENDING).exists()
            and current.stat().st_size == receipt["bytes"] <= 380_000_000,
            "The first generation is not fully published")
    monitor_dir = Path(plan["watchdog_source"]).parent / (MAIN.name + "-production")
    published = [json.loads(line) for line in (monitor_dir / "child.log").read_text().splitlines()
                 if line.startswith('{"event": "recovery_saved"')]
    published = [row for row in published if row["step"] == EXPECTED_STEP]
    require(len(published) == 1 and published[0]["sha256"] == receipt["sha256"]
            and published[0]["receipt_sha256"] == sha(receipt_path)
            and published[0]["complete_save_seconds"] < 60, "The trainer did not confirm the complete bounded save")
    published = published[0]
    frozen_paths = [plan_path, receipt_path, Path(__file__).resolve(), Path(plan["parent_checkpoint"]["path"])]
    bindings = {path: digest for path, digest in plan["source_bindings"].items() if Path(path).suffix == ".py"}
    bindings.update({str(path): sha(path) for path in frozen_paths})
    require(bindings[plan["parent_checkpoint"]["path"]] == plan["parent_checkpoint"]["sha256"], "Packed XOR parent changed")
    verify_inputs({"source_bindings": bindings})
    before = storage_snapshot(plan)
    bound = {"path": str(current), "sha256": receipt["sha256"], "receipt": str(receipt_path),
             "receipt_sha256": sha(receipt_path), "step": EXPECTED_STEP}
    import torch
    from research.direct.check_latency58_lossless_recovery_cpu_v3 import render_parity
    from research.direct.profile_latency58_lossless_checkpoint import tensor_bytes
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cpu_rng = torch.get_rng_state().clone()
    snapshot, audited = read_snapshot(bound, plan, plan_sha)
    raw, resume, averaged, ema = audited
    complete_lines = (run / "metrics.jsonl").read_bytes().splitlines(keepends=True)
    require(len(complete_lines) >= EXPECTED_STEP and all(line.endswith(b"\n") for line in complete_lines[:EXPECTED_STEP])
            and snapshot["journal"] == b"".join(complete_lines[:EXPECTED_STEP])
            and snapshot["step"] == resume["step"] == ema.updates == EXPECTED_STEP
            and snapshot["planned_stop_step"] == 1000 and snapshot["training_config"] == plan["config"]
            and state_sha256(raw.state_dict()) == receipt["raw_model_state_sha256"]
            and state_sha256(averaged.state_dict()) == receipt["ema_model_state_sha256"]
            and resume["next_sample_index"] == plan["config"]["data_start"] + EXPECTED_STEP * 16
            and len(resume["cuda_rng"]) == 1, "Saved journal, optimizer, EMA, RNG or data cursor differs")
    states = resume["optimizer"]["state"]
    ids = resume["optimizer"]["param_groups"][0]["params"]
    require(len(states) == len(ids) == len(resume["parameter_names"]) == 40, "Incomplete saved Adam inventory")
    adam = {}
    for name, identity in zip(resume["parameter_names"], ids, strict=True):
        state = states[identity]
        require(state["step"].item() == EXPECTED_STEP, "Saved Adam step differs")
        adam[name] = {"step": EXPECTED_STEP, **{key + "_sha256": hashlib.sha256(tensor_bytes(state[key])).hexdigest()
                                              for key in ("exp_avg", "exp_avg_sq")}}
    parity = {}
    for role, expected in (("raw", raw), ("ema", averaged)):
        loaded, payload = load_inference(bound, plan, plan_sha, role=role)
        require(payload["model_state_sha256"] == state_sha256(expected.state_dict())
                and payload["provenance"]["training_updates"] == plan["parent_training_updates"] + EXPECTED_STEP,
                "Independent packed role load differs")
        parity[role] = render_parity(expected, loaded)
        del loaded, payload
    require(sha(current) == bound["sha256"] and sha(receipt_path) == bound["receipt_sha256"]
            and current.stat().st_size == receipt["bytes"]
            and torch.equal(cpu_rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "The rolling file changed during inspection or CPU/RNG scope changed")
    verify_inputs({"source_bindings": bindings})
    OUT.mkdir()
    prepared = {"schema": "latency58-weighted-first-save-audit-plan-v1", "source_bindings": bindings,
        "training_plan": {"path": str(plan_path), "sha256": plan_sha}, "expected_step": EXPECTED_STEP,
        "observed_checkpoint": bound, "rolling_file_is_mutable_after_inspection": True,
        "rolling_file_is_not_a_permanent_source_binding": True, "storage_before": before}
    write(OUT / "plan.json", prepared)
    result = {"schema": "latency58-weighted-first-save-audit-result-v1", "status": "pass",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha(OUT / "plan.json"),
        "source_bindings_unchanged": True, "observed_checkpoint": bound, "packed_bytes": receipt["bytes"],
        "complete_save_seconds": published["complete_save_seconds"], "packing": published["packing"],
        "saved_step": EXPECTED_STEP, "planned_stop_step": 1000, "next_sample_index": resume["next_sample_index"],
        "raw_model_state_sha256": receipt["raw_model_state_sha256"], "ema_model_state_sha256": receipt["ema_model_state_sha256"],
        "all_40_saved_adam_states": adam, "raw_and_ema_independent_disk_load_native_parity": parity,
        "complete_saved_journal_matches_production_prefix": True, "saved_python_numpy_cpu_cuda_rng_validated": True,
        "cpu_rng_unchanged": True, "algorithmic_latency_samples": 256, "rolling_file_unchanged_during_inspection": True,
        "elapsed_seconds": time.monotonic() - began, "storage_after": storage_snapshot(plan),
        "gpu_used": False, "tensor_files_written": False, "live_gpu_weights_read": False,
        "training_interrupted": False, "training_complete": False, "quality_measured": False, "goal_complete": False}
    write(OUT / "result.json", result)
    print(json.dumps({"status": "pass", "saved_step": EXPECTED_STEP, "packed_bytes": receipt["bytes"],
        "complete_save_seconds": result["complete_save_seconds"], "both_saved_roles_native_parity": True,
        "all_40_adam_states_audited": True, "gpu_used": False}), flush=True)


if __name__ == "__main__":
    main()
