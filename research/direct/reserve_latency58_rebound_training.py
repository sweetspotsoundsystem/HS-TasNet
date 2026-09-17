"""Audit obsolete intermediate resumes before bounded plugin-build retirement."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, state_sha256, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Require frozen CPU audit")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-rebound-training-storage-plan-v1"
            and plan["steps"] == [750, 775]
            and plan["reserve_after_bytes"] == 1_000_000_000, "Different bounded retirement")
    verify_inputs(plan)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = ROOT / "research/direct/runs/latency11/ola512-right-baked-gpu-b4-bf16-projection01-lr3e-5"
    status = read(run / "status.json")
    require(status["status"] == "complete" and status["step"] == 2000
            and not Path("/proc", str(status["pid"])).exists(), "Old study is not complete")
    protected = {}
    for name in ("status.json", "latest.json", "config.json", "metrics.jsonl"):
        protected[str(run / name)] = sha(run / name)
    for name in ("resume.pt", "model.pt", "metrics.jsonl", "receipt.json"):
        path = run / "checkpoints/step-002000" / name
        protected[str(path)] = sha(path)
    for binding in plan["active_plans"]:
        require(sha(binding["path"]) == binding["sha256"], "Active plan changed")
        active = read(binding["path"])
        verify_inputs(active)
        protected.update(active["source_bindings"])
        protected[binding["path"]] = binding["sha256"]
    protected.update(plan["rollback_bindings"])
    remove = []
    for step in plan["steps"]:
        generation = run / "checkpoints" / f"step-{step:06d}"
        resume, model = generation / "resume.pt", generation / "model.pt"
        receipt = read(generation / "receipt.json")
        require(not resume.is_symlink() and sha(resume) == receipt["files"]["resume.pt"]
                and sha(model) == receipt["files"]["model.pt"] and receipt["status"] == "pass"
                and receipt["plan_sha256"] == status["latest_checkpoint"]["plan_sha256"],
                "Old generation identity differs")
        # Locally produced, hash-authenticated research resumes include RNG objects.
        original = torch.load(resume, map_location="cpu", weights_only=False)
        retained = torch.load(model, map_location="cpu", weights_only=True)
        require(original["step"] == retained["step"] == step
                and set(original["model"]) == set(retained["model"])
                and all(torch.equal(value, retained["model"][key]) for key, value in original["model"].items())
                and state_sha256(retained["model"]) == receipt["model_state_sha256"],
                "Retained inference state does not preserve every tensor")
        remove.append({"path": str(resume), "sha256": sha(resume), "bytes": resume.stat().st_size,
                       "step": step, "retained_model": str(model),
                       "all_model_tensors_preserved_exactly": True,
                       "model_state_sha256": receipt["model_state_sha256"]})
        for name in ("model.pt", "receipt.json", "metrics.jsonl"):
            path = generation / name
            protected[str(path)] = sha(path)
        del original, retained
    require(set(protected).isdisjoint(row["path"] for row in remove), "A protected input was marked for retirement")
    require(all(plan["source_bindings"].get(p) == s for p, s in protected.items()), "Unbound protected input")
    counted = sum(disk_bytes(Path(p)) for p in plan["counted_roots"])
    freed = sum(row["bytes"] for row in remove)
    require(counted - freed + plan["reserve_after_bytes"] < plan["stop_counted_bytes"],
            "Retirement does not provide the bounded build reserve")
    require(not torch.cuda.is_initialized(), "Audit used CUDA")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir(), "Unexpected audit output")
    write(out / "intent.json", {
        "schema": "latency58-rebound-training-storage-intent-v1", "status": "audited",
        "plan_sha256": args.plan_sha256, "remove": remove, "protected_bindings": protected,
        "reason": "Reserve the current training checkpoint, a further bounded training block, and CPU quality artifacts under the existing cap. Retire only two obsolete intermediate full resumes at steps 750 and 775; preserve every corresponding inference tensor and the final step-2000 full resume. Same-Adam/RNG restart at these two old intermediate steps is intentionally retired.",
        "cpu_tensor_equivalence_checked": True, "cuda_initialized": False,
        "counted_roots": plan["counted_roots"], "stop_counted_bytes": plan["stop_counted_bytes"],
        "counted_bytes_before": counted, "expected_freed_bytes": freed,
        "source_bindings_unchanged": True})
    print(json.dumps({"status": "audited", "files": len(remove), "freed_bytes_if_applied": freed}), flush=True)


if __name__ == "__main__":
    main()
