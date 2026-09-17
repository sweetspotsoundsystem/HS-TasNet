"""Fund full-model reconstruction by retiring the closed failed MUSDB Adam state."""
from __future__ import annotations

import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import disk_bytes, verify_inputs


def same(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    import torch
    torch.set_num_threads(1)
    out = PHASE / "wave-spectral-storage-001"
    require(Path.cwd() == ROOT and not out.exists(), "Preserve storage evidence")
    protected = {str(Path(__file__).resolve()): sha(__file__)}
    for name in ("direct-sdr-musdb-001", "magnitude-sdr-001"):
        root = PHASE / name
        result, execution, audit = (read(root / p) for p in ("result.json", "full14/execution.json", "checkpoint-audit.json"))
        require(result["status"] == "training_audit_and_full14_complete" and not result["target_reached"]
                and result["full_sdr_db"] < 4.0846618609770395 and execution["actual_exit_code"] == 0
                and not execution["timed_out"] and execution["source_bindings_unchanged"] and audit["status"] == "pass",
                "The MUSDB and magnitude pilots must be closed lower-SDR endpoints")
        for relative in ("plan.json", "result.json", "checkpoint-audit.json", "full14/plan.json", "full14/result.json",
                         "full14/execution.json", "production-stage/execution.json", "production-run/result.json"):
            path = root / relative
            protected.update(read(path).get("source_bindings", {}))
            protected[str(path)] = sha(path)
        for path in (root / "production-run/checkpoint").iterdir():
            if not (name == "direct-sdr-musdb-001" and path.name == "optimizer.pt"):
                protected[str(path)] = sha(path)
    baselines = read(PHASE / "c204-residual-model-001/final-review.json")
    for key in ("checkpoint", "onnx", "c204_checkpoint_preserved", "working_plugin_graph_preserved"):
        binding = baselines[key]
        require(sha(binding["path"]) == binding["sha256"], "Rollback baseline differs")
        protected[binding["path"]] = binding["sha256"]
    generation = PHASE / "direct-sdr-musdb-001/production-run/checkpoint"
    target = generation / "optimizer.pt"
    binding = read(generation / "receipt.json")["files"][target.name]
    require(target.is_file() and not target.is_symlink() and sha(target) == binding["sha256"]
            and target.stat().st_size == binding["bytes"] and str(target) not in protected, "Adam state differs or is bound")
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = proc.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(not any(arg.startswith(str(generation).encode()) for arg in arguments), "A live process uses the generation")
    verify_inputs({"source_bindings": protected})
    budget = read(PHASE / "direct-sdr-musdb-001/plan.json")
    before = sum(disk_bytes(Path(p)) for p in budget["counted_roots"])
    require(before - binding["bytes"] + 800_000_000 + 350_000_000 + 1_000_000 < 80_000_000_000,
            "Retirement cannot fund full-model waveform/spectral training")
    resume = torch.load(target, map_location="cpu", weights_only=True)
    rng = {k: v for k, v in resume.items() if k != "optimizer"}
    require(set(rng) == {"torch_rng", "cuda_rng", "python_rng", "numpy_rng", "next_sample_index", "step",
                         "model_state_sha256", "plan_sha256"}, "Unexpected resume inventory")
    out.mkdir()
    retired = {"path": str(target), **binding}
    write(out / "intent.json", {"retired": retired, "protected_bindings": protected, "counted_bytes_before": before,
          "reason": "Retire closed failed pilot Adam moments; keep all inference models and preserve the exact RNG payload separately."})
    with (out / "rng.pt").open("xb") as stream:
        torch.save(rng, stream)
        stream.flush()
        os.fsync(stream.fileno())
    require(same(rng, torch.load(out / "rng.pt", map_location="cpu", weights_only=True)), "RNG round trip differs")
    require(sha(target) == binding["sha256"], "Adam state changed before retirement")
    target.unlink()
    verify_inputs({"source_bindings": protected})
    after = sum(disk_bytes(Path(p)) for p in budget["counted_roots"])
    write(out / "receipt.json", {"status": "complete", "intent_sha256": sha(out / "intent.json"), "retired": retired,
          "retained_rng": {"path": str(out / "rng.pt"), "sha256": sha(out / "rng.pt")}, "rng_round_trip_exact": True,
          "adam_resume_available_for_retired_pilot": False, "protected_bindings_unchanged": True,
          "counted_bytes_after": after, "forecast_including_outside_and_wave_spectral": after + 800_000_000 + 350_000_000})
    print(json.dumps(read(out / "receipt.json")), flush=True)


if __name__ == "__main__":
    main()
