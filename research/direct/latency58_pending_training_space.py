"""Keep a pending quadrature save reserved during concurrent CPU qualification."""
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha
from research.direct.latency58_sdr_checkpoint import require_space


def require_cpu_space(plan, extra_bytes):
    require(plan["schema"] == "latency58-quadrature-training-plan-v1" and extra_bytes >= 0,
            "Require an authenticated quadrature budget and a nonnegative CPU reservation")
    root = Path(plan["output_directory"])
    final, pending = root / "production-run/checkpoint", root / "production-run/checkpoint.pending"
    # The training writer atomically renames its complete generation. Avoid
    # counting a partially written generation and a second full reservation.
    for _ in range(31):
        if pending.exists():
            time.sleep(1)
            continue
        reserved = 380_000_000
        if final.exists():
            receipt = read(final / "receipt.json")
            require(receipt["schema"] == "latency58-quadrature-generation-v1"
                    and receipt["plan_sha256"] == sha(root / "plan.json")
                    and receipt["step"] == plan["config"]["steps"]
                    and set(receipt["files"]) == {"model.pt", "optimizer.pt"}
                    and all((final / name).is_file() and not (final / name).is_symlink()
                            and (final / name).stat().st_size == value["bytes"]
                            for name, value in receipt["files"].items()),
                    "Completed training generation does not match its reservation")
            reserved = 0
        try:
            return require_space(plan, extra_bytes + reserved)
        except RuntimeError:
            if pending.exists() or (reserved and final.exists()):
                continue
            raise
    raise RuntimeError("Training checkpoint write has not completed; preserve it and inspect the writer")
