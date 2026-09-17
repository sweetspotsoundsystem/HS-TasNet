"""Freeze scoring adapters and bind both live prerequisites for automatic scoring."""
from __future__ import annotations
import json
import os
from pathlib import Path
from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.prove_latency58_quarter_controlled import normalized


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def identity(module, plan):
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = [part.decode() for part in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
            if "-m" not in argv or argv[argv.index("-m") + 1] != module:
                continue
            require(argv[argv.index("--plan") + 1] == str(plan)
                    and argv[argv.index("--plan-sha256") + 1] == sha(plan), "Live prerequisite plan differs")
            ticks = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
            found.append({"pid": int(proc.name), "start_ticks": ticks, "argv": argv})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    require(len(found) == 1, "Need one live prerequisite: " + module)
    return found[0]


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU freezing required")
    training_queue = PHASE / "quarter-controlled-queue-001/plan.json"
    reference_queue = PHASE / "cleanup-lr3e6-quality-prep-001/plan.json"
    training_supervisor = identity("research.direct.queue_latency58_quarter_controlled", training_queue)
    reference_supervisor = identity("research.direct.prepare_latency58_cleanup_lr3e6_quality", reference_queue)
    source_dir = ROOT / "research/direct"
    modules = ("evaluate_latency58_cleanup_lr3e6", "evaluate_latency58_cleanup_lr3e6_parallel",
               "evaluate_latency58_cleanup_lr3e6_probes", "evaluate_latency58_cleanup_lr3e6_views",
               "run_latency58_cleanup_lr3e6_quality", "run_latency58_cleanup_lr3e6_views")
    proof = []
    for name in modules:
        original = source_dir / (name + ".py")
        target = source_dir / (name.replace("cleanup_lr3e6", "quarter_controlled") + ".py")
        require(normalized(original, (("cleanup_lr3e6", "quarter_controlled"),
                                      ("cleanup-lr3e6", "quarter-controlled"))) == normalized(target),
                "Quality arithmetic changed: " + name)
        proof.append({"source": binding(original), "target": binding(target), "numeric_ast_identical_after_family_rename": True})
    out = PHASE / "quarter-controlled-quality-prep-001"
    require(not out.exists(), "Preserve existing quality preparation")
    out.mkdir()
    write(out / "adapter-proof.json", {"status": "pass", "comparisons": proof,
          "other_changes": "Additional completed lower-rate reference, matched view-frequency metadata and prerequisite terminal routing only"})
    bindings = {}
    for path in (training_queue, reference_queue):
        plan = read(path)
        verify_inputs(plan)
        bindings.update(plan["source_bindings"])
        bindings[str(path)] = sha(path)
    for path in (out / "adapter-proof.json", *source_dir.glob("*quarter_controlled*.py")):
        bindings[str(path)] = sha(path)
    prospective = {"schema": "latency58-quarter-controlled-quality-preparation-v1", "maximum_wait_seconds": 12000,
                   "output_directory": str(out), "training_supervisor": training_supervisor,
                   "reference_supervisor": reference_supervisor, "source_bindings": bindings,
                   "qualification_blocks_training": False, "quality_selection_committed": False,
                   "confirmation_policy": "Additional reserved windows remain unscored until one candidate is frozen after selection review."}
    verify_inputs(prospective)
    write(out / "plan.json", prospective)
    print(json.dumps({"status": "prepared", "plan": binding(out / "plan.json"),
                      "training_supervisor": training_supervisor, "reference_supervisor": reference_supervisor}), flush=True)


if __name__ == "__main__":
    main()
