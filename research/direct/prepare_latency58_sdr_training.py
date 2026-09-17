"""Freeze the matched teacher comparison before either arm trains."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from research.direct.train_latency58 import PRODUCTION, ROOT, read, require, sha, verify_inputs
from research.direct.run_latency58_quality import write


def main():
    phase = ROOT / "research/direct/runs/latency58"
    out = phase / "sdr-teacher-prep-001"
    require(Path.cwd() == ROOT and not out.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Prepare once on CPU with CUDA hidden")
    functional_dir = phase / "sdr-teacher-functional-001"
    functional_plan, functional = read(functional_dir / "plan.json"), read(functional_dir / "result.json")
    verify_inputs(functional_plan)
    require(functional["status"] == "pass" and read(functional_dir / "execution.json")["actual_exit_code"] == 0,
            "Teacher functional proof did not complete")
    resources = {}
    for kind in ("cropped11", "c91"):
        directory = phase / f"sdr-resource-{kind}-full-001"
        plan, result, execution = read(directory / "plan.json"), read(directory / "result.json"), read(directory / "execution.json")
        verify_inputs(plan)
        monitor = read(execution["monitor_result"])
        require(result["status"] == "pass" and result["samples"] == 88064 and result["teacher_kind"] == kind
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and monitor["status"] == monitor["supervisor_health"] == "pass"
                and monitor["post_exit_quiet_completed"] and monitor["child_exit_code"] == 0,
                "Full resource fixture or its actual monitor failed")
        resources[kind] = directory
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_sdr_teacher import STUDENT_SHA256, STUDENT_STATE_SHA256
    old = read(phase / "teacher-prep/half-plan.json")
    limits = read(phase / "single-thread-rt-prep-001/syntax-final-plan.json")
    accounting = {key: limits[key] for key in ("counted_roots", "stop_counted_bytes")}
    require_space(accounting, 1_600_000_000)
    import torch
    require(not torch.cuda.is_initialized(), "Plan preparation initialized CUDA")
    checkpoint = phase / "teacher-half-canonical-001/model.pt"
    require(sha(checkpoint) == STUDENT_SHA256, "Current model changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parent_provenance = payload["provenance"]
    require(payload["model_state_sha256"] == STUDENT_STATE_SHA256 and parent_provenance["training_updates"] == 5000,
            "Current canonical lineage differs")
    config = copy.deepcopy(old["config"])
    config.update(steps=1000, checkpoint_every=250, seed=20260908, data_start=916000)
    common = {"schema": "latency58-sdr-training-v1", "config": config, "teacher_weight": 0.5,
              "objective": "raw4_l1_plus_teacher_l1", "environment": old["environment"],
              "torch_version": old["torch_version"], "precision_policy": old["precision_policy"],
              "helper_source": old["helper_source"], "watchdog_source": old["watchdog_source"],
              "manifest_sha256": old["manifest_sha256"], "initial_model_state_sha256": STUDENT_STATE_SHA256,
              "parent_checkpoint": {"path": str(checkpoint), "sha256": STUDENT_SHA256},
              "parent_provenance": parent_provenance, **accounting}
    out.mkdir()
    design = {"schema": "latency58-sdr-matched-design-v1", "config": config,
              "teacher_kinds": ["cropped11", "c91"], "teacher_weight": 0.5,
              "parent_checkpoint": common["parent_checkpoint"], "initial_model_state_sha256": STUDENT_STATE_SHA256,
              "quality_endpoints": [250, 500, 1000], "first_saved_state_check": 2,
              "hypothesis": "A stronger training-only teacher and a longer fixed fine-tune may recover SDR without changing inference geometry.",
              "matching": "Same initial student, fresh Adam, learning-rate horizon, crop addresses and augmentation seeds. Verify every batch digest and LR; only the frozen teacher differs.",
              "continuation": "Stages pause for saved-state audit; full-panel evidence decides continuation beyond quality endpoints. No endpoint is selected from training loss.",
              "checkpoint_policy": "Atomic model/optimizer/RNG/journal generations. Preserve inference models at checkpoints; retire only replaced optimizer files after the new generation passes a separate audit.",
              "primary_protocol": "Unchanged 14 tracks, 30–45 and 75–90 seconds, CPU FP32 continuous state and native gains.",
              "inference_geometry": {"graph_delay_samples": 128, "queue_samples": 128, "sample_rate": 44100}}
    write(out / "matched-design.json", design)
    manifest = read(ROOT / "research/manifests/valid.json")
    require(all(t["frames"] >= 150 * 44100 for t in manifest["tracks"]), "Confirmation interval exceeds a track")
    confirmation = {"schema": "latency58-sdr-confirmation-plan-v1", "track_names": [t["name"] for t in manifest["tracks"]],
                    "manifest": str(ROOT / "research/manifests/valid.json"),
                    "manifest_sha256": sha(ROOT / "research/manifests/valid.json"),
                    "excerpt_starts": [105.0, 135.0], "duration_seconds": 15.0,
                    "use": "Evaluate the current baseline and selected candidate only after primary-panel selection. These intervals do not guide this matched trial's checkpoint selection.",
                    "limitation": "The tracks have already been used for development. This is additional within-track confirmation, not an unseen-track or sealed-test result."}
    write(out / "confirmation-plan.json", confirmation)
    binds = dict(functional_plan["source_bindings"])
    paths = [functional_dir / name for name in ("plan.json", "result.json", "execution.json")]
    paths += [out / "matched-design.json", out / "confirmation-plan.json", Path(__file__),
              PRODUCTION / "train_production.py", PRODUCTION / "full_config.json", PRODUCTION / "manifests/combined.manifest.json",
              ROOT / "research/experiment.py", ROOT / "research/direct/latency_ola512_training.py",
              ROOT / "research/direct/run_latency58_quality.py", Path(old["helper_source"]), Path(old["watchdog_source"])]
    paths += [ROOT / "research/direct" / name for name in
              ("latency58_sdr_checkpoint.py", "train_latency58_sdr.py", "audit_latency58_sdr.py", "run_latency58_sdr_stage.py")]
    for directory in resources.values():
        paths += [directory / name for name in ("plan.json", "result.json", "execution.json")]
        paths.append(Path(read(directory / "execution.json")["monitor_result"]))
    for path in paths:
        binds[str(path.resolve())] = sha(path)
    plans = {}
    for kind in ("cropped11", "c91"):
        directory = resources[kind]
        plan = copy.deepcopy(common)
        plan.update(teacher_kind=kind, teacher=functional_plan["teachers"][kind],
                    teacher_model_state_sha256=functional["teachers"][kind]["identity"]["model_state_sha256"],
                    run_dir=str(phase / f"sdr-teacher-{kind}-b4-lr3e5-1000"), source_bindings=binds,
                    full_resource={"path": str(directory / "result.json"), "sha256": sha(directory / "result.json")},
                    full_resource_execution={"path": str(directory / "execution.json"), "sha256": sha(directory / "execution.json")})
        verify_inputs(plan)
        path = out / f"{kind}-plan.json"
        write(path, plan)
        plans[kind] = {"path": str(path), "sha256": sha(path)}
    require(not torch.cuda.is_initialized(), "Plan preparation used CUDA")
    write(out / "preparation-result.json", {"status": "prepared", "plans": plans,
                                            "matched_design_sha256": sha(out / "matched-design.json"),
                                            "source_bindings_unchanged": True, "new_training_started": False})
    print(json.dumps(plans, indent=2))


if __name__ == "__main__":
    main()
