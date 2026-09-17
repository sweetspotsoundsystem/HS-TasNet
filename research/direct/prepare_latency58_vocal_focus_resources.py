"""Freeze three matched GPU rehearsals from the reviewed working-model parent."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(args.protocol) == args.protocol_sha256
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Require frozen CPU1 preparation")
    protocol = read(args.protocol)
    verify_inputs(protocol)
    require(protocol["schema"] == "latency58-vocal-focus-protocol-v1"
            and protocol["arms"] == ["original", "focused", "focused_mixer"]
            and protocol["maximum_production_updates"] == 750
            and protocol["maximum_unique_examples"] == 4000
            and protocol["quality_endpoints"] == [250]
            and not protocol["automatic_continuation"], "Unexpected pilot scope")
    decision_binding = protocol["preparation_decision"]
    require(sha(decision_binding["path"]) == decision_binding["sha256"], "Parent decision changed")
    decision = read(decision_binding["path"])
    verify_inputs(decision)
    require(decision["status"] == "prepare_pilots" and decision["history_review_complete"]
            and decision["baseline_vocal_views_review_complete"]
            and decision["priority"] == "vocal_cleanliness" and decision["pilot_updates_per_arm"] == 250,
            "Incomplete common-parent review")
    for proof in protocol["functional_proofs"]:
        result, execution = read(proof["result"]), read(proof["execution"])
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"], "A functional prerequisite failed")
    import torch
    from research.direct.latency58_sdr_teacher import load_initial_student, STUDENT_SHA256, STUDENT_STATE_SHA256
    from research.direct.latency58_vocal_focus_model import LocalMaskMixerModel
    from research.direct.latency58_vocal_focus_checkpoint import load_parent, require_space, validate_recipe

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = torch.get_rng_state().clone()
    parent = load_initial_student().eval()
    require(state_sha256(parent.state_dict()) == decision["parent_model_state_sha256"] == STUDENT_STATE_SHA256,
            "Common parent differs from the completed review")
    extended = LocalMaskMixerModel.from_parent(parent, initialization_seed=protocol["mixer_initialization_seed"]).eval()
    with torch.inference_mode():
        audio = torch.linspace(-.02, .02, 2048).reshape(1, 2, 1024)
        original, zero = parent.render(audio), extended.render(audio)
        require(all(torch.equal(getattr(original, k), getattr(zero, k))
                    for k in ("raw", "deployed", "spectral", "waveform", "delayed_mixture"))
                and all(torch.equal(a, b) for a, b in zip(original.state, zero.state, strict=True)),
                "Selected-parent zero mixer changes synthetic streaming output")
    descriptor = {"kind": "working", "checkpoint": {
        "kind": "inference", "path": str(PHASE / "teacher-half-canonical-001/model.pt"), "sha256": STUDENT_SHA256},
        "model_state_sha256": STUDENT_STATE_SHA256, "provenance": parent.provenance,
        "architecture": parent.architecture_metadata}
    before = require_space(protocol, 10_000_000)
    out = args.protocol.parent
    require(out.is_relative_to(PHASE) and not (out / "resource-preparation.json").exists(), "Preserve preparation")
    base = {k: copy.deepcopy(protocol[k]) for k in (
        "config", "environment", "torch_version", "precision_policy", "helper_source", "watchdog_source",
        "manifest_sha256", "geometry", "teacher_kind", "teacher", "teacher_weight", "teacher_model_state_sha256",
        "warmup_samples", "scored_samples", "carry_state", "drum_weight", "objective_version",
        "microbatch_size", "accumulation_steps", "accumulation_version", "optimizer_initialization",
        "augmentation_version", "mixer_initialization_seed", "automatic_continuation", "functional_proofs",
        "preparation_decision", "counted_roots", "stop_counted_bytes", "quality_endpoints")}
    base.update(schema="latency58-vocal-focus-training-v1", resource_only=True, parent=descriptor,
                matched_protocol=binding(args.protocol),
                source_bindings={**protocol["source_bindings"], str(args.protocol): args.protocol_sha256,
                                 str(Path(__file__).resolve()): sha(__file__)})
    prepared = []
    for arm in protocol["arms"]:
        model = extended if arm == "focused_mixer" else parent
        plan = copy.deepcopy(base)
        plan.update(arm=arm, focused_augmentation=arm != "original", local_mask_mixer=arm == "focused_mixer",
                    parameter_tensors=len(list(model.parameters())), architecture=model.architecture_metadata,
                    initialized_model_state_sha256=state_sha256(model.state_dict()),
                    run_dir=str(PHASE / ("vocal-focus-" + arm.replace("_", "-") + "-resource-run-001")))
        validate_recipe(plan)
        verify_inputs(plan)
        replay = load_parent(plan)
        require(state_sha256(replay.state_dict()) == plan["initialized_model_state_sha256"], "Loader replay differs")
        del replay
        path = out / (arm + "-resource-plan.json")
        require(not path.exists() and not Path(plan["run_dir"]).exists(), "Preserve a previously prepared arm")
        prepared.append((path, plan))
    require(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), "Preparation changed RNG or CUDA")
    verify_inputs(protocol)
    for path, plan in prepared:
        write(path, plan)
        print({"plan": str(path), "sha256": sha(path), "arm": plan["arm"], "resource_only": True}, flush=True)
    write(out / "resource-preparation.json", {
        "schema": "latency58-vocal-focus-resource-preparation-v1", "status": "pass",
        "protocol": binding(args.protocol), "parent_model_state_sha256": STUDENT_STATE_SHA256,
        "source_bindings": base["source_bindings"], "source_bindings_unchanged": True,
        "plans": {plan["arm"]: binding(path) for path, plan in prepared},
        "initialized_model_state_sha256": {plan["arm"]: plan["initialized_model_state_sha256"] for _, plan in prepared},
        "selected_parent_zero_output_and_state_exact": True, "all_three_loader_replays_exact": True,
        "cpu_rng_unchanged": True, "cuda_initialized": False, "training_updates_executed": 0,
        "optimizer_instances": 0, "weights_saved": False, "quality_selected": False,
        "counted_bytes_before": before, "counted_bytes_after": require_space(protocol, 0)})


if __name__ == "__main__":
    main()
