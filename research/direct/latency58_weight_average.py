"""One fixed equal-weight average of the retained drum-500 and drum-1000 models."""
from __future__ import annotations

import copy
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha
from research.direct.train_latency58 import state_sha256, verify_inputs

VERSION = "drum500-drum1000-equal-fp64-average-fp32-v1"


def load_components(plan):
    import torch
    from research.direct.latency58_sdr_drum_accum_checkpoint import load_model

    require(plan["schema"] == "latency58-weight-average-plan-v1" and plan["version"] == VERSION
            and [c["step"] for c in plan["components"]] == [500, 1000]
            and [c["weight"] for c in plan["components"]] == [.5, .5]
            and plan["components"][0]["training_plan"] == plan["components"][1]["training_plan"]
            and not torch.cuda.is_initialized(), "Require the fixed CPU average")
    verify_inputs(plan)
    models = []
    for component in plan["components"]:
        binding = component["training_plan"]
        require(sha(binding["path"]) == binding["sha256"], "Component training plan changed")
        model, receipt = load_model(component["generation"], read(binding["path"]),
                                    expected_plan_sha=binding["sha256"])
        require(receipt["step"] == component["step"]
                and receipt["model_state_sha256"] == component["model_state_sha256"]
                and receipt["files"]["model.pt"]["sha256"] == component["checkpoint"]["sha256"]
                and component["checkpoint"]["path"] == str(Path(component["generation"]) / "model.pt")
                and model.provenance == component["provenance"], "Different retained component")
        models.append(model.eval().requires_grad_(False))
    left, right = models
    require(left.architecture_metadata == right.architecture_metadata
            and list(dict(left.named_parameters())) == list(dict(right.named_parameters()))
            and len(list(left.parameters())) == len(list(right.parameters())) == 21
            and len(list(left.buffers())) == len(list(right.buffers())) == 6
            and all(torch.equal(value, dict(right.named_buffers())[name]) for name, value in left.named_buffers()),
            "Component architecture, parameter inventory or fixed buffers differ")
    return left, right


def expected_provenance(plan, plan_sha):
    return {
        "initialization": "equal_weight_checkpoint_average", "version": VERSION,
        "training_updates": max(c["provenance"]["training_updates"] for c in plan["components"]),
        "training_update_counter_policy": "maximum component counter; no new training",
        "components": copy.deepcopy(plan["components"]), "construction_plan_sha256": plan_sha,
        "parameter_arithmetic": "float64 pair sum times one half, then float32",
        "fixed_buffer_policy": "identical buffers retained exactly",
        "training_updates_executed": 0, "optimizer_instances": 0,
    }


def read_average(plan, *, expected_plan_sha):
    directory = Path(plan["output_directory"])
    receipt = read(directory / "receipt.json")
    require(receipt["schema"] == "latency58-weight-average-generation-v1"
            and receipt["version"] == VERSION and receipt["step"] == 0
            and receipt["component_steps"] == [500, 1000]
            and receipt["plan_sha256"] == expected_plan_sha
            and set(receipt["files"]) == {"model.pt"}, "Different averaged generation")
    path = directory / "model.pt"
    binding = receipt["files"]["model.pt"]
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == binding["bytes"]
            and sha(path) == binding["sha256"], "Derived model bytes changed")
    return receipt


def load_average(plan, *, expected_plan_sha):
    import torch
    from research.direct.latency58_sdr_drum_accum_checkpoint import load_model

    require(not torch.cuda.is_initialized(), "Load derived inference model on CPU")
    verify_inputs(plan)
    receipt = read_average(plan, expected_plan_sha=expected_plan_sha)
    component = plan["components"][0]
    binding = component["training_plan"]
    require(sha(binding["path"]) == binding["sha256"], "Component plan changed")
    model, _ = load_model(component["generation"], read(binding["path"]), expected_plan_sha=binding["sha256"])
    fixed = {name: value.clone() for name, value in model.named_buffers()}
    payload = torch.load(Path(plan["output_directory"]) / "model.pt", map_location="cpu", weights_only=True)
    require(payload["schema"] == "latency58-weight-average-inference-v1"
            and payload["plan_sha256"] == expected_plan_sha and payload["architecture"] == model.architecture_metadata
            and payload["model_state_sha256"] == receipt["model_state_sha256"]
            and payload["provenance"] == expected_provenance(plan, expected_plan_sha), "Derived provenance differs")
    model.load_state_dict(payload["model"], strict=True)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values())
            and all(torch.equal(value, fixed[name]) for name, value in model.named_buffers())
            and state_sha256(model.state_dict()) == receipt["model_state_sha256"]
            and not torch.cuda.is_initialized(), "Derived weights or fixed buffers differ")
    model.provenance = payload["provenance"]
    return model.eval().requires_grad_(False), receipt


def load_evaluation_model(plan):
    binding = plan["construction_plan"]
    require(sha(binding["path"]) == binding["sha256"], "Construction plan changed")
    construction = read(binding["path"])
    for key, stem in (("build_result", "build"), ("audit_result", "audit")):
        result_binding, execution_binding = plan[key], plan[stem + "_execution"]
        require(sha(result_binding["path"]) == result_binding["sha256"]
                and sha(execution_binding["path"]) == execution_binding["sha256"], "Derived qualification changed")
        result, execution = read(result_binding["path"]), read(execution_binding["path"])
        command = execution["argv"]
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and result["schema"] == "latency58-weight-average-" + stem + "-v1"
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"]
                and result["plan_sha256"] == execution["plan_sha256"] == binding["sha256"]
                and command[command.index("-m") + 1] == "research.direct.build_latency58_weight_average"
                and command[command.index("--mode") + 1] == stem
                and Path(command[command.index("--plan") + 1]) == Path(binding["path"])
                and Path(result_binding["path"]) == Path(construction["output_directory"]) / (stem + "-result.json"),
                "Require completed construction and independent audit")
        verify_inputs(result)
        if stem == "audit":
            require(result["independent_saved_tensor_average_verified"], "Saved midpoint arithmetic was not audited")
    model, receipt = load_average(construction, expected_plan_sha=binding["sha256"])
    require(receipt["files"]["model.pt"]["sha256"] == plan["checkpoint"]["sha256"]
            and receipt["model_state_sha256"] == plan["expected_model_state_sha256"]
            and read(plan["audit_result"]["path"])["model_state_sha256"] == receipt["model_state_sha256"],
            "Different audited averaged endpoint")
    return model, receipt
