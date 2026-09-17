"""Review all eight completed fixed-training gradient probes without inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot


ROLES = ("retained_best", "starting_parent", "raw", "ema")
FIRST_INDICES = (4116000, 4116016)
GROUPS = ("ordinary", "instrumental", "vocals_only", "auxiliary")
PAIRS = (("ordinary", "auxiliary"), ("ordinary", "instrumental"),
         ("ordinary", "vocals_only"), ("instrumental", "vocals_only"))
BLOCK_PREFIXES = {
    "spectral_encoder": ("spec_encode.", "spec_norm."),
    "waveform_encoder": ("conv_encode.", "basis_to_embed.", "waveform_norm."),
    "fusion_recurrence": ("fusion_branch.",),
    "mask_heads": ("to_spec_masks.", "to_waveform_masks."),
    "waveform_decoder": ("waveform_decoder_weight",),
    "phase_head": ("phase_reduce.", "phase_expand."),
    "fusion_refinement": ("fusion_refine_",),
    "temporal_attention": ("temporal_",),
    "spectral_memory": ("spec_memory.", "spec_memory_output."),
    "waveform_memory": ("waveform_memory.", "waveform_memory_output."),
}


def pair(left_squared, right_squared, dot):
    left, right = math.sqrt(left_squared), math.sqrt(right_squared)
    cosine = dot / (left * right) if left and right else None
    require(cosine is None or -1 - 1e-12 <= cosine <= 1 + 1e-12, "Invalid reconstructed cosine")
    return {"left_l2": left, "right_l2": right, "dot": dot,
            "cosine": None if cosine is None else max(-1., min(1., cosine)),
            "right_to_left_l2_ratio": right / left if left else None, "opposed": dot < 0}


def aggregate(rows):
    require(bool(rows), "Cannot summarize an empty parameter block")
    squared = {group: math.fsum(row["squared_l2"][group] for row in rows.values()) for group in GROUPS}
    dots = {left + "_vs_" + right: math.fsum(row["dots"][left + "_vs_" + right] for row in rows.values())
            for left, right in PAIRS}
    return {"parameter_tensors": len(rows), "parameter_names": list(rows),
            "parameter_elements": sum(row["elements"] for row in rows.values()),
            "fp32_parameter_bytes": sum(row["fp32_parameter_bytes"] for row in rows.values()),
            "two_fp32_adam_moment_bytes": sum(row["two_fp32_adam_moment_bytes"] for row in rows.values()),
            "group_l2": {group: math.sqrt(value) for group, value in squared.items()},
            "pairs": {left + "_vs_" + right: pair(squared[left], squared[right], dots[left + "_vs_" + right])
                      for left, right in PAIRS},
            "ordinary_directional_derivative_along_negative_combined_gradient":
                -(squared["ordinary"] + dots["ordinary_vs_auxiliary"]),
            "auxiliary_directional_derivative_along_negative_combined_gradient":
                -(squared["auxiliary"] + dots["ordinary_vs_auxiliary"])}


def validate_statistics(statistics):
    rows = statistics["per_parameter"]
    require(len(rows) == statistics["parameter_tensors"] == 40, "Require all 40 parameter tensors")
    for name, row in rows.items():
        require(row["elements"] == math.prod(row["shape"])
                and row["fp32_parameter_bytes"] == 4 * row["elements"]
                and row["two_fp32_adam_moment_bytes"] == 8 * row["elements"]
                and set(row["squared_l2"]) == set(GROUPS)
                and all(math.isfinite(value) and value >= 0 for value in row["squared_l2"].values())
                and all(math.isfinite(value) for value in row["dots"].values()), "Invalid parameter statistics: " + name)
        reconstructed = aggregate({name: row})
        require(reconstructed["pairs"] == row["pairs"], "Stored per-parameter pair statistics differ: " + name)
    combined = aggregate(rows)
    require(all(statistics[key] == combined[key] for key in (
        "parameter_tensors", "parameter_elements", "group_l2", "pairs",
        "ordinary_directional_derivative_along_negative_combined_gradient",
        "auxiliary_directional_derivative_along_negative_combined_gradient")),
        "Global statistics do not match the parameter reductions")
    expected_opposed = {key: sum(row["pairs"][key]["opposed"] for row in rows.values()) for key in combined["pairs"]}
    require(expected_opposed == statistics["opposed_parameter_tensors"], "Opposed-parameter counts differ")
    blocks = {block: {name: row for name, row in rows.items() if name.startswith(prefixes)}
              for block, prefixes in BLOCK_PREFIXES.items()}
    assigned = [name for values in blocks.values() for name in values]
    require(len(assigned) == len(set(assigned)) == 40 and set(assigned) == set(rows),
            "Functional blocks must partition every parameter exactly once")
    blocks = {name: aggregate(values) for name, values in blocks.items()}
    return combined, blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output-prefix", default="grouped-gradient-probe-training-review-001")
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    require(Path.cwd() == ROOT and directory.is_relative_to(PHASE)
            and args.output_prefix and all(c.isalnum() or c in "-_" for c in args.output_prefix), "Invalid review paths")
    out = PHASE / args.output_prefix
    require(not out.exists(), "Preserve previous gradient reviews")
    plan_path, result_path, execution_path = (directory / name for name in ("plan.json", "result.json", "execution.json"))
    require(sha(execution_path) == args.execution_sha256, "Actual completion receipt changed")
    plan, result, execution = read(plan_path), read(result_path), read(execution_path)
    require(result["status"] == "pass" and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"] and result["source_bindings_unchanged"]
            and execution["result_sha256"] == sha(result_path)
            and result["plan_sha256"] == execution["plan_sha256"] == sha(plan_path)
            and result["completed_probes"] == 8 and result["fixed_input_replay_bit_exact"]
            and result["checkpoint_weights_inputs_and_rng_unchanged"] and not result["gpu_used"]
            and result["optimizer_updates"] == 0 and not result["checkpoint_files_written"]
            and plan["role_order"] == list(ROLES)
            and [row["first_sample_index"] for row in plan["batches"]] == list(FIRST_INDICES),
            "Require all eight completed, unmodified fixed-training probes")
    verify_inputs(plan)
    budget = budget_snapshot(plan["storage_budget"])
    require(budget["headroom_bytes"] > 2_000_000, "Reserve scalar review artifacts")
    bindings = dict(plan["source_bindings"])
    for path in (Path(__file__).resolve(), plan_path, result_path, execution_path):
        bindings[str(path)] = sha(path)
    rows, seen = [], set()
    for binding in result["reports"]:
        path = Path(binding["path"])
        key = binding["first_sample_index"], binding["role"]
        require(path.parent == directory and key not in seen and key[0] in FIRST_INDICES and key[1] in ROLES
                and sha(path) == binding["sha256"], "Invalid or duplicated probe report")
        seen.add(key); bindings[str(path)] = binding["sha256"]
        report = read(path)
        expected = plan["batches"][FIRST_INDICES.index(key[0])]
        model = plan["models"][key[1]]
        metadata = report["metadata"]
        require(report["status"] == "pass" and report["plan_sha256"] == sha(plan_path)
                and (report["first_sample_index"], report["role"]) == key and report["model"] == model
                and metadata["model_state_sha256"] == model["model_state_sha256"]
                and metadata["input_sha256"] == expected["after_remix_sha256"]
                and metadata["weights_inputs_and_rng_unchanged"] and metadata["optimizer_updates"] == 0
                and metadata["auxiliary_contributions_from_joint_objective"]
                and metadata["policy"] == plan["policy"], "Probe identity or method differs")
        for group in ("ordinary", "auxiliary"):
            actual, counts = metadata["groups"][group], expected["groups"][group]
            require(all(actual[k] == counts[k] for k in ("examples", "active_windows", "absent_windows"))
                    and actual["replay_outputs_bit_exact"] and actual["whole_group_objective_evaluations"] == 1,
                    "Probe lost the declared whole-group reduction")
        global_statistics, blocks = validate_statistics(report["statistics"])
        rows.append({"role": key[1], "first_sample_index": key[0], "report": binding,
                     "losses": {k: v["weighted_loss"] for k, v in metadata["groups"].items()},
                     "global": global_statistics, "blocks": blocks,
                     "opposed_parameter_tensors": report["statistics"]["opposed_parameter_tensors"]})
    require(seen == {(first, role) for first in FIRST_INDICES for role in ROLES}, "Incomplete checkpoint/batch cross product")
    review_plan = {"schema": "latency58-completed-training-gradient-review-v1", "source_bindings": bindings,
                   "input_directory": str(directory), "role_order": list(ROLES), "first_sample_indices": list(FIRST_INDICES),
                   "functional_block_prefixes": BLOCK_PREFIXES, "budget_before": budget,
                   "arithmetic": "Recompute each saved FP64 parameter reduction and partition the 40 tensors into ten blocks",
                   "selection_scope": "Review only; no training recipe or trainable subset selected by this script"}
    verify_inputs(review_plan)
    out.mkdir(); write(out / "plan.json", review_plan)
    summary = {"schema": review_plan["schema"], "status": "pass", "plan_sha256": sha(out / "plan.json"),
               "source_bindings_unchanged": True, "models": plan["models"], "rows": rows,
               "all_eight_global_and_parameter_statistics_recomputed": True,
               "all_40_parameters_partitioned_once": True, "training_recipe_selected": False,
               "quality_measured": False, "gpu_used": False,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "budget_after": budget_snapshot(plan["storage_budget"]),
               "limitations": ["Batch gradients remain separate; mean cosines are not aggregate-gradient cosines.",
                               "Auxiliary vectors include the fixed 0.1 weight and unchanged joint two-view normalization.",
                               "Directional derivatives concern infinitesimal gradient descent, not Adam updates or validation quality.",
                               "Parameter and Adam byte counts exclude serialization, full inference checkpoint duplication, recovery and logs."]}
    write(out / "result.json", summary)
    print({"status": "pass", "plan_sha256": sha(out / "plan.json"), "result_sha256": sha(out / "result.json"),
           "completed_probes": len(rows)}, flush=True)


if __name__ == "__main__":
    main()
