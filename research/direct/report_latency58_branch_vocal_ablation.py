"""Summarize every fixed training ablation with explicit window support."""
import json
import math
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.diagnose_latency58_branch_vocal_ablation import STEPS, VARIANTS, VIEWS

STEMS = ("drums", "bass", "vocals", "other")


def weighted(values):
    supported = [(value, count) for value, count in values if count]
    require(all(value is not None and math.isfinite(value) and count > 0 for value, count in supported),
            "Missing or nonfinite supported diagnostic metric")
    return sum(value * count for value, count in supported) / sum(count for _, count in supported) if supported else None


def delta(value, reference):
    require((value is None) == (reference is None), "Ablation eligibility differs")
    return None if value is None else value - reference


def main():
    require(Path.cwd() == ROOT, "Use the research workspace")
    root = PHASE / "branch-parent-vocal-ablation-001"
    execution_path = PHASE / "branch-parent-vocal-ablation-stage-001/execution.json"
    result, plan, execution = read(root / "result.json"), read(root / "plan.json"), read(execution_path)
    require(result["status"] == "pass" and result["source_bindings_unchanged"]
            and result["training_examples_covered_by_batch_hashes"] == 64
            and result["all_four_augmented_batch_hashes_verified"]
            and result["all_six_upstream_states_unchanged_across_variants"]
            and result["warmup_recomputed_for_every_variant"] and result["all_projection_weights_restored"]
            and result["saved_parent_parameters_and_gradients_unchanged"]
            and result["training_only"] and not result["checkpoint_quality_measured"]
            and not result["gpu_used"] and not result["checkpoint_written"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"], "Require a completed, unchanged CPU diagnostic")
    verify_inputs(plan)
    verify_inputs(result)
    rows = result["comparisons"]
    require([row["step"] for row in rows] == STEPS and plan["training_steps"] == STEPS
            and plan["variants"] == list(VARIANTS) and plan["views"] == list(VIEWS), "Diagnostic panel changed")
    for row in rows:
        require(set(row["views"]) == set(VIEWS) and set(row["vocal_addition_response"]) == set(VARIANTS),
                "Missing diagnostic view or response")
        for view in VIEWS:
            require(set(row["views"][view]) == set(VARIANTS), "Missing memory ablation")
            reference = row["views"][view][VARIANTS[0]]
            for variant in VARIANTS:
                cell = row["views"][view][variant]
                require(cell["six_upstream_states_bit_exact"]
                        and cell["active_windows"] == reference["active_windows"]
                        and cell["absent_windows"] == reference["absent_windows"]
                        and all(a + b == 16 for a, b in zip(cell["active_windows"], cell["absent_windows"], strict=True)),
                        "Ablation states or one-window-per-example support differ")
    summaries, differences, per_batch = {}, {}, []
    for view in VIEWS:
        summaries[view], differences[view] = {}, {}
        for variant in VARIANTS:
            selected = [row["views"][view][variant] for row in rows]
            stems = {}
            for index, stem in enumerate(STEMS):
                active_count = sum(row["active_windows"][index] for row in selected)
                absent_count = sum(row["absent_windows"][index] for row in selected)
                sdr = weighted([(row["per_stem_training_sdr_db"][stem], row["active_windows"][index]) for row in selected])
                absent_db = weighted([(row["absent_output"][stem]["mean_window_output_dbfs"], row["absent_windows"][index]) for row in selected])
                absent_power = weighted([(row["absent_output"][stem]["mean_output_power"], row["absent_windows"][index]) for row in selected])
                require(all(row["absent_output"][stem]["windows"] == row["absent_windows"][index] for row in selected),
                        "Absence summary support differs")
                stems[stem] = {"active_windows": active_count, "absent_windows": absent_count,
                               "training_sdr_db": sdr, "absent_mean_window_output_dbfs": absent_db,
                               "absent_mean_output_power": absent_power,
                               "absent_global_rms_dbfs": 10 * math.log10(absent_power + 1e-12) if absent_power is not None else None}
            supported_sdr = [cell["training_sdr_db"] for cell in stems.values() if cell["active_windows"]]
            summaries[view][variant] = {"per_stem": stems,
                "window_pooled_stem_macro_training_sdr_db": sum(supported_sdr) / len(supported_sdr),
                "unweighted_mean_batch_training_sdr_db": sum(row["training_batch_sdr_db"] for row in selected) / len(selected)}
            if variant == VARIANTS[0]:
                continue
            baseline = summaries[view][VARIANTS[0]]
            differences[view][variant] = {"window_pooled_macro_sdr_delta_db":
                summaries[view][variant]["window_pooled_stem_macro_training_sdr_db"] - baseline["window_pooled_stem_macro_training_sdr_db"],
                "per_stem": {stem: {key + "_delta": delta(stems[stem][key], baseline["per_stem"][stem][key])
                             for key in ("training_sdr_db", "absent_mean_window_output_dbfs", "absent_global_rms_dbfs")}
                             for stem in STEMS}}
            for row in rows:
                candidate, reference = row["views"][view][variant], row["views"][view][VARIANTS[0]]
                per_batch.append({"step": row["step"], "view": view, "variant": variant,
                    "macro_sdr_delta_db": candidate["training_batch_sdr_db"] - reference["training_batch_sdr_db"],
                    "per_stem_sdr_delta_db": {stem: delta(candidate["per_stem_training_sdr_db"][stem], reference["per_stem_training_sdr_db"][stem])
                        if reference["active_windows"][index] else None for index, stem in enumerate(STEMS)},
                    "absent_vocal_mean_window_output_delta_db": delta(candidate["absent_output"]["vocals"]["mean_window_output_dbfs"], reference["absent_output"]["vocals"]["mean_window_output_dbfs"])})
    response = {}
    for variant in VARIANTS:
        response[variant] = {}
        for stem in STEMS:
            selected = [row["vocal_addition_response"][variant][stem] for row in rows]
            response[variant][stem] = {"active_vocal_examples": sum(row["active_vocal_examples"] for row in selected),
                **{key: weighted([(row[key], row["active_vocal_examples"]) for row in selected])
                   for key in ("response_energy_db_relative_to_vocal", "mean_projection_gain_on_vocal")}}
    paths = [root / "result.json", root / "plan.json", execution_path, Path(__file__).resolve(),
             ROOT / "research/direct/diagnose_latency58_branch_vocal_ablation.py"]
    bindings = {**plan["source_bindings"], **{str(path): sha(path) for path in paths}}
    verify_inputs({"source_bindings": bindings})
    summary = {"status": "pass", "source_bindings": bindings, "source_bindings_unchanged": True,
               "summaries": summaries, "ablation_minus_saved": differences, "all_24_batch_comparisons": per_batch,
               "vocal_addition_responses": response, "training_examples": 64, "training_steps": STEPS,
               "logged_mixture_reference_mismatch": [{"step": row["step"], "max_abs": row["logged_mixture_minus_source_sum_max_abs"],
                     "rms": row["logged_mixture_minus_source_sum_rms"]} for row in rows],
               "aggregation": "Pool per-stem one-second window support across the four fixed batches, then average supported stem SDRs. Also retain the unweighted mean batch SDR and every batch comparison. Absence means are window-weighted; global RMS is separately labeled.",
               "comparison_direction": "Temporary ablation minus saved branch memories. Lower absence output means less output on excluded or below-threshold references, not necessarily better wanted-source fidelity.",
               "training_only": True, "validation_quality_measured": False, "checkpoint_selected": False,
               "interpretation_limit": "Local effects on 64 source-summed training examples and their vocal-removed views. Vocal-addition responses include nonlinear interactions. No retrained comparison, held-out score, population inference, or deployment selection is established."}
    require(len(per_batch) == 24, "Missing batch comparison")
    write(root / "analysis.json", summary)
    print(json.dumps({"status": "pass", "all_sources_ablation_sdr_deltas": {key: value["window_pooled_macro_sdr_delta_db"] for key, value in differences["all_sources"].items()},
          "vocal_removed_ablation_vocal_output_deltas": {key: value["per_stem"]["vocals"]["absent_mean_window_output_dbfs_delta"] for key, value in differences["vocals_removed"].items()}}), flush=True)


if __name__ == "__main__":
    main()
