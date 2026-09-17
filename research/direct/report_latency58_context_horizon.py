"""Summarize measured training-history mismatch and render a standalone figure."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from research.direct.run_latency58_quality import read, require, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Summary plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-context-horizon-summary-plan-v1"
            and all(sha(p) == s for p, s in plan["source_bindings"].items())
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Summary inputs or CPU scope changed")
    directory = Path(plan["diagnostic_directory"])
    diagnostic, execution, diagnostic_plan = (read(directory / name) for name in
                                              ("result.json", "diagnostic-execution.json", "plan.json"))
    require(diagnostic["schema"] == "latency58-context-horizon-continuation-result-v1"
            and diagnostic["status"] == "pass" and diagnostic["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"]
            and diagnostic["plan_sha256"] == execution["plan_sha256"] == sha(directory / "plan.json")
            and all(sha(p) == s for p, s in diagnostic_plan["source_bindings"].items()), "Diagnostic incomplete")
    previous = diagnostic["previous_incomplete_run"]
    prior = [json.loads(s) for s in Path(previous["progress"]).read_text().splitlines()]
    current = [json.loads(s) for s in (directory / "progress.jsonl").read_text().splitlines()]
    rows = diagnostic["rows"]
    require(diagnostic["retained_prior_rows"] == len(prior) == 25 and len(current) == 5
            and rows == prior + current and len(rows) == 30, "Combined case inventory changed")
    stems = diagnostic["source_order"]
    warmups = diagnostic_plan["warmup_samples"]
    expected = [(t, s, w) for t in diagnostic_plan["track_indices"]
                for s in diagnostic_plan["start_samples"] for w in warmups]
    key = lambda r: (r["track_index"], r["start_sample"], r["warmup_samples"])
    require([key(r) for r in rows] == expected, "Diagnostic cases changed")
    lookup = {key(r): r for r in rows}
    pooled = []
    for warmup in warmups:
        selected = [r for r in rows if r["warmup_samples"] == warmup]
        regions = {}
        for region in diagnostic["scored_regions"]:
            regions[region] = {}
            for kind in ("raw", "deployed"):
                values = {}
                for index, stem in enumerate(stems):
                    error_mse = sum(r["regions"][region][kind]["difference_rms"][index] ** 2 for r in selected) / 6
                    reference_mse = sum(r["regions"][region][kind]["reference_rms"][index] ** 2 for r in selected) / 6
                    require(error_mse > 0 and reference_mse > 0, "Undefined pooled ratio")
                    values[stem] = {"difference_rms": math.sqrt(error_mse),
                                    "reference_rms": math.sqrt(reference_mse),
                                    "difference_dbfs": 10 * math.log10(error_mse),
                                    "difference_to_reference_db": 10 * math.log10(error_mse / reference_mse)}
                regions[region][kind] = values
        pooled.append({"warmup_samples": warmup, "warmup_seconds": warmup / 44100, "regions": regions})
    cases = []
    for track in diagnostic_plan["track_indices"]:
        for start in diagnostic_plan["start_samples"]:
            two, eight, sixteen = (lookup[(track, start, w)] for w in (88064, 352256, 704512))
            values = lambda r: r["regions"]["whole_score"]["deployed"]["difference_to_reference_db"]
            cases.append({"track_index": track, "track_name": two["track_name"], "start_sample": start,
                          "eight_minus_two_db": dict(zip(stems, (b - a for a, b in zip(values(two), values(eight))))),
                          "sixteen_minus_eight_db": dict(zip(stems, (b - a for a, b in zip(values(eight), values(sixteen)))))})
    out = Path(plan["output_directory"])
    require(out.is_dir() and not any((out / name).exists() for name in ("result.json", "figure.svg", "figure.png")),
            "Preserve previous summary")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                               "svg.hashsalt": "latency58-context-horizon-v1"})
    colors = {"drums": "#b45309", "bass": "#2563eb", "vocals": "#9333ea", "other": "#0f766e"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1, 1.15]})
    for stem in stems:
        axes[0].plot(range(5), [r["regions"]["whole_score"]["deployed"][stem]["difference_to_reference_db"]
                               for r in pooled], color=colors[stem], marker="o", label=stem.title())
    axes[0].set(xticks=range(5), xticklabels=["0", "2", "4", "8", "16"],
                xlabel="Preceding audio (approximately seconds)",
                ylabel="Waveform difference / reference level (dB)", title="Pooled waveform mismatch · lower is closer")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    labels = [r["track_name"].split(" - ")[1] + (" · ~20 s" if r["start_sample"] == 880640 else " · ~40 s") for r in cases]
    for index, stem in enumerate(stems):
        axes[1].scatter([r["eight_minus_two_db"][stem] for r in cases],
                        [i + (index - 1.5) * .13 for i in range(6)], color=colors[stem], s=32)
    axes[1].axvline(0, color="#6b7280", linewidth=1, linestyle="--")
    axes[1].set(yticks=range(6), yticklabels=labels, ylim=(5.6, -.6), xlim=(-12, 1),
                xlabel="8 s minus 2 s mismatch (dB)", title="Each passage · negative values are closer")
    axes[1].annotate(f"Vocals {cases[4]['eight_minus_two_db']['vocals']:+.3f} dB",
                     xy=(cases[4]["eight_minus_two_db"]["vocals"], 4.065),
                     xytext=(-5.6, 4.8), fontsize=9, color=colors["vocals"],
                     arrowprops={"arrowstyle": "-", "color": colors["vocals"]})
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x" if axis is axes[1] else "y", alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle("Past audio changes the model's starting state", fontsize=17, fontweight="bold", x=.08, ha="left")
    fig.text(.08, .90, "Accumulation 1000 · 3 fixed training tracks · 6 passages · native FP32 · unchanged weights", color="#4b5563")
    fig.text(.08, .07, "Reference: continuous predictions from each song's start. These measurements describe context mismatch.", fontsize=9)
    fig.text(.08, .03, "All 30 cases complete across two bounded runs; the first reached its 30-minute limit.", fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.20, wspace=.62)
    fig.savefig(out / "figure.svg", metadata={"Date": None})
    fig.savefig(out / "figure.png", dpi=160)
    plt.close(fig)
    require(all(sha(p) == s for p, s in plan["source_bindings"].items()), "Summary input changed")
    write(out / "result.json", {"schema": "latency58-context-horizon-summary-v1", "status": "pass",
          "plan_sha256": args.plan_sha256, "model_state_sha256": diagnostic["model_state_sha256"],
          "source_bindings_unchanged": True, "pooled": pooled, "cases": cases,
          "aggregation": "Equal sample counts per case: pool mean-square differences and reference levels before the dB ratio.",
          "figure_sha256": {name: sha(out / name) for name in ("figure.svg", "figure.png")},
          "quality_selected": False, "new_inference": False, "matplotlib_version": matplotlib.__version__,
          "limitations": diagnostic["limitations"]})
    print(json.dumps({"status": "pass", "pooled_warmups": len(pooled), "cases": len(cases)}), flush=True)


if __name__ == "__main__":
    main()
