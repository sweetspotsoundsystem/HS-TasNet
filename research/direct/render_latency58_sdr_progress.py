"""Render retained primary-panel SDR milestones and the final per-stem gap."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Figure plan changed")
    plan = read(args.plan)
    require(plan["schema"] == "latency58-sdr-progress-figure-plan-v1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use a frozen CPU figure plan")
    verify_inputs(plan)
    out = Path(plan["output_directory"])
    require(out.parent == PHASE and out.is_dir() and not (out / "result.json").exists()
            and not (out / "figure.svg").exists() and not (out / "figure.png").exists(),
            "Preserve previous figure")
    bindings, milestones = {}, []
    for item in plan["milestones"]:
        _, report = load_completed(Path(item["directory"]), bindings,
                                   canonical_baseline=item["canonical_baseline"])
        result = report["results"][0]
        require(report["track_names"] == plan["track_names"] and report["excerpt_count"] == 28
                and result["model"]["model_state_sha256"] == item["model_state_sha256"],
                "Milestone panel or state differs")
        milestones.append({"label": item["label"], "prefix": item["prefix"],
                           "model_state_sha256": item["model_state_sha256"],
                           "full_sdr_db": result["aggregate"]["full_sdr_db"]})
    require(all(plan["source_bindings"].get(p) == s for p, s in bindings.items()),
            "Unbound milestone evidence")
    paired = read(plan["paired_summary"]["path"])
    require(sha(plan["paired_summary"]["path"]) == plan["paired_summary"]["sha256"],
            "Paired summary changed")
    comparison = paired["comparisons"]["accepted_11_6ms"]["metrics"]["full_sdr_db"]
    current, working = milestones[-1]["full_sdr_db"], milestones[0]["full_sdr_db"]
    require(comparison["candidate"] == current and comparison["reference"] == plan["target_full_sdr_db"],
            "Final candidate or target differs")
    gaps = {k: -v for k, v in comparison["per_stem_delta"].items()}
    gap, gain = plan["target_full_sdr_db"] - current, current - working
    require(abs(sum(gaps.values()) / 4 - gap) < 1e-12, "Per-stem gaps do not reconstruct the aggregate")
    data = {"milestones": milestones, "target_full_sdr_db": plan["target_full_sdr_db"],
            "current_full_sdr_db": current, "gain_over_working_db": gain, "target_gap_db": gap,
            "per_stem_target_minus_current_db": gaps,
            "scope": "Point estimates on the unchanged primary development panel; no confirmation data.",
            "limitation": "Retained milestones span different recipes and data exposure; not a causal ablation."}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require(matplotlib.__version__ == plan["matplotlib_version"], "Plotting runtime differs")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": args.plan_sha256})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"width_ratios": [1.28, 1]})
    fig.subplots_adjust(left=.19, right=.97, top=.72, bottom=.23, wspace=.40)
    fig.suptitle("5.8 ms quality recovery", x=.04, y=.97, ha="left", fontsize=20, fontweight="bold")
    fig.text(.04, .87, f"Current {current:.4f} dB   ·   Gain {gain:+.4f} dB   ·   Remaining gap {gap:.4f} dB",
             fontsize=12, color="#26354a")
    left, right = axes
    values = [m["full_sdr_db"] for m in milestones]
    left.scatter(values, range(len(values)), s=[45] * (len(values) - 1) + [85],
                 c=["#9aa8b8"] * (len(values) - 1) + ["#1263a8"], zorder=3)
    for i, value in enumerate(values):
        left.annotate(f"{value:.4f}", (value, i), xytext=(8, 0), textcoords="offset points",
                      va="center", fontsize=9, color="#26354a")
    left.set_yticks(range(len(values)), [m["label"] for m in milestones])
    left.set_ylim(len(values) - .4, -.8)
    left.set_xlim(min(values) - .02, plan["target_full_sdr_db"] + .028)
    left.axvline(plan["target_full_sdr_db"], linestyle="--", color="#b05b22", linewidth=1.2)
    left.text(plan["target_full_sdr_db"], 1.04, "11.6 ms target", transform=left.get_xaxis_transform(),
              ha="right", fontsize=9, color="#b05b22")
    left.set_xlabel("Full-band SDR (dB)")
    left.set_title("Retained milestones", loc="left", pad=25, fontsize=11)
    names = list(gaps)
    right.barh(range(4), [gaps[k] for k in names], height=.53,
               color=["#c56b32" if gaps[k] >= 0 else "#238c7c" for k in names], zorder=3)
    for i, name in enumerate(names):
        value = gaps[name]
        right.annotate(f"{value:+.3f}", (value, i), xytext=(5 if value >= 0 else -5, 0),
                       textcoords="offset points", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    right.set_yticks(range(4), [k.title() for k in names])
    right.invert_yaxis()
    right.set_xlim(min(gaps.values()) - .075, max(gaps.values()) + .09)
    right.axvline(0, color="#526377", linewidth=.8)
    right.set_title("Remaining gap by stem", loc="left", pad=25, fontsize=11)
    right.set_xlabel("Target SDR minus current SDR (dB)")
    for axis in axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#c8d0d9")
        axis.grid(axis="x", color="#e8ecf1", linewidth=.7, zorder=0)
        axis.tick_params(axis="y", length=0)
    fig.text(.04, .09, "Same 14 development tracks and 28 fixed excerpts. Higher SDR is better. Positive stem gaps remain below the target.",
             fontsize=9, color="#526377")
    fig.text(.04, .045, "Milestones span different training recipes; they are not a causal ablation. Confirmation and deployment qualification remain pending.",
             fontsize=9, color="#526377")
    fig.savefig(out / "figure.svg", metadata={"Date": None, "Description": data["scope"]})
    fig.savefig(out / "figure.png", dpi=160, metadata={"Software": "HS-TasNet research"})
    plt.close(fig)
    verify_inputs(plan)
    write(out / "data.json", data)
    write(out / "result.json", {"status": "pass", "plan_sha256": args.plan_sha256,
                                "source_bindings_unchanged": True, "data_sha256": sha(out / "data.json"),
                                "figure_sha256": sha(out / "figure.svg"), "new_inference": False,
                                "png_sha256": sha(out / "figure.png"),
                                "confirmation_used": False, "model_selected": False})
    print({"status": "pass", "figure": str(out / "figure.svg"), "current_full_sdr_db": current}, flush=True)


if __name__ == "__main__":
    main()
