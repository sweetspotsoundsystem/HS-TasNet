"""Render the completed 4,000-update quality/leakage comparison as a standalone figure."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.direct.run_latency58_quality import PHASE, read, require, sha, write


def main():
    out = PHASE / "branch-long-context-recovery-006"
    paired_root = PHASE / "paired-vocal-long-context-006"
    pair, execution = read(paired_root / "result.json"), read(paired_root / "root-execution.json")
    require(pair["status"] == "pass" and execution["actual_exit_code"] == 0
            and execution["result_sha256"] == sha(paired_root / "result.json"), "Require complete paired evidence")
    source = read(PHASE / "branch-long-context-006/plan.json")
    paths = [Path(source["reference_result"]), out / "full14-raw/result.json", out / "full14-ema/result.json"]
    reports = [read(p)["results"][0] for p in paths]
    paths.extend([paired_root / "result.json", paired_root / "root-execution.json", Path(__file__).resolve()])
    bindings = {str(p): sha(p) for p in paths}
    labels, colors = ["Previous parent", "Raw, 4,000", "EMA, 4,000"], ["#79818A", "#3478A5", "#00866B"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), gridspec_kw={"height_ratios": [1, 1.65]})
    fig.subplots_adjust(left=.10, right=.97, bottom=.14, top=.89, hspace=.36, wspace=.32)
    fig.suptitle("Saved quality improves; difficult false-vocal windows remain", fontsize=17,
                 x=.08, y=.98, ha="left", fontweight="bold")
    fig.text(.08, .94, "Unchanged 14-track / 28-excerpt development panel · FP32 source checkpoints · 256-sample graph + host latency",
             fontsize=10, color="#4A5560")
    a = axes[0, 0]
    values = [x["aggregate"]["full_sdr_db"] for x in reports]
    bars = a.bar(labels, values, color=colors, width=.6)
    a.axhline(5, color="#B54E42", linestyle="--", linewidth=1.2, label="5.0 dB goal")
    for bar, value in zip(bars, values):
        a.text(bar.get_x() + bar.get_width() / 2, value + .09, f"{value:.3f}", ha="center", fontsize=11)
    a.set(ylim=(0, 5.5), ylabel="Full-band SDR (dB)")
    a.set_title("Original full-mixture score", loc="left", fontweight="bold")
    a.legend(loc="upper left", frameon=False, fontsize=9)
    a, x, stems = axes[0, 1], np.arange(4), ["drums", "bass", "vocals", "other"]
    for k, (report, label, color) in enumerate(zip(reports, labels, colors)):
        a.bar(x + (k - 1) * .24, [report["aggregate"]["per_stem"][s]["full_sdr_db"] for s in stems],
              .23, label=label, color=color)
    a.set_xticks(x, [s.title() for s in stems])
    a.set(ylim=(0, 6), ylabel="Full-band SDR (dB)")
    a.set_title("All stem means improve; local losses persist", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    a = axes[1, 0]
    comparisons = [pair["comparisons"][key] for key in ["raw_vs_released", "ema_vs_released"]]
    tracks = list(comparisons[0]["aggregate"]["views"]["instrumental"]["per_stem"]["vocals"]["metrics"]["output_rms_dbfs"]["per_track_delta"])
    for k, (c, label, color) in enumerate(zip(comparisons, labels[1:], colors[1:])):
        values = list(c["aggregate"]["views"]["instrumental"]["per_stem"]["vocals"]["metrics"]["output_rms_dbfs"]["per_track_delta"].values())
        a.barh(np.arange(14) + (k - .5) * .34, values, height=.32, label=label, color=color)
    names = [t.split(" - ")[0].replace("Clara Berry And Wooldog", "Clara Berry") for t in tracks]
    a.set_yticks(np.arange(14), names, fontsize=8)
    a.invert_yaxis()
    a.axvline(0, color="#616A73", linewidth=.8)
    a.set(xlim=(-3.8, 1.7), xlabel="Vocal output change versus release (dB); lower is quieter")
    a.set_title("Instrumental leakage by track", loc="left", fontweight="bold")
    a.grid(axis="x", alpha=.15)
    a = axes[1, 1]
    keys = [("Skelpolu - Human Mistakes", 88), ("Skelpolu - Human Mistakes", 76),
            ("Meaxic - Take A Step", 84), ("ANiMAL - Rockshow", 80)]
    for k, (c, label, color) in enumerate(zip(comparisons, labels[1:], colors[1:])):
        values = []
        for track, start in keys:
            row = next(x for x in c["windows"]["all_windows"] if x["track"] == track
                       and x["view"] == "instrumental" and x["physical_start"] == start * 44100)
            values.append(row["per_stem"]["vocals"]["metrics"]["output_rms_dbfs"]["delta"])
        a.barh(np.arange(4) + (k - .5) * .28, values, height=.26, label=label, color=color)
        for j, value in enumerate(values):
            a.text(value + .09, j + (k - .5) * .28, f"{value:+.2f}", va="center", fontsize=9)
    a.set_yticks(np.arange(4), ["Skelpolu 88–89 s", "Skelpolu 76–77 s", "Meaxic 84–85 s", "Rockshow 80–81 s"], fontsize=9)
    a.invert_yaxis()
    a.axvline(0, color="#616A73", linewidth=.8)
    a.set(xlim=(-.2, 7.3), xlabel="Extra unwanted vocal output versus release (dB)")
    a.set_title("Large and persistent leakage failures", loc="left", fontweight="bold")
    a.grid(axis="x", alpha=.15)
    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.text(.10, .063, "Rockshow at 80–81 s: EMA vocals are only 0.56 dB below the instrumental input. M4 playback and listening remain unverified.",
             fontsize=9, color="#4A5560")
    fig.text(.10, .032, "Track resampling and repeated development selection do not establish unseen-data, training-seed, deployment or listening acceptance.",
             fontsize=9, color="#4A5560")
    for suffix in ["png", "svg"]:
        fig.savefig(out / ("quality-and-leakage-review." + suffix), dpi=150, facecolor="white")
    plt.close(fig)
    require(all(sha(p) == digest for p, digest in bindings.items()), "Figure inputs changed")
    write(out / "quality-and-leakage-figure-inputs.json", {"source_bindings": bindings, "status": "pass",
          "png_sha256": sha(out / "quality-and-leakage-review.png"), "svg_sha256": sha(out / "quality-and-leakage-review.svg")})
    print({"status": "pass", "figure": str(out / "quality-and-leakage-review.png")}, flush=True)


if __name__ == "__main__":
    main()
