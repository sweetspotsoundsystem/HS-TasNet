"""Render the manuscript's vector architecture figure with Matplotlib."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "stemgenrt-paper-v0.1"})
fig, ax = plt.subplots(figsize=(14, 7.2))
fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.985)
ax.set(xlim=(0, 1400), ylim=(0, 720))
ax.axis("off")
ink, blue, green, gray = "#182b3a", "#e8f1f8", "#e5f2ed", "#647686"


def label(x, y, text, size=10, ha="center", weight="normal", color=ink):
    ax.text(x, y, text, fontsize=size, ha=ha, va="center", color=color,
            weight=weight, linespacing=1.45)


def box(x, y, w, h, title, body, fill=blue):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                               facecolor=fill, edgecolor="#8195a5", linewidth=1))
    label(x+w/2, y+h-24, title, 10, weight="bold")
    label(x+w/2, y+h/2-12, body, 9)


def arrow(start, end, color=gray):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.2, color=color))


def elbow(points):
    ax.plot(*zip(*points[:-1]), color=gray, lw=1.2)
    arrow(points[-2], points[-1])


label(30, 693, "StemgenRT-5.8  |  Real-time low-latency music source separation", 15, ha="left", weight="bold")
label(30, 665, "Historical four-state C204 architecture · stereo 44.1 kHz · 128 new samples per call", 10, ha="left", color=gray)
box(30, 427, 160, 110, "Input + history", "128 new samples\n896 stored samples")
box(240, 542, 250, 95, "Spectral encoder", "Asymmetric window + FFT1024\nReal/imaginary → 500 features")
box(240, 337, 250, 95, "Waveform encoder", "Gated Conv1024 → 1500 bases\nProjection → 500 features")
box(555, 417, 230, 130, "Shared recurrent fusion", "Concatenate → GRU\n2 layers × 1000 hidden\nResiduals → split", green)
box(860, 542, 280, 95, "Spectral synthesis", "Source masks → IFFT1024\nTail 256 + window + overlap-add")
box(860, 337, 280, 95, "Waveform synthesis", "Source masks → learned decoder\n256 samples + Hann overlap-add")
box(1190, 422, 185, 120, "Deployed outputs", "Sum branches + scales\nRetain Drums/Bass/Vocals\nOther = mixture residual", green)
elbow([(190, 507), (210, 507), (210, 589), (240, 589)])
elbow([(190, 457), (210, 457), (210, 384), (240, 384)])
elbow([(490, 589), (520, 589), (520, 511), (555, 511)])
elbow([(490, 384), (520, 384), (520, 452), (555, 452)])
elbow([(785, 511), (820, 511), (820, 589), (860, 589)])
elbow([(785, 452), (820, 452), (820, 384), (860, 384)])
elbow([(1140, 589), (1160, 589), (1160, 511), (1190, 511)])
elbow([(1140, 384), (1160, 384), (1160, 452), (1190, 452)])
label(668, 388, "Hidden state carried between calls", 9, color=gray)
label(1200, 308, "Each synthesis branch also carries a 128-sample tail.", 9, ha="right", color=gray)

ax.plot([30, 1375], [280, 280], color="#ccd6de", lw=1)
label(30, 255, "Sample-time geometry of one call", 12, ha="left", weight="bold")
# Analysis spans [-896, 128); the current input starts at zero.
x0, scale = 340, .90
start = x0
current = x0 + 896 * scale
stop = x0 + 1024 * scale
synth = x0 + 768 * scale
for y, text in [(203, "Analysis: 1024"), (145, "Synthesis tail: 256"), (87, "Emitted output: 128")]:
    label(30, y, text, 10, ha="left")
ax.add_patch(Rectangle((start, 187), 896*scale, 32, facecolor=blue, edgecolor="#8195a5"))
ax.add_patch(Rectangle((current, 187), 128*scale, 32, facecolor="#bedfce", edgecolor="#8195a5"))
label((start+current)/2, 203, "896 past samples", 9)
label((current+stop)/2, 203, "128 new", 9)
ax.add_patch(Rectangle((synth, 129), 256*scale, 32, facecolor=green, edgecolor="#8195a5"))
label((synth+stop)/2, 145, "256 retained samples", 9)
ax.add_patch(Rectangle((synth, 71), 128*scale, 32, facecolor="#386b86", edgecolor="#386b86"))
label((synth+current)/2, 87, "Previous hop", 9, color="white")
for x, text in [(start, "−896"), (synth, "−128"), (current, "0"), (stop, "+128")]:
    ax.plot([x, x], [62, 228], ls=(0, (3, 3)), color="#8497a6", lw=.7, zorder=0)
    label(x, 46, text, 9)
label(30, 16, "Zero marks the start of this call's input. Graph delay: 128 samples. The specified host queue adds another 128.",
      9, ha="left", color=gray)
output = HERE / "architecture.svg"
fig.savefig(output, facecolor="white", metadata={"Date": None})
plt.close(fig)
output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
