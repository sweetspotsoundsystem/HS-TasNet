"""Write candidate documentation from the completed, packaged measurements."""
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.latency58_m4_followup_budget import snapshot


def main():
    followup = ROOT / "research/m4_followup_20260916"
    target = followup / "attention-qkv-review-plugin-001"
    quality = read(followup / "attention-qkv-int8-quality-001/result.json")
    review = read(followup / "attention-qkv-int8-review-001/result.json")
    package = read(followup / "attention-qkv-int8-package-review-001/result.json")
    native = read(target / "model/linux-validation.json")
    require(all(value["status"] == "pass" for value in (quality, review, package, native)), "Complete all evidence first")
    require(sha(target / "model/quality-deployment.json") == package["destination_sha256"], "Published quality copy differs")
    aggregate = quality["results"][0]["aggregate"]
    parent = read(ROOT / "research/direct/runs/latency58/branch-output-int8-quality-001/result.json")["results"][0]["aggregate"]
    full = aggregate["full_sdr_db"]
    delta = full - parent["full_sdr_db"]
    summary = review["music_regression_summary"]["full_sdr_db"]
    worst = min(row["delta"] for row in summary["worst_changes"])
    sir_worst = review["music_regression_summary"]["sir_db"]["worst_changes"][0]
    absence_worst = review["music_regression_summary"]["absent_fp_dbfs"]["worst_changes"][0]
    instrumental = review["vocal_views"]["views"]["instrumental"]["per_stem"]["vocals"]["metrics"]
    isolated = review["vocal_views"]["views"]["vocals_only"]["per_stem"]["vocals"]["metrics"]
    levels = instrumental["output_rms_dbfs"]
    vocal_sdr = isolated["desired_full_sdr_db"]
    gain = isolated["signed_desired_projection_gain"]
    windows = review["paired_windows"]["all_windows"]
    leakage_changes = [row["per_stem"]["vocals"]["metrics"]["output_rms_dbfs"]["delta"]
                       for row in windows if row["view"] == "instrumental" and row["input_active"]]
    require(len(windows) == 840 and summary["eligible_cells"] == 56, "Incomplete review")
    rows = []
    for stem in ("drums", "bass", "vocals", "other"):
        a, b = (value["per_stem"][stem]["full_sdr_db"] for value in (parent, aggregate))
        rows.append(f"| {stem.title()} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    model_text = f'''# Streaming separation model

`model.onnx` separates Drums, Bass, Vocals and Other at 44.1 kHz. It accepts
128-sample stereo hops, uses a 1024-sample asymmetric analysis window and a
256-sample synthesis frame, and carries eight FP32 states. The graph delay
is 128 samples; the asynchronous host queue adds 128 samples at the supported
128-sample host buffer. Total graph-plus-host delay is 256 samples.

The authoritative identity and interface are in
`cmake/QualifiedModelContract.cmake`. This self-contained graph has SHA-256
`{quality['graph_sha256']}` and is {quality['graph_bytes']:,} bytes.
It retains the source checkpoint, 39,250 training updates, raw input levels,
four outputs, residual policy and streaming/reset/EOF contract. The confidence
envelope and output routing are unchanged.

## Inference changes

The three attention input products are packed into one dynamic U8/S8 product
with per-column signed weight scales. The query's last-frame slice follows
the packed projection, preserving its time selection. Seventeen products now
use reduced signed weights in [-64,64]. The other graph nodes and initializers
retain their definitions. No extra audio buffering or persistent state is added.

The plugin also sets `mlas.disable_kleidiai=1`. The retained
[M4 Pro parent diagnostic](macos-runtime-parity-diagnostic.json) failed all
eight independent reference cases with ORT defaults and passed all eight with
KleidiAI disabled. The setting and graph optimization are separate commits;
the setting-only parent is `af06fac`. This candidate still needs its own M4
numerical and timing measurements.

## Deployment quality

The exact graph scores **{full:.9f} dB full-band SDR**, a **{delta:+.9f} dB**
change from PR #17's sixteen-product graph on the unchanged 14-track,
28-excerpt development panel. The source FP32 checkpoint scores 4.465157422 dB;
v0.4.0's deployment graph scores 4.455188055 dB. These are separate endpoints.
The 5.0 dB target remains unmet.

| Stem | PR #17 SDR (dB) | Candidate SDR (dB) | Change (dB) |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

Full-band SDR decreases in {summary['regressing_cells']}/56 track/stem cells;
the worst change is {worst:+.9f} dB. The complete
[deployment report](quality-deployment.json) retains all track/stem/band/absence
regressions, paired bootstrap summaries, and all 840 source-view windows.
The worst SIR change is {sir_worst['delta']:+.6f} dB for {sir_worst['stem']}
on {sir_worst['track']}. The largest natural-absence output increase is
{absence_worst['delta']:+.6f} dB for {absence_worst['stem']} on
{absence_worst['track']}.

On the exact instrumental remixes, mean unwanted vocal output is
{levels['candidate']:.6f} dBFS, a {levels['delta']:+.6f} dB change from PR #17
(positive means more leakage). The largest one-second instrumental-window
increase is {max(leakage_changes):+.6f} dB. On isolated vocals, desired-vocal
SDR changes by {vocal_sdr['delta']:+.6f} dB and signed desired projection gain
changes by {gain['delta']:+.9f}. Read these with Other quality and the retained
worst windows. They do not establish improved instrumental listening.
The small instrumental-vocal increase occurs on all fourteen tracks; this
runtime experiment does not solve the reported vocal-leakage problem.

Scoring uses the original continuous input, physical intervals, alignment,
residual reconstruction and metric code. CPU ORT 1.26.0 runtime binaries match
those used for the parent measurements. There is no new confirmation panel;
source-view references can contain recording bleed. Listening acceptance and
representative real instrumental material remain outstanding.

## Numerical and native checks

[Streaming validation](streaming-validation.json) retains independent PyTorch
reconstruction of all seventeen integer products, short and long carried-state
cases, nonzero initial states, partial EOF and exact reset replay. Expected
fixture outputs are generated without importing ONNX Runtime.

The [Linux native suite](linux-validation.json) passed 164 tests, with one
platform-specific skip and seven disabled performance tests. Both four-stem
parity tests passed at the unchanged 1e-5 waveform limit. The suite also checks
state/reset/EOF behavior, timestamp admission, queue recovery, alignment,
reconstruction, variable offline callbacks and callback heap traffic. Its
record includes 50 plugin/test/contract/fixture input hashes.

All 24 [Linux backend diagnostic cases](runtime-parity-diagnostic.json) passed
with maximum error 1.63912773132e-7. The diagnostic retains ORT-default,
KleidiAI-disabled and optimization-disabled sessions. The plugin uses the
KleidiAI-disabled setting. Linux correctness does not establish physical M4
correctness for this new graph.

## Timing and M4 acceptance

An eight-block preallocated native ORT 1.26.0 comparison on an AMD Ryzen 5 5500
under WSL2 measured median block p50 of 3.154038 ms for PR #17 and 2.991973 ms
for this graph: **5.14% lower**, with all four paired medians faster. Each block
used 256 warmup and 2,048 measured hops, one ORT thread and disabled spinning.
Timing tails varied under concurrent training. This is a relative local
graph measurement; the two legacy `linux-*-timing.log` files are historical.

The parent M4 Pro diagnostic's longest short clip averaged 0.932 ms with ORT
defaults and 1.047 ms with KleidiAI disabled. Those averages and the Linux graph
comparison cannot establish the combined candidate's M4 performance.

The user reported 1,920 fallback samples after ten minutes with PR #17 in
Ableton on M4 at 44.1 kHz / 128 samples. Follow [the M4 test instructions](../M4_TESTING.md)
for this candidate's own numerical checks, repeated 30-minute untraced soaks,
complete worker traces when needed, and installed-AU playback. Retain raw
startup/reset counters and require zero additional steady-playback fallback.
Exercise transport changes and listen to instrumental passages, quiet real
vocals and Other. The candidate is for evaluation; no M4 qualification, release
replacement or completion of the broader quality goal is claimed.
'''
    model_path = target / "model/README.md"
    readme_path = target / "README.md"
    previous = {str(p): sha(p) for p in (model_path, readme_path)}
    model_path.write_text(model_text)
    text = readme_path.read_text()
    start = text.index("## Model and checks")
    end = text.index("Built with [JUCE]", start)
    text = text[:start] + f'''## Model and checks

The model combines spectrogram and waveform estimates with causal attention
and separate recurrent memories. This candidate packs the attention query,
key and value projections into one integer product, bringing the total to
seventeen. The exact graph scores **{full:.6f} dB SDR** on the unchanged
development panel, versus {parent['full_sdr_db']:.6f} dB for PR #17. The source
FP32 checkpoint scores 4.465157 dB. The [deployment report](model/quality-deployment.json)
retains all 56 track/stem comparisons and 840 paired source-view windows.
The 5 dB goal and instrumental listening acceptance remain unmet.

Raw input levels, the linked near-silence confidence fade and
`Other = Main - Drums - Bass - Vocals` are preserved. The native Linux suite
passed 164 tests, with one platform-specific skip and seven disabled timing
tests. Coverage includes independent PyTorch parity, resets, partial EOF,
alignment, queue recovery, reconstruction and callback heap traffic.

The plugin now disables KleidiAI through ORT's session configuration. The
retained [M4 Pro parent evidence](model/macos-runtime-parity-diagnostic.json)
fails all eight cases with backend defaults and passes all eight with KleidiAI
disabled. All 24 diagnostic cases pass on Linux for this candidate; its own
physical M4 numerical checks remain outstanding.

The graph reduced local median block p50 by 5.14% compared with PR #17 under
concurrent training. The backend setting's cost on M4 must be measured together
with this graph. The latest user report is 1,920 fallback samples after ten
minutes with PR #17 on M4 in Ableton at 44.1 kHz / 128 samples. Sustained zero
fallback is still unqualified for this candidate.

Follow [the M4 test instructions](M4_TESTING.md) for numerical checks, repeated
extended soaks and installed-DAW playback. Keep one inference worker and retain
the previous complete plugin bundle. See [the model report](model/README.md)
for exact quality changes, timing evidence and limitations.

''' + text[end:]
    readme_path.write_text(text)
    out = followup / "attention-qkv-documentation.json"
    require(not out.exists(), "Preserve documentation receipt")
    write(out, {"status": "pass", "previous_hashes": previous,
        "current_hashes": {str(p): sha(p) for p in (model_path, readme_path)},
        "quality_result_sha256": sha(followup / "attention-qkv-int8-quality-001/result.json"),
        "review_result_sha256": sha(followup / "attention-qkv-int8-review-001/result.json"),
        "full_sdr_db": full, "full_sdr_delta_db": delta,
        "goal_complete": False, "native_host_qualified": False, "budget_after": snapshot()})
    print({"status": "pass", "full_sdr_db": full, "delta_db": delta})


if __name__ == "__main__":
    main()
