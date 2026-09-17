"""Package completed quality evidence and the model/PR descriptions for M4 testing."""
import json
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha

STEMS = ("drums", "bass", "vocals", "other")


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items() if key != "source_bindings"}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
        return value[len(str(ROOT)) + 1:]
    return value


def main():
    out = PHASE / "best-model-stemgen-rt-001"
    full = PHASE / "branch-plugin-full14-001"
    result, execution = read(full / "result.json"), read(full / "execution.json")
    require(result["status"] == "pass" and result["quality_handoff_gate_passed"]
            and result["source_bindings_unchanged"] and execution["actual_exit_code"] == 0
            and result["track_count"] == 14 and result["excerpt_count"] == 28,
            "Complete the actual deployment evaluation before packaging its claims")
    require(sha(out / "model/model.onnx") == result["graph_sha256"]
            and read(out / "model/streaming-validation.json")["long"]["status"] == "pass"
            and read(out / "model/linux-validation.json")["correctness"]["status"] == "pass",
            "The PR graph, long reference checks or native tests differ")
    keys = ("status", "quality_handoff_gate_passed", "results", "source_checkpoint_comparison",
            "previous_pr_comparison", "c204_comparison", "source_checkpoint_all_track_stem_cells",
            "previous_pr_all_track_stem_cells", "c204_all_track_stem_cells", "track_count", "excerpt_count",
            "runtime", "graph_sha256", "graph_bytes", "graph_delay_samples", "host_queue_samples")
    public = {"schema": "stemgenrt-deployment-quality-v2",
              "scope": "Exact saved eight-state integer ONNX graph on the unchanged development panel; includes final residual reconstruction. Source FP32 quality is separate.",
              "original_result_sha256": sha(full / "result.json"), "execution": execution,
              "source_checkpoint_full14_sdr_db": 4.46515742201644,
              **{key: result[key] for key in keys}}
    public["runtime"] = {"version": result["runtime"]["version"],
                         "python_module_sha256": sha(result["runtime"]["python_module"]),
                         "execution_provider": "CPUExecutionProvider", "intra_op_threads": 1,
                         "inter_op_threads": 1, "execution_mode": "sequential", "spinning": False}
    (out / "model/quality-deployment.json").write_text(json.dumps(portable(public), indent=2, allow_nan=False) + "\n")
    aggregate = result["results"][0]["aggregate"]
    score = aggregate["full_sdr_db"]
    prior = result["previous_pr_comparison"]
    c204 = result["c204_comparison"]
    source = result["source_checkpoint_comparison"]
    track_deltas = prior["metrics"]["full_sdr_db"]["per_track_macro_delta"]
    up, down = sum(v > 0 for v in track_deltas.values()), sum(v < 0 for v in track_deltas.values())
    ci = prior["metrics"]["full_sdr_db"]["paired_track_bootstrap_95_percent"]
    regressed_cells = []
    for track, stems in result["previous_pr_all_track_stem_cells"].items():
        for stem, values in stems.items():
            delta = values["metrics"]["full_sdr_db"]["delta"]
            if delta is not None and delta < 0:
                regressed_cells.append((delta, track, stem))
    lines = [
        f"The exact saved deployment graph scores **{score:.6f} dB full-band SDR** on",
        "the unchanged development panel: 14 tracks, two 15-second excerpts per track,",
        f"four stems. This is **{prior['metrics']['full_sdr_db']['delta']:+.6f} dB** versus the prior PR graph",
        f"({prior['metrics']['full_sdr_db']['reference']:.6f} dB) and **{c204['metrics']['full_sdr_db']['delta']:+.6f} dB** versus C204",
        f"({c204['metrics']['full_sdr_db']['reference']:.6f} dB). Its difference from the source FP32 checkpoint is",
        f"**{source['metrics']['full_sdr_db']['delta']:+.6f} dB**.", "",
        "| Deployment metric, dB | Drums | Bass | Vocals | Other |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Full-band SDR | " + " | ".join(f"{aggregate['per_stem'][s]['full_sdr_db']:.3f}" for s in STEMS) + " |",
    ]
    for label, comparison, metric in (("SDR change vs prior PR", prior, "full_sdr_db"),
                                      ("SIR change vs prior PR", prior, "bleed_sir_db"),
                                      ("SDR change vs C204", c204, "full_sdr_db")):
        lines.append("| " + label + " | " + " | ".join(
            f"{comparison['metrics'][metric]['per_stem_delta'][s]:+.3f}" for s in STEMS) + " |")
    lines.append("| Absent-source output change vs prior PR (lower is better) | " + " | ".join(
        f"{prior['absence_dbfs_delta'][s]:+.3f}" if prior['absence_dbfs_delta'][s] is not None else "n/a"
        for s in STEMS) + " |")
    lines.append("| Absent-source output change vs C204 (lower is better) | " + " | ".join(
        f"{c204['absence_dbfs_delta'][s]:+.3f}" if c204['absence_dbfs_delta'][s] is not None else "n/a"
        for s in STEMS) + " |")
    lines += ["", f"Against the prior PR, {up} tracks improve and {down} regress in mean SDR.",
              f"The paired track-bootstrap interval for the average gain is {ci[0]:+.4f} to",
              f"{ci[1]:+.4f} dB. An average gain can still include individual stem or",
              "absence regressions; the linked report includes every cell."]
    lines += [f"Of the 56 track/stem SDR cells, {len(regressed_cells)} regress versus the prior PR."]
    absence_worse = [s for s in STEMS if prior["absence_dbfs_delta"][s] is not None
                     and prior["absence_dbfs_delta"][s] > 0]
    absence_note = ("Mean absent-source output rises for " + ", ".join(absence_worse)
                    + " versus the prior PR. Lower is better for this metric; include instrumental and quiet passages in listening tests."
                    if absence_worse else "Mean absent-source output does not increase versus the prior PR.")
    lines += ["", absence_note]
    negative = sorted((delta, name) for name, delta in c204["metrics"]["full_sdr_db"]["per_track_macro_delta"].items()
                      if delta < 0)
    if negative:
        lines += ["", "Tracks with lower mean SDR than C204: " + "; ".join(
            f"{name} ({delta:+.3f} dB)" for delta, name in negative) + "."]
    template = (PHASE / "branch-plugin-model-readme-001.md").read_text()
    require(template.count("@@QUALITY@@") == 1, "Documentation template differs")
    (out / "model/README.md").write_text(template.replace("@@QUALITY@@", "\n".join(lines)))
    body = f"""Update the bundled model to the saved EMA checkpoint with separate spectral and waveform memories. The new deployment graph scores **{score:.6f} dB full-band SDR**, versus **{prior['metrics']['full_sdr_db']['reference']:.6f} dB** for the previous PR graph and **{c204['metrics']['full_sdr_db']['reference']:.6f} dB** for C204, on the unchanged 14-track / 28-excerpt development protocol. The source FP32 checkpoint scores **4.465157 dB**; both scores are recorded separately.

The runtime now carries eight states through initialization, validation, inference and reset. The graph still uses 128-sample hops at 44.1 kHz, with **256 samples / 5.80 ms** of graph-plus-host delay at a 128-sample host buffer and one inference worker. The self-contained Git LFS graph is 48,754,181 bytes, SHA-256 `{result['graph_sha256']}`.

Validation completed:

- Independent signed-weight reconstruction and PyTorch reference parity with ORT optimizations disabled/enabled; all eight states and four outputs checked, including nonzero states and partial EOF.
- Two 30-second music runs (10,338 calls each), exact reset replay, all three GRU state tensors and both attention caches exact, maximum waveform error 3.5763e-7.
- Release native correctness: 162 tests passed, one Windows-only test skipped on Linux, seven disabled by default. Coverage includes queue/reset races, PDC/Main alignment, one-flush recovery, variable offline callbacks, non-finite input, reconstruction and callback allocation checks.
- Full deployment quality and every track/stem/band/absence comparison retained in `model/quality-deployment.json`; source quality and numerical/native evidence updated alongside it.

Mean track SDR improves on {up} tracks and declines on {down} versus the prior PR; {len(regressed_cells)} of 56 individual track/stem SDR cells regress. These are development-panel results from repeated model selection. {absence_note} Local regressions are documented in `model/README.md`; the 5 dB research goal remains active.

**M4/M4 Pro timing and DAW acceptance are pending.** This larger model needs fresh measurements using `M4_TESTING.md`. Linux correctness ran alongside training and CPU scoring; no quiet timing qualification is claimed. C204 (`6fc2382`) and the prior attention candidate (`c848050`) remain available for rollback. Merge to main and a GitHub release follow the user's target-Mac and DAW tests.
"""
    (PHASE / "branch-plugin-pr-body-001.md").write_text(body)
    print(json.dumps({"deployment_sdr_db": score, "delta_previous_pr": prior["metrics"]["full_sdr_db"]["delta"],
                      "delta_source_fp32": source["metrics"]["full_sdr_db"]["delta"], "tracks_up": up,
                      "tracks_down": down, "per_stem_sdr": {s: aggregate["per_stem"][s]["full_sdr_db"] for s in STEMS}}))


if __name__ == "__main__":
    main()
