"""Bundle C204 and the quadrature ten-projection integer graph for a balanced M4 comparison."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs
from research.direct.package_latency58_m4_benchmark import RUNNER
from research.direct.benchmark_latency58_quadrature_candidates_native import C204_INT8_SHA, QUADRATURE_INT8_SHA


def main():
    from research.direct.latency58_pending_training_space import require_cpu_space
    require(Path.cwd() == ROOT, "Use the reviewed workspace")
    source_path = PHASE / "quadrature-continuation-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    counted = require_cpu_space(source, 62_000_000)
    quality_paths = {"int8_c204": PHASE / "m4-int8-precise-core-full14-001/result.json",
                     "int8_quadrature": PHASE / "m4-quadrature-magint8-full14-memory-001/result.json"}
    quality = {name: read(path) for name, path in quality_paths.items()}
    require(all(r["status"] == "pass" and r["track_count"] == 14 and r["source_bindings_unchanged"]
                for r in quality.values()), "Both full-panel evaluations must be complete")
    saved_root = PHASE / "m4-quadrature-magint8-saved-001"
    saved = read(saved_root / "result.json")
    require(saved["status"] == "pass" and saved["graph_saved"] and saved["source_bindings_unchanged"]
            and saved["file_identical_to_quality_tested_memory_graph"]
            and saved["quality_result"]["sha256"] == sha(quality_paths["int8_quadrature"])
            and saved["checkpoint"]["sha256"] == quality["int8_quadrature"]["results"][0]["checkpoint"]["sha256"],
            "Saved graph must match the completed in-memory quality test")
    verify_inputs(read(saved_root / "plan.json"))
    graphs = {"int8_c204": Path(quality["int8_c204"]["results"][0]["checkpoint"]["path"]),
              "int8_quadrature": Path(saved["checkpoint"]["path"])}
    expected = {"int8_c204": C204_INT8_SHA, "int8_quadrature": QUADRATURE_INT8_SHA}
    require(all(sha(path) == expected[name] for name, path in graphs.items()), "Reviewed graphs changed")
    short_paths = [PHASE / name / "result.json" for name in
                   ("m4-int8-precise-core-001", "m4-quadrature-magint8-screen-001")]
    long_paths = [PHASE / name / "result.json" for name in
                  ("m4-int8-precise-core-long-001", "m4-quadrature-magint8-long-001")]
    require(all(read(path)["graph_sha256"] == expected[label] and len(read(path)["cases"]) == 6
                for label, path in zip(expected, short_paths, strict=True))
            and all(len(read(path)["cases"]) == 2 for path in long_paths),
            "Parity case inventory or graph identity differs")
    require(all(read(p)["strict_parity_passed"] and read(p)["source_bindings_unchanged"] for p in short_paths)
            and all(read(p)["source_bindings_unchanged"] and
                    all(row["existing_strict_tolerances_passed"] for row in read(p)["cases"]) for p in long_paths),
            "Strict independent parity evidence incomplete")
    native_path = PHASE / "m4-quadrature-memory-native-001/result.json"
    native = read(native_path)
    require(native["status"] == "pass" and len(native["cycles"]) == 18 and native["source_bindings_unchanged"],
            "Balanced native comparison incomplete")
    execution_paths = [PHASE / name / "execution.json" for name in (
        "m4-quadrature-magint8-screen-stage-001", "m4-quadrature-magint8-long-stage-001",
        "m4-quadrature-magint8-full14-memory-stage-001", "m4-quadrature-memory-native-stage-001", "m4-quadrature-magint8-saved-stage-001",
        "m4-quadrature-candidates-portable-smoke-stage-001")]
    require(all(read(p)["actual_exit_code"] == 0 and not read(p)["timed_out"]
                and read(p)["source_bindings_unchanged"] for p in execution_paths),
            "New candidate checks require actual successful wrapper executions")
    native_review_path = native_path.with_name("selection-review.json")
    native_review = read(native_review_path)
    require(native_review["status"] == "eligible_for_saved_graph_and_M4_package"
            and native_review["actual_root_exit_code"] == 0 and native["native_speed_gate_passed"],
            "Native comparison must pass its predefined gate")
    verify_inputs(native_review)
    smoke_root = PHASE / "m4-quadrature-candidates-portable-smoke-001"
    smoke = read(smoke_root / "result.json")
    require(smoke["status"] == "pass" and len(smoke["cycles"]) == 18
            and smoke["source_bindings_unchanged"], "Portable saved-graph runner did not complete")
    verify_inputs(read(smoke_root / "plan.json"))
    review_path = PHASE / "m4-quadrature-magint8-full14-memory-001/selection-review.json"
    review = read(review_path)
    require(review["status"] == "advance_to_native_comparison" and review["actual_root_exit_code"] == 0,
            "Require reviewed full-panel quality and exact saved-byte provenance")
    verify_inputs(review)
    native_plan_path = native_path.with_name("plan.json")
    verify_inputs(read(native_plan_path))
    cpp = ROOT / "research/direct/benchmark_latency58_native.cpp"
    python = ROOT / "research/direct/benchmark_latency58_quadrature_candidates_native.py"
    old_helper = ROOT / "research/direct/package_latency58_m4_benchmark.py"
    sdk_script = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/scripts/download-onnxruntime.sh")
    paths = [source_path, *quality_paths.values(), *graphs.values(), *short_paths, *long_paths,
             native_path, native_plan_path, native_review_path, *execution_paths, review_path,
             saved_root / "result.json", saved_root / "plan.json", smoke_root / "result.json", smoke_root / "plan.json",
             cpp, python, old_helper, sdk_script, Path(__file__).resolve()]
    bindings = {str(p): sha(p) for p in paths}
    out = PHASE / "m4-quadrature-candidates-benchmark-package-001"
    require(not out.exists(), "Preserve benchmark packages")
    runner = RUNNER.replace("benchmark_latency58_native.py", python.name)
    runner = runner.replace('--candidate "$PROBE_ROOT/quantized.onnx"',
                            '--c204-int8 "$PROBE_ROOT/c204-int8.onnx" --quadrature-int8 "$PROBE_ROOT/quadrature-int8.onnx"')
    sdk = sdk_script.read_text()
    footer = 'echo "Now rebuild your project:"'
    require(footer in sdk, "SDK script footer changed")
    sdk = sdk.split(footer)[0]
    for script in (runner, sdk):
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    scores = {name: result["results"][0]["aggregate"]["full_sdr_db"] for name, result in quality.items()}
    baseline_score = quality["int8_c204"]["comparison"]["metrics"]["full_sdr_db"]["reference"]
    medians = native["median_of_cycle_p50_ms"]
    misses = {name: sum(row["run"]["calls_over_hop_budget"] for row in native["cycles"]
                        if row["variant"] == name) for name in medians}
    c204_delta = review["full14_vs_c204_fp32"]
    sir_delta = c204_delta["metrics"]["bleed_sir_db"]["per_stem_delta"]
    worst_sdr = min(c204_delta["metrics"]["full_sdr_db"]["per_track_macro_delta"].items(), key=lambda x: x[1])
    worst_sir = min(c204_delta["metrics"]["bleed_sir_db"]["per_track_macro_delta"].items(), key=lambda x: x[1])
    evidence = {"full14": {name: {"checkpoint": result["results"][0]["checkpoint"],
                                 "aggregate": result["results"][0]["aggregate"], "comparison": result["comparison"],
                                 "comparison_to_c204": result.get("c204_comparison", result["comparison"])}
                           for name, result in quality.items()},
                "short_independent_parity": [read(p) for p in short_paths],
                "long_independent_parity": [read(p)["cases"] for p in long_paths],
                "linux_balanced_native_during_training": native,
                "native_selection_review": native_review, "saved_graph_provenance": saved,
                "portable_runner_functional_smoke": smoke,
                "quadrature_selection_review": review,
                "new_candidate_actual_executions": [read(p) for p in execution_paths],
                "mac_execution_performed": False, "native_host_qualified": False, "plugin_modified": False}
    readme = f"""# StemgenRT M4 candidate comparison

Extract this folder on the M4 mini and M4 Pro, then run in a native Terminal:

```bash
bash run-macos.sh
```

The probe finds the installed C204 baseline, downloads the plugin's official
ONNX Runtime 1.26.0 CPU SDK with checksum verification, and compiles a small
native C++ benchmark with Apple's Command Line Tools. It runs all six orders
of the three models, with 256 warmup and 2,048 measured 128-sample hops for each
trial. Each model occupies each position twice. All use one CPU worker, the
plugin's Mac QoS and preallocated tensors. No Python inference library is needed.

Use AC power, close the DAW, and keep the background workload comparable across
the two Macs. Downloaded files stay inside the extracted folder; temporary
compiler output is removed. The probe does not install a model or edit a plugin.

| Model | Full14 SDR (dB) |
|---|---:|
| C204 FP32 baseline | {baseline_score:.6f} |
| C204 precise integer | {scores['int8_c204']:.6f} |
| Quadrature precise integer | {scores['int8_quadrature']:.6f} |

The quadrature graph is byte-identical to the graph scored in memory on all
14 tracks. File readback verifies its hash; full14 scoring was not rerun from
the file. It derives from the saved 4.227700 dB FP32 research checkpoint.
Both packaged integer graphs passed six independent parity cases and a
30-second music check using the original strict tolerances. These checks
establish execution fidelity of each integer variant. Their quality effects
are measured separately. Quadrature uses ten integer projections including
magnitude; the two phase factors remain FP32.

A separate balanced Linux native comparison ran while GPU training was active,
with CPU quality evaluation stopped. Median inference times were
{medians['c204_integer']:.3f} ms for C204 integer,
{medians['quadrature_nine']:.3f} ms for quadrature with nine integer projections,
and {medians['quadrature_ten']:.3f} ms for the packaged ten-projection graph.
The ten-projection version took 14.4% less time than the nine-projection version
and was faster in all six paired trials. It took about 2.2% more median time
than C204 integer. Across 12,288 hops each, calls above the 2.9025 ms budget were
{misses['c204_integer']}, {misses['quadrature_nine']} and {misses['quadrature_ten']}, respectively.
These measurements do not establish deadline reliability. They must not be
combined with earlier timings collected without training load. C204 FP32 was
not timed in this particular experiment. The nine-projection variant is not
packaged and has no full14 quality score. Actual M4 timing remains unmeasured.

Full per-stem SDR, interference, absence leakage and parity results are in
quality-evidence.json. The separate training goal remains at least 5 dB.

C204 integer has the closer quality match to the current plugin. Quadrature
integer has a higher panel average, with uneven
changes: compared with C204 FP32, drum SIR changes by {sir_delta['drums']:+.3f}
dB and vocal SIR by {sir_delta['vocals']:+.3f} dB. Its worst track SDR change is
{worst_sdr[1]:+.3f} dB on {worst_sdr[0]}, and its worst track SIR change is
{worst_sir[1]:+.3f} dB on {worst_sir[0]}. Most of these regressions are inherited
from the research parent. Neither candidate has replaced the working plugin.

Compared with its FP32 parent, the packaged integer graph loses 0.008912 dB
panel SDR and 0.113152 dB SIR. All four stem SIRs decline. Absent drum, bass
and vocal outputs increase slightly; absent other output decreases. The
aggregate SDR gain over C204 does not resolve those per-stem regressions.
The full review includes every stem and track delta. Further training is
running independently from this fixed, identified package.

The generated results-*/result.json and cycle files record the Mac identity,
runtime, exact graph hashes, medians, p95/p99/max times and calls exceeding
the 2.9025 ms hop budget. These unpaced inference measurements do not establish
DAW callback or worker-queue performance. A DAW playback check is still needed.

If C204 is in a source checkout rather than an installed plugin:

```bash
bash run-macos.sh /path/to/stemgen-rt/model/model.onnx
```
"""
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "counted_bytes_before": counted,
          "reserved_archive_bytes": 62_000_000, "pending_training_reservation_managed_by": "require_cpu_space",
          "purpose": "One balanced comparison of C204 and both precise integer candidates on M4 and M4 Pro"})
    archive = out / "StemgenRT-M4-quadrature-candidates.zip"
    prefix = "StemgenRT-M4-quadrature-candidates/"
    named_graphs = {"c204-int8.onnx": graphs["int8_c204"], "quadrature-int8.onnx": graphs["int8_quadrature"]}
    payloads = {"run-macos.sh": runner, "scripts/download-onnxruntime.sh": sdk,
                "README.md": readme, "quality-evidence.json": json.dumps(evidence, indent=2)}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, path in {**named_graphs, cpp.name: cpp, python.name: python}.items():
            bundle.write(path, prefix + name)
        for name, text in payloads.items():
            bundle.writestr(prefix + name, text)
    require(archive.stat().st_size < 62_000_000, "Archive exceeded its reservation")
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None and set(bundle.namelist()) == {
            prefix + name for name in (*named_graphs, cpp.name, python.name, *payloads)}, "Archive CRC or contents failed")
        for name, path in named_graphs.items():
            require(hashlib.sha256(bundle.read(prefix + name)).hexdigest() == sha(path), "Packaged graph differs")
    verify_inputs({"source_bindings": bindings})
    write(out / "result.json", {"status": "pass", "archive": {"path": str(archive), "sha256": sha(archive),
          "bytes": archive.stat().st_size}, "source_bindings_unchanged": True,
          "mac_execution_performed": False, "native_host_qualified": False,
          "counted_bytes_after": require_cpu_space(source, 0)})
    print(json.dumps(read(out / "result.json")), flush=True)


if __name__ == "__main__":
    main()
