"""Package the independently checked precise integer graph for native M4 timing."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs
from research.direct.package_latency58_m4_benchmark import RUNNER, README
from research.direct.benchmark_latency58_precise_core_native import CANDIDATE_SHA


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT, "Use the reviewed workspace")
    source_path = PHASE / "full-magnitude-sdr-001/plan.json"
    source = read(source_path)
    saved, closed = (source_path.parent / name for name in
                     ("production-run/result.json", "production-stage/execution.json"))
    require(read(saved)["checkpoint_written"] and read(closed)["actual_exit_code"] == 0,
            "Previous training must be saved and closed")
    counted = require_space(source, 415_000_000)
    quality_path = PHASE / "m4-int8-precise-core-full14-001/result.json"
    screen_path = PHASE / "m4-int8-precise-core-001/result.json"
    long_path = PHASE / "m4-int8-precise-core-long-001/result.json"
    native_path = PHASE / "m4-native-precise-core-001/result.json"
    quality, screen, long, native = map(read, (quality_path, screen_path, long_path, native_path))
    require(quality["status"] == native["status"] == "pass" and quality["track_count"] == 14
            and screen["strict_parity_passed"]
            and all(row["existing_strict_tolerances_passed"] for row in long["cases"])
            and all(r["source_bindings_unchanged"] for r in (quality, screen, long, native)),
            "Quality, independent parity and native evidence must all be complete")
    checkpoint = quality["results"][0]["checkpoint"]
    graph = Path(checkpoint["path"])
    require(sha(graph) == checkpoint["sha256"] == CANDIDATE_SHA, "Reviewed graph changed")
    cpp = ROOT / "research/direct/benchmark_latency58_native.cpp"
    python = ROOT / "research/direct/benchmark_latency58_precise_core_native.py"
    old_helper = ROOT / "research/direct/package_latency58_m4_benchmark.py"
    sdk_script = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/scripts/download-onnxruntime.sh")
    paths = [source_path, saved, closed, quality_path, screen_path, long_path, native_path,
             graph, cpp, python, old_helper, sdk_script, Path(__file__).resolve()]
    bindings = {str(path): sha(path) for path in paths}
    out = PHASE / "m4-precise-benchmark-package-001"
    require(not out.exists(), "Preserve benchmark packages")
    runner = RUNNER.replace("benchmark_latency58_native.py", python.name)
    sdk = sdk_script.read_text()
    footer = 'echo "Now rebuild your project:"'
    require(footer in sdk, "SDK script footer changed")
    sdk = sdk.split(footer)[0]
    for text in (runner, sdk):
        subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    readme = README.split("The candidate is experimental.")[0] + '''The candidate uses U8U8 matrix projections, FP64 arithmetic before activation
quantization, and FP32 decoding and public state. Its unchanged 14-track score
under ONNX Runtime 1.26.0 is 4.067043 dB versus 4.069079 dB for C204 (-0.002036 dB).
Per-stem SDR changes (drums, bass, vocals, other) are -0.00236, +0.00171,
-0.00121 and -0.00629 dB. These results do not establish a quality improvement.
The full comparisons, including interference and absent-stem leakage, are in
quality-evidence.json.

All six independent short parity cases and a 30-second music check passed the
original strict tolerances. In that music check, recurrent state was bit exact
and maximum waveform disagreement was 4.03e-7. This comparison checks execution
of the revised integer model, not equality with the C204 float model.

Four alternating native Linux trials measured 53.7% lower median inference time
than C204 under host load. Both models missed deadlines on that host. Actual M4
timing and DAW playback remain unmeasured. The current plugin has not been
changed. This package supersedes the earlier integer benchmark package; the
separate quality-training goal remains at least 5 dB.
'''
    evidence = {"checkpoint": checkpoint, "comparison": quality["comparison"],
                "original_integer_comparison": quality["original_integer_comparison"],
                "strict_short_parity_passed": screen["strict_parity_passed"],
                "strict_long_parity_cases": long["cases"],
                "linux_native_median_ms": native["median_of_cycle_p50_ms"],
                "linux_native_candidate_over_baseline_ratio": native["candidate_over_baseline_ratio"],
                "mac_execution_performed": False, "native_host_qualified": False, "plugin_modified": False}
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "counted_bytes_before": counted,
          "reserved_package_bytes": 35_000_000, "reserved_next_training_bytes": 380_000_000,
          "purpose": "Portable native inference timing on M4 mini and M4 Pro"})
    archive = out / "StemgenRT-M4-precise-comparison.zip"
    prefix = "StemgenRT-M4-precise-comparison/"
    payloads = {"run-macos.sh": runner, "scripts/download-onnxruntime.sh": sdk,
                "README.md": readme, "quality-evidence.json": json.dumps(evidence, indent=2)}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path, name in ((graph, "quantized.onnx"), (cpp, cpp.name), (python, python.name)):
            bundle.write(path, prefix + name)
        for name, text in payloads.items():
            bundle.writestr(prefix + name, text)
    require(archive.stat().st_size < 35_000_000, "Package exceeded reservation")
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None and set(bundle.namelist()) == {
            prefix + name for name in ("quantized.onnx", cpp.name, python.name, *payloads)},
            "Archive contents or CRC check failed")
        require(hashlib.sha256(bundle.read(prefix + "quantized.onnx")).hexdigest() == CANDIDATE_SHA,
                "Packaged graph differs")
    verify_inputs({"source_bindings": bindings})
    write(out / "result.json", {"status": "pass", "archive": {"path": str(archive), "sha256": sha(archive),
          "bytes": archive.stat().st_size}, "source_bindings_unchanged": True,
          "mac_execution_performed": False, "native_host_qualified": False,
          "counted_bytes_after": require_space(source, 380_000_000)})
    print(json.dumps(read(out / "result.json")), flush=True)


if __name__ == "__main__":
    main()
