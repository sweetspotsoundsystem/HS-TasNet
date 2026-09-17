"""Package the reviewed integer artifact and portable native Mac timing probe."""
import json
from pathlib import Path
import zipfile

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require
from research.direct.train_latency58 import verify_inputs

RUNNER = '''#!/bin/bash
set -euo pipefail
PROBE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || {
  echo "Run from a native Apple silicon Terminal." >&2; exit 1;
}
xcode-select -p >/dev/null 2>&1 || {
  echo "Install Apple's Command Line Tools with: xcode-select --install" >&2; exit 1;
}
BASELINE="${1:-}"
if [[ -z "$BASELINE" ]]; then
  for probe in \
    "$HOME/Library/Audio/Plug-Ins/VST3/StemgenRT.vst3/Contents/Resources/model.onnx" \
    "$HOME/Library/Audio/Plug-Ins/Components/StemgenRT.component/Contents/Resources/model.onnx" \
    "/Library/Audio/Plug-Ins/VST3/StemgenRT.vst3/Contents/Resources/model.onnx" \
    "/Library/Audio/Plug-Ins/Components/StemgenRT.component/Contents/Resources/model.onnx"; do
    if [[ -f "$probe" && "$(shasum -a 256 "$probe" | awk '{print $1}')" == \
       "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3" ]]; then
      BASELINE="$probe"; break
    fi
  done
fi
[[ -n "$BASELINE" && -f "$BASELINE" ]] || {
  echo "C204 baseline not found. Run: bash run-macos.sh /path/to/C204/model.onnx" >&2; exit 1;
}
SDK="$PROBE_ROOT/libs/onnxruntime"
if [[ ! -f "$SDK/include/onnxruntime_cxx_api.h" || ! -f "$SDK/lib/libonnxruntime.dylib" ]]; then
  bash "$PROBE_ROOT/scripts/download-onnxruntime.sh"
fi
OUTPUT="$PROBE_ROOT/results-$(date -u +%Y%m%dT%H%M%SZ)"
python3 "$PROBE_ROOT/benchmark_latency58_native.py" --sdk "$SDK" \
  --baseline "$BASELINE" --candidate "$PROBE_ROOT/quantized.onnx" --output "$OUTPUT"
echo "Timing evidence: $OUTPUT/result.json"
'''

README = '''# StemgenRT M4 inference comparison

Extract this folder on the M4 mini and M4 Pro, then run from Terminal:

```bash
bash run-macos.sh
```

The probe finds the installed C204 model, downloads the same official ONNX
Runtime 1.26.0 CPU SDK used by the plugin (32 MB, checksum verified), and builds
a small native benchmark using Apple's Command Line Tools. All downloaded and
generated files stay in this extracted folder or a temporary build directory.
It does not install a model or modify a plugin bundle.

Run on AC power with the same background workload on both Macs. Keep the DAW
closed for this initial inference comparison. Each model receives the same
synthetic stereo signal, four alternating trials, 256 warmup hops and 2,048
measured hops per trial. Both use one CPU worker, the plugin's Mac QoS, and
preallocated tensors. No Python inference library is needed.

The resulting `results-*/result.json` records the Mac model, runtime, exact
graph hashes, median/p95/p99/maximum times and calls exceeding the 2.9025 ms
hop budget. The full plugin callback and worker queue require a separate DAW
playback check; this benchmark cannot qualify that path.

If the baseline is in a source checkout instead of an installed bundle:

```bash
bash run-macos.sh /path/to/stemgen-rt/model/model.onnx
```

The candidate is experimental. Its full 14-track score is 4.067196 dB versus
4.069079 dB for the C204 float model, with the same result under ORT 1.26 and
1.27. Linux native trials showed about 54% lower median inference time under
load. Mac timing has not been measured. Its long independent PyTorch comparison
exceeds the original float model's strict tolerances, so the current plugin
has not been changed to accept this graph. The separate quality-training goal
remains at least 5 dB.
'''


def main():
    from research.direct.latency58_sdr_checkpoint import require_space
    require(Path.cwd() == ROOT, "Use the reviewed workspace")
    source_path = PHASE / "full-magnitude-fast16-001/plan.json"
    source = read(source_path)
    require_space(source, 405_000_000)
    screen_path = PHASE / "m4-int8-screen-002/result.json"
    quality_path = PHASE / "m4-int8-ort126-full14-001/result.json"
    screen, quality = read(screen_path), read(quality_path)
    require(quality["status"] == "pass" and quality["track_count"] == 14, "Full quality check incomplete")
    graph = Path(screen["quantized"]["path"])
    require(sha(graph) == screen["quantized"]["sha256"], "Quantized model changed")
    cpp, python = (ROOT / "research/direct" / ("benchmark_latency58_native" + suffix) for suffix in (".cpp", ".py"))
    sdk_script = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/scripts/download-onnxruntime.sh")
    paths = [source_path, screen_path, quality_path, graph, cpp, python, sdk_script, Path(__file__).resolve()]
    bindings = {str(path): sha(path) for path in paths}
    out = PHASE / "m4-benchmark-package-001"
    require(not out.exists(), "Preserve benchmark packages")
    out.mkdir()
    write(out / "plan.json", {"source_bindings": bindings, "native_host_qualified": False,
          "purpose": "Portable native timing comparison on the user's M4 and M4 Pro"})
    sdk = sdk_script.read_text()
    footer = 'echo "Now rebuild your project:"'
    require(footer in sdk, "SDK script footer changed")
    sdk = sdk.split(footer)[0]
    evidence = {"quantized": screen["quantized"], "comparison": quality["comparison"]["metrics"]["full_sdr_db"],
                "runtime_126_and_127_full_score_delta": quality["runtime_version_comparison"]["metrics"]["full_sdr_db"]["delta"],
                "native_host_qualified": False, "plugin_modified": False}
    archive = out / "StemgenRT-M4-comparison.zip"
    prefix = "StemgenRT-M4-comparison/"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        bundle.write(graph, prefix + "quantized.onnx")
        bundle.write(cpp, prefix + cpp.name)
        bundle.write(python, prefix + python.name)
        bundle.writestr(prefix + "run-macos.sh", RUNNER)
        bundle.writestr(prefix + "scripts/download-onnxruntime.sh", sdk)
        bundle.writestr(prefix + "README.md", README)
        bundle.writestr(prefix + "quality-evidence.json", json.dumps(evidence, indent=2))
    require(archive.stat().st_size < 35_000_000, "Package exceeded its reservation")
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None and set(bundle.namelist()) == {
            prefix + name for name in ("quantized.onnx", cpp.name, python.name, "run-macos.sh",
                                      "scripts/download-onnxruntime.sh", "README.md", "quality-evidence.json")},
            "Archive contents or CRC check failed")
        import hashlib
        require(hashlib.sha256(bundle.read(prefix + "quantized.onnx")).hexdigest() == screen["quantized"]["sha256"],
                "Packaged graph differs")
    verify_inputs({"source_bindings": bindings})
    write(out / "result.json", {"status": "pass", "archive": {"path": str(archive), "sha256": sha(archive),
          "bytes": archive.stat().st_size}, "source_bindings_unchanged": True, "mac_execution_performed": False,
          "native_host_qualified": False, "counted_bytes_after": require_space(source, 370_000_000)})
    print(json.dumps(read(out / "result.json")), flush=True)


if __name__ == "__main__":
    main()
