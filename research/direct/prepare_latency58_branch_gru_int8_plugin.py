"""Stage the qualified numerical experiment in its isolated native-test checkout."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items() if k != "source_bindings"}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
        return value[len(str(ROOT)) + 1:]
    return value


def replace(path, old, new):
    text = path.read_text()
    require(text.count(old) == 1, "Expected one replacement in " + str(path))
    path.write_text(text.replace(old, new))


def main():
    import onnx
    from research.direct.train_latency58 import verify_inputs
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    target = PHASE / "branch-gru-int8-plugin-001"
    base = PHASE / "branch-plugin-release-040-source"
    revision = "ef09113c90b8259268308c1f11d4087f80ca88d9"
    for directory in (base, target):
        require(subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip() == revision
                and not subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True),
                "Require clean isolated v0.4.0 source checkouts")
    out = PHASE / "branch-gru-int8-plugin-preparation-001"
    require(not out.exists(), "Preserve prior preparation evidence")
    roots = [PHASE / name for name in ("branch-gru-int8-screen-002", "branch-gru-int8-long-001", "branch-gru-int8-fixture-001")]
    evidence = []
    bindings = {str(Path(__file__).resolve()): sha(__file__)}
    for directory in roots:
        plan, result, execution = (read(directory / name) for name in ("plan.json", "result.json", "execution.json"))
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"], "Numerical or fixture evidence incomplete")
        # The short and long numerical checks authenticate their original
        # sources; fixture generation reverified the inference dependencies.
        if directory == roots[-1]:
            verify_inputs(plan)
        for name in ("plan.json", "result.json", "execution.json"):
            path = directory / name
            bindings[str(path)] = sha(path)
        evidence.append({"result_sha256": sha(directory / "result.json"),
                         "result": portable(result), "execution": portable(execution)})
    short, long, fixture = (item["result"] for item in evidence)
    require(short["graph_sha256"] == long["graph_sha256"] == fixture["graph_sha256"], "Numerical graph identity differs")
    graph_path = Path(read(roots[0] / "result.json")["saved_graph_path"])
    require(sha(graph_path) == fixture["graph_sha256"], "Saved graph changed")
    binary, metadata = (roots[-1] / ("cropped1024-pytorch" + suffix) for suffix in (".bin", ".json"))
    require(sha(binary) == fixture["fixture_sha256"] and read(metadata)["deployment_graph_sha256"] == fixture["graph_sha256"],
            "Fixture changed")
    paths = [graph_path, binary, metadata, PHASE / "branch-gru-int8-plugin-storage-001.json"]
    paths.extend(ROOT / "research/direct" / name for name in (
        "run_latency58_quality.py", "train_latency58.py", "run_latency58_deployed_vocal_views.py",
        "run_latency58_macos_extended_soak.sh"))
    base_names = subprocess.check_output(["git", "-C", str(base), "ls-files", "-z"]).decode().split("\0")[:-1]
    paths.extend(base / name for name in base_names)
    bindings.update({str(p): sha(p) for p in paths})
    budget_plan = {**read(PHASE / "deployed-vocal-views-001/plan.json"), "diagnostic_artifact_allowance_bytes": 2_040_000_000}
    before = budget_snapshot(budget_plan)
    out.mkdir()
    plan = {"schema": "latency58-fourteen-projection-plugin-preparation-v1", "source_bindings": bindings,
        "base_revision": revision, "target": str(target), "graph_sha256": fixture["graph_sha256"],
        "fixture_sha256": fixture["fixture_sha256"], "budget_before": before,
        "scope": "Isolated development checkout and native correctness preparation; quality and M4 acceptance remain pending."}
    write(out / "plan.json", plan)
    graph = onnx.load(graph_path)
    props = {p.key: p.value for p in graph.metadata_props}
    require(len(props) == len(graph.metadata_props) == 88, "Unexpected metadata inventory")
    cmake = target / "cmake/QualifiedModelContract.cmake"
    prefix = cmake.read_text().split("set(STEMGENRT_QUALIFIED_METADATA_KIND", 1)[0]
    prefix = prefix.replace("branch-memory-ema-s8-hop128-step39250-m4-test-pdc256",
                            "branch-memory-fourteen-s8-hop128-step39250-m4-test-pdc256")
    prefix = prefix.replace("d2945742d27fe23469614aef4f5b79e46fb1a11696ee2c8e6055c494163bcffa", fixture["graph_sha256"])
    prefix = prefix.replace('"48754181"', '"%d"' % graph_path.stat().st_size)
    constants = {key: "STEMGENRT_QUALIFIED_METADATA_" + re.sub(r"[^A-Z0-9]", "_", key.removeprefix("hs_tasnet.").upper())
                 for key in props}
    require(len(set(constants.values())) == len(props)
            and all("]==]" not in v and ')meta"' not in v for v in props.values()), "Metadata delimiter collision")
    cmake.write_text(prefix + "\n".join(f"set({constants[k]} [==[{v}]==])" for k, v in props.items()) + "\n")
    template = target / "cmake/QualifiedModelContract.h.in"
    content = template.read_text()
    first = content.index("inline constexpr auto kMetadata = ")
    last = content.index("\n});", first) + len("\n});")
    rows = ["inline constexpr auto kMetadata = std::to_array<MetadataEntry>({"]
    rows.extend('    {"' + key + '",\n     R"meta(@' + constants[key] + '@)meta"},' for key in props)
    rows.append("});")
    template.write_text(content[:first] + "\n".join(rows) + content[last:])
    shutil.copyfile(graph_path, target / "model/model.onnx")
    for source in (binary, metadata):
        shutil.copyfile(source, target / "test/fixtures" / source.name)
    replace(target / "test/source/OrtRunBenchmarkTest.cpp", "contract::kMetadata.size(), 84U", "contract::kMetadata.size(), 88U")
    replace(target / "AGENTS.md", "84 metadata entries", "88 metadata entries")
    # Existing quality/native reports described the released graph. Replace the
    # copied claims with explicit pending records until this graph completes.
    for name, purpose in (("quality-deployment.json", "Exact-graph full14 and vocal-view evaluation is running."),
                          ("linux-validation.json", "Native Release correctness suite has not yet run for this graph.")):
        (target / "model" / name).write_text(json.dumps({"status": "pending", "graph_sha256": fixture["graph_sha256"],
            "scope": purpose, "quality_selected": False, "native_host_qualified": False}, indent=2) + "\n")
    (target / "model/streaming-validation.json").write_text(json.dumps({"schema": "stemgenrt-fourteen-projection-streaming-v1",
        "graph_sha256": fixture["graph_sha256"], "short": evidence[0], "long": evidence[1], "fixture": evidence[2],
        "native_host_qualified": False}, indent=2, allow_nan=False) + "\n")
    readme = (base / "model/README.md").read_text().split("## Quality\n", 1)[0]
    readme = readme.replace("d2945742d27fe23469614aef4f5b79e46fb1a11696ee2c8e6055c494163bcffa", fixture["graph_sha256"])
    readme = readme.replace("48,754,181", "39,789,914").replace("84 metadata entries", "88 metadata entries")
    readme += """## Quality and current scope

This development graph adds signed integer weights and dynamic unsigned
activations to four branch-memory GRU matrix products. Fourteen projections
now use integer products; GRU biases, nonlinearities and output projections
retain their previous floating arithmetic. The source EMA checkpoint still
scores 4.465157 dB full-band SDR. That is not this changed graph's score.

The exact-graph 14-track quality and vocal-leakage evaluation is running.
The first completed tracks show essentially unchanged leakage. This is a
runtime cost experiment; it does not yet resolve the 5 dB quality target or
instrumental vocal separation. The preserved v0.4.0 graph scores 4.455188 dB.

## Numerical and native checks

The saved graph passed ten short cases and 8,216 longer graph calls per
implementation, with exact reset replay and maximum waveform error 8.381903e-8
against its independent fourteen-projection reference. Existing tolerances
remain fixed. The portable fixture covers eight clip lengths, including
partial EOF, with expectations derived in PyTorch and runtime imports blocked.
See [streaming evidence](streaming-validation.json).

The native Release suite remains pending for this checkout. The separate local
preallocated inference comparison measured about 40% lower median block p50
than v0.4.0 under concurrent load; every measured call still exceeded the local
hop budget. That x86 comparison establishes no M4 or complete-plugin timing.

## M4 playback

The user reported 1,408 fallback samples after a few minutes on M4 with v0.4.0.
No M4 test of this graph has run. The target remains zero additional fallback
during repeated 30-minute steady-playback tests and the installed DAW workload,
with startup/reset counters retained separately. Follow [M4 testing](../M4_TESTING.md).
The original v0.4.0, earlier attention model and C204 remain rollback baselines.
"""
    (target / "model/README.md").write_text(readme)
    (target / "M4_TESTING.md").write_text("""# Model tests on M4

This development checkout tests the fourteen-projection graph (SHA-256 prefix
`878c74694fa4`). It retains one inference worker, 44.1 kHz, 128-sample model hops
and 256 samples of graph-plus-host delay with a 128-sample host buffer.
Full quality review, native correctness and target-Mac playback are pending.
See [the model report](model/README.md) for current evidence.

After authenticating this checkout and building its native arm64 Release
binary with official ONNX Runtime 1.26.0, run the existing correctness and
paced qualification, then run `scripts/extended-soak-macos.sh` with the source
directory and a new evidence directory. The extended runner repeats 30-minute
paced tests and retains machine/model/source identities, raw logs and exits.
It does not qualify installed-DAW playback by itself.

In the DAW, record sample rate, buffer, workload, duration and cumulative
fallback counters. Separate startup/reset increments from steady playback;
require zero new steady-playback fallback. Include start/stop, seeks and loops,
instrumental passages, quiet real vocals, and Other-stem listening. Retain the
complete v0.4.0 bundle outside the plugin directory for rollback.
""")
    soak = ROOT / "research/direct/run_latency58_macos_extended_soak.sh"
    shutil.copyfile(soak, target / "scripts/extended-soak-macos.sh")
    (target / "scripts/extended-soak-macos.sh").chmod(0o755)
    verify_inputs(plan)
    require(sha(target / "model/model.onnx") == fixture["graph_sha256"]
            and sha(target / "test/fixtures/cropped1024-pytorch.bin") == fixture["fixture_sha256"], "Staged inputs differ")
    write(out / "result.json", {"status": "pass", "plan_sha256": sha(out / "plan.json"),
        "source_bindings_unchanged": True, "graph_sha256": fixture["graph_sha256"], "metadata_entries": len(props),
        "fixture_sha256": fixture["fixture_sha256"], "runtime_implementation_changed": False,
        "quality_selected": False, "native_host_qualified": False, "budget_after": budget_snapshot(budget_plan)})
    print(json.dumps({"status": "pass", "target": str(target), "metadata_entries": len(props)}))


if __name__ == "__main__":
    main()
