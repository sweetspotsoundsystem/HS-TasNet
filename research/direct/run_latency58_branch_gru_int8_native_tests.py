"""Build and run the isolated fourteen-projection plugin's Release correctness suite."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    target, prep = (PHASE / name for name in ("branch-gru-int8-plugin-001", "branch-gru-int8-plugin-preparation-001"))
    preparation, execution = read(prep / "result.json"), read(prep / "execution.json")
    require(preparation["status"] == "pass" and preparation["source_bindings_unchanged"]
            and execution["actual_exit_code"] == 0 and not execution["timed_out"]
            and execution["source_bindings_unchanged"], "Native checkout preparation incomplete")
    verify_inputs(read(prep / "plan.json"))
    out = PHASE / "branch-gru-int8-native-tests-001"
    require(not out.exists(), "Preserve prior native build/test evidence")
    sdk = PHASE / "best-model-stemgen-rt-001/libs/onnxruntime"
    require((sdk / "VERSION_NUMBER").read_text().strip() == "1.26.0"
            and sha(sdk / "lib/libonnxruntime.so.1.26.0") == "5bd5bedf736fc501692435d0ec4f6e8b2bdf48cd30af8e6d00d61b3ddc9a7ab8",
            "Wrong native ORT SDK")
    dependencies = {"JUCE": (Path("/home/axel/autoresearch/codex/stemgen-rt-cropped1024-11ms/libs/juce"),
                             "51a8a6d7aeae7326956d747737ccf1575e61e209"),
                    "GOOGLETEST": (Path("/home/axel/autoresearch/codex/stemgen-rt-cropped1024-11ms/libs/googletest"),
                                   "6910c9d9165801d8827d628cb72eb7ea9dd538c5")}
    for directory, revision in dependencies.values():
        require(subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip() == revision,
                "Native dependency revision changed")
    budget_plan = {**read(PHASE / "deployed-vocal-views-001/plan.json"), "diagnostic_artifact_allowance_bytes": 2_040_000_000}
    before = budget_snapshot(budget_plan)
    destination = target / "libs/onnxruntime"
    require(not destination.exists() and not (target / "build-release").exists(), "Require a fresh native build")
    shutil.copytree(sdk, destination, symlinks=True)
    paths = [Path(__file__).resolve(), ROOT / "research/direct/run_latency58_quality.py",
             ROOT / "research/direct/train_latency58.py", ROOT / "research/direct/run_latency58_deployed_vocal_views.py"]
    paths.extend(prep / name for name in ("plan.json", "result.json", "execution.json"))
    tracked = subprocess.check_output(["git", "-C", str(target), "ls-files", "-z"]).decode().split("\0")[:-1]
    paths.extend(target / name for name in tracked if name.startswith(("cmake/", "plugin/", "test/"))
                 or name in ("CMakeLists.txt", "CMakePresets.json", "model/model.onnx"))
    paths.extend(p for p in destination.rglob("*") if p.is_file())
    for directory, _ in dependencies.values():
        names = subprocess.check_output(["git", "-C", str(directory), "ls-files", "-z"]).decode().split("\0")[:-1]
        paths.extend(directory / name for name in names if (directory / name).is_file())
    bindings = {str(p): sha(p) for p in paths}
    require(bindings[str(target / "model/model.onnx")] == preparation["graph_sha256"], "Staged graph changed")
    verify_inputs({"source_bindings": bindings})
    out.mkdir()
    plan = {"schema": "latency58-fourteen-projection-native-tests-v1", "source_bindings": bindings,
        "target": str(target), "graph_sha256": preparation["graph_sha256"], "budget_before": before,
        "build_jobs": 2, "test_jobs": 1, "inference_workers": 1, "build_type": "Release",
        "runtime": "1.26.0 CPU", "native_host_qualified": False,
        "scope": "Correctness suite under concurrent training/validation; paced tests remain disabled."}
    write(out / "plan.json", plan)
    build = target / "build-release"
    configure = ["cmake", "-S", str(target), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                 "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DSTEMGENRT_REQUIRE_QUALIFIED_ORT=ON"]
    configure.extend("-DCPM_%s_SOURCE=%s" % (name, spec[0]) for name, spec in dependencies.items())
    commands = [("configure", configure, 1800),
                ("build", ["cmake", "--build", str(build), "--target", "AudioPluginTest", "RealtimeSafetyTest", "-j", "2"], 3600),
                ("correctness", ["ctest", "--test-dir", str(build), "--output-on-failure", "--output-junit",
                                 str(out / "correctness.xml"), "-j", "1"], 3600)]
    for label, argv, timeout in commands:
        execute(argv, out, label, timeout, bindings, {"plan_sha256": sha(out / "plan.json"), "graph_sha256": preparation["graph_sha256"]})
        budget_snapshot(budget_plan)
    cases = ET.parse(out / "correctness.xml").getroot().findall(".//testcase")
    require(cases and all(c.find("failure") is None and c.find("error") is None for c in cases), "Native tests failed")
    passed = [c.attrib["name"] for c in cases if c.find("skipped") is None and c.attrib.get("status") == "run"]
    skipped = [c.attrib["name"] for c in cases if c.find("skipped") is not None or c.attrib.get("status") != "run"]
    require(len(cases) == 170 and len(passed) == 162 and len(skipped) == 8, "Native test inventory changed")
    binaries = {name: sha(build / "test" / name) for name in ("AudioPluginTest", "RealtimeSafetyTest")}
    verify_inputs(plan)
    write(out / "result.json", {"schema": "latency58-fourteen-projection-native-tests-result-v1", "status": "pass",
        "plan_sha256": sha(out / "plan.json"), "source_bindings_unchanged": True,
        "graph_sha256": preparation["graph_sha256"], "test_binaries": binaries,
        "passed": passed, "skipped_or_disabled": skipped, "total": len(cases),
        "correctness_xml_sha256": sha(out / "correctness.xml"), "native_host_qualified": False,
        "quality_selected": False, "budget_before": before, "budget_after": budget_snapshot(budget_plan)})
    print(json.dumps({"status": "pass", "passed": len(passed), "skipped_or_disabled": len(skipped), "total": len(cases),
                      "native_host_qualified": False}))


if __name__ == "__main__":
    main()
