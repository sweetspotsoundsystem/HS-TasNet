"""Retry the unchanged plugin with the retained Linux SDK environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write, execute
from research.direct.train_latency58 import verify_inputs


def main():
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    previous = PHASE / "branch-gru-int8-native-tests-001"
    original, failure = read(previous / "plan.json"), read(previous / "execution.json")
    require(failure["actual_exit_code"] == 1 and failure["source_bindings_unchanged"]
            and not failure["timed_out"] and not (previous / "correctness.xml").exists(), "Unexpected prior attempt")
    verify_inputs(original)
    sysroot = Path("/home/axel/autoresearch/codex/stemgen-rt-cropped1024-11ms/.local-linux-deps/sysroot")
    native_lib = sysroot / "usr/lib/x86_64-linux-gnu"
    environment = {"CPATH": ":".join(str(sysroot / "usr/include" / tail) for tail in ("", "freetype2", "libpng16")),
        "LIBRARY_PATH": str(native_lib), "LD_LIBRARY_PATH": str(native_lib),
        "PKG_CONFIG_PATH": str(native_lib / "pkgconfig"), "PKG_CONFIG_SYSROOT_DIR": str(sysroot),
        "CMAKE_BUILD_PARALLEL_LEVEL": "2"}
    require("PKG_CONFIG_LIBDIR" not in os.environ, "Do not hide system zlib package metadata")
    os.environ.update(environment)
    bindings = dict(original["source_bindings"])
    paths = [Path(__file__).resolve(), *[previous / name for name in ("plan.json", "execution.json", "configure-execution.json")]]
    paths.extend(p for p in sysroot.rglob("*") if p.is_file())
    bindings.update({str(p): sha(p) for p in paths})
    verify_inputs({"source_bindings": bindings})
    budget_plan = {**read(PHASE / "deployed-vocal-views-001/plan.json"), "diagnostic_artifact_allowance_bytes": 2_040_000_000}
    before = budget_snapshot(budget_plan)
    out = PHASE / "branch-gru-int8-native-tests-002"
    target, build = Path(original["target"]), Path(original["target"]) / "build-native-002"
    require(not out.exists() and not build.exists(), "Preserve prior retry artifacts")
    out.mkdir()
    plan = {**original, "schema": "latency58-fourteen-projection-native-tests-retry-v1",
        "source_bindings": bindings, "build_directory": str(build), "environment": environment,
        "budget_before": before, "retry_reason": "Use the existing Linux development sysroot; model, fixture and runtime code unchanged."}
    write(out / "plan.json", plan)
    configure = read(previous / "configure-execution.json")["argv"]
    configure[configure.index("-B") + 1] = str(build)
    commands = [("configure", configure, 1800),
                ("build", ["cmake", "--build", str(build), "--target", "AudioPluginTest", "RealtimeSafetyTest", "-j", "2"], 3600),
                ("correctness", ["ctest", "--test-dir", str(build), "--output-on-failure", "--output-junit",
                                 str(out / "correctness.xml"), "-j", "1"], 3600)]
    for label, argv, timeout in commands:
        execute(argv, out, label, timeout, bindings, {"plan_sha256": sha(out / "plan.json"), "graph_sha256": original["graph_sha256"]})
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
        "graph_sha256": original["graph_sha256"], "test_binaries": binaries,
        "passed": passed, "skipped_or_disabled": skipped, "total": len(cases),
        "correctness_xml_sha256": sha(out / "correctness.xml"), "native_host_qualified": False,
        "quality_selected": False, "budget_before": before, "budget_after": budget_snapshot(budget_plan)})
    print(json.dumps({"status": "pass", "passed": len(passed), "skipped_or_disabled": len(skipped), "total": len(cases),
                      "native_host_qualified": False}))


if __name__ == "__main__":
    main()
