"""Stage the numerically checked candidate for local plugin verification."""
from pathlib import Path
import re
import shutil
import subprocess
from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.prepare_latency58_branch_gru_int8_plugin import portable


def main():
    import onnx
    from research.direct.train_latency58 import verify_inputs
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
    target = PHASE / "m4-inference-diagnostics-plugin-001"
    require(subprocess.check_output(["git", "-C", str(target), "branch", "--show-current"], text=True).strip()
            == "codex/m4-inference-diagnostics", "Wrong development worktree")
    out = PHASE / "branch-output-int8-plugin-preparation-001"
    require(not out.exists(), "Preserve preparation evidence")
    names = ("branch-output-int8-screen-003", "branch-output-int8-long-001", "branch-output-int8-fixture-001")
    roots = [PHASE / n for n in names]
    evidence, bindings = [], {str(Path(__file__).resolve()): sha(__file__)}
    for directory in roots:
        plan, result, execution = (read(directory / n) for n in ("plan.json", "result.json", "execution.json"))
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "Incomplete numerical or fixture evidence")
        verify_inputs(plan)
        for name in ("plan.json", "result.json", "execution.json"):
            p = directory / name
            bindings[str(p)] = sha(p)
        evidence.append({"result_sha256": sha(directory / "result.json"), "result": portable(result),
                         "execution": portable(execution)})
    short, long, fixture = (e["result"] for e in evidence)
    require(short["graph_sha256"] == long["graph_sha256"] == fixture["graph_sha256"], "Graph identity differs")
    graph_path = Path(read(roots[0] / "result.json")["saved_graph_path"])
    binary, metadata = (roots[-1] / ("cropped1024-pytorch" + suffix) for suffix in (".bin", ".json"))
    require(sha(graph_path) == fixture["graph_sha256"] and sha(binary) == fixture["fixture_sha256"]
            and read(metadata)["deployment_graph_sha256"] == fixture["graph_sha256"], "Staging inputs changed")
    bindings.update({str(p): sha(p) for p in (graph_path, binary, metadata,
        ROOT / "research/direct/prepare_latency58_branch_gru_int8_plugin.py")})
    budget_plan = {**read(PHASE / "deployed-vocal-views-001/plan.json"),
                   "diagnostic_artifact_allowance_bytes": 700_000_000}
    before = budget_snapshot(budget_plan)
    out.mkdir()
    edited = ["cmake/QualifiedModelContract.cmake", "cmake/QualifiedModelContract.h.in", "model/model.onnx",
        "test/fixtures/cropped1024-pytorch.bin", "test/fixtures/cropped1024-pytorch.json",
        "test/source/OrtRunBenchmarkTest.cpp", "AGENTS.md", "model/streaming-validation.json",
        "model/quality-deployment.json", "model/linux-validation.json"]
    tracked = subprocess.check_output(["git", "-C", str(target), "ls-files", "-z"], text=True).split("\0")[:-1]
    before_files = {n: sha(target / n) for n in tracked}
    plan = {"schema": "latency58-sixteen-projection-plugin-preparation-v1", "source_bindings": bindings,
        "target": str(target), "graph_sha256": fixture["graph_sha256"], "fixture_sha256": fixture["fixture_sha256"],
        "budget_before": before, "planned_mutations": edited, "target_files_before": before_files,
        "quality_selected": False, "native_host_qualified": False,
        "scope": "Local development model staging; quality and native suite still pending"}
    write(out / "plan.json", plan)
    graph = onnx.load(graph_path)
    props = {p.key: p.value for p in graph.metadata_props}
    require(len(props) == len(graph.metadata_props) == 90, "Unexpected metadata inventory")
    cmake = target / "cmake/QualifiedModelContract.cmake"
    prefix = cmake.read_text().split("set(STEMGENRT_QUALIFIED_METADATA_KIND", 1)[0]
    for old, new in (("branch-memory-fourteen-s8-hop128-step39250-m4-test-pdc256", "branch-memory-sixteen-s8-hop128-step39250-m4-test-pdc256"),
        ("878c74694fa4c558de1c5a75837893a0afeadcf57f6e3b860d5904cab04e9fc9", fixture["graph_sha256"]),
        ('"39789914"', '"%d"' % graph_path.stat().st_size)):
        require(prefix.count(old) == 1, "Contract identity changed")
        prefix = prefix.replace(old, new)
    constants = {key: "STEMGENRT_QUALIFIED_METADATA_" + re.sub(r"[^A-Z0-9]", "_", key.removeprefix("hs_tasnet.").upper()) for key in props}
    require(len(set(constants.values())) == len(props) and all("]==]" not in v and ')meta"' not in v for v in props.values()), "Metadata collision")
    cmake.write_text(prefix + "\n".join(f"set({constants[k]} [==[{v}]==])" for k,v in props.items()) + "\n")
    template = target / "cmake/QualifiedModelContract.h.in"
    content = template.read_text();first = content.index("inline constexpr auto kMetadata = ");last = content.index("\n});", first) + len("\n});")
    rows = ["inline constexpr auto kMetadata = std::to_array<MetadataEntry>({"]
    rows.extend('    {"' + key + '",\n     R"meta(@' + constants[key] + '@)meta"},' for key in props)
    rows.append("});")
    template.write_text(content[:first] + "\n".join(rows) + content[last:])
    shutil.copyfile(graph_path, target / "model/model.onnx")
    for source in (binary, metadata):shutil.copyfile(source, target / "test/fixtures" / source.name)
    for name, old, new in (("test/source/OrtRunBenchmarkTest.cpp", "contract::kMetadata.size(), 88U", "contract::kMetadata.size(), 90U"),
                           ("AGENTS.md", "88 metadata entries", "90 metadata entries")):
        path = target / name;content = path.read_text();require(content.count(old) == 1, "Metadata count changed")
        path.write_text(content.replace(old,new))
    for name in ("quality-deployment.json", "linux-validation.json"):
        (target / "model" / name).write_text(__import__("json").dumps({"status":"pending", "graph_sha256":fixture["graph_sha256"],
            "scope":"This development graph is staged for ongoing quality/native verification.","native_host_qualified":False},indent=2)+"\n")
    (target / "model/streaming-validation.json").write_text(__import__("json").dumps({"schema":"stemgenrt-sixteen-projection-streaming-v1", "graph_sha256":fixture["graph_sha256"],
        "short":evidence[0],"long":evidence[1],"fixture":evidence[2],"native_host_qualified":False},indent=2,allow_nan=False)+"\n")
    verify_inputs(plan)
    require(all(sha(target/n) == h for n,h in before_files.items() if n not in edited), "Unrelated tracked file changed")
    require(sha(target / "model/model.onnx") == fixture["graph_sha256"] and sha(target / "test/fixtures/cropped1024-pytorch.bin") == fixture["fixture_sha256"], "Staged inference inputs differ")
    write(out / "result.json", {"status":"pass", "source_bindings_unchanged":True,"plan_sha256":sha(out/"plan.json"),
        "graph_sha256":fixture["graph_sha256"],"fixture_sha256":fixture["fixture_sha256"],"metadata_entries":len(props),
        "target_files_after":{n:sha(target/n) for n in edited}, "unrelated_tracked_files_unchanged":True,
        "quality_selected":False,"native_host_qualified":False,"budget_after":budget_snapshot(budget_plan)})
    print({"status":"pass","target":str(target),"metadata_entries":len(props)})


if __name__ == "__main__":
    main()
