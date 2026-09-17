"""Stage the numerically checked candidate for local plugin verification."""
from pathlib import Path
import re
import shutil
import subprocess
from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.prepare_latency58_branch_gru_int8_plugin import portable
from research.direct.latency58_m4_followup_budget import POLICY, snapshot

FOLLOWUP = ROOT / "research/m4_followup_20260916"


def main():
    import onnx
    from research.direct.train_latency58 import verify_inputs
    target = FOLLOWUP / "attention-qkv-plugin-001"
    require(subprocess.check_output(["git", "-C", str(target), "branch", "--show-current"], text=True).strip()
            == "codex/m4-attention-qkv", "Wrong development worktree")
    out = FOLLOWUP / "attention-qkv-int8-plugin-preparation-001"
    require(not out.exists(), "Preserve preparation evidence")
    names = ("attention-qkv-int8-screen-001", "attention-qkv-int8-long-001", "attention-qkv-int8-fixture-001")
    roots = [FOLLOWUP / n for n in names]
    receipts = [FOLLOWUP / "attention-qkv-int8-execution.json", FOLLOWUP / "attention-qkv-long-execution.json", roots[-1] / "execution.json"]
    evidence, bindings = [], {str(Path(__file__).resolve()): sha(__file__)}
    for directory, receipt in zip(roots, receipts, strict=True):
        plan, result, execution = (read(p) for p in (directory / "plan.json", directory / "result.json", receipt))
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
                and not execution["timed_out"], "Incomplete numerical or fixture evidence")
        verify_inputs(plan)
        for p in (directory / "plan.json", directory / "result.json", receipt):
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
    bindings.update({str(p): sha(p) for p in (POLICY, ROOT / "research/direct/latency58_m4_followup_budget.py")})
    before = snapshot()
    out.mkdir()
    edited = ["cmake/QualifiedModelContract.cmake", "cmake/QualifiedModelContract.h.in", "model/model.onnx",
        "test/fixtures/cropped1024-pytorch.bin", "test/fixtures/cropped1024-pytorch.json",
        "test/source/OrtRunBenchmarkTest.cpp", "AGENTS.md", "model/streaming-validation.json",
        "model/quality-deployment.json", "model/linux-validation.json"]
    tracked = subprocess.check_output(["git", "-C", str(target), "ls-files", "-z"], text=True).split("\0")[:-1]
    require("CLAUDE.md" in tracked and not (target / "CLAUDE.md").exists(), "Expected sparse exclusion of tracked documentation symlink")
    before_files = {n: sha(target / n) for n in tracked if n != "CLAUDE.md"}
    plan = {"schema": "latency58-seventeen-projection-plugin-preparation-v1", "source_bindings": bindings,
        "target": str(target), "graph_sha256": fixture["graph_sha256"], "fixture_sha256": fixture["fixture_sha256"],
        "budget_before": before, "planned_mutations": edited, "target_files_before": before_files,
        "quality_selected": False, "native_host_qualified": False,
        "scope": "Local development model staging; quality and native suite still pending"}
    write(out / "plan.json", plan)
    graph = onnx.load(graph_path)
    props = {p.key: p.value for p in graph.metadata_props}
    require(len(props) == len(graph.metadata_props) == 94, "Unexpected metadata inventory")
    cmake = target / "cmake/QualifiedModelContract.cmake"
    prefix = cmake.read_text().split("set(STEMGENRT_QUALIFIED_METADATA_KIND", 1)[0]
    for old, new in (("branch-memory-sixteen-s8-hop128-step39250-m4-test-pdc256", "branch-memory-seventeen-qkv-s8-hop128-step39250-m4-test-pdc256"),
        ("c7ea50ac67bf4bfddf1f5ff41c6eb419af00fe420ce1a0b0eaeef11a1861cd61", fixture["graph_sha256"]),
        ('"38298020"', '"%d"' % graph_path.stat().st_size)):
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
    for name, old, new in (("test/source/OrtRunBenchmarkTest.cpp", "contract::kMetadata.size(), 90U", "contract::kMetadata.size(), 94U"),
                           ("AGENTS.md", "90 metadata entries", "94 metadata entries")):
        path = target / name;content = path.read_text();require(content.count(old) == 1, "Metadata count changed")
        path.write_text(content.replace(old,new))
    for name in ("quality-deployment.json", "linux-validation.json"):
        (target / "model" / name).write_text(__import__("json").dumps({"status":"pending", "graph_sha256":fixture["graph_sha256"],
            "scope":"This development graph is staged for ongoing quality/native verification.","native_host_qualified":False},indent=2)+"\n")
    (target / "model/streaming-validation.json").write_text(__import__("json").dumps({"schema":"stemgenrt-seventeen-projection-streaming-v1", "graph_sha256":fixture["graph_sha256"],
        "short":evidence[0],"long":evidence[1],"fixture":evidence[2],"native_host_qualified":False},indent=2,allow_nan=False)+"\n")
    verify_inputs(plan)
    require(all(sha(target/n) == h for n,h in before_files.items() if n not in edited), "Unrelated tracked file changed")
    require(sha(target / "model/model.onnx") == fixture["graph_sha256"] and sha(target / "test/fixtures/cropped1024-pytorch.bin") == fixture["fixture_sha256"], "Staged inference inputs differ")
    write(out / "result.json", {"status":"pass", "source_bindings_unchanged":True,"plan_sha256":sha(out/"plan.json"),
        "graph_sha256":fixture["graph_sha256"],"fixture_sha256":fixture["fixture_sha256"],"metadata_entries":len(props),
        "target_files_after":{n:sha(target/n) for n in edited}, "unrelated_tracked_files_unchanged":True,
        "quality_selected":False,"native_host_qualified":False,"budget_after":snapshot()})
    print({"status":"pass","target":str(target),"metadata_entries":len(props)})


if __name__ == "__main__":
    main()
