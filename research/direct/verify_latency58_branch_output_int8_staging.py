"""Read-only completion audit accounting for the instruction-file symlink."""
from pathlib import Path
import os
from research.direct.run_latency58_quality import ROOT, PHASE, read, write, require, sha
from research.direct.train_latency58 import verify_inputs
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot


def main():
    old = PHASE / "branch-output-int8-plugin-preparation-001"
    p, failure = read(old / "plan.json"), read(old / "execution.json")
    target = Path(p["target"])
    require(failure["actual_exit_code"] == 1 and failure["source_bindings_unchanged"], "Unexpected earlier exit")
    require((target / "CLAUDE.md").is_symlink() and os.readlink(target / "CLAUDE.md") == "AGENTS.md", "Unexpected instruction alias")
    verify_inputs(p)
    edited = set(p["planned_mutations"]) | {"CLAUDE.md"}
    require(all(sha(target / n) == h for n,h in p["target_files_before"].items() if n not in edited), "An unrelated file changed")
    short = read(PHASE / "branch-output-int8-screen-003/result.json")
    fixture = read(PHASE / "branch-output-int8-fixture-001/result.json")
    require(sha(target / "model/model.onnx") == p["graph_sha256"] == short["graph_sha256"], "Staged model differs")
    for suffix in (".bin", ".json"):
        name = "cropped1024-pytorch" + suffix
        require(sha(target / "test/fixtures" / name) == sha(PHASE / "branch-output-int8-fixture-001" / name), "Staged fixture differs")
    require(read(target / "test/fixtures/cropped1024-pytorch.json")["deployment_graph_sha256"] == p["graph_sha256"], "Fixture/model identity differs")
    contract = (target / "cmake/QualifiedModelContract.cmake").read_text()
    require(p["graph_sha256"] in contract and '"38298020"' in contract and 'branch-memory-sixteen-s8-hop128-step39250-m4-test-pdc256' in contract, "Staged contract differs")
    for name in ("quality-deployment.json", "linux-validation.json"):
        report = read(target / "model" / name)
        require(report["status"] == "pending" and report["graph_sha256"] == p["graph_sha256"], "Old qualification claim remains")
    streaming = read(target / "model/streaming-validation.json")
    require(streaming["graph_sha256"] == p["graph_sha256"] and streaming["fixture"]["result"]["fixture_sha256"] == fixture["fixture_sha256"], "Staged streaming evidence differs")
    out = PHASE / "branch-output-int8-plugin-preparation-002"
    require(not out.exists(), "Preserve completion audit")
    bindings = dict(p["source_bindings"])
    bindings.update({str(f):sha(f) for f in (Path(__file__).resolve(), old / "plan.json", old / "execution.json")})
    plan = {"source_bindings":bindings,"target":str(target),"parent_preparation":str(old),"read_only":True,
        "alias_correction":"CLAUDE.md is a symlink to the intentionally updated AGENTS.md; symlink bytes unchanged",
        "graph_sha256":p["graph_sha256"],"fixture_sha256":p["fixture_sha256"]}
    out.mkdir();write(out / "plan.json",plan)
    verify_inputs(plan)
    budget = budget_snapshot({**read(PHASE / "deployed-vocal-views-001/plan.json"),"diagnostic_artifact_allowance_bytes":700_000_000})
    write(out / "result.json",{"status":"pass","source_bindings_unchanged":True,"graph_sha256":p["graph_sha256"],
        "fixture_sha256":p["fixture_sha256"],"metadata_entries":90,"target_files_after":{n:sha(target/n) for n in edited},
        "unrelated_tracked_files_unchanged":True,"quality_selected":False,"native_host_qualified":False,"budget_after":budget})
    print({"status":"pass","metadata_entries":90,"alias_verified":True})


if __name__ == "__main__":
    main()
