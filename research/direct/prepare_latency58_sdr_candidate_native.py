"""Bind the preserved hop128 native diagnostic to a numerically verified model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write

BASE = PHASE / "teacher-half250-native-prep-001/latency58_teacher_native_async_qualifier.cpp"
BASE_SHA256 = "1af0daf50cd0bd2342f3a988a2e37738248581f37ccbb6e7b014ef410d1442e9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--verification-sha256", required=True)
    parser.add_argument("--export-execution", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and sha(BASE) == BASE_SHA256 and sha(args.verification) == args.verification_sha256,
            "Preserved native source or candidate verification changed")
    verified, execution = read(args.verification), read(args.export_execution)
    require(verified["schema"] == "latency58-sdr-candidate-onnx-verification-v1"
            and verified["status"] == "passed_cpu_numerical_verification_only" and verified["verification"]["passed"]
            and verified["source_bindings_unchanged"] and execution["actual_exit_code"] == 0
            and not execution["timed_out"] and execution["source_bindings_unchanged"]
            and execution["plan_sha256"] == verified["plan_sha256"]
            and sha(args.model) == verified["onnx_sha256"]
            and all(sha(p) == s for p, s in verified["source_bindings"].items()), "Candidate export did not pass with unchanged inputs")
    base_receipt = read(BASE.parent / "source-preparation.json")
    base_execution = read(BASE.parent / "compile-execution.json")
    require(base_receipt["source"]["sha256"] == BASE_SHA256 and base_execution["actual_exit_code"] == 0
            and not base_execution["timed_out"] and base_execution["source_bindings_unchanged"], "Preserved native source was not compiled")
    meta, provenance = verified["metadata"], verified["model_provenance"]
    expected = {"hs_tasnet.architecture_version": "cropped1024-asymmetric256-hop128-v1",
                "hs_tasnet.hop_samples": "128", "hs_tasnet.graph_output_delay_samples": "128",
                "hs_tasnet.intended_total_latency_samples": "256", "hs_tasnet.initial_state": "all_zeros",
                "hs_tasnet.teacher_used_in_inference": "false", "hs_tasnet.external_data": "false",
                "hs_tasnet.checkpoint_sha256": verified["checkpoint"]["sha256"],
                "hs_tasnet.model_state_sha256": verified["model_state_sha256"],
                "hs_tasnet.snapshot_step": str(verified["step"]),
                "hs_tasnet.training_updates": str(provenance["training_updates"])}
    require(all(meta[k] == v for k, v in expected.items()) and verified["step"] > 0
            and provenance["parent_training_updates"] + verified["step"] == provenance["training_updates"]
            and 2250 + provenance["pilot_updates"] == provenance["training_updates"], "Candidate geometry or cumulative lineage differs")
    source = BASE.read_text()
    replacements = {
        base_receipt["checkpoint_sha256"]: verified["checkpoint"]["sha256"],
        base_receipt["model_state_sha256"]: verified["model_state_sha256"],
        'expect("hs_tasnet.snapshot_step", "250");': f'expect("hs_tasnet.snapshot_step", "{verified["step"]}");',
        'expect("hs_tasnet.training_updates", "5000");': f'expect("hs_tasnet.training_updates", "{provenance["training_updates"]}");',
    }
    for old, new in replacements.items():
        require(source.count(old) == 1, "Unexpected native identity occurrence: " + old)
        source = source.replace(old, new)
    out = args.output_directory.absolute()
    require(out.parent == PHASE and not out.exists(), "Use a fresh native source directory")
    out.mkdir()
    cpp = out / "latency58_sdr_candidate_native_async_qualifier.cpp"
    cpp.write_text(source)
    paths = (Path(__file__).resolve(), BASE, BASE.parent / "source-preparation.json", BASE.parent / "compile-execution.json",
             args.verification.resolve(), args.export_execution.resolve(), args.model.resolve())
    bindings = {**verified["source_bindings"], **{str(p): sha(p) for p in paths}}
    write(out / "source-preparation.json", {
        "schema": "latency58-sdr-candidate-native-source-v1", "source": {"path": str(cpp), "sha256": sha(cpp)},
        "source_bindings": bindings, "identity_replacements": replacements, "native_algorithm_unchanged": True,
        "checkpoint_sha256": verified["checkpoint"]["sha256"], "model_state_sha256": verified["model_state_sha256"],
        "snapshot_step": verified["step"], "cumulative_training_updates": provenance["training_updates"],
        "model_provenance": provenance, "export_purpose": verified["export_purpose"],
        "compiled": False, "native_executed": False, "native_host_qualified": False,
    })
    print(json.dumps({"output_directory": str(out), "source_sha256": sha(cpp), "identity_replacements": 4}), flush=True)


if __name__ == "__main__":
    main()
