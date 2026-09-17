"""Run bounded quality scoring of one immutable, independently audited generation.

This fixed derived model has no optimizer or new training updates. The
original music, probe and parallel-aggregation protocols are preserved.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from research.direct.run_latency58_quality import (
    PHASE, PYTHON, ROOT, checkpoint_bindings, execute, read, require, sha, write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-plan", type=Path, required=True)
    parser.add_argument("--construction-plan-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--modes", nargs="+", choices=("full14", "actions60", "probes"), default=["full14"])
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.output_prefix and all(c.isalnum() or c in "-_" for c in args.output_prefix)
            and len(set(args.modes)) == len(args.modes), "Invalid output prefix, cwd or duplicate modes")
    require(sha(args.construction_plan) == args.construction_plan_sha256, "Construction plan changed")
    construction = read(args.construction_plan)
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.latency58_weight_average import read_average
    require_space(construction, 1_400_000_000 + (25_000_000 if "actions60" in args.modes else 2_000_000))
    model_dir = Path(construction["output_directory"])
    receipt = read_average(construction, expected_plan_sha=args.construction_plan_sha256)
    qualification = {}
    for mode in ("build", "audit"):
        result_path, execution_path = model_dir / (mode + "-result.json"), model_dir / (mode + "-execution.json")
        result, execution = read(result_path), read(execution_path)
        require(result["status"] == "pass" and result["source_bindings_unchanged"]
                and execution["actual_exit_code"] == 0 and not execution["timed_out"]
                and execution["source_bindings_unchanged"]
                and result["plan_sha256"] == execution["plan_sha256"] == args.construction_plan_sha256
                and result["model_state_sha256"] == receipt["model_state_sha256"], "Incomplete averaged model")
        qualification[mode + "_result"] = {"path": str(result_path), "sha256": sha(result_path)}
        qualification[mode + "_execution"] = {"path": str(execution_path), "sha256": sha(execution_path)}
    require(read(qualification["audit_result"]["path"])["independent_saved_tensor_average_verified"],
            "Saved midpoint arithmetic was not independently audited")
    bindings = {**construction["source_bindings"], str(args.construction_plan.resolve()): args.construction_plan_sha256}
    paths = [model_dir / n for n in ("build-result.json", "build-execution.json", "audit-result.json", "audit-execution.json", "model.pt", "receipt.json")]
    paths += [Path(__file__).resolve(), ROOT / "research/direct/latency58_weight_average.py",
              ROOT / "research/direct/evaluate_latency58_weight_average.py",
              ROOT / "research/direct/compare_latency58_sdr.py", ROOT / "research/direct/compare.py"]
    if "probes" in args.modes:
        paths.append(ROOT / "research/direct/evaluate_latency58_weight_average_probes.py")
    if "full14" in args.modes:
        proof_path = PHASE / "sdr-parallel-baseline-qualification-full14-001/qualification.json"
        proof = read(proof_path)
        require(proof["status"] == "pass" and proof["exact_track_reports"] and proof["exact_stream_metadata"]
                and proof["exact_original_aggregate"] and proof["track_indices"] == list(range(6))
                and all(sha(p) == v for p, v in proof["source_bindings"].items()), "Original parallel qualification changed")
        bindings.update(proof["source_bindings"])
        paths += [proof_path, ROOT / "research/direct/evaluate_latency58_weight_average_parallel.py"]
    bindings.update({str(p): sha(p) for p in paths})
    prepared = []
    for mode in args.modes:
        template_path = PHASE / ("pilot2000-" + mode + "-001") / "plan.json"
        template, template_execution = read(template_path), read(template_path.parent / "execution.json")
        require(template_execution["actual_exit_code"] == 0 and not template_execution["timed_out"]
                and template_execution["source_bindings_unchanged"]
                and template_execution["plan_sha256"] == sha(template_path), "Require a previously executed protocol")
        old = checkpoint_bindings(template["checkpoint"])
        require(all(template["source_bindings"].get(p) == s for p, s in old.items()), "Old checkpoint inventory differs")
        unchanged = {p: s for p, s in template["source_bindings"].items() if p not in old}
        require(all(sha(p) == s for p, s in unchanged.items()), "Preserved evaluation protocol changed")
        out = PHASE / (args.output_prefix + "-" + mode + "-001")
        require(not out.exists(), "Preserve existing evaluation: " + str(out))
        plan = copy.deepcopy(template)
        plan.update(schema="latency58-weight-average-" + ("probe" if mode == "probes" else "music") + "-plan-v1",
                    mode=mode, label=args.output_prefix, step=0, component_steps=[500, 1000],
                    construction_plan={"path": str(args.construction_plan.resolve()), "sha256": args.construction_plan_sha256},
                    **qualification, expected_model_state_sha256=receipt["model_state_sha256"],
                    checkpoint={"path": str(model_dir / "model.pt"), "sha256": receipt["files"]["model.pt"]["sha256"]}, output_directory=str(out),
                    source_bindings={**unchanged, **bindings, str(template_path): sha(template_path)},
                    protocol_template={"path": str(template_path), "sha256": sha(template_path)})
        if mode == "full14":
            plan.update(schema="latency58-weight-average-parallel-music-plan-v1", workers=2,
                        model_kind="weight_average_candidate", track_indices=list(range(14)),
                        parallel_qualification={"path": str(proof_path), "sha256": sha(proof_path)})
        prepared.append((mode, out, plan))
    for mode, out, plan in prepared:
        out.mkdir()
        plan_path = out / "plan.json"
        write(plan_path, plan)
        module = "research.direct.evaluate_latency58_weight_average" + ("_probes" if mode == "probes" else "_parallel" if mode == "full14" else "")
        argv = [PYTHON, "-u", "-m", module, "--plan", str(plan_path), "--plan-sha256", sha(plan_path)]
        print(json.dumps({"event": "evaluate", "component_steps": [500, 1000], "step": receipt["step"],
                          "mode": mode, "plan_sha256": sha(plan_path)}), flush=True)
        execute(argv, out, "evaluation", plan["timeout_seconds"], plan["source_bindings"], {"plan_sha256": sha(plan_path)})
        if mode == "full14":
            from research.direct.compare_latency58_sdr import REFERENCE_PATHS
            source = ROOT / "research/direct/compare_latency58_sdr.py"
            comparison_bindings = {str(p): sha(p) for p in (source, ROOT / "research/direct/compare.py",
                out / "result.json", out / "execution.json", out / "plan.json", *REFERENCE_PATHS.values())}
            argv = [PYTHON, "-u", "-m", "research.direct.compare_latency58_sdr", "--candidate", str(out)]
            execute(argv, out, "comparison", 120, comparison_bindings, {"source_bindings": comparison_bindings})
    print(json.dumps({"event": "quality_bundle_completed", "component_steps": [500, 1000],
                      "step": receipt["step"], "quality_selected": False}), flush=True)


if __name__ == "__main__":
    main()
