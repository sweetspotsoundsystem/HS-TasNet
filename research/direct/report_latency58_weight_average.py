"""Compare the one fixed midpoint with both components and the working model."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write
from research.direct.report_latency58_sdr import load_completed, probe_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.prefix == "sdr-weight-average-500-1000" and not args.output.exists()
            and args.output.parent.is_dir(), "Require the single predeclared midpoint and fresh output")
    from research.direct.compare import compare
    bindings = {str(p): sha(p) for p in (Path(__file__).resolve(), ROOT / "research/direct/report_latency58_sdr.py",
                                        ROOT / "research/direct/compare.py")}
    candidates, plans, states = {}, {}, {}
    for mode in ("full14", "actions60", "probes"):
        plans[mode], candidates[mode] = load_completed(PHASE / (args.prefix + "-" + mode + "-001"), bindings)
        candidate = candidates[mode]
        if mode == "probes":
            require(candidate["status"] == "pass" and candidate["source_bindings_unchanged"], "Probe run failed")
            states[mode] = candidate["model_state_sha256"]
        else:
            require(candidate["inputs_unchanged"], "Music inputs changed")
            states[mode] = candidate["results"][0]["model"]["model_state_sha256"]
    require(len(set(states.values())) == 1 and all(plans[m]["construction_plan"] == plans["full14"]["construction_plan"]
            and plans[m]["checkpoint"] == plans["full14"]["checkpoint"] for m in plans), "Mixed derived endpoints")
    construction_binding = plans["full14"]["construction_plan"]
    require(sha(construction_binding["path"]) == construction_binding["sha256"], "Construction plan changed")
    construction = read(construction_binding["path"])
    require(construction["schema"] == "latency58-weight-average-plan-v1" and not construction["coefficient_search"]
            and [c["step"] for c in construction["components"]] == [500, 1000]
            and [c["weight"] for c in construction["components"]] == [.5, .5], "Different coefficient rule")
    bindings[construction_binding["path"]] = construction_binding["sha256"]
    comparisons = {}
    for key, prefix, component_index in (("versus_working_baseline", "teacher-half250", None),
                                        ("versus_drum500", "sdr-drum-accum-500", 0),
                                        ("versus_drum1000", "sdr-drum-accum-1000", 1)):
        comparisons[key] = {}
        for mode in ("full14", "actions60", "probes"):
            _, reference = load_completed(PHASE / (prefix + "-" + mode + "-001"), bindings,
                                          canonical_baseline=component_index is None)
            state = reference["model_state_sha256"] if mode == "probes" else reference["results"][0]["model"]["model_state_sha256"]
            if component_index is not None:
                require(state == construction["components"][component_index]["model_state_sha256"], "Different scored component")
            if mode == "probes":
                comparisons[key][mode] = probe_comparison(reference, candidates[mode])
            else:
                require(all(reference[k] == candidates[mode][k] for k in
                            ("manifest_sha256", "output_policy", "precision", "metrics", "metric_source_sha256")),
                        "Music scoring protocols differ")
                comparisons[key][mode] = compare(reference["results"][0], candidates[mode]["results"][0])
    require(all(sha(p) == s for p, s in bindings.items()), "A completed input changed")
    write(args.output, {"schema": "latency58-weight-average-quality-summary-v1", "prefix": args.prefix,
          "model_state_sha256": states["full14"], "source_bindings": bindings,
          "source_bindings_unchanged": True, "construction_plan": construction_binding,
          "component_steps": [500, 1000], "component_weights": [.5, .5], "coefficient_search": False,
          "training_updates_executed": 0, **comparisons, "quality_selected": False, "listening_completed": False,
          "limitations": ["Primary development panel; reserved confirmation excerpts remain unused.",
                          "One predeclared arithmetic midpoint of two checkpoints from one training seed.",
                          "Track intervals omit training-seed uncertainty and model-selection correction.",
                          "The one-track Actions interval is degenerate, not sampling uncertainty.",
                          "Native levels and unexplained probe energy do not establish audibility."]})
    print({"result": str(args.output), "versus_drum500": comparisons["versus_drum500"]["full14"]["metrics"]}, flush=True)


if __name__ == "__main__":
    main()
