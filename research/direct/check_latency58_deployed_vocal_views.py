"""Qualify source-view transport against independent physical arrays and calls."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from research.direct.run_latency58_quality import read, require, sha, write
from research.direct.train_latency58 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Qualification plan changed")
    plan = read(args.plan)
    verify_inputs(plan)
    from research.direct.run_latency58_deployed_vocal_views import budget_snapshot, require_cpu
    require_cpu(plan)
    before = budget_snapshot(plan)
    import numpy as np
    import soundfile as sf
    from research.direct.latency58_attention_int8_verify import session_for
    from research.direct.latency58_evaluate import plan_latency58_stream
    from research.direct.latency58_onnx_vocal_views import stream_onnx_views
    from research.direct.latency58_vocal_views import VIEWS, combine_sources
    from research.direct.evaluate import shipping_residual
    from research.metrics import SOURCE_ORDER

    out = Path(plan["output_directory"]) / "qualification"
    contract = plan["interface"]
    data = Path(plan["checkpoint"]["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == plan["checkpoint"]["sha256"], "Wrong graph bytes")
    session = session_for(data, contract)
    began = time.monotonic()
    sources = np.random.default_rng(20260913).normal(0, .025, (4, 2, 16384)).astype(np.float32)
    sources[2, :, 700:850] = 0
    sources[3, :, 10000:10200] *= 3
    paths = []
    for stem, value in zip(SOURCE_ORDER, sources, strict=True):
        path = out / ("synthetic-" + stem + ".wav")
        require(not path.exists(), "Preserve fixture files")
        sf.write(path, value.T, 44100, subtype="FLOAT")
        decoded, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 44100 and np.array_equal(decoded.T, value), "Fixture samples changed")
        paths.append(path)
    intervals = [{"reference_start": a, "reference_end": b,
                  "estimate_start": a + 128, "estimate_end": b + 128}
                 for a, b in ((33, 321), (9901, 14497))]
    stream_plan = plan_latency58_stream(intervals, sources.shape[-1], unroll_hops=1, io_block_hops=1)
    independent = {}
    # Separate full-prefix arrays and one complete view at a time. This checks
    # transport/state isolation, not graph numerics against a native oracle;
    # the saved graph's existing independent numerical qualifications are bound.
    for view, included in VIEWS.items():
        physical = sources[list(included), :, :stream_plan.receive_end].sum(axis=0, dtype=np.float32)
        states = [np.zeros(s, np.float32) for s in contract["state_shapes"]]
        raw_outputs = []
        for start in range(0, stream_plan.receive_end, 128):
            chunk = np.ascontiguousarray(physical[None, :, start:start + 128])
            values = session.run(contract["output_names"], dict(zip(
                contract["input_names"], [chunk, *states], strict=True)))
            raw_outputs.append(values[0][0].copy())
            states = values[1:]
        full = np.concatenate(raw_outputs, axis=-1)
        independent[view] = [shipping_residual(full[..., row["estimate_start"]:row["estimate_end"]],
            physical[:, row["reference_start"]:row["reference_end"]]) for row in intervals]
    checks = []
    for block_hops in (1, 3, 64):
        stream_plan = plan_latency58_stream(intervals, sources.shape[-1], unroll_hops=1, io_block_hops=block_hops)
        refs, outputs, mixtures, metadata = stream_onnx_views(session, contract, paths, stream_plan)
        require(all(np.array_equal(ref, sources[..., row["reference_start"]:row["reference_end"]])
                    for ref, row in zip(refs, intervals, strict=True)), "Source capture or gap coordinates differ")
        for view, included in VIEWS.items():
            physical = sources[list(included), :, :stream_plan.receive_end].sum(axis=0, dtype=np.float32)
            digest = hashlib.sha256(np.ascontiguousarray(physical.T).tobytes()).hexdigest()
            require(metadata["input_stream_sha256"][view] == digest, "Remix or prefix input differs")
            require(all(np.array_equal(a, b) for a, b in zip(outputs[view], independent[view], strict=True)),
                    "Interleaved views, independent states, reset replay or output capture differs")
            require(all(np.array_equal(m, physical[:, row["reference_start"]:row["reference_end"]])
                        for m, row in zip(mixtures[view], intervals, strict=True)), "Independent alignment differs")
        checks.append({"io_block_hops": block_hops, "bit_exact_to_independent_view_execution": True,
                       "metadata": metadata})
    # Music captures must use real future input through the full final callback.
    try:
        plan_latency58_stream([{"reference_start": 16000, "reference_end": 16384,
            "estimate_start": 16128, "estimate_end": 16512}], 16384, unroll_hops=1)
    except ValueError:
        eof_rejected = True
    else:
        eof_rejected = False
    require(eof_rejected, "Music transport silently fabricated EOF input")
    # Reuse already qualified metric fixtures only while the scoring code and
    # fixture evidence retain their original hashes. This code adds no metric.
    template = read(plan["protocol_template"]["path"])
    metric_qualification = read(template["qualification"]["path"])
    metrics = metric_qualification["metric_fixtures"]
    require(metric_qualification["status"] == "pass" and metric_qualification["source_bindings_unchanged"],
            "Original metric qualification is incomplete")
    for view, included in VIEWS.items():
        for key in ("ideal", "known_leakage", "muted"):
            require(metrics[key][view]["input_active_windows"] == 1, "Metric support fixture changed")
        for index in included:
            stem = SOURCE_ORDER[index]
            wanted = metrics["ideal"][view]["native_output_levels"][stem]
            muted = metrics["muted"][view]["native_output_levels"][stem]
            require(abs(wanted["signed_desired_projection_gain"] - 1) < 1e-10
                    and muted["signed_desired_projection_gain"] == 0,
                    "Metric evidence fails to expose muting")
    verify_inputs(plan)
    write(out / "result.json", {"status": "pass", "plan_sha256": args.plan_sha256,
        "source_bindings_unchanged": True, "graph_sha256": plan["checkpoint"]["sha256"],
        "fixture_bindings": {str(p): sha(p) for p in paths}, "checks": checks,
        "independent_source_arrays_and_physical_coordinates_verified": True,
        "reset_replay_and_view_state_isolation_bit_exact": True, "near_eof_capture_rejected": eof_rejected,
        "metric_fixtures_reused_from_unchanged_qualified_code": template["qualification"],
        "graph_numerical_parity_evidence": plan["graph_numerical_parity_evidence"],
        "new_native_graph_parity_claimed": False, "host_qualified": False,
        "budget_before": before, "budget_after": budget_snapshot(plan),
        "elapsed_seconds": time.monotonic() - began})
    print({"status": "pass", "transport": "bit exact", "metric_muting_guard": "preserved"}, flush=True)


if __name__ == "__main__":
    main()
