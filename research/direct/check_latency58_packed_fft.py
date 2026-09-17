"""Independently check and time two non-learned FFT packing rewrites."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, write, sha, require


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_residual_model import load_checkpoint, BASE_STATE
    from research.direct.latency58_asymmetric_onnx import TOLERANCES, verification_cases
    from research.direct.latency58_packed_fft import rewrite
    from research.direct.latency58_int8_precise_core import make_reference
    from research.direct.check_latency58_rank_int8 import check_case
    from research.direct.check_latency58_fused_gru import session_for, benchmark
    from research.direct.train_latency58 import state_sha256, verify_inputs
    from research.direct.latency58_sdr_checkpoint import require_space

    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
            and ort.__version__ == "1.26.0", "Require CPU1 and the shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "remix-magnitude-001/plan.json"
    source = read(source_path)
    counted = require_space(source, 410_000_000)
    baseline = PHASE / "m4-int8-precise-core-full14-001/model.onnx"
    require(sha(baseline) == "a550c904ef501fe98f3afa010eca5a53abd5bf66d63906e018511a01c5076d61",
            "Precise C204 graph changed")
    integer_binding = read(PHASE / "m4-int8-screen-002/result.json")["quantized"]
    require(sha(integer_binding["path"]) == integer_binding["sha256"], "Original quantized graph changed")
    parent_binding = read(PHASE / "full-magnitude-001/plan.json")["parent_checkpoint"]
    parent, _ = load_checkpoint(parent_binding["path"], parent_binding["sha256"])
    native = Latency58AsymmetricModel()
    native.load_state_dict({k: v for k, v in parent.state_dict().items() if k != "fixed_residual_share"}, strict=True)
    native.eval().requires_grad_(False)
    require(state_sha256(native.state_dict()) == BASE_STATE, "Wrong C204 source")
    del parent
    reference, proof = make_reference(native, onnx.load(integer_binding["path"], load_external_data=False))
    music = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile/mixture.wav")
    out = PHASE / "m4-packed-fft-001"
    require(not out.exists(), "Preserve prior FFT experiments")
    module = Path(ort.__file__).resolve()
    paths = [source_path, baseline, Path(integer_binding["path"]), Path(parent_binding["path"]), music,
             Path(__file__).resolve(), module, *sorted((module.parent / "capi").glob("*.so*"))]
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_packed_fft.py", "check_latency58_rank_int8.py", "latency58_int8_precise_core.py",
        "latency58_int8_precise_float.py", "latency58_int8_reference.py", "check_latency58_fused_gru.py",
        "latency58_asymmetric_onnx.py"))
    bindings = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    orders = list(itertools.permutations(("baseline", "synthesis", "both")))
    plan = {"source_bindings": bindings, "source_checkpoint": parent_binding, "tolerances": TOLERANCES,
            "parent_graph": {"path": str(baseline), "sha256": sha(baseline)},
            "counted_bytes_before": counted, "pending_training_save_reserved_bytes": 370_000_000,
            "candidate_reservation_bytes": 40_000_000, "runtime": {"version": ort.__version__, "module": str(module)},
            "hypotheses": ["Pack the eight real inverse transforms into four complex transforms",
                           "Also pack the stereo analysis into one complex transform"],
            "oracle": "Unchanged independent PyTorch precise integer reference; no FFT-packing code in reference",
            "independent_weight_proof": proof, "screen_hops": 1024, "reset_repetitions": 2,
            "timing_orders": orders, "warmup_hops": 64, "timed_hops_per_cycle": 512,
            "advance_rule": "All original parity tolerances; pooled median <=0.97 baseline; faster cycle median in at least 4 of 6 cycles. Keep only the faster eligible graph.",
            "validation_used": False, "new_training_updates": 0, "graph_delay_samples": 128,
            "host_queue_samples": 128, "native_host_qualified": False,
            "operator_reference": "https://onnx.ai/onnx/operators/onnx__DFT.html",
            "runtime_reference": "https://github.com/microsoft/onnxruntime/blob/v1.26.0/onnxruntime/core/providers/cpu/signal/dft.cc",
            "compatibility_note": "Use full complex DFTs; the 1.26.0 inverse+onesided probe produced [8,513,2], not the requested [8,1024,1]."}
    verify_inputs({"source_bindings": bindings})
    out.mkdir()
    write(out / "plan.json", plan)
    started = time.monotonic()
    graph = onnx.load(baseline, load_external_data=False)
    data, constructions, results = {"baseline": baseline.read_bytes()}, {}, {}
    for label, analysis in (("synthesis", False), ("both", True)):
        candidate, construction = rewrite(graph, analysis=analysis)
        encoded = candidate.SerializeToString()
        require(len(encoded) < 34_000_000, "Candidate exceeded the graph reservation")
        data[label] = encoded
        construction.update(graph_sha256=hashlib.sha256(encoded).hexdigest(), graph_bytes=len(encoded))
        constructions[label] = construction
        write(out / (label + "-construction.json"), construction)
        runtime = session_for(encoded)
        cases = []
        for case in verification_cases(1024, [music]):
            row = check_case(reference, runtime, case)
            cases.append(row)
            write(out / (label + f"-parity-{len(cases)}.json"), row)
            print(json.dumps({"event": "parity", "variant": label, "case": row["case"],
                              "passed": row["passed"], **row["repetitions"][0]}), flush=True)
        results[label] = {"cases": cases, "strict_parity_passed": all(row["passed"] for row in cases)}
        del runtime, candidate
    measurements, summaries = [], {}
    eligible, selected = [], None
    passed = [label for label in results if results[label]["strict_parity_passed"]]
    if passed:
        audio = next(verification_cases(600, []))["audio"]
        for cycle, order in enumerate(orders):
            for label in order:
                if label != "baseline" and label not in passed:
                    continue
                row = {"cycle": cycle, "variant": label, **benchmark(session_for(data[label]), audio, 64, 512)}
                measurements.append(row)
                print(json.dumps({"event": "timing", **{k: row[k] for k in ("cycle", "variant", "p50_ms", "p95_ms")}}), flush=True)
        for label in ("baseline", *passed):
            values = [v for row in measurements if row["variant"] == label for v in row["times_ms"]]
            summaries[label] = {f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 95, 99)}
        medians = {(row["cycle"], row["variant"]): row["p50_ms"] for row in measurements}
        for label in passed:
            ratio = summaries[label]["p50_ms"] / summaries["baseline"]["p50_ms"]
            cycles = [medians[(i, label)] / medians[(i, "baseline")] for i in range(6)]
            accepted = ratio <= .97 and sum(r < 1 for r in cycles) >= 4
            results[label].update(pooled_p50_ratio=ratio, cycle_p50_ratios=cycles, passes_speed_screen=accepted)
            if accepted:
                eligible.append(label)
        if eligible:
            selected = min(eligible, key=lambda label: summaries[label]["p50_ms"])
    verify_inputs({"source_bindings": bindings})
    require(not torch.cuda.is_initialized() and state_sha256(native.state_dict()) == BASE_STATE,
            "Source or CPU-only scope changed")
    if selected:
        with (out / "model.onnx").open("xb") as stream:
            stream.write(data[selected])
    result = {"status": "screen_complete", "plan_sha256": sha(out / "plan.json"), "variants": results,
              "timings": measurements, "timing_summary": summaries, "selected_variant": selected,
              "graph_saved": selected is not None,
              "selected_graph_sha256": None if selected is None else constructions[selected]["graph_sha256"],
              "advance_to_native_timing_long_parity_and_quality": selected is not None,
              "source_bindings_unchanged": True, "native_host_qualified": False, "plugin_modified": False,
              "validation_used": False, "quality_measured": False, "elapsed_seconds": time.monotonic() - started,
              "counted_bytes_after": require_space(source, 370_000_000),
              "limitations": "Linux Python timing under concurrent GPU training; native timing, long parity and saved full14 quality are required before packaging."}
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "timing_summary", "selected_variant")}), flush=True)


if __name__ == "__main__":
    main()
