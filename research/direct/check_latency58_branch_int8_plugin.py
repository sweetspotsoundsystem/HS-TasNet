"""Export and screen the saved branch-memory EMA model for the existing M4 PR."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write


def cases(model, hops=64):
    import numpy as np
    from research.direct.latency58_asymmetric_onnx import verification_cases
    for case in verification_cases(hops, []):
        state = model.initial_state(1)
        nonzero = case["name"].startswith("nonzero_")
        rng = np.random.default_rng(202609150)
        extras = []
        for index, value in enumerate(state[4:], start=4):
            scale = .03 * (2**-18 if index in (6, 7) else 1)
            extras.append(rng.normal(0, scale, value.shape).astype(np.float32)
                          if nonzero else np.zeros(value.shape, np.float32))
        yield {**case, "initial_states": [*case["initial_states"], *extras]}


def check_float_wrapper(model, wrapper, case):
    import numpy as np
    import torch
    from research.direct.latency58_branch_onnx import TOLERANCES
    audio = np.pad(case["audio"], ((0, 0), (0, (-case["audio"].shape[-1]) % 128 + 128)))
    own = tuple(torch.from_numpy(v.copy()) for v in case["initial_states"])
    state = type(model.initial_state(1))(*(v.clone() for v in own))
    maxima = np.zeros(9)
    max_rms = 0.
    with torch.inference_mode():
        for start in range(0, audio.shape[-1], 128):
            chunk = torch.from_numpy(np.ascontiguousarray(audio[None, :, start:start+128]))
            expected, state = model.forward_chunk(chunk, state)
            values = wrapper(chunk, *own)
            own = values[1:]
            for index, (a, b) in enumerate(zip((expected, *state), values, strict=True)):
                require(torch.isfinite(a).all() and torch.isfinite(b).all(), "Nonfinite FP32 wrapper output")
                error = (a.double() - b.double()).abs()
                maxima[index] = max(maxima[index], error.max().item() * (2**18 if index in (2, 7, 8) else 1))
                if index == 0:
                    max_rms = max(max_rms, error.square().mean((2, 3)).sqrt().max().item())
    return {"case": case["name"], "calls": audio.shape[-1] // 128,
            "maximum_errors_in_physical_units": maxima.tolist(), "maximum_callback_rms": max_rms,
            "passed": bool(maxima[0] <= TOLERANCES["waveform_max_abs"]
                           and maxima[1:].max() <= TOLERANCES["state_max_abs_decoded_units"]
                           and max_rms <= TOLERANCES["stem_callback_rms"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Use CPU1 without CUDA")
    import onnxruntime as ort
    import torch
    from research.direct.latency58_branch_plugin_endpoint import selected_endpoint, export_bindings
    from research.direct.latency58_branch_onnx import interface, make_export_copy, TOLERANCES
    from research.direct.latency58_branch_int8 import build, make_reference
    from research.direct.latency58_branch_int8_verify import session_for, run_case
    from research.direct.latency58_sdr_checkpoint import require_space
    from research.direct.train_latency58 import state_sha256, verify_inputs
    require(ort.__version__ == "1.26.0", "Use shipping ONNX Runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    print(json.dumps({"stage": "authenticate_saved_endpoint"}), flush=True)
    model, payload, training, checkpoint, quality_path, review_path, bindings = selected_endpoint()
    # The 800 MB outside-root reservation is already excluded in this plan's
    # 89.2 GB stop threshold. Preserve another 600 MB for the live training save.
    counted = require_space(training, 600_000_000 + 300_000_000)
    out = args.output.resolve()
    require(out.parent == PHASE and not out.exists(), "Preserve existing export evidence")
    out.mkdir()
    bindings = {**export_bindings(bindings), str(Path(__file__).resolve()): sha(__file__)}
    contract = interface(model)
    write(out / "plan.json", {"source_bindings": bindings, "checkpoint": checkpoint,
          "quality_result": str(quality_path), "selection_review": str(review_path), "interface": contract,
          "tolerances": TOLERANCES, "budget_counted_before": counted,
          "live_training_save_reservation_bytes": 600_000_000,
          "purpose": "Update PR13 for user M4/M4 Pro testing; merge and release follow user testing"})
    wrapper = make_export_copy(model)
    float_rows = [check_float_wrapper(model, wrapper, case) for case in cases(model)]
    write(out / "fp32-wrapper-parity.json", {"status": "pass" if all(r["passed"] for r in float_rows) else "fail",
          "reports": float_rows, "tolerances": TOLERANCES})
    require(all(r["passed"] for r in float_rows), "Branch-memory FP32 wrapper differs from saved model")
    print(json.dumps({"stage": "build_integer_graph", "fp32_wrapper_passed": True}), flush=True)
    integer, graph, conversion = build(model, payload, training, checkpoint, quality_path)
    data = graph.SerializeToString()
    graph_path = out / "model.onnx"
    with graph_path.open("xb") as stream:
        stream.write(data)
    write(out / "graph.json", {"graph_sha256": hashlib.sha256(data).hexdigest(), "graph_bytes": len(data),
          "metadata": {v.key: v.value for v in graph.metadata_props}, "conversion": conversion})
    reference, independent = make_reference(model, integer, conversion)
    write(out / "independent-reference.json", independent)
    reports = []
    for optimization in ("disabled", "all"):
        session = session_for(data, contract, optimization)
        for case in cases(model):
            first = run_case(reference, session, case, contract)
            replay = run_case(reference, session, case, contract)
            exact = first["all_output_and_state_trajectory_sha256"] == replay["all_output_and_state_trajectory_sha256"]
            first.update({"optimization": optimization, "reset_replay_bit_exact": exact,
                          "passed": first["passed"] and replay["passed"] and exact})
            reports.append(first)
            write(out / (optimization + "-" + case["name"] + ".json"), first)
            print(json.dumps({"optimization": optimization, "case": case["name"], "passed": first["passed"],
                              "maximum_errors": first["maximum_errors_in_physical_units"]}), flush=True)
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(model.state_dict()) == payload["model_state_sha256"]
            and not torch.cuda.is_initialized() and sha(graph_path) == hashlib.sha256(data).hexdigest(),
            "Source model, saved graph or CPU scope changed")
    passed = all(r["passed"] for r in reports)
    write(out / "result.json", {"status": "pass" if passed else "fail", "strict_parity_passed": passed,
          "checkpoint": checkpoint, "model_state_sha256": payload["model_state_sha256"],
          "graph_sha256": sha(graph_path), "graph_bytes": len(data), "reports": reports,
          "source_bindings_unchanged": True, "tolerances_changed": False, "gpu_used": False,
          "quality_measured": False, "native_host_qualified": False,
          "counted_bytes_after": require_space(training, 600_000_000),
          "elapsed_seconds": time.monotonic() - began})
    require(passed, "Independent branch-memory integer parity failed")
    print(json.dumps({"status": "pass", "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
