"""Screen a fixed rank-128 mask head using independent parity and training audio."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, state_sha256, verify_inputs


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(k) == "1" for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Use CPU1")
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from research.direct.latency58_asymmetric import Latency58AsymmetricModel
    from research.direct.latency58_residual_model import load_checkpoint, BASE_STATE
    from research.direct.latency58_asymmetric_onnx import (
        INPUT_NAMES, OUTPUT_NAMES, TOLERANCES, _initial_states, verification_cases, _run_case, make_export_copy)
    from research.direct.latency58_low_rank_mask import convert
    from research.direct.latency58_direct_sdr import objective
    from research.direct.latency58_recorded301_data import select_tracks
    from research.direct.check_latency58_fused_gru import session_for, benchmark
    from research.direct.latency58_sdr_checkpoint import require_space

    require(ort.__version__ == "1.26.0", "Use shipping runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source_path = PHASE / "full-magnitude-sdr-001/plan.json"
    source = read(source_path)
    verify_inputs(source)
    counted_before = require_space(source, 372_000_000)
    graph_path = Path("/home/axel/autoresearch/codex/stemgen-rt-hop128-5ms/model/model.onnx")
    require(sha(graph_path) == "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3",
            "C204 graph changed")
    parent_plan_path = PHASE / "full-magnitude-001/plan.json"
    parent_binding = read(parent_plan_path)["parent_checkpoint"]
    parent, _ = load_checkpoint(parent_binding["path"], parent_binding["sha256"])
    native = Latency58AsymmetricModel()
    native.load_state_dict({k: v for k, v in parent.state_dict().items() if k != "fixed_residual_share"}, strict=True)
    native.eval().requires_grad_(False)
    require(state_sha256(native.state_dict()) == BASE_STATE, "Wrong C204 source")
    del parent
    out = PHASE / "m4-rank128-mask-001"
    require(not out.exists(), "Preserve previous screen")
    paths = [source_path, parent_plan_path, graph_path, Path(parent_binding["path"]), Path(__file__).resolve(),
             ROOT / "research/direct/latency58_low_rank_mask.py",
             ROOT / "research/direct/check_latency58_fused_gru.py",
             ROOT / "research/direct/latency58_asymmetric_onnx.py"]
    bindings = {**source["source_bindings"], **{str(p): sha(p) for p in paths}}
    indices = list(range(2_400_000, 2_400_008))
    plan = {"source_bindings": bindings, "rank": 128, "precision": "FP32 inference; FP64 SVD",
            "rank_selected_using": "C204 weight spectrum only", "training_crop_indices": indices,
            "validation_used": False, "graph_saved": False, "tolerances": TOLERANCES,
            "pending_training_save_reserved_bytes": 370_000_000, "counted_bytes_before": counted_before,
            "quality_screen_rule": "Advance only if eight-crop mean SDR falls by no more than 0.02 dB and no stem falls by more than 0.05 dB; this is a training screen, not a qualification.",
            "timing_screen_rule": "Require lower pooled median over four alternating-order cycles; target Mac remains unmeasured."}
    out.mkdir()
    write(out / "plan.json", plan)
    began = time.monotonic()
    candidate = copy.deepcopy(native)
    graph = onnx.load(graph_path, load_external_data=False)
    proof = convert(candidate, graph)
    onnx.checker.check_model(graph, full_check=True)
    candidate_bytes = graph.SerializeToString()
    candidate_state = state_sha256(candidate.state_dict())
    wrapper = make_export_copy(candidate)
    runtime = session_for(candidate_bytes)
    parity = []
    for case in verification_cases(256, []):
        first = _run_case(candidate, wrapper, runtime, case)
        second = _run_case(candidate, wrapper, runtime, case)
        exact = first["all_output_and_state_trajectory_sha256"] == second["all_output_and_state_trajectory_sha256"]
        row = {**first, "reset_replay_bit_exact": exact,
               "passed": first["passed"] and second["passed"] and exact}
        parity.append(row)
        write(out / f"parity-{len(parity)}.json", row)
        print(json.dumps({"event": "parity", "case": row["input"], "passed": row["passed"]}), flush=True)
    require(all(r["passed"] for r in parity), "Rank128 implementation fails unchanged parity thresholds")
    del wrapper, candidate, graph, runtime

    sys.path.insert(0, str(PRODUCTION))
    import train_production as production
    config = source["config"]
    _, tracks, manifest_sha, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
        expected_file_sha256=source["manifest_sha256"], config=read(PRODUCTION / "full_config.json"))
    require(manifest_sha == source["manifest_sha256"], "Corpus changed")
    tracks = select_tracks(tracks, source["training_selection"])
    dataset = production.CounterAddressedCropDataset(tracks, root_weights=config["root_weights"], seed=config["data_seed"],
        crop_samples=config["crop_samples"], vocal_active_probability=config["vocal_active_probability"],
        final_sample_index=indices[-1] + 1)
    examples = [dataset[index] for index in indices]
    input_sha = hashlib.sha256()
    for mixture, truth in examples:
        for value in (mixture, truth):
            input_sha.update(value.contiguous().numpy().tobytes())
    truth = torch.stack([value[..., source["warmup_samples"]:] for _, value in examples])
    mixture = torch.stack([value[..., source["warmup_samples"]:] for value, _ in examples])

    def render(session, audio):
        count = audio.shape[-1]
        padded = np.pad(audio, ((0, 0), (0, (-count) % 128 + 128)))
        states, outputs = _initial_states(), []
        for offset in range(0, padded.shape[-1], 128):
            chunk = np.ascontiguousarray(padded[None, :, offset:offset + 128])
            values = session.run(list(OUTPUT_NAMES), dict(zip(INPUT_NAMES, [chunk, *states], strict=True)))
            require(all(np.isfinite(v).all() for v in values), "Nonfinite training render")
            states = values[1:]
            outputs.append(values[0])
        return np.concatenate(outputs, axis=-1)[..., 128:128 + count]

    data = {"c204": graph_path.read_bytes(), "rank128": candidate_bytes}
    diagnostics, closures = {}, {}
    for name, graph_bytes in data.items():
        session = session_for(graph_bytes)
        outputs = []
        closure = 0.
        for index, (audio, _) in zip(indices, examples, strict=True):
            result = render(session, audio.numpy())
            closure = max(closure, float(np.abs(result.sum(axis=1) - audio.numpy()[None]).max()))
            outputs.append(result[..., source["warmup_samples"]:])
            print(json.dumps({"event": "training_crop", "variant": name, "index": index}), flush=True)
        estimate = torch.from_numpy(np.concatenate(outputs))
        terms = objective(estimate, estimate, truth, mixture)
        diagnostics[name] = {"sdr_db": -float(terms.negative_sdr_db),
                             "per_stem_sdr_db": (-terms.per_stem_negative_sdr_db).tolist()}
        closures[name] = closure
        require(closure <= TOLERANCES["reconstruction_max_abs"], "Training mixture closure failed")
        del session, outputs, estimate
        write(out / f"training-{name}.json", diagnostics[name])
        print(json.dumps({"event": "training_quality", "variant": name, **diagnostics[name]}), flush=True)
    delta = diagnostics["rank128"]["sdr_db"] - diagnostics["c204"]["sdr_db"]
    stem_deltas = [a - b for a, b in zip(diagnostics["rank128"]["per_stem_sdr_db"],
                                       diagnostics["c204"]["per_stem_sdr_db"], strict=True)]
    quality_pass = delta >= -.02 and min(stem_deltas) >= -.05
    timing_audio = examples[0][0][:, :128 * 576].numpy()
    timings = []
    for cycle, order in enumerate((("c204", "rank128"), ("rank128", "c204"),
                                   ("rank128", "c204"), ("c204", "rank128"))):
        for label in order:
            session = session_for(data[label])
            row = {"cycle": cycle, "variant": label, **benchmark(session, timing_audio, 64, 512)}
            timings.append(row)
            del session
            print(json.dumps({k: row[k] for k in ("cycle", "variant", "p50_ms", "p95_ms", "p99_ms")}), flush=True)
    pooled = {name: [t for row in timings if row["variant"] == name for t in row["times_ms"]] for name in data}
    summary = {name: {f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 95, 99)}
               for name, values in pooled.items()}
    ratio = summary["rank128"]["p50_ms"] / summary["c204"]["p50_ms"]
    verify_inputs({"source_bindings": bindings})
    require(state_sha256(native.state_dict()) == BASE_STATE and not torch.cuda.is_initialized(), "Source or CPU scope changed")
    result = {"status": "screen_complete", "factorization": proof, "parity_passed": True, "parity_cases": parity,
              "graph_sha256": hashlib.sha256(candidate_bytes).hexdigest(), "graph_bytes": len(candidate_bytes),
              "candidate_model_state_sha256": candidate_state, "graph_saved": False,
              "training_inputs_sha256": input_sha.hexdigest(), "training_diagnostics": diagnostics,
              "mean_sdr_delta_db": delta, "per_stem_sdr_delta_db": stem_deltas,
              "quality_screen_passed": quality_pass, "maximum_closure": closures,
              "timings": timings, "timing_summary": summary, "p50_ratio_to_c204": ratio,
              "advance_to_full14": quality_pass and ratio < 1., "source_bindings_unchanged": True,
              "target_m4_qualified": False, "plugin_modified": False, "gpu_used": False,
              "elapsed_seconds": time.monotonic() - began,
              "counted_bytes_after": require_space(source, 370_000_000),
              "limitation": "Eight training crops and local Python ORT CPU1 timing under concurrent GPU training cannot establish held-out quality or Mac callback deadlines."}
    write(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("parity_cases", "timings")}), flush=True)


if __name__ == "__main__":
    main()
