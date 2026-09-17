"""Save the improved residual-policy checkpoint and check physical streaming."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write


def qualify(parent, model, reloaded):
    import numpy as np
    import soundfile as sf
    import torch
    from research.direct.latency58_asymmetric import AsymmetricState
    from research.direct.latency58_residual_share import residual_share
    from research.direct.latency58_evaluate import model_state_sha256
    rng = np.random.default_rng(202609241)
    signal = rng.normal(0, .03, (2, 24 * 128 + 37)).astype(np.float32)
    impulses = np.zeros_like(signal)
    impulses[:, [0, 127, 128, 255, 256, signal.shape[-1] - 1]] = .2
    fixtures = [("noise_partial", signal), ("boundary_impulses", impulses),
                ("silence", np.zeros_like(signal))]
    manifest = read(ROOT / "research/manifests/valid.json")
    # Real physical mixtures supplement synthetic boundaries and nonzero states.
    for index in (1, 10):
        track = manifest["tracks"][index]
        audio, rate = sf.read(Path(manifest["root"]) / track["mixture"], start=30 * 44100,
                              frames=signal.shape[-1], dtype="float32", always_2d=True)
        require(rate == 44100 and audio.shape == signal.T.shape, "Invalid qualification music")
        fixtures.append((track["name"], np.ascontiguousarray(audio.T)))
    fingerprints = [model_state_sha256(m) for m in (parent, model, reloaded)]
    cpu_rng = torch.get_rng_state().clone()
    rows = []
    with torch.inference_mode():
        for name, physical in fixtures:
            physical = np.pad(physical, ((0, 0), (0, (-physical.shape[-1]) % 128)))
            # Two zeros flush the graph and the separate one-hop host queue.
            received = torch.from_numpy(np.pad(physical, ((0, 0), (0, 256)))[None])
            traces = []
            arithmetic_error = closure = roundtrip_error = state_error = 0.0
            for partition in (1, 3, 7):
                states = [m.initial_state(1) for m in (parent, model, reloaded)]
                outputs = []
                for start in range(0, received.shape[-1], partition * 128):
                    chunk = received[..., start:start + partition * 128].clone()
                    unchanged = chunk.clone()
                    values = [m.render(chunk, s) for m, s in zip((parent, model, reloaded), states, strict=True)]
                    states = [v.state for v in values]
                    native, corrected, saved = values
                    oracle = residual_share(native.raw[0].numpy(), native.delayed_mixture[0].numpy())
                    arithmetic_error = max(arithmetic_error, float(np.abs(corrected.deployed[0].numpy() - oracle).max()))
                    require(torch.equal(corrected.native_raw, native.raw) and torch.equal(chunk, unchanged),
                            "Correction changed native estimates or physical input")
                    roundtrip_error = max(roundtrip_error, float((saved.deployed - corrected.deployed).abs().max()))
                    for reference_state, corrected_state, saved_state in zip(native.state, corrected.state, saved.state, strict=True):
                        state_error = max(state_error, float((reference_state - corrected_state).abs().max()),
                                          float((corrected_state - saved_state).abs().max()))
                    closure = max(closure, float((corrected.deployed.sum(dim=1) - native.delayed_mixture).abs().max()))
                    outputs.append(corrected.deployed)
                traces.append(torch.cat(outputs, dim=-1))
            partition_error = max(float((trace - traces[0]).abs().max()) for trace in traces[1:])
            # Explicit queue: callback n returns graph output from callback n-1.
            queued = torch.cat((torch.zeros_like(traces[0][..., :128]), traces[0][..., :-128]), dim=-1)
            recovered = queued.sum(dim=1)[0, :, 256:256 + physical.shape[-1]].numpy()
            host_error = float(np.abs(recovered - physical).max())
            require(arithmetic_error == roundtrip_error == state_error == 0.0
                    and partition_error < 1e-4 and closure < 1e-6 and host_error < 1e-6,
                    "Saved correction failed arithmetic, state, partition, or physical delay checks")
            rows.append({"name": name, "numpy_oracle_max_abs": arithmetic_error,
                         "saved_roundtrip_max_abs": roundtrip_error, "state_max_abs": state_error,
                         "partition_max_abs": partition_error, "closure_max_abs": closure,
                         "host_delay_256_reconstruction_max_abs": host_error})
        # Same received prefix and incoming state must ignore all later callbacks.
        state = model.initial_state(1)
        state = AsymmetricState(*(v + torch.from_numpy(rng.normal(0, scale, tuple(v.shape)).astype(np.float32))
                                  for v, scale in zip(state, (.02, 1e-7, .001, .001), strict=True)))
        state_copy = [v.clone() for v in state]
        prefix = torch.from_numpy(signal[:, :8 * 128].copy())[None]
        left = torch.cat((prefix, torch.zeros_like(prefix)), dim=-1)
        right = torch.cat((prefix, torch.ones_like(prefix) * .1), dim=-1)
        first = model.render(left, state).deployed[..., :prefix.shape[-1]]
        second = model.render(right, state).deployed[..., :prefix.shape[-1]]
        future_error = float((first - second).abs().max())
        n, c = parent.render(prefix, state), model.render(prefix, state)
        nonzero_oracle = residual_share(n.raw[0].numpy(), n.delayed_mixture[0].numpy())
        require(future_error == 0 and np.array_equal(c.deployed[0].numpy(), nonzero_oracle)
                and all(torch.equal(a, b) for a, b in zip(state, state_copy, strict=True)),
                "Future callback or incoming-state preservation failed")
    require([model_state_sha256(m) for m in (parent, model, reloaded)] == fingerprints
            and torch.equal(cpu_rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            "Qualification mutated checkpoint, RNG, or CPU scope")
    return {"status": "pass", "fixtures": rows, "future_callback_max_abs": future_error,
            "nonzero_incoming_state_numpy_oracle_exact": True, "incoming_state_unchanged": True,
            "graph_delay_samples": 128, "host_queue_samples": 128, "total_delay_samples": 256,
            "sample_rate": 44100, "total_delay_ms": 256 / 44100 * 1000,
            "extra_buffering_samples": 0, "host_wall_clock_deadlines_tested": False,
            "model_state_sha256": fingerprints[1], "parent_state_sha256": fingerprints[0]}


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Use CUDA-hidden repository CPU")
    import torch
    from research.direct.latency58_evaluate import model_state_sha256
    from research.direct.latency58_residual_model import VERSION, Latency58ResidualModel, load_checkpoint
    from research.direct.evaluate_latency58_leader_cleanup import load_evaluation_model
    from research.direct.latency58_sdr_checkpoint import require_space
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    diagnostic_dir = PHASE / "c204-residual-share-full14-001"
    diagnostic = read(diagnostic_dir / "result.json")
    execution = read(diagnostic_dir / "diagnostic-execution.json")
    require(diagnostic["status"] == "pass" and diagnostic["exact_stored_track_reports_and_aggregate"]
            and diagnostic["comparison"]["metrics"]["full_sdr_db"]["delta"] > 0
            and execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"]
            and not execution["timed_out"], "Require the completed improving full14 diagnostic")
    budget = read(PHASE / "latency58-reduced-teacher-001/training-plan.json")
    counted = require_space(budget, 120_000_000)
    paths = [Path(__file__).resolve(), ROOT / "research/direct/latency58_residual_model.py",
             ROOT / "research/direct/evaluate_latency58_residual_model.py",
             diagnostic_dir / "result.json", diagnostic_dir / "diagnostic-execution.json"]
    bindings = {**diagnostic["source_bindings"], **{str(p): sha(p) for p in paths}}
    require(all(sha(p) == digest for p, digest in bindings.items()), "Residual candidate inputs changed")
    out = PHASE / "c204-residual-model-001"
    require(not out.exists(), "Preserve saved candidates")
    parent, _ = load_evaluation_model(read(PHASE / "leader-cleanup-250-full14-001/plan.json"))
    model = Latency58ResidualModel.from_c204(parent)
    payload = {"schema": VERSION, "model": model.state_dict(), "architecture": model.architecture_metadata,
               "model_state_sha256": model_state_sha256(model), "provenance": model.provenance,
               "source_bindings": bindings}
    out.mkdir()
    path = out / "model.pt"
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checkpoint = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    reloaded, _ = load_checkpoint(path, checkpoint["sha256"])
    qualification = qualify(parent, model, reloaded)
    require(all(sha(p) == digest for p, digest in bindings.items()), "Candidate preparation inputs changed")
    qualification.update(checkpoint=checkpoint, source_bindings=bindings, source_bindings_unchanged=True,
                         counted_bytes_before=counted, counted_bytes_after=require_space(budget, 2_000_000),
                         elapsed_seconds=time.monotonic() - began)
    write(out / "qualification.json", qualification)
    write(out / "checkpoint.json", checkpoint)
    print(json.dumps({"status": "pass", "checkpoint": checkpoint,
                      "model_state_sha256": model_state_sha256(model),
                      "latency_samples": qualification["total_delay_samples"]}), flush=True)


if __name__ == "__main__":
    main()
