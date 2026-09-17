"""Literal-hop source views for a saved deployment graph, with independent states.

This is supplementary development scoring. It neither implements the host queue
nor changes the full-mixture validation protocol.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from research.direct.run_latency58_quality import require
from research.direct.latency58_vocal_views import VIEWS, combine_sources


def stream_onnx_views(session, contract, source_paths, plan, progress=None):
    import numpy as np
    from research import evaluate as legacy
    from research.direct.evaluate import shipping_residual
    from research.direct.latency58_evaluate import latency58_stream_metadata

    require(len(source_paths) == 4 and plan.unroll_hops == 1
            and len(contract["state_shapes"]) == 8, "Require four sources and the eight-state hop128 graph")
    readers = legacy._open_blocked_readers([Path(p) for p in source_paths],
        [plan.expected_frames] * 4, hop=128, block_hops=plan.io_block_hops)
    reference_capture = legacy._Capture(plan.reference_intervals, (4, 2))
    captures = {v: legacy._Capture(plan.capture_intervals, (4, 2)) for v in VIEWS}
    delayed_captures = {v: legacy._Capture(plan.capture_intervals, (2,)) for v in VIEWS}
    states = {v: [np.zeros(s, np.float32) for s in contract["state_shapes"]] for v in VIEWS}
    previous = {v: np.zeros((2, 128), np.float32) for v in VIEWS}
    digests = {v: hashlib.sha256() for v in VIEWS}
    raw_closure = dict.fromkeys(VIEWS, 0.)
    cursor = calls = reads = 0
    try:
        for start, stop in plan.call_slices:
            require(start == cursor and stop - start == 128, "Skipped or nonliteral source input")
            sources = np.empty((4, 2, 128), np.float32)
            for index, reader in enumerate(readers):
                block = reader.read_hop()
                require(block.shape == (128, 2) and block.dtype == np.float32
                        and np.isfinite(block).all(), "Invalid source audio or early EOF")
                sources[index] = block.T
                reads += 1
            reference_capture.add(start, sources)
            for view in VIEWS:
                mixture = combine_sources(sources, view)
                digests[view].update(np.ascontiguousarray(mixture.T).tobytes())
                audio = np.ascontiguousarray(mixture[None])
                expected_history = np.concatenate((states[view][0][..., 128:], audio), axis=-1)
                values = session.run(contract["output_names"], dict(zip(
                    contract["input_names"], [audio, *states[view]], strict=True)))
                require(len(values) == len(contract["output_shapes"])
                        and all(v.shape == tuple(s) and v.dtype == np.float32 and np.isfinite(v).all()
                                for v, s in zip(values, contract["output_shapes"], strict=True)),
                        "Invalid graph output or state")
                require(np.array_equal(values[1], expected_history), "Graph audio history changed")
                error = float(np.abs(values[0][0].sum(axis=0, dtype=np.float32) - previous[view]).max())
                raw_closure[view] = max(raw_closure[view], error)
                require(error <= 2e-6, "Graph reconstruction or physical delay changed")
                # Same post-ORT Other reconstruction as the shipping writer.
                estimates = shipping_residual(values[0][0], previous[view])
                captures[view].add(start, estimates)
                delayed_captures[view].add(start, previous[view])
                states[view], previous[view] = values[1:], mixture
            cursor, calls = stop, calls + 1
            if progress is not None and (calls % 4096 == 0 or calls == plan.literal_hop_count):
                progress(calls, plan.literal_hop_count)
    finally:
        for reader in readers:
            reader.close()
    require(cursor == plan.receive_end and calls == plan.literal_hop_count and reads == 4 * calls,
            "Incomplete physical source coverage")
    references = reference_capture.finish()
    outputs, mixtures, reconstruction = {}, {}, {}
    for view in VIEWS:
        outputs[view], mixtures[view] = captures[view].finish(), delayed_captures[view].finish()
        reconstruction[view] = 0.
        for refs, estimates, delayed in zip(references, outputs[view], mixtures[view], strict=True):
            expected = combine_sources(refs, view)
            require(np.array_equal(delayed, expected), "Physical remix alignment differs")
            error = float(np.abs(estimates.sum(axis=0, dtype=np.float32) - expected).max())
            reconstruction[view] = max(reconstruction[view], error)
        require(reconstruction[view] <= 1e-6, "Shipping output reconstruction failed")
    metadata = {**latency58_stream_metadata(plan), "views": list(VIEWS),
        "runtime": "ONNX Runtime CPU1 literal hop128", "public_float32_states": 8,
        "actual_forward_call_count_per_view": calls, "actual_source_file_hop_reads": reads,
        "input_stream_sha256": {v: h.hexdigest() for v, h in digests.items()},
        "source_subset_fixed_from_track_origin": True, "separate_state_per_view": True,
        "physical_alignment_verified_by_delayed_mixture": True,
        "all_public_output_shapes_dtypes_finiteness_and_history_checked": True,
        "maximum_raw_graph_closure": raw_closure, "reconstruction_max_abs": reconstruction,
        "shipping_other_residual_applied": True}
    return references, outputs, mixtures, metadata
