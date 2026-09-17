"""Continuous vocals-only and instrumental remixes for vocal spill diagnostics.

The original validation protocol is unchanged. These supplementary views use
its physical excerpts, with the same source subset throughout the prefix.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from research.direct.run_latency58_quality import require

VIEWS = {"vocals_only": (2,), "instrumental": (0, 1, 3)}
VERSION = "latency58-continuous-vocal-counterfactuals-v1"


def combine_sources(sources, view):
    """Sum stored FP32 stems in fixed source order without level normalization."""
    import numpy as np
    require(view in VIEWS and sources.shape[0:2] == (4, 2)
            and sources.dtype == np.float32, "Invalid source view")
    return sources[list(VIEWS[view])].sum(axis=0, dtype=np.float32)


def stream_views(model, source_paths, plan):
    import numpy as np
    import torch
    from research import evaluate as legacy
    from research.direct.latency58_evaluate import _require_cpu1_model, latency58_stream_metadata
    _require_cpu1_model(model)
    require(len(source_paths) == 4, "Require all four original source files")
    readers = legacy._open_blocked_readers([Path(p) for p in source_paths],
        [plan.expected_frames] * 4, hop=128, block_hops=plan.io_block_hops)
    source_capture = legacy._Capture(plan.reference_intervals, (4, 2))
    captures = {view: legacy._Capture(plan.capture_intervals, (4, 2)) for view in VIEWS}
    mixture_captures = {view: legacy._Capture(plan.capture_intervals, (2,)) for view in VIEWS}
    digests = {view: hashlib.sha256() for view in VIEWS}
    cursor = calls = literal_reads = 0
    try:
        with torch.inference_mode():
            states = {view: model.initial_state(1, device=torch.device("cpu")) for view in VIEWS}
            for start, stop in plan.call_slices:
                require(start == cursor, "Counterfactual stream skipped physical input")
                count = stop - start
                sources = np.empty((4, 2, count), dtype=np.float32)
                for offset in range(0, count, 128):
                    for index, reader in enumerate(readers):
                        block = reader.read_hop()
                        require(block.shape == (128, 2) and np.isfinite(block).all(),
                                "Original source ended early or contains nonfinite audio")
                        sources[index, :, offset:offset + 128] = block.T
                        literal_reads += 1
                source_capture.add(start, sources)
                for view in VIEWS:
                    mixture = combine_sources(sources, view)
                    # Stereo samples are interleaved before hashing, so the
                    # digest is independent of the render-group boundaries.
                    digests[view].update(np.ascontiguousarray(mixture.T).tobytes())
                    result = model.render(torch.from_numpy(mixture[None]), states[view])
                    states[view] = result.state
                    estimates, delayed = result.deployed[0].numpy(), result.delayed_mixture[0].numpy()
                    require(estimates.shape == (4, 2, count) and delayed.shape == (2, count)
                            and estimates.dtype == delayed.dtype == np.float32
                            and np.isfinite(estimates).all() and np.isfinite(delayed).all(),
                            "Counterfactual output shape, precision or finiteness differs")
                    captures[view].add(start, estimates)
                    mixture_captures[view].add(start, delayed)
                cursor, calls = stop, calls + 1
    finally:
        for reader in readers:
            reader.close()
    require(cursor == plan.receive_end and calls == len(plan.call_slices)
            and literal_reads == 4 * plan.literal_hop_count, "Counterfactual input coverage differs")
    references = source_capture.finish()
    outputs, mixtures = {}, {}
    reconstruction = {}
    for view in VIEWS:
        outputs[view], mixtures[view] = captures[view].finish(), mixture_captures[view].finish()
        reconstruction[view] = 0.0
        for refs, estimates, delayed in zip(references, outputs[view], mixtures[view], strict=True):
            expected = combine_sources(refs, view)
            require(np.array_equal(delayed, expected), "Counterfactual physical alignment differs")
            error = float(np.max(np.abs(estimates.sum(axis=0, dtype=np.float32) - expected)))
            reconstruction[view] = max(reconstruction[view], error)
        require(reconstruction[view] <= 1e-6, "Counterfactual mixture closure failed")
    metadata = latency58_stream_metadata(plan)
    metadata.update(actual_forward_call_count_per_view=calls, views=list(VIEWS),
                    actual_source_file_hop_reads=literal_reads,
                    input_stream_sha256={view: value.hexdigest() for view, value in digests.items()},
                    physical_alignment_verified_by_delayed_mixture=True,
                    reconstruction_max_abs=reconstruction, source_subset_fixed_from_track_origin=True)
    return references, outputs, mixtures, metadata


def score_views(name, intervals, references, outputs, mixtures, config):
    """Report native leakage alongside desired-source SDR and signed gain.

    Every input-active window contributes to leakage, including silent model
    output. Quiet inputs are excluded using the unchanged -50 dBFS criterion.
    Absolute levels accompany ratios to avoid mistaking attenuation for purity.
    """
    import numpy as np
    from research import evaluate as legacy
    from research.metrics import SOURCE_ORDER, db_ratio, frame_ranges, mean_or_none, rms_dbfs
    result = {}
    for view, included in VIEWS.items():
        desired, windows = [], []
        for excerpt_index, (refs, estimates, mixture) in enumerate(zip(
                references, outputs[view], mixtures[view], strict=True)):
            target = np.zeros_like(refs)
            target[list(included)] = refs[list(included)]
            desired.append(target)
            for start, stop in frame_ranges(mixture.shape[-1], config.window_samples, config.hop_samples):
                segment = mixture[:, start:stop].astype(np.float64)
                input_dbfs = rms_dbfs(segment, config.epsilon)
                active = input_dbfs > config.activity_dbfs
                input_energy = float(np.square(segment).sum())
                cells = {}
                for index, stem in enumerate(SOURCE_ORDER):
                    value = estimates[index, :, start:stop].astype(np.float64)
                    truth = target[index, :, start:stop].astype(np.float64)
                    truth_energy = float(np.square(truth).sum())
                    cells[stem] = {
                        "output_rms_dbfs": rms_dbfs(value, config.epsilon),
                        "output_to_input_db": db_ratio(float(np.square(value).sum()), input_energy,
                            epsilon=config.epsilon, floor=config.db_floor, ceiling=config.db_ceiling),
                        "desired_active": rms_dbfs(truth, config.epsilon) > config.activity_dbfs,
                        "signed_desired_projection_gain": float((value * truth).sum() / (truth_energy + config.epsilon))
                            if rms_dbfs(truth, config.epsilon) > config.activity_dbfs else None,
                        "off_target": index not in included,
                    }
                windows.append({"excerpt_index": excerpt_index,
                    "physical_start": int(intervals[excerpt_index]["reference_start"]) + start,
                    "physical_end": int(intervals[excerpt_index]["reference_start"]) + stop,
                    "input_rms_dbfs": input_dbfs, "input_active": active, "per_stem": cells})
        score = legacy._score_track(name, intervals, mixtures[view], desired, outputs[view], config)
        levels = {}
        for stem_index, stem in enumerate(SOURCE_ORDER):
            active_cells = [row["per_stem"][stem] for row in windows if row["input_active"]]
            levels[stem] = {"off_target": stem_index not in included,
                           "input_active_windows": len(active_cells),
                           "output_rms_dbfs": mean_or_none(c["output_rms_dbfs"] for c in active_cells),
                           "output_to_input_db": mean_or_none(c["output_to_input_db"] for c in active_cells),
                           "signed_desired_projection_gain": mean_or_none(
                               c["signed_desired_projection_gain"] for c in active_cells)}
        result[view] = {"desired_stems": [SOURCE_ORDER[i] for i in included],
                        "standard_scores_on_remixed_references": score,
                        "input_active_windows": sum(r["input_active"] for r in windows),
                        "total_windows": len(windows), "native_output_levels": levels, "windows": windows}
    return result
