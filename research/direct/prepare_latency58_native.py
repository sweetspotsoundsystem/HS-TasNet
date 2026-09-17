"""Adapt the preserved native queue diagnostic to a verified hop128 graph.

Only source and its provenance are written. Compilation and execution are
separate actions. This preserves the accepted hop256 diagnostic byte for byte.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.latency58_checkpoint import require, sha

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/direct/cropped1024_native_async_qualifier.cpp"
BASE_SHA = "f6dfc31de936bf6263f9eb329dc4342252cb315dd3bf98cdecec7a4d3f7100d1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--verification-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(sha(BASE) == BASE_SHA and sha(args.verification) == args.verification_sha256,
            "Preserved native source or ONNX verification changed")
    verified = json.loads(args.verification.read_text())
    require(verified["schema"] == "latency58-onnx-verification-v1"
            and verified["status"] == "passed_cpu_numerical_verification_only"
            and verified["verification"]["passed"] and verified["source_bindings_unchanged"]
            and sha(args.model) == verified["onnx_sha256"], "Require the actual successfully verified hop128 graph")
    metadata = verified["metadata"]
    require(metadata["hs_tasnet.architecture_version"] == "cropped1024-hann256-hop128-v1"
            and metadata["hs_tasnet.hop_samples"] == metadata["hs_tasnet.graph_output_delay_samples"] == "128"
            and metadata["hs_tasnet.intended_total_latency_samples"] == "256",
            "Native preparation requires the distinct hop128 family")
    checkpoint_sha = metadata["hs_tasnet.checkpoint_sha256"]
    state_sha = metadata["hs_tasnet.model_state_sha256"]
    step, total_updates = metadata["hs_tasnet.snapshot_step"], metadata["hs_tasnet.training_updates"]
    provenance = verified["model_provenance"]
    require(all(len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in (checkpoint_sha, state_sha))
            and step.isdecimal() and int(step) > 0 and total_updates.isdecimal()
            and int(step) == verified["step"] and checkpoint_sha == verified["checkpoint"]["sha256"]
            and state_sha == verified["model_state_sha256"]
            and provenance["version"] == metadata["hs_tasnet.architecture_version"]
            and provenance["training_plan_sha256"] == metadata["hs_tasnet.training_plan_sha256"]
            and type(provenance["training_updates"]) is int
            and type(provenance["pilot_updates"]) is int
            and int(total_updates) == provenance["training_updates"] == 2250 + provenance["pilot_updates"],
            "Verified checkpoint identity or cumulative update lineage differs")
    # A follow-up resets its local counter and Adam while retaining all prior
    # hop128 updates. Its snapshot step is therefore not its total hop128 count.
    if "tail_updates" in provenance:
        require(type(provenance["tail_updates"]) is int and provenance["tail_updates"] == int(step)
                and type(provenance["parent_training_updates"]) is int
                and provenance["parent_training_updates"] >= 2250
                and int(total_updates) == provenance["parent_training_updates"] + int(step)
                and provenance["training_objective"] in ("raw4_l1", "deployed4_l1")
                and provenance["parent_checkpoint"]["kind"] == "inference"
                and all(len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                        for v in (provenance["parent_checkpoint"]["sha256"],
                                  provenance["parent_model_state_sha256"])),
                "Matched follow-up parent or local update lineage differs")
    else:
        require(provenance["pilot_updates"] == int(step), "Original pilot local update count differs")
    source = BASE.read_text()
    replacements = {
        "// Source preparation for the listened cropped1024 raw-four L1 +250 graph.":
            "// Native diagnostic for a verified hop128 research graph; audible quality is assessed separately.",
        "a paced 256-sample callback": "a paced 128-sample callback",
        "Graph256 + queue256 =512": "Graph128 + queue128 =256",
        "constexpr int kHopSamples = 256;": "constexpr int kHopSamples = 128;",
        "constexpr int kHistorySamples = 768;": "constexpr int kHistorySamples = 896;",
        '"ola-cropped1024-hann512-hop256-v1"': '"cropped1024-hann256-hop128-v1"',
        "// These identify the already completed and listened checkpoint, not a future graph.":
            "// These identify the checkpoint recorded by the verified graph's metadata.",
        "ac46729e5e4d379b09914a6e40ae927e09089b43fd4eef219ae7e034f355da65": checkpoint_sha,
        "a12c215810026c603a1fd394383c1646219b8b3f764ebe9c2a83856404443aa4": state_sha,
        "{1, 2, 256}": "{1, 2, kHopSamples}",
        "{1, 2, 768}": "{1, 2, kHistorySamples}",
        "{1, 4, 2, 256}": "{1, 4, 2, kHopSamples}",
        'expect("hs_tasnet.kind", "cropped1024_ola");': 'expect("hs_tasnet.kind", "cropped1024_hop128");',
        'expect("hs_tasnet.hop_samples", "256");': 'expect("hs_tasnet.hop_samples", "128");',
        'expect("hs_tasnet.spectral_output_crop", "[512,1024]");': 'expect("hs_tasnet.spectral_output_crop", "[768,1024]");',
        'expect("hs_tasnet.synthesis_frame_samples", "512");': 'expect("hs_tasnet.synthesis_frame_samples", "256");',
        'expect("hs_tasnet.waveform_decoder_samples", "512");': 'expect("hs_tasnet.waveform_decoder_samples", "256");',
        'expect("hs_tasnet.analysis_history_samples", "768");': 'expect("hs_tasnet.analysis_history_samples", "896");',
        'expect("hs_tasnet.graph_output_delay_samples", "256");': 'expect("hs_tasnet.graph_output_delay_samples", "128");',
        'expect("hs_tasnet.alignment_samples", "256");': 'expect("hs_tasnet.alignment_samples", "128");',
        'expect("hs_tasnet.intended_external_host_queue_samples", "256");': 'expect("hs_tasnet.intended_external_host_queue_samples", "128");',
        'expect("hs_tasnet.intended_total_latency_samples", "512");': 'expect("hs_tasnet.intended_total_latency_samples", "256");',
        '"[[1, 2, 768], [2, 1, 1000], [1, 4, 2, 256], [1, 4, 2, 256]]"':
            '"[[1, 2, 896], [2, 1, 1000], [1, 4, 2, 128], [1, 4, 2, 128]]"',
        'expect("hs_tasnet.snapshot_step", "250");': f'expect("hs_tasnet.snapshot_step", "{step}");',
        'expect("hs_tasnet.training_updates", "2250");': f'expect("hs_tasnet.training_updates", "{total_updates}");',
        "last256 samples of the incoming 768 history": "last128 samples of the incoming 896 history",
        "* kHopSamples == 512U": "* kHopSamples == 2U * kHopSamples",
        "{1U, 255U, 256U, 257U, 511U, 512U, 513U, 769U}": "{1U, 127U, 128U, 129U, 255U, 256U, 257U, 385U}",
        "graph256 + queue256": "graph128 + queue128",
        "callback sample positions physical+512": "callback sample positions physical+256",
        "arguments.captureStart + 512U": "arguments.captureStart + 2U * kHopSamples",
        "arguments.captureEnd + 512U": "arguments.captureEnd + 2U * kHopSamples",
        "exact_512_pdc_mapping_count": "exact_256_pdc_mapping_count",
        "paced 256-sample callback boundaries": "paced 128-sample callback boundaries",
        r'\"hop_samples\":256,\"graph_delay_samples\":256,\"async_queue_delay_samples\":256':
            r'\"hop_samples\":128,\"graph_delay_samples\":128,\"async_queue_delay_samples\":128',
        r'\"simulated_pdc_samples\":512': r'\"simulated_pdc_samples\":256',
        "(1000.0 * 512.0 / 44100.0)": "(1000.0 * 2.0 * kHopSamples / kSampleRate)",
    }
    counts = {}
    for before, after in replacements.items():
        counts[before] = source.count(before)
        require(counts[before] > 0, "Preserved source no longer contains the expected field: " + before)
        source = source.replace(before, after)
    out = args.output_directory.absolute()
    require(out.is_relative_to(ROOT / "research/direct/runs/latency58") and not out.exists(),
            "Use a fresh native source directory within the phase")
    out.mkdir()
    cpp = out / "latency58_native_async_qualifier.cpp"
    cpp.write_text(source)
    require(sha(BASE) == BASE_SHA and sha(args.verification) == args.verification_sha256
            and sha(args.model) == verified["onnx_sha256"], "Preparation inputs changed")
    receipt = {"schema": "latency58-native-source-preparation-v1", "source": {"path": str(cpp), "sha256": sha(cpp)},
               "generator_sha256": sha(__file__), "preserved_source": {"path": str(BASE), "sha256": BASE_SHA},
               "verification": {"path": str(args.verification), "sha256": args.verification_sha256},
               "graph": {"path": str(args.model), "sha256": verified["onnx_sha256"], "bytes": args.model.stat().st_size},
               "checkpoint_sha256": checkpoint_sha, "model_state_sha256": state_sha, "snapshot_step": int(step),
               "cumulative_training_updates": int(total_updates),
               "cumulative_hop128_updates": provenance["pilot_updates"], "model_provenance": provenance,
               "explicit_replacement_counts": counts, "compiled": False, "native_executed": False,
               "native_host_qualified": False}
    (out / "source-preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output_directory": str(out), "cpp_sha256": sha(cpp), "compiled": False}))


if __name__ == "__main__":
    main()
