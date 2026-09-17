"""Check saved-model parity and 256-sample mapping through the preserved native queue.

The original paced deadline result is retained separately. This run does not
install a plugin or qualify target hardware timing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from research.direct.run_latency58_quality import PHASE, ROOT, PYTHON, read, require, sha, write, execute

BASE = PHASE / "leader-cleanup-250-native-prep-001"


def run(plan_path, plan_sha):
    require(sha(plan_path) == plan_sha, "Native residual plan changed")
    plan = read(plan_path)
    bindings = plan["source_bindings"]
    require(plan["schema"] == "latency58-direct-sdr-native-plan-v1"
            and all(sha(p) == s for p, s in bindings.items()) and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Native residual inputs or CPU scope differ")
    from research.direct.latency58_sdr_checkpoint import require_space
    budget = read(plan["training_plan"])
    counted = require_space(budget, 35_000_000)
    import numpy as np
    import soundfile as sf
    import torch
    from research import evaluate as legacy
    from research.direct import evaluate as shared
    from research.direct.latency58_evaluate import plan_latency58_stream, stream_latency58_track
    from research.direct.latency58_magnitude_checkpoint import load_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(plan["output_directory"])
    checkpoint = plan["checkpoint"]
    model, _ = load_model(checkpoint)
    manifest, config = read(ROOT / "research/manifests/valid.json"), read(ROOT / "research/eval_config.json")
    tracks, config = shared.select_panel(manifest, config, panel="full", track_indices=[1],
                                         excerpt_starts=[60.0], duration=15.0, alignment_samples=128)
    track = tracks[0]
    rows = legacy._reference_intervals(track, config)
    streaming = plan_latency58_stream(rows, int(track["frames"]))
    audio_path = Path(manifest["root"]) / track["mixture"]
    raw, mixture, metadata = stream_latency58_track(model, audio_path, streaming)
    estimates = shared.shipping_residual(raw[0], mixture[0])
    references = {}
    for stem, samples in zip(("drums", "bass", "vocals", "other"), estimates, strict=True):
        path = out / (stem + ".wav")
        sf.write(path, samples.T, 44100, subtype="FLOAT")
        decoded, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 44100 and np.array_equal(decoded.T, samples), "Native parity reference samples changed")
        references[stem] = {"path": str(path), "sha256": sha(path)}
        bindings[str(path)] = sha(path)
    write(out / "reference.json", {"checkpoint": checkpoint, "references": references,
                                   "stream": metadata, "branch_diagnostics_requested": False})
    print(json.dumps({"event": "reference_rendered", "track": track["name"]}), flush=True)
    export = read(plan["export_verification"])
    preparation = read(BASE / "source-preparation.json")
    source_path = Path(preparation["source"]["path"])
    require(sha(source_path) == preparation["source"]["sha256"], "Preserved C204 native source changed")
    source = source_path.read_text()
    old_policy = "complete deployed four stems; Other = previous physical mixture - sum(unchanged DBV), once"
    replacements = {preparation["checkpoint_sha256"]: checkpoint["sha256"],
                    preparation["model_state_sha256"]: export["model_state_sha256"],
                    old_policy: export["metadata"]["hs_tasnet.output_policy"],
                    'expect("hs_tasnet.architecture_version", kFamily);':
                        'expect("hs_tasnet.architecture_version", "' + export["metadata"]["hs_tasnet.architecture_version"] + '");'}
    for old, new in replacements.items():
        require(source.count(old) == 1, "Unexpected native metadata occurrence")
        source = source.replace(old, new)
    cpp, binary = out / "native.cpp", out / "native-qualifier"
    cpp.write_text(source)
    bindings[str(cpp)] = sha(cpp)
    compile_command = read(BASE / "compile-execution.json")["argv"]
    compile_command = [str(cpp) if value == str(source_path) else value for value in compile_command]
    compile_command[compile_command.index("-o") + 1] = str(binary)
    execute(compile_command, out, "compile", 120, bindings, {"native_queue_algorithm_unchanged": True})
    bindings[str(binary)] = sha(binary)
    command = read(BASE / "qualification-plan-001.json")["command"]
    command[0] = str(binary)
    changes = {"--model": export["onnx"]["path"], "--model-sha256": export["onnx"]["sha256"],
               "--model-bytes": str(export["onnx"]["bytes"]), "--output": str(out / "native-result.json")}
    for stem, binding in references.items():
        changes["--reference-" + stem] = binding["path"]
        changes["--reference-" + stem + "-sha256"] = binding["sha256"]
    for option, value in changes.items():
        command[command.index(option) + 1] = value
    require(int(command[command.index("--prefix-frames") + 1]) == streaming.receive_end
            and int(command[command.index("--capture-start") + 1]) == rows[0]["reference_start"]
            and int(command[command.index("--capture-end") + 1]) == rows[0]["reference_end"], "Native stream geometry differs")
    write(out / "native-command.json", {"argv": command, "source_bindings": bindings, "replacements": replacements})
    began, timed_out = time.monotonic(), False
    with (out / "native-console.log").open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait(timeout=15)
    unchanged = all(sha(p) == s for p, s in bindings.items())
    write(out / "native-execution.json", {"actual_exit_code": code, "timed_out": timed_out,
          "elapsed_seconds": time.monotonic() - began, "source_bindings_unchanged": unchanged, "argv": command})
    require(code in (0, 3) and not timed_out and unchanged, "Native process did not finish its diagnostic")
    native = read(out / "native-result.json")
    music, latency = native["music_parity"], native["latency_contract"]
    require(music["executed"] and music["passed"] and music["inputs_unchanged"]
            and music["stream"]["all_consumer_checks"] and music["stream"]["last_real_sample_recovered"]
            and music["stream"]["valid_frames_recovered"] == music["prefix_frames"]
            and music["capture_callback_start"] - music["capture_physical_start"] == 256
            and music["capture_callback_end"] - music["capture_physical_end"] == 256
            and latency["graph_delay_samples"] == latency["async_queue_delay_samples"] == 128
            and native["identity"]["model_sha256"] == export["onnx"]["sha256"]
            and native["gate"]["model_and_runtime_unchanged"], "Continuous native parity or 256-sample mapping failed")
    write(out / "review.json", {"status": "pass_continuous_audio_alignment_and_parity", "checkpoint": checkpoint,
          "onnx": export["onnx"], "native_result_sha256": sha(out / "native-result.json"),
          "full_native_harness_status": native["status"], "full_native_harness_gate": native["gate"],
          "actual_native_exit_code": code, "latency_contract": latency, "music_parity": music,
          "source_bindings": bindings, "source_bindings_unchanged": True,
          "counted_bytes_before": counted, "counted_bytes_after": require_space(budget, 2_000_000),
          "plugin_modified": False, "target_hardware_deadlines_qualified": False})
    print(json.dumps({"status": "pass_continuous_audio_alignment_and_parity", "actual_native_exit_code": code,
                      "full_native_harness_status": native["status"], "total_delay_samples": 256}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-name")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.plan is not None:
        run(args.plan, args.plan_sha256)
        return
    require(args.pilot_name and all(c.isalnum() or c in "-_" for c in args.pilot_name), "Specify an existing pilot name")
    pilot = PHASE / args.pilot_name
    export_path = pilot / "onnx/verification.json"
    export = read(export_path)
    require(export["status"] == "pass" and export["schema"] == "latency58-direct-sdr-export-result-v1"
            and float(export["metadata"]["hs_tasnet.full14_sdr_db"]) >= 5.0,
            "The saved 5 dB endpoint must pass export parity before native execution")
    source = read(BASE / "source-preparation.json")["source"]["path"]
    base_plan = read(BASE / "qualification-plan-001.json")
    paths = [Path(__file__).resolve(), pilot / "plan.json", export_path, Path(export["onnx"]["path"]), Path(source),
             BASE / "source-preparation.json", BASE / "compile-execution.json", BASE / "qualification-plan-001.json"]
    bindings = {**base_plan["source_bindings"], **export["source_bindings"], **{str(p): sha(p) for p in paths}}
    require(all(sha(p) == s for p, s in bindings.items()), "Native prerequisite files changed")
    out = pilot / "native"
    require(not out.exists(), "Preserve native results")
    plan = {"schema": "latency58-direct-sdr-native-plan-v1", "checkpoint": export["checkpoint"],
            "training_plan": str(pilot / "plan.json"), "export_verification": str(export_path),
            "source_bindings": bindings, "output_directory": str(out)}
    out.mkdir()
    write(out / "plan.json", plan)
    execute([PYTHON, "-u", "-m", "research.direct.qualify_latency58_magnitude_native", "--plan", str(out / "plan.json"),
             "--plan-sha256", sha(out / "plan.json")], out, "qualification", 1000, bindings,
            {"plan_sha256": sha(out / "plan.json")})


if __name__ == "__main__":
    main()
