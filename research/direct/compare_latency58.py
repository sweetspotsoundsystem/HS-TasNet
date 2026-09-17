"""Compare completed hop128 reports against preserved accepted-model evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.direct.latency58_checkpoint import require, sha

ROOT = Path(__file__).resolve().parents[2]
ACCEPTED = ROOT / ("research/direct/runs/latency11/"
                  "cropped1024-matched-raw4_control-b4-bf16-lr3e-5/"
                  "evaluation-cpu-fp32-pilot000250")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mode", choices=("actions60", "full14"), required=True)
    args = parser.parse_args()
    candidate_path = args.candidate / "result.json"
    candidate = json.loads(candidate_path.read_text())
    execution = json.loads((args.candidate / "execution.json").read_text())
    require(execution["actual_exit_code"] == 0 and execution["source_bindings_unchanged"],
            "Candidate evaluation did not finish successfully")
    bindings = {str(candidate_path): sha(candidate_path), str(Path(__file__).resolve()): sha(__file__)}
    if args.mode == "full14":
        from research.direct.compare import compare
        reference_path = ACCEPTED / "music/result.json"
        reference = json.loads(reference_path.read_text())["music"]
        for field in ("manifest_sha256", "output_policy", "precision", "metrics"):
            require(reference[field] == candidate[field], "Evaluation protocols differ: " + field)
        report = compare(reference["results"][0], candidate["results"][0])
        report["candidate_quality_accepted"] = False
        output = args.candidate / "paired-panel-summary.json"
    else:
        import numpy as np
        import soundfile as sf
        reference_path = ACCEPTED / "actions60/result.json"
        reference = json.loads(reference_path.read_text())
        start, end = 60 * 44100, 75 * 44100
        require(candidate["results"][0]["stream_batches"][0]["reference_intervals"] == [[start, end]],
                "Audition physical interval differs")
        rows = {}
        for stem in ("drums", "bass", "vocals", "other"):
            name = f"estimate-{stem}.wav"
            old = Path(reference["wav_files"][name]["path"])
            require(sha(old) == reference["wav_files"][name]["sha256"], "Accepted audio changed")
            source = Path("/home/axel/HS-TasNet/data/musdb18hq/train/Actions - One Minute Smile") / f"{stem}.wav"
            require(sha(source) == candidate["root_source_bindings"][str(source)], "Physical source changed")
            target, rate = sf.read(source, start=start, stop=end, dtype="float64", always_2d=True)
            require(rate == 44100, "Source sample rate differs")
            bindings[str(source)] = sha(source)
            row = {}
            for label, path in (("accepted", old), ("candidate", args.candidate / "audio" / name)):
                audio, rate = sf.read(path, dtype="float64", always_2d=True)
                require(rate == 44100 and audio.shape == target.shape and np.isfinite(audio).all(),
                        "Audition sample rate, length or finite values differ")
                error = (audio - target) ** 2
                sdr = float(10 * np.log10(np.sum(target ** 2) / np.sum(error)))
                if label == "accepted":
                    saved = reference["phase"][stem]["cropped1024_raw4_control_pilot250_total2250"]["raw_sdr_db"]
                    require(abs(sdr - saved) < 1e-9, "Accepted saved SDR was not reproduced")
                phase = (np.arange(len(audio)) + start) % 128
                profile = np.bincount(phase, weights=error.mean(1)) / np.bincount(phase)
                harmonic = float(2 * np.abs(np.fft.rfft(profile)[1]) / len(profile) / profile.mean())
                row[label] = {"raw_waveform_sdr_db": sdr,
                              "rms_dbfs": float(10 * np.log10(np.mean(audio * audio))),
                              "error_phase128_first_harmonic_fraction": harmonic, "sha256": sha(path)}
                bindings[str(path)] = sha(path)
            row["delta_sdr_db"] = row["candidate"]["raw_waveform_sdr_db"] - row["accepted"]["raw_waveform_sdr_db"]
            rows[stem] = row
        report = {"physical_interval_samples": [start, end], "output_gain": "unchanged native",
                  "per_stem": rows, "accepted_saved_sdr_reproduced": True, "normalization": "none",
                  "candidate_quality_accepted": False,
                  "phase_interpretation": "Error energy modulation by physical sample modulo 128; not a listening verdict"}
        output = args.candidate / "paired-audio-summary.json"
    bindings[str(reference_path)] = sha(reference_path)
    require(all(sha(path) == digest for path, digest in bindings.items()), "Comparison input changed")
    report["source_bindings"] = bindings
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
