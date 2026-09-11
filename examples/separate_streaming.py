"""Separate a stereo 44.1 kHz file into four floating-point WAV files."""

import argparse
from pathlib import Path

import soundfile as sf

from hs_tasnet.streaming import SOURCE_ORDER, StreamingSeparator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/hop128.onnx"))
    args = parser.parse_args()
    audio, rate = sf.read(args.input, dtype="float32", always_2d=True)
    separator = StreamingSeparator(args.model, sample_rate=rate)
    stems = separator.separate(audio.T)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for name, stem in zip(SOURCE_ORDER, stems):
        destination = args.output_directory / f"{name}.wav"
        if destination.exists():
            raise FileExistsError(destination)
    for name, stem in zip(SOURCE_ORDER, stems):
        sf.write(args.output_directory / f"{name}.wav", stem.T, rate, subtype="FLOAT")


if __name__ == "__main__":
    main()
