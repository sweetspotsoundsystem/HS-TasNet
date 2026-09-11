"""Inventory dedicated stem-folder training roots without modifying the audio."""
import argparse
from pathlib import Path
from hs_tasnet.streaming_data import build_manifest


def assignments(values, cast):
    result = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key or key in result:
            raise ValueError("Use unique NAME=VALUE assignments")
        result[key] = cast(raw)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="NAME=PATH; repeat for multiple training roots")
    parser.add_argument("--weight", action="append", default=[], help="NAME=WEIGHT; supply one for each root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_manifest(assignments(args.root, Path), args.output,
                         weights=assignments(args.weight, float) if args.weight else None))


if __name__ == "__main__":
    main()
