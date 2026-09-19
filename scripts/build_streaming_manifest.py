"""Inventory dedicated stem-folder training roots without modifying the audio."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hs_tasnet.data import build_manifest


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
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="Exclude mixture paths/bytes from an existing held-out manifest")
    parser.add_argument("--exclude-name", action="append", default=[],
                        help="Exclude a relative song-folder name; repeat for held-out songs")
    args = parser.parse_args()
    print(build_manifest(assignments(args.root, Path), args.output,
                         weights=assignments(args.weight, float) if args.weight else None,
                         split=args.split, exclude_manifests=args.exclude_manifest,
                         excluded_names=args.exclude_name))


if __name__ == "__main__":
    main()
