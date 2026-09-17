"""Convert the four-state FP32 ONNX training weights to an identical trainable PyTorch checkpoint."""
import argparse
from pathlib import Path

from hs_tasnet.streaming_checkpoint import import_released_onnx, save_streaming_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("models/hop128-trainable.onnx"))
    parser.add_argument("--output", type=Path, default=Path("models/hop128-trainable.pt"))
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    model = import_released_onnx(args.onnx)
    print(save_streaming_checkpoint(model, args.output))


if __name__ == "__main__":
    main()
