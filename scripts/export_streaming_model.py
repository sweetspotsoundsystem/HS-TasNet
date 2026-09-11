"""Export a PyTorch streaming checkpoint and verify every recurrent output."""
import argparse
from pathlib import Path

import torch
from hs_tasnet.streaming_checkpoint import load_streaming_checkpoint
from hs_tasnet.streaming_export import export_streaming_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = export_streaming_model(load_streaming_checkpoint(args.checkpoint), args.output)
    print(f"Verified {args.output}: SHA-256 {report['onnx_sha256']}")


if __name__ == "__main__":
    main()
