"""Export authenticated current-model weights as a verified eight-state ONNX graph."""
import argparse
from pathlib import Path

import torch
from stemgenrt.checkpoint import load_model
from stemgenrt.export import export_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True,
                        help="Expected SHA-256 of the native or portable training checkpoint")
    parser.add_argument("--role", choices=("raw", "ema"), required=True,
                        help="Explicit weight role; native inference checkpoints must match their stored role")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-hops", type=int, default=48)
    parser.add_argument("--variant", choices=("fp32", "integer"), default="fp32",
                        help="FP32 native parity or the seventeen-product integer deployment variant")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_model(args.checkpoint, expected_sha256=args.checkpoint_sha256, role=args.role)
    report = export_model(model, args.output, verify_hops=args.verify_hops,
                          checkpoint_sha256=args.checkpoint_sha256, variant=args.variant)
    print(f"Verified {args.output}: SHA-256 {report['onnx_sha256']}")


if __name__ == "__main__":
    main()
