"""Fine-tune or resume the released streaming architecture on aligned stem WAVs."""
import argparse
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from hs_tasnet.streaming_trainer import StreamingTrainConfig, train_streaming


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, help="Optional manifest checked for exact data overlap")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--stop-after", type=int, help="Save a planned intermediate endpoint within the configured horizon")
    parser.add_argument("--teacher-factory", help="MODULE:FUNCTION returning a frozen, physically aligned teacher nn.Module")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Initialize from portable model weights with fresh Adam")
    source.add_argument("--resume", type=Path, help="Restore model, Adam, RNG and data position in a new output directory")
    source.add_argument("--scratch", action="store_true", help="Initialize a new untrained streaming model")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = StreamingTrainConfig(**json.loads(args.config.read_text())).validate()
    teacher = None
    if args.teacher_factory:
        module, separator, name = args.teacher_factory.partition(":")
        if not separator or not name:
            raise ValueError("Teacher factory must be MODULE:FUNCTION")
        teacher = getattr(importlib.import_module(module), name)()
    result = train_streaming(config, args.manifest, args.output, checkpoint=args.checkpoint,
                            resume=args.resume, teacher=teacher, validation_manifest=args.validation_manifest,
                            device=args.device, stop_after=args.stop_after)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
