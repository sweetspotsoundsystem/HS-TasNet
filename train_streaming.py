"""Train the current eight-state streaming model; see docs/training.md."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from stemgenrt.trainer import main

if __name__ == "__main__":
    main()
