"""Download the exact released ONNX model shared with StemgenRT."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


MODEL_URL = (
    "https://media.githubusercontent.com/media/sweetspotsoundsystem/stemgen-rt/"
    "f8fb3beb95f8a17f80e1c195964f9cf1c42f2cee/model/model.onnx"
)
MODEL_SHA256 = "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3"
MODEL_BYTES = 111344465


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(destination):
    """Publish verified bytes, preserving any existing destination."""
    destination = Path(destination)
    if destination.is_symlink():
        raise FileExistsError(f"Refusing a symlink destination: {destination}")
    if destination.exists():
        if destination.stat().st_size == MODEL_BYTES and digest(destination) == MODEL_SHA256:
            return destination
        raise FileExistsError(f"Existing file is a different model: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".hop128-", delete=False) as output:
            temporary = Path(output.name)
            checksum = hashlib.sha256()
            count = 0
            with urlopen(MODEL_URL, timeout=30) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > MODEL_BYTES:
                        raise ValueError("Downloaded model exceeds its expected size")
                    checksum.update(block)
                    output.write(block)
        if count != MODEL_BYTES or checksum.hexdigest() != MODEL_SHA256:
            raise ValueError("Downloaded model size or SHA-256 differs")
        # Same-directory hard link publishes complete bytes and fails if another
        # process created the destination while the download was in progress.
        os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models/hop128.onnx")
    args = parser.parse_args()
    print(f"Model ready: {download(args.output)}")


if __name__ == "__main__":
    main()
