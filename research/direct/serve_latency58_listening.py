"""Serve existing local listening pages with seekable WAV byte ranges."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re


class ListeningHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_head(self):
        self.remaining_audio_bytes = None
        path = Path(self.translate_path(self.path))
        if path.suffix.lower() != ".wav":
            return super().send_head()
        root = Path(self.directory).resolve()
        if not path.resolve().is_relative_to(root):
            self.send_error(403)
            return None
        try:
            stream = path.open("rb")
        except (OSError, ValueError):
            self.send_error(404)
            return None
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = self.headers.get("Range")
        if header is not None:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header) if len(header) <= 128 else None
            valid = match is not None and bool(match[1] or match[2]) and size > 0
            if valid:
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), size - 1) if match[2] else size - 1
                else:
                    suffix = int(match[2])
                    start = max(0, size - suffix)
                valid = 0 <= start <= end < size
            if not valid:
                stream.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return None
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        stream.seek(start)
        self.remaining_audio_bytes = end - start + 1
        return stream

    def copyfile(self, source, outputfile):
        if self.remaining_audio_bytes is None:
            return super().copyfile(source, outputfile)
        remaining = self.remaining_audio_bytes
        try:
            while remaining:
                block = source.read(min(65536, remaining))
                if not block:
                    break
                outputfile.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers cancel pending media requests when the source changes.
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "runs"
    if not 1024 <= args.port <= 65535 or not root.is_dir():
        raise ValueError("Use an unprivileged port and the existing research runs directory")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(ListeningHandler, directory=str(root)))
    print(json.dumps({"listening": f"http://127.0.0.1:{args.port}", "root": str(root),
                      "wav_byte_ranges": True, "source_audio_changed": False}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
