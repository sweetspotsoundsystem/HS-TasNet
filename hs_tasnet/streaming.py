"""CPU inference for the released stereo, hop-128 ONNX model.

This convenience API is synchronous and allocates memory. Use a dedicated
worker when integrating it with audio playback; instances are not thread-safe.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np


MODEL_SHA256 = "b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3"
SAMPLE_RATE = 44100
HOP_SAMPLES = 128
SOURCE_ORDER = ("drums", "bass", "vocals", "other")
_STATES = {
    "audio_history": (1, 2, 896),
    "fusion_hidden": (2, 1, 1000),
    "spectral_numerator_tail": (1, 4, 2, 128),
    "waveform_tail": (1, 4, 2, 128),
}


class StreamingSeparator:
    """Separate float32 audio at 44.1 kHz, preserving four recurrent states.

    ``process_chunk`` accepts [2, 128] and returns [4, 2, 128], aligned to
    the previous input hop. Its first result after reset is ``None``.
    ``flush`` emits the last pending hop and resets the stream.
    """

    def __init__(self, model_path, *, sample_rate=SAMPLE_RATE):
        if sample_rate != SAMPLE_RATE:
            raise ValueError("The released model requires 44100 Hz stereo audio")
        path = Path(model_path)
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Model SHA-256 differs; run scripts/download_streaming_model.py")

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = {"audio_chunk": (1, 2, 128), **_STATES}
        outputs = {"separated_chunk": (1, 4, 2, 128)}
        outputs.update({"next_" + name: shape for name, shape in _STATES.items()})
        for nodes, expected in ((self._session.get_inputs(), inputs),
                                (self._session.get_outputs(), outputs)):
            actual = {node.name: tuple(node.shape) for node in nodes}
            if actual != expected or any(node.type != "tensor(float)" for node in nodes):
                raise ValueError("Model streaming interface differs")
        self._output_names = list(outputs)
        self.reset()

    def reset(self):
        """Clear every state after a seek, input gap, or new audio stream."""
        self._state = {name: np.zeros(shape, dtype=np.float32)
                       for name, shape in _STATES.items()}
        self._pending = False

    @staticmethod
    def _audio(audio):
        audio = np.asarray(audio)
        if audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError("Audio must be a float32 array with shape [2, samples]")
        if not np.isfinite(audio).all():
            raise ValueError("Audio must contain only finite samples")
        return np.ascontiguousarray(audio)

    def process_chunk(self, audio):
        """Advance one complete hop; return None only for initial pre-roll.

        Input and inference errors reset all state before being raised.
        No input gain normalization or output confidence fade is applied.
        """
        try:
            audio = self._audio(audio)
            if audio.shape[1] != HOP_SAMPLES:
                raise ValueError("process_chunk requires exactly 128 samples")
            values = self._session.run(
                self._output_names, {"audio_chunk": audio[None], **self._state}
            )
            if not all(np.isfinite(value).all() for value in values):
                raise RuntimeError("Model returned non-finite output or state")
            valid = self._pending
            self._state = dict(zip(_STATES, values[1:]))
            self._pending = True
            return values[0][0] if valid else None
        except Exception:
            self.reset()
            raise

    def flush(self):
        """Emit the final pending hop with one zero hop, then clear state.

        Returns None for an empty stream or a repeated flush. Trim the result
        to the real input length if the caller padded the final input hop.
        """
        if not self._pending:
            return None
        try:
            return self.process_chunk(np.zeros((2, HOP_SAMPLES), dtype=np.float32))
        finally:
            self.reset()

    def separate(self, audio):
        """Render a complete [2, T] clip to aligned [4, 2, T] stem audio.

        Starts from zero state, pads a partial final hop, flushes exactly once,
        and removes padding. Empty clips return shape [4, 2, 0].
        """
        self.reset()
        try:
            audio = self._audio(audio)
            frames = audio.shape[1]
            if frames == 0:
                return np.empty((4, 2, 0), dtype=np.float32)
            rendered = []
            for offset in range(0, frames, HOP_SAMPLES):
                chunk = audio[:, offset:offset + HOP_SAMPLES]
                if chunk.shape[1] < HOP_SAMPLES:
                    chunk = np.pad(chunk, ((0, 0), (0, HOP_SAMPLES - chunk.shape[1])))
                output = self.process_chunk(chunk)
                if output is not None:
                    rendered.append(output)
            rendered.append(self.flush())
            return np.concatenate(rendered, axis=-1)[..., :frames]
        finally:
            self.reset()
