# Stereo streaming model

This trained HS-TasNet variant separates a stereo music mixture into Drums,
Bass, Vocals and Other. Spectral and waveform branches share recurrent context.
The asymmetric analysis window spans 1024 samples; synthesis uses 256 samples
with a 128-sample hop. Four explicit states retain context between calls.

`hop128.onnx` is a single float32 ONNX file, with no external weight file.
Use ONNX Runtime **1.26.0 CPU**. Download the shared StemgenRT model with
`python scripts/download_streaming_model.py`; the script pins an immutable
source revision and verifies size and SHA-256 before making the file available.
SHA-256: `b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3`.
Size: **111,344,465 bytes**.

## Streaming interface

All tensors are float32. Audio is stereo at exactly 44.1 kHz, without per-hop
normalization, clipping or resampling. Output source order is **Drums, Bass,
Vocals, Other**. The graph calculates Other as the delayed mixture minus the
first three stems, so the four outputs reconstruct that mixture within floating-
point rounding. This does not imply exact recovery of the original recordings.

| Input | Shape | Output | Shape |
| --- | --- | --- | --- |
| `audio_chunk` | `[1,2,128]` | `separated_chunk` | `[1,4,2,128]` |
| `audio_history` | `[1,2,896]` | `next_audio_history` | `[1,2,896]` |
| `fusion_hidden` | `[2,1,1000]` | `next_fusion_hidden` | `[2,1,1000]` |
| `spectral_numerator_tail` | `[1,4,2,128]` | `next_spectral_numerator_tail` | `[1,4,2,128]` |
| `waveform_tail` | `[1,4,2,128]` | `next_waveform_tail` | `[1,4,2,128]` |

1. Initialize all four state tensors to zero.
2. Submit consecutive 128-sample hops and carry every returned state unchanged.
   Ignore the first output after reset; call N emits input N-1.
3. Pad a partial final input hop with zeros, then submit exactly one zero hop
   to recover the pending output. Trim the concatenated output to the real length.
4. Reset all state after a seek, a gap, invalid input, or a new stream.

The `StreamingSeparator` Python API handles state and alignment. `process_chunk`
returns `None` for the initial hop, `flush` returns the last pending output and
resets the stream, and `separate` renders a whole clip with padding removed.
Use one instance per stream; do not call an instance concurrently.

## Validation and limits

The model scores **4.07 dB mean full-band SDR** on a 14-track development panel,
up from 3.85 dB for the previous model. Additional intervals on those same tracks
support the improvement; controlled source inputs show less vocal/instrument
spill. These are development measurements, with remaining variation by source
and passage, including quiet instruments.

Tests compare all four outputs against independent CPU float32 PyTorch results
for lengths 1, 127, 128, 129, 255, 256, 257 and 16521. They cover reset replay,
one-hop alignment, final-sample recovery and mixture reconstruction. The
[fixture metadata](../tests/fixtures/hop128-pytorch.json) records its format,
model identity and reference implementation. Run `pytest tests/test_streaming.py`.

The 128-sample graph delay describes alignment. CPU throughput and any playback
queue add separate constraints. Use StemgenRT for host scheduling, compensation,
confidence fades and deadline fallback; these are outside the Python wrapper.

Based on the [HS-TasNet architecture](https://arxiv.org/abs/2402.17701) and
Phil Wang's implementation. See the repository license and citations.
