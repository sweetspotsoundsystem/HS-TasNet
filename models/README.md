# Stereo streaming model

This trained HS-TasNet variant separates a stereo music mixture into Drums,
Bass, Vocals and Other. Spectral and waveform branches share recurrent context, with causal attention
and separate spectral and waveform GRU memories.
The asymmetric analysis window spans 1024 samples; synthesis uses 256 samples
with a 128-sample hop. Eight explicit states retain context between calls.

`hop128.onnx` is a self-contained ONNX file with float32 inputs and outputs.
Seventeen learned products use integer arithmetic, including a single fused
query/key/value attention projection.
Use ONNX Runtime **1.26.0 CPU**. Download the shared StemgenRT model with
`python scripts/download_streaming_model.py`; the script pins an immutable
source revision and verifies size and SHA-256 before making the file available.
SHA-256: `08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748`.
Size: **37,532,574 bytes**.

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
| `attention_keys` | `[1,31,64]` | `next_attention_keys` | `[1,31,64]` |
| `attention_values` | `[1,31,128]` | `next_attention_values` | `[1,31,128]` |
| `spec_memory_hidden` | `[1,1,500]` | `next_spec_memory_hidden` | `[1,1,500]` |
| `waveform_memory_hidden` | `[1,1,500]` | `next_waveform_memory_hidden` | `[1,1,500]` |

1. Initialize all eight state tensors to zero.
2. Submit consecutive 128-sample hops and carry every returned state unchanged.
   Ignore the first output after reset; call N emits input N-1.
3. Pad a partial final input hop with zeros, then submit exactly one zero hop
   to recover the pending output. Trim the concatenated output to the real length.
4. Reset all state after a seek, a gap, invalid input, or a new stream.

The `StreamingSeparator` Python API handles state and alignment. `process_chunk`
returns `None` for the initial hop, `flush` returns the last pending output and
resets the stream, and `separate` renders a whole clip with padding removed.
Use one instance per stream; do not call an instance concurrently. Fusion and
branch GRU hidden values use public scale `2**-18`: carry returned values without
rescaling. The runtime disables KleidiAI (`mlas.disable_kleidiai=1`) to match
the numerically checked plugin configuration.

## Validation and limits

The exact released graph scores **4.455173 dB mean full-band SDR** on the
unchanged 14-track, 28-excerpt development panel. Its FP32 source checkpoint
scores 4.465157 dB; these are distinct measurements. See the
[deployment report](https://github.com/sweetspotsoundsystem/stemgen-rt/blob/35b533017b32099f417b6c965e37214b72a8ccea/model/quality-deployment.json)
for per-stem results and source-view leakage. The broader 5 dB research goal
remains open.

The user reported zero fallback in their M4 plugin test. No run duration or
raw counter trace was supplied. The Python API has no deadline fallback counter
and its synchronous timing is not a DAW playback qualification.

Tests compare all four outputs against independent CPU PyTorch results for the declared integer graph
for lengths 1, 127, 128, 129, 255, 256, 257 and 16521. They cover reset replay,
one-hop alignment, final-sample recovery and mixture reconstruction. The
[fixture metadata](../tests/fixtures/hop128-pytorch.json) records its format,
model identity and reference implementation. Run `pytest tests/test_streaming.py`.

The 128-sample graph delay describes alignment. CPU throughput and any playback
queue add separate constraints. Use StemgenRT for host scheduling, compensation,
confidence fades and deadline fallback; these are outside the Python wrapper.

## Four-state training compatibility

`python scripts/download_streaming_model.py --variant trainable` downloads
`models/hop128-trainable.onnx`, the earlier FP32 model used by
`StreamingHSTasNet`, the weight importer, and the training/export guide.
It retains its original checksum
`b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3`
and 111,344,465-byte size. Existing portable checkpoints remain supported.

The default download preserves an existing file with a different checksum;
choose a new `--output` path when retaining an earlier `hop128.onnx` file.
To run the training baseline through `StreamingSeparator`, pass
`expected_sha256=TRAINABLE_MODEL_SHA256` from `hs_tasnet.streaming`. For a new
training export, pass its verification report's digest. The API checks and
carries all states in either supported interface.

Based on the [HS-TasNet architecture](https://arxiv.org/abs/2402.17701) and
Phil Wang's implementation. See the repository license and citations.
