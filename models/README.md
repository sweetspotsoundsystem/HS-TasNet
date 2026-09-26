# StemgenRT-5.8 model interface

The released `hop128.onnx` separates 44.1 kHz stereo audio into drums, bass,
vocals and other. Analysis is 1024 samples, synthesis 256 samples, and each
input/output hop 128 samples. Causal attention and branch GRUs carry eight
explicit FP32 states.

See the [model overview](../README.md) for the latency suffix definition.

Download with `python scripts/download_streaming_model.py`. The file's pinned
identity and immutable download URL are in `stemgenrt/streaming_models.json`.
The StemgenRT v0.6.2 graph is 37,529,132 bytes, SHA-256
`77164d6a581fafb2a31f53fd8ffde44c07cf618472952a4cdba14e68dda3b8b9`.
Use ONNX Runtime 1.26.0 CPU; the public wrapper sets one thread and disables
KleidiAI to match the release. Weights are not bundled with Python packages.
When upgrading, move an older `models/hop128.onnx` aside before downloading;
the downloader preserves existing files and refuses to overwrite a different model.

| Input | Shape | Output |
| --- | --- | --- |
| `audio_chunk` | `[1,2,128]` | `separated_chunk`: `[1,4,2,128]` |
| `audio_history` | `[1,2,896]` | `next_audio_history` |
| `fusion_hidden` | `[2,1,1000]` | `next_fusion_hidden` |
| `spectral_numerator_tail` | `[1,4,2,128]` | `next_spectral_numerator_tail` |
| `waveform_tail` | `[1,4,2,128]` | `next_waveform_tail` |
| `attention_keys` | `[1,31,64]` | `next_attention_keys` |
| `attention_values` | `[1,31,128]` | `next_attention_values` |
| `spec_memory_hidden` | `[1,1,500]` | `next_spec_memory_hidden` |
| `waveform_memory_hidden` | `[1,1,500]` | `next_waveform_memory_hidden` |

Each next-state shape equals its input-state shape. Initialize all states to
zero. Each output corresponds to the previous input hop; discard initial
pre-roll and provide one zero hop to flush the last input. Pad partial input
hops and trim their output to the original sample count. `StreamingSeparator`
handles this lifecycle and resets on invalid input, seek or a new clip.

The graph forms Other from the delayed mixture minus the first three stems,
so stems reconstruct the mixture within floating-point rounding. This is a
mixture-consistency property, not proof of perfect source recovery. Input
levels are not normalized or clipped.

Custom current-model exports require an explicit `expected_sha256` and the
same eight-state contract. Four-state graphs and window/hop overrides are no
longer supported. Native FP32 checkpoints and licensed corpus audio must be
supplied separately; the integer release is not a reversible training archive.
