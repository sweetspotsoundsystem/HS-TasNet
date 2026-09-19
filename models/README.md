# Current stereo streaming model

The released `hop128.onnx` separates 44.1 kHz stereo audio into drums, bass,
vocals and other. Analysis is 1024 samples, synthesis 256 samples, and each
input/output hop 128 samples. Causal attention and branch GRUs carry eight
explicit FP32 states.

Download with `python scripts/download_streaming_model.py`. The file's pinned
identity and immutable download URL are in `hs_tasnet/streaming_models.json`.
The graph is 37,532,574 bytes, SHA-256
`08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748`.
Use ONNX Runtime 1.26.0 CPU; the public wrapper sets one thread and disables
KleidiAI to match the release. Weights are not bundled with Python packages.

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
