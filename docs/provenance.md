# Migration and provenance

StemgenRT-5.8 is a divergent fork of
[Phil Wang's HS-TasNet implementation](https://github.com/lucidrains/HS-TasNet),
with ancestry in [HS-TasNet](https://arxiv.org/abs/2402.17701). It targets the
StemgenRT plugin's current eight-state streaming model.

## Breaking Python API change

The distribution is `StemgenRT`, the import package is `stemgenrt`, the model
class is `StemgenRT58`, and the training command is `stemgenrt-train`.
The old `hs_tasnet` imports and configurable model are no longer supported.
Use the upstream repository for the original implementation. The current
model is a different architecture; renaming an old import does not convert
its weights or constructor arguments.

Python package 0.6.2 corresponds to plugin v0.6.2. The model name's 5.8 suffix
means approximately 5.8 ms of algorithmic latency at 44.1 kHz, rather than a
package version.

## Supported model and checkpoints

The package includes the current model and its matching training, data,
objectives, teacher supervision, Adam/EMA recovery, evaluation and export.
It uses eight states, 32 attention frames and fixed 1024/256/128-sample
analysis/synthesis/hop geometry. Earlier model variants and the C204 draft
paper remain in git history and are outside this release.

Parameter names, fixed buffers, tensor arithmetic and the current model's
native checkpoint identifiers are preserved. Native eight-state checkpoints
load with an explicit SHA-256. Older packed recovery archives require their
original decoder before import. New training checkpoints are self-contained
and retain raw weights, Adam, EMA, RNG state, configuration and data position.
An integer ONNX graph cannot reconstruct the original FP32 training weights.

Tests compare model outputs, states and gradients, deterministic data
augmentation, checkpoint recovery and ONNX waveform output. These checks do
not establish separation quality on new audio or real-time host performance.

## Frozen baselines

The research baseline is the teacher004 EMA at 45,750 cumulative updates
(2,000 updates in its final stage), with 4.564401858994641 dB full-band SDR
on the original 14-track, 28-excerpt development panel. Its packed checkpoint
SHA-256 is `7fcd444f83985c0aab0c76923c4355fea3c3e11aa32c81410a6bb9388b755154`;
the decoded EMA state is
`f78c49b3755d6a71b7890482d3a40a5662b417b7a63ed3ad9d393cfb0da0037b`.
The final training plan digest is
`2ef8b5720e7a232936b404106a4d526421090aec0262e317f0ed26802a4c36cd`.
Its matching portable recipe, teacher supervision, deterministic data pipeline,
Adam/EMA recovery, evaluation and export are maintained together here.

The frozen historical product baseline is StemgenRT v0.6.1 at plugin commit
`990df8ee5baa621f042d4534dacc65afee0a96ce`, with deployed graph
`08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748`
and 4.455172594 dB on that same development panel. It remains the comparison
and rollback reference.

The current download matches [StemgenRT v0.6.2](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/tag/v0.6.2)
at plugin commit `61df8f4aa1555ef110308d01ea92b54ace770979`. It exports the frozen
research checkpoint above, with graph SHA-256
`77164d6a581fafb2a31f53fd8ffde44c07cf618472952a4cdba14e68dda3b8b9`
and **4.564148170856732 dB** measured on the same development panel using ONNX
Runtime 1.26.0. The graph retains eight states, 32 attention frames and the
256-sample graph-plus-host delay at a 128-sample host buffer. This quality
measurement does not establish a fresh held-out result or a new sustained
real-time hardware qualification.
