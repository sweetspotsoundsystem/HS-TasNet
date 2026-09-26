# Provenance and migration

The current model is named **StemgenRT-5.8**; see the
[model overview](../README.md) for the latency suffix definition. Its ancestry
remains [HS-TasNet](https://arxiv.org/abs/2402.17701). The Python distribution is
`StemgenRT`: update imports to `stemgenrt`, construct the native model with
`StemgenRT58()`, and use the `stemgenrt-train` command. Previous package and
class aliases are removed. New exports use the `stemgenrt.*` metadata namespace;
the authenticated released graph and native checkpoint formats are preserved.

The complete research publication before cleanup is preserved in commit
[`a70da5ead2babb01f5d70d04b85538150ad5a41e`](https://github.com/sweetspotsoundsystem/HS-TasNet/tree/a70da5ead2babb01f5d70d04b85538150ad5a41e).
It includes the original source inventory, frozen four-second training plan,
its two recovery predecessors, historical experiment implementations, native
checkpoint loaders, packed XOR recovery decoder and monitoring launchers.
The cleanup leaves that commit in the PR's history. No additional public
archive tag is required.

## Maintained implementation

The original eight-state architecture was extracted without changing parameter
names, buffer names, state shapes, checkpoint schema identifiers or neural and
synthesis arithmetic. Its full implementation now lives in the installed
`stemgenrt` package. The default remains the eight-state, 32-frame-attention
architecture of the frozen research baseline. The optional nine-state model
retains the closed past-frame filter experiment; checkpoint loading preserves
the saved state interface. The historical class inheritance chain is replaced
by one current model and small DSP helpers.

The crop/pitch/remix recipes and weighted ordinary/auxiliary objectives retain
their scientific definitions. New-run orchestration uses portable manifests
and complete checkpoints, with explicit config/data identities. The new
checkpoint container is self-contained and lossless; it does not rely on an
old filesystem layout or an XOR parent checkpoint.

Historical native branch-memory inference checkpoints remain loadable with an
explicit expected digest. Historical packed recovery checkpoints require their
original decoder, bound plan and authenticated parent from the preserved
snapshot. Do not pass a historical live recovery plan to the portable runner.
Keep a running historical job on its original source checkout.

## Removed APIs and local research

The original configurable model, trainer, dataset class and ONNX exporter,
and the earlier four-state implementation are removed. `StemgenRT58` defaults
to the current eight-state model and loads supported eight- and nine-state
checkpoints.
Four-state imports, graph downloads and legacy training configurations are no
longer supported. Existing users who require them should pin the preserved
pre-cleanup commit.

`research/` is ignored local experimentation, not an installation dependency.
Historical service inventories, process receipts, storage approval records,
trial queues, failed prototypes and superseded model implementations are not
included in the package or source distribution. Port selected future changes
with their required scientific code and focused validation; do not restore a
blanket research-tree sync.

## Evidence boundaries

Migration checks compare current model initialization, tensor states, audio
trajectories and gradients against the frozen implementation. Objective/data
checks preserve scientific arithmetic and addressing. CPU tests also exercise
complete Adam/EMA/RNG restoration, export and physical evaluation alignment.
They do not reproduce the archived hardware telemetry, requalify a GPU run,
measure plugin timing, or establish a new separation-quality result. The
historical 5.0 dB target remains a research target.

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

The product baseline is StemgenRT v0.6.1 at plugin commit
`990df8ee5baa621f042d4534dacc65afee0a96ce`, with deployed graph
`08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748`
and 4.455172594 dB on that same development panel. A new plugin PR proposes
exporting the research baseline; the source score is not the exported graph's
score, and the prior product remains the rollback reference.

The FP32/BF16 precision diagnostic, longer-attention and shared-mask variants
are closed. No new training run is selected by this publication.
