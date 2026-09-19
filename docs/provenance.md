# Provenance and migration

The complete research publication before cleanup is preserved in commit
[`a70da5ead2babb01f5d70d04b85538150ad5a41e`](https://github.com/sweetspotsoundsystem/HS-TasNet/tree/a70da5ead2babb01f5d70d04b85538150ad5a41e).
It includes the original source inventory, frozen four-second training plan,
its two recovery predecessors, historical experiment implementations, native
checkpoint loaders, packed XOR recovery decoder and monitoring launchers.
The cleanup leaves that commit in the PR's history. No additional public
archive tag is required.

## Maintained implementation

The current eight-state architecture was extracted without changing parameter
names, buffer names, state shapes, metadata/schema identifiers or neural and
synthesis arithmetic. Its full implementation now lives in the installed
`hs_tasnet` package. The historical class inheritance chain is replaced by one
current model and small DSP helpers.

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

The original configurable `HSTasNet`, `Trainer`, `MusDB18HQ`, original ONNX
exporter, and the earlier four-state `StreamingHSTasNet` implementation are
removed. The public `StreamingHSTasNet` name now refers to the current model.
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
