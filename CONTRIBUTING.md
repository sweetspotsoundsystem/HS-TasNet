# Maintained source and research

The supported implementation is the eight-state streaming model in
`hs_tasnet/model.py`. Keep its data pipeline, weighted objective, Adam/EMA
checkpointing, evaluation and deployment transformations in the same review.
The model has fixed 1024-sample analysis, 256-sample synthesis and 128-sample
hops. Its parameter names, fixed buffers, state interface and historical native
checkpoint schema are part of the compatibility contract.

`research/` and `runs/` are ignored local work. The previous blanket research
sync script has been removed. Move only an intentionally selected implementation
and its necessary reusable dependencies into `hs_tasnet/`, together with tests
and user documentation. Do not force-add ignored experiments. Keep historical
machine supervision, temporary probes, operator receipts, storage approvals,
corpus audio and checkpoints out of the maintained source and distributions.

The complete pre-cleanup snapshot remains at commit
`a70da5ead2babb01f5d70d04b85538150ad5a41e`. See [provenance](docs/provenance.md).
Preserve that history instead of rewriting frozen experiment hashes. Do not
edit the checkout of an active monitored run or try to resume its old plan with
new package code. The portable trainer starts a new run or restores its own
versioned complete checkpoints.

## Validation

Install CPU PyTorch or the appropriate CUDA build, then:

```bash
python -m pip install -e '.[streaming,training,onnx,test]' build
python scripts/download_streaming_model.py
python scripts/run_cpu_tests.py -q --basetemp=/tmp/hs-tasnet-tests
python -m build
```

Current tests cover model states and detached warmup, scientific scalar and
output-gradient references, deterministic addressing/remixes, complete exact
Adam/EMA/RNG restore, export trajectory parity, physical evaluation alignment,
released graph waveform parity, and package boundaries. Update relevant tests
when changing these contracts. Source extraction must preserve arithmetic and
must be compared against the frozen implementation before deleting its local
reference.

CPU tests do not establish separation quality, CUDA resource use or real-time
host performance. Changes to training precision, data recipes or deployment
arithmetic need their respective quality and hardware qualification.

## Comparing a research port

Use `scripts/sync_research.py` to record the live source inventory and the
portable working files before publishing an intentional port. This comparison
command replaces the historical copy operation; it does not restore the
research snapshot. Supply the actual active plan and production helpers:

```bash
python scripts/sync_research.py --write \
  --source-root /path/to/live-research \
  --production-root /path/to/production-helpers \
  --active-plan /path/to/live-research/active-plan.json \
  --manifest /allocated/local-review/source-comparison.json
```

Inspect the local manifest, run the current research integration checks and
portable training/export/inference suite, and repeat the command without
`--write` immediately before pushing. It includes uncommitted source bytes and
rejects changes to plan-bound code. Keep the manifest and machine-specific
evidence outside the public checkout. Source hashes establish identity;
independent numerical comparisons establish that the selected port preserves
the research behavior. Neither substitutes for saved-checkpoint quality or
hardware qualification.

New evaluation or recovery helpers may live outside `research/direct/` and may
not be inputs to the active training plan. Include each intentionally selected
file with `--extra-source research/path/to/helper.py` when writing and verifying
the comparison. Relative paths use `--source-root`. These files are recorded as
unbound inputs; the command still verifies every included plan-bound source
against its frozen digest. Do not edit the active plan to add later review code.
