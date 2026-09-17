# Keeping research and deployment in sync

Publish the current model, training, loss, data preparation, recovery,
evaluation and export changes together. An inference update must include
the matching native training implementation and its dependencies. Preserve
older public APIs and identify baseline architectures explicitly.

When publishing from the research checkout, run this command in the public
checkout with the actual source, production-helper and active-plan paths:

```bash
python scripts/sync_research.py --write \
  --source-root /path/to/research-checkout \
  --production-root /path/to/production-helpers \
  --active-plan /path/to/research-checkout/path/to/active-plan.json
```

The script includes every direct-research source file, follows repository
imports and Python file loads, includes all repository Python sources bound
by the active plan, and copies the production helper sources and tests.
It preserves source bytes and records their hashes. It does not copy datasets,
weights, run logs or generated audio. Historical dynamically loaded Python
helpers that live beneath ignored run directories are included explicitly.
Review the manifest and stage those named source files with `git add -f`.

Review removed files explicitly before updating the snapshot. Re-run the
same command without `--write` immediately before pushing; this checks for
new, changed or missing research files, including uncommitted changes and
production helpers. Run `python scripts/sync_research.py` in CI to verify the
published files and the release binding. Do not modify source hashes merely
to silence a mismatch; publish and review the corresponding source changes.

Update the training guide and frozen recipe when the selected workflow changes.
Run `tests/test_current_research.py` for the current implementation, the
four-state training/export tests for the retained baseline, and the streaming
inference parity tests for the released graph. Changes to the original model
or export helpers also require `tests/test_export_onnx.py`. CPU checks are
separate from the monitored GPU qualification and saved-checkpoint quality
gates. Keep model selection and experimental results accurate.

Work in a separate checkout while a source-bound run is active. Preserve its
source files, original frozen inputs, rollback models and monitoring.
For local tests sharing a monitored artifact allocation, use
`python scripts/run_cpu_tests.py ... --basetemp=/allocated/path/to/new-test-directory`.
This suppresses pytest's optional `current` symlinks throughout execution;
cleaning them after the tests is too late for a concurrent storage audit.
Keep all generated test checkpoints inside the allocation and remove the
finished temporary directory after reviewing the result.
