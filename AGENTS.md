Maintain the current eight-state model and its matching training, loss, data,
checkpoint, evaluation and export code together under stemgenrt/. See
CONTRIBUTING.md. The supported geometry is 1024-sample analysis, 256-sample
synthesis and 128-sample hops. Do not reintroduce the removed configurable or
four-state models.

research/ is ignored local experimental work. Do not force-add it, copy whole
research checkouts into the package, or publish machine-specific records,
service snapshots, data, weights or private paths. Port an intentional feature
with its direct dependencies and tests. Historical source remains in git history.

Never modify a checkout used by an active monitored run. Perform source cleanup
and validation in a separate checkout. If tests share an allocation that rejects
symlinks, use scripts/run_cpu_tests.py with a fresh --basetemp; it suppresses
pytest's temporary symlinks before creation.
