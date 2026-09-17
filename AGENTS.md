Follow CONTRIBUTING.md for every training or model publication. Keep the
native model, trainer, objectives, data pipeline, recovery, evaluation and
deployment source in sync in the same update. Include uncommitted research
sources when publishing an active implementation. Run scripts/sync_research.py
against the research and production-helper checkouts immediately before pushing.
Do not publish an inference-only update that omits matching training changes.
Do not edit source-bound files in a checkout with an active monitored run.
Use scripts/run_cpu_tests.py for local pytest runs within an allocation that
rejects symlinks. Removing pytest's temporary symlinks only after completion
does not prevent a concurrent training storage audit from rejecting them.
