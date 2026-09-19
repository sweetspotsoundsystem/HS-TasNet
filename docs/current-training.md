# Current eight-state training and deployment

The current training model is
`research.direct.latency58_branch_memory.Latency58BranchMemoryModel`. It has
32,775,840 learned parameters in 40 tensors, causal attention, two additional
branch GRUs, and eight explicit FP32 states. Its state names and shapes match
the current StemgenRT ONNX release. Graph alignment is 128 samples; the host
adds 128, giving 256 samples at 44.1 kHz.

This checkout includes the complete `research/direct` source tree, its imported
repository dependencies, the production corpus helpers, and the active frozen
training plan. `research/source-manifest.json` records a SHA-256 for every
published source/config file, including changes that were uncommitted in the
research checkout. It also binds the current released ONNX digest.

The research workflow is available from a **source checkout or source
distribution**, using Python 3.12 and PyTorch 2.8. The inference wheel remains
small. The [four-state training guide](streaming-training.md) describes the
older baseline and its portable fine-tuning CLI.

## Source map

| Component | Implementation |
| --- | --- |
| Eight-state model and causal memory | `research/direct/latency58_branch_memory.py` |
| Active training loop | `research/direct/train_latency58_four_second_shared.py` |
| Current frozen recipe | `research/recipes/active-plan.json` |
| Crop addressing, pitch/tempo and remix | `latency58_four_second_data.py`, `latency58_recorded301_data.py`, `latency58_remix_augmentation.py` in `research/direct` |
| Whole-group loss and weighted source views | `latency58_logical_batch_loss.py`, `latency58_weighted_vocal_auxiliary.py`, `latency58_weighted_vocal_canonical.py` |
| Shared saved BF16 GRU weights | `research/direct/latency58_bf16_saved_gru_weights.py` |
| Adam, EMA and exact restore | `latency58_branch_memory_checkpoint.py`, `latency58_branch_ema.py`, `latency58_grouped_vocal_recovery.py` |
| Packed recovery and atomic publication | `latency58_lossless_recovery_codec_v3.py`, `latency58_four_second_recovery_files.py` |
| Monitored launch and adoption | `run_latency58_four_second_shared.py`, `watch_latency58_four_second.py`, `attach_latency58_four_second.py` |
| Monitored resume after a storage abort | `resume_latency58_four_second_shared.py`, `supervise_latency58_four_second_recovery.py` |
| Saved raw/EMA evaluation | `evaluate_latency58_four_second_shared_memory.py`, `latency58_four_second_shared_evaluation.py`, `run_latency58_four_second_shared_quality.py` |
| Streaming evaluation and metrics | `research/direct/latency58_evaluate.py`, `research/metrics.py` |
| FP32 export | `latency58_branch_onnx.py`, `latency58_branch_onnx_export.py`, root `export_onnx.py` |
| Integer deployment and independent reference | `latency58_branch_int8.py`, `latency58_branch_int8_verify.py`, `latency58_attention_qkv_int8.py`, `latency58_attention_qkv_int8_reference.py` |
| Corpus manifests, deterministic loader, original trainer | `research/production/train_production.py`, `research/production/prepare_corpus.py` |

Unqualified prototypes and historical stages retain their original names,
comments and acceptance checks. The active plan identifies the current trainer
and its exact dependencies. Publishing a prototype does not select it for
training or deployment.

## Current recipe

The active `branch-four-second-015` stage starts from its authenticated EMA
parent with fresh Adam. It trains all 40 learned tensors for 2,000 updates,
using 16 ordinary examples and a joint pair of instrumental-only/vocals-only
views. Crops have 88,064 samples of detached warmup and 176,512 scored samples,
for 264,576 samples total. Each scored region has four complete one-second
metric windows. The ordinary batch is rendered together; the auxiliary pair
is rendered together.

The objective combines waveform/spectral reconstruction and a direct SDR term
with weight 0.2. Whole-group activity counts and denominators are preserved
when replaying output gradients. The auxiliary group coefficient is 0.1;
instrumental and vocal contributions have multipliers 1 and 0.25, without
renormalization. One gradient clip, Adam update and EMA update follow the
complete ordinary and auxiliary groups. EMA decay is 0.995. The learning rate
warms up for 100 updates to 3e-5 and decays to 3e-6. This stage uses ground-truth
stems without an online teacher.

Learned CUDA kernels use BF16; parameters, public states, FFT, synthesis and
losses remain FP32. The shared-weight hook checks equal saved GRU transposes
and shares their storage without changing forward or backward arithmetic.
Recovery retains raw weights, Adam moments, EMA, CPU/CUDA/Python/NumPy RNG,
sample addresses, the journal and schedule. Byte-plane compression and XOR
references are lossless and are checked against the authenticated parent.

The active plan resumes at update 1,900 after the event-monitor worker exited
at update 1,939. The interrupted runs and saved checkpoints are retained.
The final 100 updates restore raw weights, Adam, EMA and all RNG streams;
the 39 unsaved updates are replayed. The original training recipe, 2,000-update
schedule and latency are unchanged. Checkpoint inspection, complete Windows
event coverage and a fresh monitored idle check precede the restart. Missing
historical process-exit receipts remain explicitly unknown.

`resume_latency58_four_second_shared_v2.py` prepares and launches this recovery,
and `supervise_latency58_four_second_recovery_v2.py` records its actual exit.
Both predecessor plans accompany the active recipe. Local tests sharing the
monitored allocation use `scripts/run_cpu_tests.py` to suppress pytest's
temporary symlinks before they are created.

After the resumed run completes, use
`python -m research.direct.run_latency58_four_second_recovery_quality_v2 --training-plan /path/to/plan-recovery002.json`.
The saved-state audit follows the original, first-recovery and second-recovery
receipt segments under their respective plan hashes, checks every retained
parent save and the complete journal, and requires a successfully closed final
monitor. Scoring, rendering, per-stem comparisons and selection decisions are
unchanged. The frozen plan's original controller fields retain their
historical values. The optional
`supervise_latency58_four_second_quality_v2` wrapper waits for the actual
training completion receipt and runs this CPU evaluation automatically.
The frozen workflow retains its historical v0.4 deployed baseline. The separate
`review_latency58_four_second_current_release` supplement authenticates the
v0.6.1 Git LFS model identity and recorded quality measurements, then compares
both completed candidates against that graph on all 56 track/stem cells and
840 source-view windows. It preserves the original comparisons and performs
no inference or model selection.

## Run and inspect

Install a suitable PyTorch build first, then from this checkout:

```bash
python -m pip install -e '.[streaming,onnx,test]' 'torchcodec==0.6.0'
python scripts/download_streaming_model.py
python scripts/sync_research.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/run_cpu_tests.py tests/test_current_research.py tests/test_recovery_lineage.py -q
python -m research.direct.train_latency58_four_second_shared --help
```

Constructing the native model initializes **untrained weights**:

```python
import torch
from research.direct.latency58_branch_memory import Latency58BranchMemoryModel
from research.direct.latency58_branch_memory_context import render_scored_context

model = Latency58BranchMemoryModel().train()
model.training_precision = "fp32"  # CUDA training uses the qualified BF16 policy.
audio = torch.randn(1, 2, 768) * 0.02
output = render_scored_context(model, audio, warmup_samples=256, carry_state=True)
output.raw.square().mean().backward()  # Small execution example, not the training loss.
```

For a saved native checkpoint, use
`latency58_branch_memory_checkpoint.load_model({"path": path, "sha256": digest})`.
It validates the architecture, all tensor bytes, fixed buffers and lineage.
Packed endpoints use `latency58_four_second_recovery_files.load_inference`
with an explicit `raw` or `ema` role and their frozen plan. The current integer
ONNX graph does not contain the original FP32 parameters or Adam state, and
cannot reconstruct them losslessly. Native checkpoints and licensed training
audio are not bundled with this source publication.

The original experiment launchers remain **bound to their recorded environment**.
The frozen plan contains original absolute paths, source hashes, corpus and
parent identities, monitor receipts and storage allocations. A plain clone
cannot replay that historical run without those inputs. The production helper
was originally installed at the `PRODUCTION` path in `train_latency58.py`; its
source and configuration are now included under `research/production`.
FFmpeg with the Rubber Band filter is required for the pitch/tempo path.
The original GPU supervision also uses Windows event telemetry from WSL.

For that prepared environment, the controller interface is:

```bash
python -m research.direct.run_latency58_four_second_shared \
  --plan /path/to/frozen-plan.json \
  --previous-execution /path/to/completed-monitor-receipt.json --stage resource
# After the resource qualification and prerequisites pass:
python -m research.direct.run_latency58_four_second_shared \
  --plan /path/to/frozen-plan.json \
  --previous-execution /path/to/completed-resource-receipt.json --stage production
python -m research.direct.run_latency58_four_second_shared_quality \
  --training-plan /path/to/frozen-plan.json
```

Prepare a new plan and the corresponding resource/restart qualifications for
a different environment or recipe. Preserve the scientific settings and
rebind paths and prerequisites explicitly; do not bypass the historical
validator or point new work at an existing run directory.

## Retained inference experiment

`profile_latency58_current_operators.py` profiles the unchanged fused-QKV graph
with the released ONNX Runtime 1.26.0. The subsequent
`check_latency58_small_projections_fp32.py` experiment kept all trained weights
and changed only three small projection products to FP32. Its independent
reference failed the existing audio parity limit in the multitone/noise/silence
fixture with both optimization settings (maximum error about 3.07e-5 versus
1e-5). The experiment is rejected; no speed or quality claim is made. The
shipped graph retains its existing arithmetic. These source files preserve the
experiment and its rejection logic for reproducibility.

## Evaluation and release checks

The included evaluation protocol retains the 14-track, 28-excerpt validation
panel, physical sample alignment, one-second windows, native audio levels and
per-stem metrics. `research/manifests/valid.json` identifies the panel;
`research/eval_config.json` preserves its metric/excerpt settings. The
hop128 evaluator uses its own 128-sample graph alignment; the older evaluator's
512-sample alignment field does not change the current model delay. Selecting
a candidate requires evaluation of saved weights and review of per-stem
regressions. The current 5.0 dB research target is not a claimed result.

CI checks the native model against the **actual released ONNX interface**,
carried state and reset behavior, mixture reconstruction, detached warmup
gradients, independent weighted-loss scalars and derivatives, full native
Adam/EMA/RNG restore through the lossless codec, and FP32 ONNX trajectory
parity. Existing inference tests independently check the released integer
graph. CPU integration checks do not requalify GPU memory, hardware monitoring,
separation quality, or real-time performance on a host.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the required source sync before
every model or training publication.
