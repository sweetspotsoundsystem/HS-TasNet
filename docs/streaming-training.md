# Training the released streaming architecture

`hs_tasnet.streaming_model.StreamingHSTasNet` implements the released stereo
44.1 kHz network: 1024-sample asymmetric spectral analysis, a learned waveform
branch, a two-layer GRU, 256-sample synthesis and a 128-sample hop. It has 21
trainable parameter tensors and four explicit streaming states. Its outputs
are ordered drums, bass, vocals, other.

The raw model predicts all four stems. Deployment preserves drums, bass and
vocals and sets other to the delayed mixture minus those three predictions.
`render()` exposes both estimates for training. `forward()` returns deployed
audio and state, delayed by 128 samples. `separate()` starts from zero state,
adds the final flush hop and returns audio aligned to the input sample count.
Host scheduling adds its own latency; see the [released interface](../models/README.md).

The older configurable `HSTasNet`/`Trainer` and `train.py` remain available.
Their checkpoints use a different architecture and format.

## Install and recover trainable weights

Use Python 3.12 and PyTorch 2.8 or later. CI verifies the portable CPU path with
PyTorch 2.8.0, NumPy 2.2.6, ONNX 1.19.1 and ONNX Runtime 1.26.0. Install a
CUDA-enabled PyTorch build for GPU training, then run from this checkout:

```bash
pip install -e '.[streaming,onnx]'
python scripts/download_streaming_model.py
python scripts/import_streaming_weights.py --onnx models/hop128.onnx --output models/hop128.pt
```

The importer accepts only the checksum-pinned release. It recovers every
parameter and buffer, reverses constant-folded linear transposes and checks
the exact original PyTorch state fingerprint:

```text
c204b0fcb9627ca7fecd287db42fb869a1ae6783a1bc24cf2d8864c3b4a565fb
```

This produces model weights for fresh fine-tuning; the ONNX graph contains no
historical optimizer state. The imported checkpoint is approximately 106 MB.

```python
import torch
from hs_tasnet import StreamingHSTasNet

model = StreamingHSTasNet.from_checkpoint("models/hop128.pt")
stems = model.separate(torch.zeros(1, 2, 44100))  # [1,4,2,44100]

state = model.initial_state(batch_size=1)
with torch.no_grad():
    output, state = model.forward_chunk(torch.zeros(1, 2, 128), state)
    # The first output hop represents pre-input history.
    final_output, state = model.flush(state)
```

Use a fresh initial state after a seek, gap or new stream. For long files, feed
bounded blocks whose length is divisible by 128 and carry all four states.
`separate()` processes its input as one block and is intended for short clips.

## Prepare training audio

Each song folder must contain `mixture.wav`, `drums.wav`, `bass.wav`,
`vocals.wav` and `other.wav`, all stereo 44.1 kHz and sample-aligned with equal
lengths. Preserve physical audio levels. The manifest builder neither resamples
nor normalizes the files. Use dedicated training folders; keep validation
songs in a separate root.

```text
/data/my-training-stems/
  song-a/
    mixture.wav
    drums.wav
    bass.wav
    vocals.wav
    other.wav
  song-b/
    ...
```

```bash
python scripts/build_streaming_manifest.py \
  --root train=/data/my-training-stems \
  --output data/train-manifest.json
```

Repeat `--root NAME=PATH` for multiple corpora. To control their sampling
probabilities, supply one `--weight NAME=NUMBER` for each root. Weights are
normalized across roots; songs within a selected root are sampled uniformly.
Manifest root order and stable track IDs are part of crop selection.

The manifest records file hashes, sizes, format and complete one-second vocal
activity anchors. Every training crop is determined by its data seed and
absolute sample index, independently of worker scheduling. Training validates
the files before and after the stage. The default 176128-sample crop requires
songs at least that long. Do not modify audio while a stage is running.

You can build a second manifest and pass `--validation-manifest` to reject
identical mixture paths or hashes across the two sets. This is an exact file
overlap check, not an acoustic duplicate detector or a validation evaluation.

## Fine-tune and resume

The supervised configuration starts a finite 250-update stage from the
released weights, using no external teacher:

```bash
python train_streaming.py \
  --manifest data/train-manifest.json \
  --config configs/hop128-supervised.json \
  --checkpoint models/hop128.pt \
  --output streaming-runs/finetune-001
```

It uses an effective batch of 16 accumulated from four microbatches of 4.
Each crop contains 88064 samples of detached history followed by 88064 scored
samples. A flush hop and alignment trim ensure the loss compares the same
physical sample times. CUDA BF16 applies only to learned dense kernels;
parameters, FFT, nonlinear masks, synthesis, losses and public state stay FP32.
Adam uses a warmup/cosine schedule, with one gradient clip at norm 5 and one
optimizer update per effective batch. The CUDA allocator is limited to 75%
of the selected device's memory; available memory still depends on other work.

The controlled-view cycle assigns one instrumental-only example, one
vocal-only example and two ordinary augmented examples per microbatch. The
ordinary augmentation includes stem subsets and vocal derangement. All views
receive raw four-stem L1 supervision, normalized with stem weights 2:1:1:1,
plus a capped deranged-vocal projection penalty. Controlled views also receive
deployed-output ground-truth L1 with weight 0.5. Masked terms are averaged over
the complete microbatch, including excluded rows. Setting `controlled_views`
to false retains the ordinary augmentation for every example.

`metrics.jsonl` records each update's losses, gradient norm, crop addresses and
input fingerprints. Checkpoints include model weights, Adam, RNG state, the
configuration, manifest hash and next absolute data index. A training
checkpoint is approximately 319 MB. Checkpoints accumulate at
`checkpoint_every` and at the requested endpoint; choose enough disk space for
the configured stage. Output directories and checkpoint files must be new.

To test an interruption at a planned boundary, keep the full schedule in the
config and stop early, then resume into a new directory:

```bash
python train_streaming.py \
  --manifest data/train-manifest.json --config configs/hop128-supervised.json \
  --checkpoint models/hop128.pt --stop-after 25 --output streaming-runs/first-25

python train_streaming.py \
  --manifest data/train-manifest.json --config configs/hop128-supervised.json \
  --resume streaming-runs/first-25/step-000025.pt --output streaming-runs/resumed
```

Resume requires the same configuration, manifest bytes, teacher identity and
CPU/CUDA device family. Keep the same software and hardware for deterministic
replay. CI verifies exact model, Adam and RNG equality after an interrupted
CPU update. For a new schedule, use `--checkpoint` to start with fresh Adam.
`--scratch` initializes untrained weights and needs a separate training plan.

For a small CPU integration check, use `--device cpu` with
`configs/hop128-cpu-smoke.json`. Its two updates and short synthetic-test-sized
context verify execution; they are not a quality training recipe. The portable
CUDA trainer has not received a separate end-to-end GPU qualification in CI.

## Historical teacher-assisted recipe

`configs/hop128-release-cleanup.json` records the settings for the released
model's final 250-update cleanup stage: learning rate 3e-5 decaying to 3e-6,
data indices starting at 972000, teacher weight 0.5 on ordinary views and
controlled deployed-output truth weight 0.5. Replaying that historical stage
also requires its earlier parent checkpoint, the original ordered 501-track
manifest/corpus and the frozen C91 teacher. Those are not bundled here.
Starting this config from the released weights performs an additional stage;
it does not reproduce the already completed historical stage.

For teacher-assisted training, supply `--teacher-factory your_module:make_teacher`.
The function must return a `torch.nn.Module` whose forward method maps a full
FP32 `[B,2,T]` crop to detached, physically aligned native-level FP32
`[B,4,2,T]` source estimates on the same device. The adapter must own any reset,
flush and delay compensation its teacher requires and should be stateless
between crops. The trainer freezes it, records its class and tensor-state hash,
and excludes controlled rows from teacher loss. Factory code and any behavior
outside its tensor state must also remain unchanged for replay. No C91 adapter
or teacher weights are provided. A positive teacher weight without a teacher
is rejected.

## Export and evaluate a fine-tuned checkpoint

```bash
python scripts/export_streaming_model.py \
  --checkpoint streaming-runs/finetune-001/step-000250.pt \
  --output models/finetuned.onnx
```

Export uses a copy of the FP32 CPU model and preserves its weights, training
flags and RNG. It checks native PyTorch, the export implementation and ONNX
Runtime across recurrent trajectories, nonzero initial states, partial
endpoints, impulses, DC/Nyquist signals and resets. It writes a self-contained
graph and `models/finetuned.verification.json` only after numerical checks pass.
The report includes the graph SHA-256, model fingerprint and measured errors.
Re-exporting the released state may change graph bytes without changing its
trained parameters.

Supply the new graph's digest explicitly when using the inference API:

```python
import json
from hs_tasnet.streaming import StreamingSeparator

report = json.load(open("models/finetuned.verification.json"))
separator = StreamingSeparator("models/finetuned.onnx", expected_sha256=report["onnx_sha256"])
```

The file example accepts the same digest through `--sha256`. Omitting it keeps
the released-model checksum pin. Export parity checks numerical correctness;
it does not establish separation quality, perceptual improvement or DAW timing.
Evaluate held-out songs and listen before selecting new weights for deployment.
