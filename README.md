# HS-TasNet

Stereo streaming music separation into **drums, bass, vocals and other**.
The current model combines spectral and waveform branches, causal attention,
and recurrent branch memories. It uses **1024-sample analysis, 256-sample
synthesis, 128-sample hops and eight explicit FP32 states** at 44.1 kHz.
The graph delays audio by 128 samples. StemgenRT's additional 128-sample worker
queue gives 256 samples / 5.80 ms total algorithmic latency.

## Run the released model

```bash
python -m pip install -e '.[streaming]'
python scripts/download_streaming_model.py
python examples/separate_streaming.py --help
```

```python
import numpy as np
from hs_tasnet import StreamingSeparator

separator = StreamingSeparator("models/hop128.onnx")
audio = np.zeros((2, 44100), dtype=np.float32)
stems = separator.separate(audio)  # [4, 2, samples], aligned to the input
```

The download is pinned by size and SHA-256 to the integer graph shared with
[StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt). ONNX Runtime
1.26.0 runs on one CPU thread with KleidiAI disabled, matching that release's
runtime settings. This Python API allocates memory and runs synchronously; use
it from a worker for playback integration. See [model interface](models/README.md).

## Native model, training and export

Install the desired CPU or CUDA build of PyTorch 2.8.0 first, then:

```bash
python -m pip install -e '.[training,onnx,test]'
```

```python
import torch
from hs_tasnet import StreamingHSTasNet, render_scored_context

model = StreamingHSTasNet()  # Untrained weights; this does not load the release.
audio = torch.randn(1, 2, 768) * .02
scored = render_scored_context(model, audio, warmup_samples=256, carry_state=True)
scored.raw.square().mean().backward()
```

The maintained package includes deterministic crop/pitch/remix augmentation,
the whole-group weighted source-view objective, Adam and EMA, lossless complete
recovery, native/ONNX evaluation and verified export. Read the
[training and evaluation guide](docs/training.md) for checkpoint requirements,
portable manifests, the four-second recipe and commands. Native FP32 training
weights cannot be reconstructed losslessly from the released integer graph;
provide an authenticated native checkpoint or explicitly start from scratch.

## Supported source

| Module | Responsibility |
| --- | --- |
| `hs_tasnet.model` | Current native model, states and detached context |
| `hs_tasnet.data` | Portable manifests and deterministic training augmentation |
| `hs_tasnet.losses` | Whole-group objectives and one Adam/EMA update |
| `hs_tasnet.checkpoint` | Native checkpoint authentication and complete recovery |
| `hs_tasnet.trainer` | Portable finite training and resume |
| `hs_tasnet.evaluation` | Physical alignment and per-stem metrics |
| `hs_tasnet.export` | Current fixed-geometry ONNX export |
| `hs_tasnet.streaming` | Released and checksum-pinned custom ONNX inference |

Version 0.4 removes the original configurable `HSTasNet`, its trainer, and the
earlier four-state model. `StreamingHSTasNet` now denotes only the current
eight-state architecture. `research/` is ignored local experimentation; it is
not required for imports, tests or distributions. The complete earlier source
snapshot remains in git history; see [provenance and migration](docs/provenance.md).

The implementation builds on [HS-TasNet](https://arxiv.org/abs/2402.17701) and
[Phil Wang's implementation](https://github.com/lucidrains/hs-tasnet).
