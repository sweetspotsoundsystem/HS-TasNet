<img src="./fig1.png" width="350px"></img>

## HS-TasNet

Implementation of [HS-TasNet](https://arxiv.org/abs/2402.17701), "Real-time Low-latency Music Source Separation using Hybrid Spectrogram-TasNet", proposed by the research team at L-Acoustics

## Pretrained streaming model

The released model separates **stereo 44.1 kHz audio** into **Drums, Bass,
Vocals and Other**. It combines spectrogram and waveform estimates with
recurrent state, using a 1024-sample analysis window, a 256-sample synthesis
frame and a 128-sample hop. Its output is delayed by one hop (2.90 ms).
[StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt) adds asynchronous
scheduling for DAW use: 256 samples / 5.80 ms total with a 128-sample host buffer.

Install from this checkout and download the self-contained ONNX weights:

```bash
python scripts/download_streaming_model.py
pip install -e '.[streaming]'
python examples/separate_streaming.py stereo-44100.wav stems --model models/hop128.onnx
```

The example writes four floating-point WAVs with the original length and sample
alignment. It requires stereo input already at 44.1 kHz and preserves its level.

```python
import soundfile as sf
from hs_tasnet.streaming import StreamingSeparator

audio, sample_rate = sf.read("stereo-44100.wav", dtype="float32", always_2d=True)
separator = StreamingSeparator("models/hop128.onnx", sample_rate=sample_rate)
stems = separator.separate(audio.T)  # [4, 2, samples]: drums, bass, vocals, other
```

For chunked integration, use `process_chunk`, `flush` and `reset`; see the
[model interface](models/README.md). The Python API runs synchronously on CPU
and allocates memory, so call it from a worker when integrating with playback.
The download is pinned to the same model used by StemgenRT and verified by
size and SHA-256. Weights are not included in the Python wheel.
The original PyTorch architecture and training API below remain available;
`HSTasNet()` constructs an untrained model and does not load these weights.

## Install

```bash
$ pip install HS-TasNet
```

## Usage

```python
import torch
from hs_tasnet import HSTasNet

model = HSTasNet()

audio = torch.randn(1, 2, 204800) # ~5 seconds of stereo

separated_audios, _ = model(audio)

assert separated_audios.shape == (1, 4, 2, 204800) # second dimension is the separated tracks
```

With the `Trainer`

```python
# model

from hs_tasnet import HSTasNet, Trainer

model = HSTasNet()

# trainer

trainer = Trainer(
    model,
    dataset = None,               # add your in-house Dataset
    concat_musdb_dataset = True,  # concat the musdb dataset automatically
    batch_size = 2,
    max_steps = 2,
    cpu = True,
)

trainer()

# after much training
# inferencing

model.sounddevice_stream(
    duration_seconds = 2,
    return_reduced_sources = [0, 2]
)

# or from the exponentially smoothed model (in the trainer)

trainer.ema_model.sounddevice_stream(...)

# or you can load from a specific checkpoint

model.load('./checkpoints/path.to.desired.ckpt.pt')
model.sounddevice_stream(...)

# to load an HS-TasNet from any of the saved checkpoints, without having to save its hyperparameters, just run

model = HSTasNet.init_and_load_from('./checkpoints/path.to.desired.ckpt.pt')

```

## Training script

First make sure dependencies are there by running

```shell
$ sh scripts/install.sh
```

Then make sure `uv` is installed

```shell
$ pip install uv
```

Finally run the following to train a newly initialized model on a small subset of MusDB, and make sure the loss goes down

```shell
$ uv run train.py
```

For distributed training, you just need to run `accelerate config` first, courtesy of [`accelerate` from 🤗](https://huggingface.co/docs/accelerate/en/index) but single machine is fine too

## Experiment tracking

To enable online experiment monitoring / tracking, you need to have `wandb` installed and logged in

```shell
$ pip install wandb && wandb login
```

Then

```shell
$ uv run train.py --use-wandb
```

To wipe the previous checkpoints and evaluated results, append `--clear-folders`


## Alternative RNNs

The architecture defaults to using PyTorch's `LSTM` (or `GRU`), but you can easily substitute it for any other module by passing an `rnn_klass` to the `HSTasNet` constructor, as long as it adheres to a specific interface (read `alternative_rnns.py`)

For example, to use the [minGRU](https://github.com/lucidrains/minGRU-pytorch) architecture:

```python
import torch
from hs_tasnet import HSTasNet
from hs_tasnet.alternative_rnns import minGRUWrapper

model = HSTasNet(rnn_klass = minGRUWrapper)

audio = torch.randn(1, 2, 204800)
separated_audios, _ = model(audio)
```

## Test

```shell
$ uv pip install '.[test]' --system
```

Then

```shell
$ pytest tests
```

## Sponsors

This open sourced work is sponsored by [Sweet Spot](https://github.com/sweetspotsoundsystem)

## Citations

```bibtex
@misc{venkatesh2024realtimelowlatencymusicsource,
    title    = {Real-time Low-latency Music Source Separation using Hybrid Spectrogram-TasNet},
    author   = {Satvik Venkatesh and Arthur Benilov and Philip Coleman and Frederic Roskam},
    year     = {2024},
    eprint   = {2402.17701},
    archivePrefix = {arXiv},
    primaryClass = {eess.AS},
    url      = {https://arxiv.org/abs/2402.17701},
}
```

```bibtex
@inproceedings{Feng2024WereRA,
    title   = {Were RNNs All We Needed?},
    author  = {Leo Feng and Frederick Tung and Mohamed Osama Ahmed and Yoshua Bengio and Hossein Hajimirsadegh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273025630}
}
```
