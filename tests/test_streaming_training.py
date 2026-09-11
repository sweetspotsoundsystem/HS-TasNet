"""Exercise source masks, physical context, deterministic crops and exact resume."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from hs_tasnet.streaming_checkpoint import import_released_onnx, save_streaming_checkpoint, read_streaming_checkpoint
from hs_tasnet.streaming_data import build_manifest, load_manifest, require_disjoint, CounterAddressedCropDataset
from hs_tasnet.streaming_trainer import StreamingTrainConfig, learning_rate, train_streaming
from hs_tasnet.streaming_training import augment_training_batch, render_scored_context, streaming_objective

ROOT = Path(__file__).resolve().parents[1]


def make_audio(root, seed=5):
    root.mkdir(parents=True)
    generator = np.random.default_rng(seed)
    targets = generator.normal(0, .02, (4, 1024, 2)).astype(np.float32)
    for name, data in zip(("mixture", "drums", "bass", "vocals", "other"), (targets.sum(axis=0), *targets)):
        sf.write(root / (name + ".wav"), data, 44100, subtype="FLOAT")
    return targets


def test_controlled_views_preserve_physical_targets_and_rng():
    targets = torch.randn(4, 4, 2, 64, generator=torch.Generator().manual_seed(3)) * .02
    mixture = targets.sum(dim=1)
    torch.manual_seed(99)
    ordinary = augment_training_batch(mixture, targets, first_sample_index=0, enabled=False)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(99)
    controlled = augment_training_batch(mixture, targets, first_sample_index=0, enabled=True)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert controlled.view_codes == (0, 1, 2, 3)
    assert torch.count_nonzero(controlled.targets[0, 2]) == 0
    assert torch.count_nonzero(controlled.targets[1, [0, 1, 3]]) == 0
    assert torch.equal(controlled.targets[0, [0, 1, 3]], targets[0, [0, 1, 3]])
    assert torch.equal(controlled.targets[1, 2], targets[1, 2])
    assert torch.equal(controlled.mixture[:2], controlled.targets[:2].sum(dim=1))
    assert not bool(controlled.vocal_derangement[:2].any())
    for actual, expected in zip((controlled.mixture, controlled.targets, controlled.vocal_derangement),
                                (ordinary.mixture, ordinary.targets, ordinary.vocal_derangement)):
        assert torch.equal(actual[2:], expected[2:])


def test_teacher_gradient_excludes_controlled_rows_with_full_batch_divisor():
    raw = torch.zeros(4, 4, 2, 8, requires_grad=True)
    deployed = torch.zeros_like(raw, requires_grad=True)
    targets = torch.zeros_like(raw)
    teacher = torch.ones_like(raw)
    loss = streaming_objective(raw, deployed, targets, torch.zeros(4, dtype=torch.bool),
        view_codes=(0, 1, 2, 3), teacher_targets=teacher, teacher_weight=.5, deployed_truth_weight=0)
    assert loss.teacher_l1.item() == .5
    loss.total.backward()
    assert torch.count_nonzero(deployed.grad[:2]) == 0
    expected = -torch.tensor([2., 1., 1., 1.]) / (5 * 4 * 2 * 8) * .5
    torch.testing.assert_close(deployed.grad[2, :, 0, 0], expected, rtol=0, atol=0)


def test_controlled_deployed_loss_and_projection_cap():
    raw = torch.zeros(4, 4, 2, 8, requires_grad=True)
    deployed = torch.full_like(raw, 2., requires_grad=True)
    targets = torch.zeros_like(raw)
    terms = streaming_objective(raw, deployed, targets, torch.zeros(4, dtype=torch.bool), view_codes=(0, 1, 2, 3))
    assert terms.controlled_deployed_l1.item() == 1.
    assert terms.total.item() == .5
    terms.total.backward()
    assert torch.count_nonzero(deployed.grad[2:]) == 0
    signal = torch.linspace(-.2, .2, 32).reshape(1, 1, 1, 32).expand(4, 4, 2, 32).clone()
    terms = streaming_objective(signal * 2, signal, signal, torch.ones(4, dtype=torch.bool), view_codes=(2, 2, 2, 2))
    assert terms.projection.item() > 0
    assert terms.projection_contribution.item() <= .005 * signal.abs().mean().item() + 1e-8
    with pytest.raises(ValueError, match="teacher"):
        streaming_objective(raw, deployed, targets, torch.zeros(4, dtype=torch.bool),
                            view_codes=(0, 1, 2, 3), teacher_weight=.5)


def test_manifest_sampling_is_absolute_and_rejects_changed_audio(tmp_path):
    root = tmp_path / "audio"
    make_audio(root / "song-a", seed=4)
    make_audio(root / "song-b", seed=7)
    manifest = build_manifest({"train": root}, tmp_path / "manifest.json")
    tracks, weights, _ = load_manifest(manifest, crop_samples=512)
    dataset = CounterAddressedCropDataset(tracks, root_weights=weights, seed=60, crop_samples=512,
                                         vocal_active_probability=.85, final_sample_index=100)
    first = dataset[17]
    dataset[3]
    replay = dataset[17]
    assert all(torch.equal(actual, expected) for actual, expected in zip(first, replay))
    with pytest.raises(ValueError, match="overlap"):
        require_disjoint(tracks, tracks)
    with (root / "song-a/drums.wav").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(manifest, crop_samples=512)


def digest_tree(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


class AlignedTeacherFixture(torch.nn.Module):
    """Deterministic synthetic teacher used only to exercise the adapter contract."""
    def __init__(self):
        super().__init__()
        self.register_buffer("fractions", torch.tensor([.4, .3, .2, .1]))

    def forward(self, audio):
        return audio[:, None] * self.fractions[None, :, None, None]


@pytest.mark.parametrize("with_teacher", [False, True])
def test_real_model_adam_resume_replays_next_update(tmp_path, with_teacher):
    torch.set_num_threads(1)
    make_audio(tmp_path / "audio/song")
    manifest = build_manifest({"train": tmp_path / "audio"}, tmp_path / "manifest.json")
    config = StreamingTrainConfig(**json.loads((ROOT / "configs/hop128-cpu-smoke.json").read_text()))
    teacher = AlignedTeacherFixture() if with_teacher else None
    if with_teacher:
        config = replace(config, teacher_weight=.5)
    checkpoint = ROOT / "models/hop128.pt"
    if not checkpoint.exists():
        checkpoint = tmp_path / "initial.pt"
        save_streaming_checkpoint(import_released_onnx(ROOT / "models/hop128.onnx"), checkpoint)
    reference = train_streaming(config, manifest, tmp_path / "reference", checkpoint=checkpoint, teacher=teacher, device="cpu")
    reference_path = tmp_path / "reference/step-000002.pt"
    expected = read_streaming_checkpoint(reference_path)
    expected_optimizer = digest_tree(expected["training"]["optimizer"])
    expected_rng = digest_tree(expected["training"]["rng"])
    del expected
    reference_path.unlink()
    train_streaming(config, manifest, tmp_path / "first", checkpoint=checkpoint, teacher=teacher, device="cpu", stop_after=1)
    first_path = tmp_path / "first/step-000001.pt"
    resumed = train_streaming(config, manifest, tmp_path / "resumed", resume=first_path, teacher=teacher, device="cpu")
    actual_path = tmp_path / "resumed/step-000002.pt"
    actual = read_streaming_checkpoint(actual_path)
    try:
        assert actual["model_state_sha256"] == reference["model_state_sha256"] == resumed["model_state_sha256"]
        assert digest_tree(actual["training"]["optimizer"]) == expected_optimizer
        assert digest_tree(actual["training"]["rng"]) == expected_rng
        assert actual["training"]["next_sample_index"] == 8
        assert resumed["start_step"] == 1 and resumed["step"] == 2
        with pytest.raises(ValueError, match="differs"):
            train_streaming(replace(config, lr=2 * config.lr), manifest, tmp_path / "wrong-recipe",
                            resume=first_path, teacher=teacher, device="cpu")
        if teacher is not None:
            teacher.fractions.add_(.1)
            with pytest.raises(ValueError, match="teacher"):
                train_streaming(config, manifest, tmp_path / "wrong-teacher", resume=first_path,
                                teacher=teacher, device="cpu")
    finally:
        first_path.unlink()
        actual_path.unlink()
        if checkpoint.parent == tmp_path:
            checkpoint.unlink()


def test_learning_rate_endpoints_and_invalid_config():
    config = StreamingTrainConfig(lr=3e-5, min_lr=3e-6)
    assert learning_rate(0, config) == config.lr / config.warmup
    assert learning_rate(config.warmup - 1, config) == config.lr
    assert learning_rate(config.steps - 1, config) == config.min_lr
    with pytest.raises(ValueError):
        replace(config, batch_size=3).validate()


@pytest.mark.parametrize("device, expected_index", [("cuda", 2), ("cuda:3", 3)])
def test_cuda_setup_resolves_device_before_precision_check(monkeypatch, tmp_path, device, expected_index):
    import hs_tasnet.streaming_trainer as trainer

    class DatasetReached(Exception):
        pass

    current = {"index": 2}
    calls = []

    def set_device(selected):
        if selected.index is None:
            raise ValueError("CUDA setup requires an indexed device")
        current["index"] = selected.index
        calls.append(("device", selected.index))

    def supports_bf16():
        calls.append(("bf16", current["index"]))
        return True

    def memory_fraction(fraction, selected):
        calls.append(("memory", fraction, selected.index))

    def load_manifest(*args, **kwargs):
        raise DatasetReached

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supports_bf16)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", memory_fraction)
    monkeypatch.setattr(trainer, "load_manifest", load_manifest)
    with pytest.raises(DatasetReached):
        trainer.train_streaming(StreamingTrainConfig(), tmp_path / "manifest.json", tmp_path / "run", device=device)
    assert calls == [("device", expected_index), ("bf16", expected_index), ("memory", .75, expected_index)]
