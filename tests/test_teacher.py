"""Teacher math and CPU-provider contracts without downloading model weights."""
import copy
import hashlib
import random

import numpy as np
import pytest
import torch

from stemgenrt import losses, teacher
from stemgenrt._losses.teacher import contribution


@pytest.fixture(autouse=True)
def cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def provider(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'teacher.th'
    checkpoint.write_bytes(b'unit fixture, not a neural checkpoint')
    monkeypatch.setattr(teacher, 'CHECKPOINT_BYTES', checkpoint.stat().st_size)
    monkeypatch.setattr(teacher, 'CHECKPOINT_SHA256', hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    return teacher.CPUTrainingTeacher(checkpoint, coefficient=1.)


def test_zero_teacher_requires_no_weights_backend_or_audio(monkeypatch):
    monkeypatch.setattr(teacher, '_load_model', lambda *_: pytest.fail('Disabled teacher loaded a model'))
    monkeypatch.setattr(teacher, '_sha', lambda *_: pytest.fail('Disabled teacher read a file'))
    assert teacher.CPUTrainingTeacher(object(), coefficient=0.).render(object()) is None
    assert teacher.specification(0.) is None
    assert losses.policy() == losses.policy(teacher_coefficient=0.)


@pytest.mark.parametrize('coefficient', [-1., float('nan'), float('inf'), True, 1, '1'])
def test_invalid_teacher_coefficient(coefficient):
    with pytest.raises(ValueError): teacher.specification(coefficient)


@pytest.mark.parametrize('fail', [False, True])
def test_loading_restores_all_cpu_rng_even_on_failure(provider, monkeypatch, fail):
    def load(_):
        random.random(); np.random.rand(); torch.rand(3)
        if fail: raise RuntimeError('loader failure')
        return object(), object()
    monkeypatch.setattr(teacher, '_load_model', load)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    initialized = torch.cuda.is_initialized()
    if fail:
        with pytest.raises(RuntimeError, match='loader failure'): provider._load()
        assert provider.model is None
    else:
        provider._load()
        monkeypatch.setattr(teacher, '_load_model', lambda *_: pytest.fail('Teacher loaded twice'))
        provider._load()
    assert random.getstate() == states[0]
    after = np.random.get_state()
    assert after[0] == states[1][0] and after[2:] == states[1][2:]
    np.testing.assert_array_equal(after[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert torch.cuda.is_initialized() == initialized


def test_checkpoint_replacement_before_loading_is_rejected(provider):
    provider.checkpoint.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='changed before loading'): provider._load()


def test_cpu_provider_uses_full_context_native_order_and_residual_other(provider, monkeypatch):
    model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    calls = []
    def apply(candidate, normalized, **kwargs):
        assert candidate is model
        assert kwargs == dict(shifts=0, split=False, device='cpu', num_workers=0)
        calls.append(normalized.clone())
        # Demucs order is drums, bass, other, vocals.
        return normalized[:, None] * normalized.new_tensor([1., 2., 3., 4.])[None, :, None, None]
    monkeypatch.setattr(teacher, '_load_model', lambda _: (model, apply))
    mixture = torch.linspace(-.2, .3, teacher.CROP_SAMPLES).repeat(1, 2, 1)
    mixture[..., :teacher.WARMUP_SAMPLES] += .4
    original = mixture.clone()
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
        targets = provider.render(mixture)
    assert len(calls) == 1 and calls[0].shape == (1, 2, teacher.CROP_SAMPLES)
    assert torch.equal(mixture, original)
    assert targets.shape == (1, 4, 2, teacher.SCORED_SAMPLES)
    assert targets.dtype == torch.float32 and not targets.requires_grad and not torch.is_inference(targets)
    mean = mixture.mean(1).mean(-1)[:, None, None]
    expected = torch.stack((mixture, 2 * mixture - mean, 4 * mixture - 3 * mean,
                            -6 * mixture + 4 * mean), dim=1)[..., teacher.WARMUP_SAMPLES:]
    torch.testing.assert_close(targets, expected, atol=3e-7, rtol=1e-5)
    torch.testing.assert_close(targets.sum(1), mixture[..., teacher.WARMUP_SAMPLES:], atol=2e-7, rtol=0)


@pytest.mark.parametrize('case', ['short', 'dtype', 'grad', 'inference', 'nan'])
def test_provider_rejects_invalid_input_before_loading(provider, monkeypatch, case):
    monkeypatch.setattr(teacher, '_load_model', lambda *_: pytest.fail('Invalid audio reached teacher'))
    mixture = torch.zeros(1, 2, teacher.CROP_SAMPLES)
    if case == 'short': mixture = mixture[..., :-1]
    elif case == 'dtype': mixture = mixture.double()
    elif case == 'grad': mixture.requires_grad_()
    elif case == 'inference':
        with torch.inference_mode(): mixture = mixture.clone()
    else: mixture[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='fixed normal FP32 CPU'): provider.render(mixture)


def test_teacher_scalar_and_derivative_match_independent_numpy_reference():
    generator = torch.Generator().manual_seed(821)
    samples = 2 * losses.WINDOW + 128
    truth = .02 * torch.randn(2, 4, 2, samples, generator=generator)
    truth[0, 1, :, :losses.WINDOW] = 0
    truth[:, 2] = 0
    mixture = truth.sum(1)
    target = .03 * torch.randn(truth.shape, generator=generator)
    estimate = (.02 * torch.randn(truth.shape, generator=generator)).requires_grad_()
    reduction = losses.prepare_reduction(truth)
    actual = contribution(estimate, target, truth, mixture, reduction)
    t, m, ref, pred = [x.detach().numpy().astype(np.float64) for x in (truth, mixture, target, estimate)]
    expected_gradient = np.zeros_like(pred)
    cells = [[] for _ in range(4)]
    for stem in range(4):
        for batch in range(2):
            for start in (0, losses.WINDOW):
                section = slice(start, start + losses.WINDOW)
                power = np.mean(t[batch, stem, :, section] ** 2)
                if power > 1e-5:
                    scale = max(np.sqrt(power), .1 * np.sqrt(np.mean(m[batch, :, section] ** 2)), 1e-3)
                    error = pred[batch, stem, :, section] - ref[batch, stem, :, section]
                    cells[stem].append((batch, section, scale, np.mean(np.abs(error)) / scale))
    eligible_stems = sum(bool(c) for c in cells)
    expected = sum(sum(c[3] for c in stem) / len(stem) for stem in cells if stem) / eligible_stems
    for stem, entries in enumerate(cells):
        for batch, section, scale, _ in entries:
            expected_gradient[batch, stem, :, section] = np.sign(
                pred[batch, stem, :, section] - ref[batch, stem, :, section]) / (
                2 * losses.WINDOW * scale * len(entries) * eligible_stems)
    assert actual.total.item() == pytest.approx(expected, abs=2e-7)
    gradient, = torch.autograd.grad(actual.total, estimate, retain_graph=True)
    np.testing.assert_allclose(gradient.numpy(), expected_gradient, atol=1e-10, rtol=2e-6)
    split = sum(contribution(estimate[i:i+1], target[i:i+1], truth[i:i+1], mixture[i:i+1], reduction).total for i in range(2))
    torch.testing.assert_close(split, actual.total, atol=2e-7, rtol=0)
    split_gradient, = torch.autograd.grad(split, estimate)
    torch.testing.assert_close(split_gradient, gradient, atol=1e-10, rtol=2e-6)
    assert not torch.count_nonzero(gradient[:, 2]) and not torch.count_nonzero(gradient[..., 2 * losses.WINDOW:])


@pytest.mark.parametrize('case', ['valid', 'coefficient', 'teacher_hash', 'flag', 'disabled', 'missing_spec'])
def test_portable_teacher_checkpoint_identity(case):
    spec = teacher.specification(1.)
    model = torch.nn.Linear(1, 1)
    teacher.attach(model, spec)
    config = {'teacher_coefficient': 1., 'teacher_supervision': copy.deepcopy(spec)}
    if case == 'coefficient': config['teacher_coefficient'] = .5
    elif case == 'teacher_hash': config['teacher_supervision']['checkpoint']['sha256'] = '0' * 64
    elif case == 'flag': model.provenance[teacher.FLAGS[0]] = False
    elif case == 'disabled': config['teacher_coefficient'] = 0.
    elif case == 'missing_spec': config.pop('teacher_supervision')
    if case == 'valid': teacher.validate_checkpoint(config, model.provenance)
    else:
        with pytest.raises(ValueError): teacher.validate_checkpoint(config, model.provenance)
