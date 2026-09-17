from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("c91_production", ROOT / "train_production.py")
assert SPEC is not None and SPEC.loader is not None
production = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = production
SPEC.loader.exec_module(production)


def tiny_model():
    config, _ = production.load_json(ROOT / "full_config.json")
    _, _, HSTasNet = production.resolve_source(config)
    model = HSTasNet(
        dim=16,
        small=True,
        stereo=True,
        num_basis=32,
        use_gru=True,
        use_branch_rnns=False,
        residual_source_softmax=True,
    )
    return HSTasNet, model


def test_frozen_manifest_matches_unique_production_inventory() -> None:
    config, _ = production.load_json(ROOT / "full_config.json")
    manifest_path = ROOT / "manifests/combined.manifest.json"
    _, tracks, file_hash, content_hash = production.load_corpus_manifest(
        manifest_path,
        expected_file_sha256="300b0bfbd835e2ce40c8832d10219a35941d6453392fa50688b236499375061a",
        config=config,
    )
    counts = {
        root_id: sum(track.root_id == root_id for track in tracks)
        for root_id in config["sampling"]["expected_track_counts"]
    }
    assert counts == {
        "musdb18hq_train": 83,
        "moisesdb_train": 218,
        "recordpool_best200_v1": 200,
    }
    assert len(tracks) == 501
    assert "moisesdb_train:Firefly - Up Close And Personal" not in {
        track.track_id for track in tracks
    }
    assert file_hash == "300b0bfbd835e2ce40c8832d10219a35941d6453392fa50688b236499375061a"
    assert content_hash == "a2414fb30ca5fc68fd37517da0f934bcbb0383e4c476ea9a667b2a83fcb703f9"


def test_run_lock_is_exclusive(tmp_path: Path) -> None:
    with production.exclusive_run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another trainer"):
            with production.exclusive_run_lock(tmp_path):
                raise AssertionError("unreachable")


def test_counter_addressed_crop_is_order_independent(tmp_path: Path) -> None:
    frames = 44_100 * 3
    time = np.arange(frames, dtype=np.float32) / 44_100
    files = {}
    for index, label in enumerate(production.FILE_NAMES):
        audio = np.stack(
            (
                np.sin(2 * np.pi * (110 + index) * time),
                np.cos(2 * np.pi * (130 + index) * time),
            ),
            axis=1,
        ).astype(np.float32) * 0.05
        path = tmp_path / f"{label}.wav"
        sf.write(path, audio, 44_100, subtype="PCM_16")
        info = sf.info(path)
        files[label] = production.FrozenAudioFile(
            path=path,
            sha256=production.sha256_file(path),
            size_bytes=path.stat().st_size,
            frames=info.frames,
            sample_rate=info.samplerate,
            channels=info.channels,
        )
    track = production.FrozenTrack(
        track_id="synthetic:track",
        root_id="synthetic",
        name="track",
        effective_frames=frames,
        files=files,
        vocal_active_seconds=(0, 1, 2),
    )
    dataset = production.CounterAddressedCropDataset(
        [track],
        root_weights={"synthetic": 1.0},
        seed=60,
        crop_samples=2048,
        vocal_active_probability=0.85,
        final_sample_index=100,
    )
    mixture_a, targets_a = dataset[37]
    _ = dataset[2]
    _ = dataset[99]
    mixture_b, targets_b = dataset[37]
    assert torch.equal(mixture_a, mixture_b)
    assert torch.equal(targets_a, targets_b)
    assert mixture_a.shape == (2, 2048)
    assert targets_a.shape == (4, 2, 2048)


def test_checkpoint_falls_back_after_current_corruption(tmp_path: Path) -> None:
    HSTasNet, model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer._autoresearch_base_lrs = (3e-4,)
    optimizer._autoresearch_step = 0
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    identity = "a" * 64

    first = production.make_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=0,
        contract_identity=identity,
        recent_losses=[],
    )
    first_written = production.write_verified_checkpoint(
        run_dir=tmp_path,
        payload=first,
        contract_identity=identity,
        HSTasNet=HSTasNet,
        learning_rate=3e-4,
    )
    optimizer._autoresearch_step = 1
    second = production.make_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=1,
        contract_identity=identity,
        recent_losses=[1.0],
    )
    second_written = production.write_verified_checkpoint(
        run_dir=tmp_path,
        payload=second,
        contract_identity=identity,
        HSTasNet=HSTasNet,
        learning_rate=3e-4,
    )

    current_path = Path(second_written["path"])
    with current_path.open("ab") as handle:
        handle.write(b"corruption")
    recovered, errors = production.recover_checkpoint(
        tmp_path,
        contract_identity=identity,
        HSTasNet=HSTasNet,
        learning_rate=3e-4,
    )
    assert errors
    assert recovered is not None and recovered["step"] == 0
    assert current_path.exists(), "recovery must be read-only"
    assert Path(first_written["path"]).exists()
    pointer = tmp_path / production.CHECKPOINT_DIRECTORY_NAME / production.CHECKPOINT_POINTER_NAME
    assert pointer.is_file()


def test_incomplete_new_generation_cannot_hide_valid_fallback(tmp_path: Path) -> None:
    HSTasNet, model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer._autoresearch_base_lrs = (3e-4,)
    optimizer._autoresearch_step = 0
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    identity = "c" * 64
    payload = production.make_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=0,
        contract_identity=identity,
        recent_losses=[],
    )
    production.write_verified_checkpoint(
        run_dir=tmp_path,
        payload=payload,
        contract_identity=identity,
        HSTasNet=HSTasNet,
        learning_rate=3e-4,
    )
    incomplete = (
        tmp_path
        / production.CHECKPOINT_DIRECTORY_NAME
        / f"step-{2:012d}-{'f' * 32}.pt"
    )
    incomplete.write_bytes(b"interrupted publish")
    recovered, errors = production.recover_checkpoint(
        tmp_path,
        contract_identity=identity,
        HSTasNet=HSTasNet,
        learning_rate=3e-4,
    )
    assert recovered is not None and recovered["step"] == 0
    assert any(record["path"] == str(incomplete) for record in errors)


def test_finalize_copy_is_transactional_and_streamable(tmp_path: Path) -> None:
    HSTasNet, model = tiny_model()
    raw_hash = production.tensor_state_sha256(model.state_dict())
    output = tmp_path / "deployment.pt"
    metadata = production.finalize_deployment_copy(
        model=model,
        path=output,
        gains=(1.0, 1.0, 0.8, 1.12),
        HSTasNet=HSTasNet,
        streaming_callbacks=2,
    )
    assert output.is_file()
    assert metadata["decoder_hann_baked"] is True
    assert metadata["streaming_shape"] == [4, 2, 512]
    assert production.model_is_raw(model)
    assert production.tensor_state_sha256(model.state_dict()) == raw_hash
    loaded = HSTasNet.init_and_load_from(output, strict=True)
    assert loaded.conv_decode.hann_window_baked
    assert torch.equal(
        loaded.output_source_scales,
        torch.tensor([0.5, 0.5, 0.4, 0.56]),
    )


def test_interrupted_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    config, _ = production.load_json(ROOT / "full_config.json")
    _, experiment, HSTasNet = production.resolve_source(config)
    experiment.DERANGED_PROJECTION_RAMP_START_STEP = 0
    experiment.DERANGED_PROJECTION_RAMP_END_STEP = 1
    experiment.LEARNING_RATE_DECAY_START_STEP = 2
    experiment.LEARNING_RATE_DECAY_END_STEP = 4
    experiment_config = experiment.ExperimentConfig(
        batch_size=2,
        crop_seconds=2048 / 44_100,
        precision="fp32",
        learning_rate=3e-4,
        grad_clip_norm=5.0,
        log_every=1,
    )

    def initialize(seed: int):
        production.configure_determinism(seed)
        model = HSTasNet(
            dim=16,
            small=True,
            stereo=True,
            num_basis=32,
            use_gru=True,
            use_branch_rnns=False,
            residual_source_softmax=True,
        )
        optimizer = experiment.build_optimizer(model=model, config=experiment_config)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return model, optimizer, scaler

    def batch(step: int):
        axis = torch.arange(2048, dtype=torch.float32)
        targets = []
        for batch_index in range(2):
            sources = []
            for source_index in range(4):
                wave = torch.sin(axis * (0.002 + 0.0001 * (step + batch_index + source_index)))
                sources.append(torch.stack((wave, wave * 0.9)) * (0.01 + 0.002 * source_index))
            targets.append(torch.stack(sources))
        target = torch.stack(targets)
        return target.sum(dim=1), target

    def advance(model, optimizer, scaler, first: int, stop: int):
        losses = []
        for step in range(first, stop):
            mixture, targets = batch(step)
            losses.append(
                experiment.train_step(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    mixture=mixture,
                    targets=targets,
                    config=experiment_config,
                    amp_dtype=None,
                )
            )
        return losses

    uninterrupted_model, uninterrupted_optimizer, uninterrupted_scaler = initialize(777)
    uninterrupted_losses = advance(
        uninterrupted_model,
        uninterrupted_optimizer,
        uninterrupted_scaler,
        0,
        4,
    )
    uninterrupted_model_hash = production.tensor_state_sha256(uninterrupted_model.state_dict())
    uninterrupted_optimizer_hash = production.structured_state_sha256(uninterrupted_optimizer.state_dict())
    uninterrupted_rng_hash = production.structured_state_sha256(production.capture_rng_state())

    interrupted_model, interrupted_optimizer, interrupted_scaler = initialize(777)
    interrupted_losses = advance(
        interrupted_model,
        interrupted_optimizer,
        interrupted_scaler,
        0,
        2,
    )
    identity = "b" * 64
    payload = production.make_checkpoint_payload(
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        scaler=interrupted_scaler,
        step=2,
        contract_identity=identity,
        recent_losses=interrupted_losses,
    )
    production.write_verified_checkpoint(
        run_dir=tmp_path,
        payload=payload,
        contract_identity=identity,
        HSTasNet=HSTasNet,
        learning_rate=3e-4,
    )

    resumed_model, resumed_optimizer, resumed_scaler = initialize(123456)
    recovered, errors = production.recover_checkpoint(
        tmp_path,
        contract_identity=identity,
        HSTasNet=HSTasNet,
        learning_rate=3e-4,
    )
    assert not errors and recovered is not None
    restored_step, restored_losses = production.restore_training_state(
        payload=recovered,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        device=torch.device("cpu"),
    )
    assert restored_step == 2
    resumed_losses = advance(
        resumed_model,
        resumed_optimizer,
        resumed_scaler,
        restored_step,
        4,
    )

    assert uninterrupted_losses == restored_losses + resumed_losses
    assert production.tensor_state_sha256(resumed_model.state_dict()) == uninterrupted_model_hash
    assert production.structured_state_sha256(resumed_optimizer.state_dict()) == uninterrupted_optimizer_hash
    assert production.structured_state_sha256(production.capture_rng_state()) == uninterrupted_rng_hash
