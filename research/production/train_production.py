#!/usr/bin/env python3
"""Crash-safe long-horizon trainer for the exact HS-TasNet c91 recipe.

This program deliberately lives outside the sealed autoresearch checkout.  It
imports c91 only after verifying its commit and critical file hashes, consumes
only a frozen corpus manifest, keeps every resume checkpoint uncalibrated and
decoder-unbaked, and creates calibrated/baked deployment copies separately.
"""

from __future__ import annotations

import os

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG) != _CUBLAS_WORKSPACE_CONFIG:
    raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be {_CUBLAS_WORKSPACE_CONFIG!r}")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hs-tasnet-matplotlib")

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
import pickle
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
import fcntl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = SCRIPT_PATH.with_name("full_config.json")
CHECKPOINT_SCHEMA = 1
CHECKPOINT_DIRECTORY_NAME = "checkpoints"
CHECKPOINT_POINTER_NAME = "latest.json"
CHECKPOINT_RETAINED_GENERATIONS = 2
SOURCE_NAMES = ("drums", "bass", "vocals", "other")
FILE_NAMES = ("mixture", *SOURCE_NAMES)
STOP_REQUESTED = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def load_json(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"missing JSON file: {path}")
    file_hash = sha256_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value, file_hash


def resolve_source(config: Mapping[str, Any]) -> tuple[Path, Any, Any]:
    source = config["source"]
    repo = Path(source["repo"]).expanduser().resolve()
    require(repo.is_dir(), f"c91 source checkout is missing: {repo}")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(commit == source["commit"], f"c91 commit changed: {commit}")
    require(tree == source["tree"], f"c91 tree changed: {tree}")
    tracked_changes = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(not tracked_changes.strip(), f"c91 checkout has tracked worktree/index changes:\n{tracked_changes}")
    for relative, expected in source["files"].items():
        actual = sha256_file(repo / relative)
        require(actual == expected, f"c91 source hash changed for {relative}: {actual}")

    # The package initializer imports trainer.py, so checking only the model
    # file would still execute mutable worktree code.  Bind every tracked
    # Python module in the package to the exact commit before importing it.
    tracked_package_files = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD", "--", "hs_tasnet"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_python = {relative for relative in tracked_package_files if relative.endswith(".py")}
    on_disk_python = {
        path.relative_to(repo).as_posix()
        for path in (repo / "hs_tasnet").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    require(on_disk_python == tracked_python, f"c91 package Python inventory changed: extra={sorted(on_disk_python - tracked_python)}, missing={sorted(tracked_python - on_disk_python)}")
    for relative in sorted(tracked_python):
        committed = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        require(hashlib.sha256(committed).hexdigest() == sha256_file(repo / relative), f"c91 tracked module differs from HEAD: {relative}")
    research_init = repo / "research/__init__.py"
    if research_init.exists():
        tracked = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", "research/__init__.py"],
            capture_output=True,
        )
        require(tracked.returncode == 0, "untracked research/__init__.py would alter c91 import resolution")
    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))
    import research.experiment as experiment
    from hs_tasnet import HSTasNet

    require(Path(experiment.__file__).resolve() == repo / "research/experiment.py", "imported the wrong c91 experiment module")
    return repo, experiment, HSTasNet


@dataclass(frozen=True)
class FrozenAudioFile:
    path: Path
    sha256: str
    size_bytes: int
    frames: int
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class FrozenTrack:
    track_id: str
    root_id: str
    name: str
    effective_frames: int
    files: Mapping[str, FrozenAudioFile]
    vocal_active_seconds: tuple[int, ...]


def _manifest_content_hash(manifest: Mapping[str, Any]) -> str:
    unhashed = dict(manifest)
    unhashed.pop("content_sha256", None)
    return canonical_sha256(unhashed)


def load_corpus_manifest(
    path: Path,
    *,
    expected_file_sha256: str | None,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[FrozenTrack], str, str]:
    manifest, file_hash = load_json(path)
    if expected_file_sha256 is not None:
        require(file_hash == expected_file_sha256.lower(), f"manifest file SHA-256 mismatch: expected {expected_file_sha256}, got {file_hash}")
    require(manifest.get("schema_version") == 1, "corpus manifest schema_version must be 1")
    require(manifest.get("kind") == "hs_tasnet_training_corpus_manifest", "wrong corpus manifest kind")
    require(manifest.get("sample_rate") == 44_100, "corpus manifest sample_rate must be 44100")
    require(manifest.get("channels") == 2, "corpus manifest channels must be 2")
    require(tuple(manifest.get("source_order", ())) == SOURCE_NAMES, f"corpus source order must be {SOURCE_NAMES}")
    activity_policy = manifest.get("validation_policy", {}).get("vocal_activity", {})
    require(
        activity_policy
        == {
            "block_frames": 44_100,
            "mean_square_floor_int16": int(config["sampling"]["vocal_active_int16_mean_square_floor"]),
            "comparison": ">",
            "partial_final_second": "excluded",
        },
        "manifest vocal-activity policy differs from the configured sampler",
    )
    content_hash = _manifest_content_hash(manifest)
    require(manifest.get("content_sha256") == content_hash, "corpus manifest content_sha256 mismatch")

    roots_raw = manifest.get("roots")
    require(isinstance(roots_raw, (dict, list)), "manifest roots must be a mapping or list")
    roots: dict[str, Path] = {}
    root_records: dict[str, Mapping[str, Any]] = {}
    if isinstance(roots_raw, dict):
        iterable = []
        for root_id, raw in roots_raw.items():
            if isinstance(raw, str):
                iterable.append({"root_id": root_id, "path": raw})
            else:
                require(isinstance(raw, dict), f"invalid root record for {root_id}")
                iterable.append({"root_id": root_id, **raw})
    else:
        iterable = roots_raw
    for raw in iterable:
        require(isinstance(raw, dict), "manifest root entry must be an object")
        root_id = raw.get("root_id") or raw.get("id")
        root_path = raw.get("path") or raw.get("root")
        require(isinstance(root_id, str) and root_id, "manifest root_id is missing")
        require(isinstance(root_path, str) and root_path, f"manifest root path is missing for {root_id}")
        unresolved = Path(root_path).expanduser()
        resolved = (unresolved if unresolved.is_absolute() else path.parent / unresolved).resolve()
        require(resolved.is_dir(), f"manifest root does not exist: {resolved}")
        require(root_id not in roots, f"duplicate manifest root_id: {root_id}")
        roots[root_id] = resolved
        root_records[root_id] = raw

    configured_weights = config["sampling"]["root_weights"]
    require(set(roots) == set(configured_weights), f"manifest roots {sorted(roots)} differ from configured roots {sorted(configured_weights)}")
    declared_probabilities = manifest.get("sampling_policy", {}).get("root_probabilities")
    require(declared_probabilities == configured_weights, "manifest and training root probabilities differ")
    raw_tracks = manifest.get("tracks")
    require(isinstance(raw_tracks, list) and raw_tracks, "manifest tracks must be a non-empty list")
    require(manifest.get("track_count", len(raw_tracks)) == len(raw_tracks), "manifest track_count is inconsistent")
    tracks: list[FrozenTrack] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_tracks):
        require(isinstance(raw, dict), f"tracks[{index}] must be an object")
        root_id = raw.get("root_id")
        name = raw.get("name")
        track_id = raw.get("track_id") or raw.get("id") or f"{root_id}:{name}"
        require(root_id in roots, f"unknown root_id in track {track_id}: {root_id}")
        require(isinstance(name, str) and name, f"missing track name at index {index}")
        require(isinstance(track_id, str) and track_id and track_id not in seen_ids, f"invalid or duplicate track id: {track_id}")
        seen_ids.add(track_id)
        effective_frames = raw.get("effective_frames")
        require(isinstance(effective_frames, int) and effective_frames > 0, f"invalid effective_frames for {track_id}")
        files_raw = raw.get("files")
        require(isinstance(files_raw, dict) and set(files_raw) == set(FILE_NAMES), f"{track_id}: files must contain exactly {FILE_NAMES}")
        files: dict[str, FrozenAudioFile] = {}
        for label in FILE_NAMES:
            record = files_raw[label]
            require(isinstance(record, dict), f"{track_id}/{label}: file record must be an object")
            relative = record.get("relative_path") or record.get("path")
            require(isinstance(relative, str) and relative, f"{track_id}/{label}: relative path is missing")
            candidate = (roots[root_id] / relative).resolve()
            require(candidate.is_relative_to(roots[root_id]), f"{track_id}/{label}: path escapes root")
            digest = record.get("sha256") or record.get("file_sha256")
            require(is_sha256(digest), f"{track_id}/{label}: SHA-256 is missing or malformed")
            frames = record.get("frames")
            sample_rate = record.get("sample_rate", 44_100)
            channels = record.get("channels", 2)
            size_bytes = record.get("size_bytes")
            require(all(isinstance(v, int) and v > 0 for v in (frames, sample_rate, channels, size_bytes)), f"{track_id}/{label}: invalid file metadata")
            require(sample_rate == 44_100 and channels == 2, f"{track_id}/{label}: file must be stereo 44.1 kHz")
            files[label] = FrozenAudioFile(candidate, digest.lower(), size_bytes, frames, sample_rate, channels)
        require(effective_frames == min(item.frames for item in files.values()), f"{track_id}: effective_frames must equal the shortest frozen stream")
        require(effective_frames >= int(config["model"]["crop_samples"]), f"{track_id}: shorter than the configured crop")
        active_raw = raw.get("vocal_active_seconds", [])
        require(isinstance(active_raw, list), f"{track_id}: vocal_active_seconds must be a list")
        active = tuple(int(value) for value in active_raw)
        require(active == tuple(sorted(set(active))) and all(value >= 0 for value in active), f"{track_id}: vocal active seconds must be sorted unique non-negative integers")
        require(all((value + 1) * 44_100 <= effective_frames for value in active), f"{track_id}: vocal active second exceeds effective training frames")
        tracks.append(FrozenTrack(track_id, root_id, name, effective_frames, files, active))

    expected_counts = {
        str(root_id): int(count)
        for root_id, count in config["sampling"]["expected_track_counts"].items()
    }
    counts = {root_id: sum(track.root_id == root_id for track in tracks) for root_id in roots}
    require(counts == expected_counts, f"unexpected corpus counts: {counts}")
    for root_id, count in counts.items():
        require(root_records[root_id].get("track_count") == count, f"manifest root track_count differs for {root_id}")
        require(float(root_records[root_id].get("sample_weight")) == float(configured_weights[root_id]), f"manifest root sample weight differs for {root_id}")
    expected_total = int(config["sampling"]["expected_total_tracks"])
    require(len(tracks) == expected_total, f"expected {expected_total} tracks, found {len(tracks)}")

    exclusions = manifest.get("excluded_identities", {})
    expected_exclusion_counts = {
        "musdb_validation": 14,
        "musdb_test": 46,
        "recordpool_holdout": 12,
    }
    for group, expected_count in expected_exclusion_counts.items():
        values = exclusions.get(group, {}).get("values")
        require(isinstance(values, list) and len(values) == expected_count and len(values) == len(set(values)), f"invalid frozen exclusion group: {group}")

    integrity = manifest.get("integrity", {})
    require(integrity.get("exact_mixture_file_sha256_collisions") == [], "included corpus has exact mixture-file collisions")
    require(integrity.get("recordpool_prior_validation_hashes_matched") is True, "RecordPool prior hashes were not validated")
    collision_seal = integrity.get("acoustic_collision_report", {})
    require(collision_seal.get("status") == "pass" and collision_seal.get("coverage_status") == "pass", "acoustic collision report did not pass")
    require(collision_seal.get("input_tracks") == 502 and collision_seal.get("fingerprinted_input_tracks") == 502, "acoustic collision coverage is incomplete")
    collision_name = collision_seal.get("artifact")
    collision_hash = collision_seal.get("content_sha256")
    require(isinstance(collision_name, str) and collision_name == "acoustic_collision_report.json", "unexpected acoustic collision artifact")
    collision_path = path.parent / collision_name
    collision_report, _ = load_json(collision_path)
    require(_manifest_content_hash(collision_report) == collision_hash == collision_report.get("content_sha256"), "acoustic collision report content hash mismatch")
    require(collision_report.get("status") == "pass", "acoustic collision artifact is not clean")
    coverage = collision_report.get("training_coverage", {})
    require(coverage.get("fingerprinted_input_tracks") == 502 and coverage.get("input_coverage_fraction") == 1.0, "acoustic collision artifact lacks full input coverage")

    inventory_path = path.parent / "artifact_inventory.json"
    inventory, _ = load_json(inventory_path)
    require(_manifest_content_hash(inventory) == inventory.get("content_sha256"), "artifact inventory content hash mismatch")
    artifacts = inventory.get("artifacts", {})
    require(artifacts.get(path.name, {}).get("file_sha256") == file_hash, "artifact inventory does not bind the combined manifest file")
    require(artifacts.get(path.name, {}).get("content_sha256") == content_hash, "artifact inventory does not bind the combined manifest content")
    collision_file_hash = sha256_file(collision_path)
    require(artifacts.get(collision_name, {}).get("file_sha256") == collision_file_hash, "artifact inventory does not bind the acoustic collision file")
    require(artifacts.get(collision_name, {}).get("content_sha256") == collision_hash, "artifact inventory does not bind the acoustic collision content")
    for root_id, raw_root in root_records.items():
        root_manifest_name = raw_root.get("root_manifest")
        root_manifest_hash = raw_root.get("root_manifest_content_sha256")
        require(isinstance(root_manifest_name, str) and is_sha256(root_manifest_hash), f"missing root-manifest seal for {root_id}")
        root_manifest_path = path.parent / root_manifest_name
        root_manifest, root_manifest_file_hash = load_json(root_manifest_path)
        require(_manifest_content_hash(root_manifest) == root_manifest_hash == root_manifest.get("content_sha256"), f"root manifest content mismatch for {root_id}")
        require(artifacts.get(root_manifest_name, {}).get("file_sha256") == root_manifest_file_hash, f"artifact inventory does not bind root manifest {root_id}")
    return manifest, tracks, file_hash, content_hash


def validate_audio_inventory(tracks: Sequence[FrozenTrack], *, full_hash: bool) -> dict[str, Any]:
    started = time.monotonic()
    checked = 0
    bytes_checked = 0
    for track in tracks:
        for label, frozen in track.files.items():
            require(frozen.path.is_file(), f"audio file disappeared: {frozen.path}")
            stat = frozen.path.stat()
            require(stat.st_size == frozen.size_bytes, f"audio size changed: {frozen.path}")
            info = sf.info(str(frozen.path))
            actual = (int(info.frames), int(info.samplerate), int(info.channels))
            expected = (frozen.frames, frozen.sample_rate, frozen.channels)
            require(actual == expected, f"audio header changed for {track.track_id}/{label}: {actual} != {expected}")
            if full_hash:
                actual_hash = sha256_file(frozen.path)
                require(actual_hash == frozen.sha256, f"audio SHA-256 changed: {frozen.path}")
                bytes_checked += stat.st_size
            checked += 1
    return {
        "files_checked": checked,
        "full_hash": full_hash,
        "bytes_hashed": bytes_checked,
        "seconds": time.monotonic() - started,
    }


class CounterAddressedCropDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Every sample is a pure function of (seed, absolute sample index)."""

    def __init__(
        self,
        tracks: Sequence[FrozenTrack],
        *,
        root_weights: Mapping[str, float],
        seed: int,
        crop_samples: int,
        vocal_active_probability: float,
        final_sample_index: int,
    ) -> None:
        self.seed = int(seed)
        self.crop_samples = int(crop_samples)
        self.vocal_active_probability = float(vocal_active_probability)
        self.final_sample_index = int(final_sample_index)
        self.root_ids = tuple(root_weights)
        weights = tuple(float(root_weights[root_id]) for root_id in self.root_ids)
        require(all(weight > 0 and math.isfinite(weight) for weight in weights), "root weights must be finite and positive")
        total = sum(weights)
        self.cumulative_weights = tuple(sum(weights[: index + 1]) / total for index in range(len(weights)))
        grouped: dict[str, list[FrozenTrack]] = {root_id: [] for root_id in self.root_ids}
        for track in tracks:
            grouped[track.root_id].append(track)
        self.tracks_by_root = {root_id: tuple(sorted(grouped[root_id], key=lambda item: item.track_id)) for root_id in self.root_ids}
        require(all(self.tracks_by_root.values()), "every configured root must contain tracks")

    def __len__(self) -> int:
        return self.final_sample_index

    def _rng(self, sample_index: int) -> random.Random:
        material = f"hs-tasnet-c91-sample-v1:{self.seed}:{sample_index}".encode("ascii")
        seed = int.from_bytes(hashlib.blake2b(material, digest_size=16).digest(), "big")
        return random.Random(seed)

    @staticmethod
    def _read(file: FrozenAudioFile, *, offset: int, frames: int) -> torch.Tensor:
        with sf.SoundFile(str(file.path), "r") as handle:
            handle.seek(offset)
            audio = handle.read(frames, dtype="float32", always_2d=True)
        require(audio.shape == (frames, 2), f"short or malformed crop from {file.path}: {audio.shape}")
        return torch.from_numpy(np.ascontiguousarray(audio.T))

    def __getitem__(self, sample_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        require(0 <= sample_index < self.final_sample_index, f"sample index out of range: {sample_index}")
        rng = self._rng(int(sample_index))
        draw = rng.random()
        root_index = next((index for index, boundary in enumerate(self.cumulative_weights) if draw < boundary), len(self.root_ids) - 1)
        root_id = self.root_ids[root_index]
        root_tracks = self.tracks_by_root[root_id]
        track = root_tracks[rng.randrange(len(root_tracks))]
        max_start = track.effective_frames - self.crop_samples
        if track.vocal_active_seconds and rng.random() < self.vocal_active_probability:
            anchor = track.vocal_active_seconds[rng.randrange(len(track.vocal_active_seconds))] * 44_100
            offset = max(0, min(anchor - rng.randrange(self.crop_samples), max_start))
        else:
            offset = rng.randrange(max_start + 1)
        mixture = self._read(track.files["mixture"], offset=offset, frames=self.crop_samples)
        targets = torch.stack([self._read(track.files[source], offset=offset, frames=self.crop_samples) for source in SOURCE_NAMES])
        require(bool(torch.isfinite(mixture).all()) and bool(torch.isfinite(targets).all()), f"non-finite crop from {track.track_id}")
        return mixture, targets


class AbsoluteIndexSampler(Sampler[int]):
    def __init__(self, first: int, stop: int) -> None:
        self.first = int(first)
        self.stop = int(stop)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.first, self.stop))

    def __len__(self) -> int:
        return max(0, self.stop - self.first)


def worker_init(worker_id: int) -> None:
    # Crop selection uses its own counter-addressed RNG.  These seeds merely
    # prevent accidental future worker-local randomness from being inherited.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % (2**32))


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        require(
            len(state["torch_cuda"]) == torch.cuda.device_count(),
            "checkpoint CUDA RNG-state count differs from visible CUDA devices",
        )
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextlib.contextmanager
def preserve_rng() -> Iterator[None]:
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()

    def add(blob: bytes) -> None:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)

    for name, value in state.items():
        require(isinstance(value, torch.Tensor), f"model state is not a tensor: {name}")
        cpu = value.detach().to("cpu").contiguous()
        add(name.encode("utf-8"))
        add(str(cpu.dtype).encode("ascii"))
        add(canonical_json_bytes(list(cpu.shape)))
        add(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def structured_state_sha256(value: Any) -> str:
    """Hash nested optimizer/scaler/RNG state with explicit type boundaries."""

    digest = hashlib.sha256()

    def add(blob: bytes) -> None:
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            cpu = item.detach().to("cpu").contiguous()
            add(b"torch.Tensor")
            add(str(cpu.dtype).encode("ascii"))
            add(canonical_json_bytes(list(cpu.shape)))
            add(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            add(b"numpy.ndarray")
            add(str(array.dtype).encode("ascii"))
            add(canonical_json_bytes(list(array.shape)))
            add(array.view(np.uint8).tobytes())
        elif isinstance(item, dict):
            add(b"dict")
            ordered = sorted(item.items(), key=lambda pair: (type(pair[0]).__name__, repr(pair[0])))
            add(str(len(ordered)).encode("ascii"))
            for key, child in ordered:
                visit(key)
                visit(child)
        elif isinstance(item, list):
            add(b"list")
            add(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif isinstance(item, tuple):
            add(b"tuple")
            add(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif isinstance(item, bytes):
            add(b"bytes")
            add(item)
        elif isinstance(item, str):
            add(b"str")
            add(item.encode("utf-8"))
        elif item is None:
            add(b"None")
        elif isinstance(item, bool):
            add(b"bool:true" if item else b"bool:false")
        elif isinstance(item, (int, np.integer)):
            add(b"int")
            add(str(int(item)).encode("ascii"))
        elif isinstance(item, (float, np.floating)):
            add(b"float")
            add(float(item).hex().encode("ascii"))
        else:
            raise TypeError(f"unsupported checkpoint state type: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def recursive_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def model_is_raw(model: torch.nn.Module) -> bool:
    config = pickle.loads(model._config)
    scales = model.output_source_scales.detach().cpu()
    return (
        not bool(config.get("decoder_hann_baked", False))
        and not bool(model.conv_decode.hann_window_baked)
        and torch.equal(scales, torch.full_like(scales, 0.5))
    )


def build_static_contract(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    manifest_path: Path,
    manifest_file_sha256: str,
    manifest_content_sha256: str,
    repo: Path,
    device: torch.device,
    requested_torch_threads: int,
) -> dict[str, Any]:
    cuda_identity: dict[str, Any] | None = None
    if device.type == "cuda":
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(index)
        cuda_identity = {
            "visible_device_count": torch.cuda.device_count(),
            "index": index,
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": int(properties.total_memory),
        }
    return {
        "schema_version": 1,
        "config": config,
        "config_path": str(config_path.resolve()),
        "config_file_sha256": config_sha256,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_content_sha256": manifest_content_sha256,
        "trainer_path": str(SCRIPT_PATH),
        "trainer_sha256": sha256_file(SCRIPT_PATH),
        "source_repo": str(repo),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "libsndfile": getattr(sf, "__libsndfile_version__", None),
        "execution": {
            "device": str(device),
            "cuda_identity": cuda_identity,
            "requested_torch_threads": int(requested_torch_threads),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "amp_dtype": "bfloat16" if device.type == "cuda" else "float32",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def open_run_contract(run_dir: Path, static: Mapping[str, Any]) -> tuple[dict[str, Any], str, bool]:
    path = run_dir / "run_contract.json"
    identity = canonical_sha256(static)
    if path.exists():
        existing, _ = load_json(path)
        require(existing.get("static_identity_sha256") == identity, "run contract differs from current config, manifest, code, or runtime")
        require(existing.get("static") == static, "run contract static payload mismatch")
        return existing, identity, False
    unexpected = [item for item in run_dir.iterdir() if item.name not in {"stdout.log", "events.jsonl", ".run.lock"}]
    require(not unexpected, f"refusing to initialize a non-empty run directory: {unexpected[:5]}")
    contract = {
        "schema_version": 1,
        "run_uuid": str(uuid.uuid4()),
        "created_at_utc": utc_now(),
        "static_identity_sha256": identity,
        "static": static,
    }
    atomic_write_json(path, contract)
    return contract, identity, True


@contextlib.contextmanager
def exclusive_run_lock(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another trainer already owns run directory {run_dir}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_at={utc_now()}\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def make_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    contract_identity: str,
    recent_losses: Sequence[float],
) -> dict[str, Any]:
    require(model_is_raw(model), "resume checkpoint model must be uncalibrated and decoder-unbaked")
    model_state = recursive_to_cpu(model.state_dict())
    optimizer_state = recursive_to_cpu(optimizer.state_dict())
    scaler_state = recursive_to_cpu(scaler.state_dict())
    rng_state = recursive_to_cpu(capture_rng_state())
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "kind": "hs_tasnet_c91_raw_resume",
        "saved_at_utc": utc_now(),
        "contract_identity_sha256": contract_identity,
        "step": int(step),
        "model_config": model._config,
        "model": model_state,
        "model_state_sha256": tensor_state_sha256(model_state),
        "optimizer": optimizer_state,
        "optimizer_state_sha256": structured_state_sha256(optimizer_state),
        "optimizer_meta": {
            "autoresearch_step": int(optimizer._autoresearch_step),
            "base_lrs": tuple(float(value) for value in optimizer._autoresearch_base_lrs),
        },
        "scaler": scaler_state,
        "scaler_state_sha256": structured_state_sha256(scaler_state),
        "rng_state": rng_state,
        "rng_state_sha256": structured_state_sha256(rng_state),
        "recent_losses": [float(value) for value in recent_losses[-1000:]],
    }


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    contract_identity: str,
    HSTasNet: Any,
    learning_rate: float,
) -> None:
    require(payload.get("schema_version") == CHECKPOINT_SCHEMA, "resume checkpoint schema mismatch")
    require(payload.get("kind") == "hs_tasnet_c91_raw_resume", "resume checkpoint kind mismatch")
    require(payload.get("contract_identity_sha256") == contract_identity, "resume checkpoint run contract mismatch")
    step = payload.get("step")
    require(isinstance(step, int) and step >= 0, "resume checkpoint has an invalid step")
    require(payload.get("optimizer_meta", {}).get("autoresearch_step") == step, "resume optimizer step disagrees with global step")
    state = payload.get("model")
    require(isinstance(state, dict), "resume model state is missing")
    require(tensor_state_sha256(state) == payload.get("model_state_sha256"), "resume model state hash mismatch")
    require(structured_state_sha256(payload.get("optimizer")) == payload.get("optimizer_state_sha256"), "resume optimizer state hash mismatch")
    require(structured_state_sha256(payload.get("scaler")) == payload.get("scaler_state_sha256"), "resume scaler state hash mismatch")
    require(structured_state_sha256(payload.get("rng_state")) == payload.get("rng_state_sha256"), "resume RNG state hash mismatch")
    with preserve_rng():
        config = pickle.loads(payload["model_config"])
        require(not bool(config.get("decoder_hann_baked", False)), "refusing a finalized decoder in a resume checkpoint")
        fresh = HSTasNet(**config)
        fresh.load_state_dict(state, strict=True)
        require(model_is_raw(fresh), "resume checkpoint contains calibrated or baked persistent state")
        optimizer = torch.optim.Adam(fresh.parameters(), lr=learning_rate)
        optimizer.load_state_dict(payload["optimizer"])
        require(len(optimizer.param_groups) > 0, "resume optimizer contains no parameter groups")


def _write_torch_file(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _checkpoint_generation_step(path: Path) -> int | None:
    prefix = "step-"
    suffix = ".pt"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    body = name[len(prefix) : -len(suffix)]
    step_text, separator, generation_id = body.partition("-")
    if (
        separator != "-"
        or len(step_text) != 12
        or not step_text.isdigit()
        or len(generation_id) != 32
        or any(character not in "0123456789abcdef" for character in generation_id)
    ):
        return None
    return int(step_text)


def _checkpoint_generation_candidates(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.is_dir():
        return []
    candidates: set[Path] = set()
    for item in checkpoint_dir.iterdir():
        candidate = item.with_suffix("") if item.name.endswith(".pt.sha256") else item
        if _checkpoint_generation_step(candidate) is not None:
            candidates.add(candidate)
    return sorted(
        candidates,
        key=lambda path: (_checkpoint_generation_step(path), path.name),
        reverse=True,
    )


def _checkpoint_candidates(run_dir: Path) -> list[Path]:
    candidates = _checkpoint_generation_candidates(run_dir / CHECKPOINT_DIRECTORY_NAME)
    # Read legacy pairs so an in-progress run can migrate on its next save.
    for name in ("resume.pt", "resume.prev.pt"):
        path = run_dir / name
        if path.exists() or _sidecar_path(path).exists():
            candidates.append(path)
    return candidates


def _read_checkpoint_sidecar(path: Path) -> str:
    sidecar = _sidecar_path(path)
    require(sidecar.is_file(), f"checkpoint SHA sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").split()
    require(len(fields) == 2, f"malformed checkpoint SHA sidecar: {sidecar}")
    digest, recorded_name = fields
    digest = digest.lower()
    require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"invalid checkpoint SHA-256 in {sidecar}",
    )
    allowed_names = {path.name}
    if path.name == "resume.prev.pt":
        # The old rotation renamed this sidecar without rewriting its filename.
        allowed_names.add("resume.pt")
    require(recorded_name in allowed_names, f"checkpoint SHA sidecar names the wrong file: {sidecar}")
    return digest


def _checkpoint_file_hash_matches(path: Path) -> bool:
    try:
        return path.is_file() and sha256_file(path) == _read_checkpoint_sidecar(path)
    except (OSError, UnicodeError, RuntimeError):
        return False


def _prune_verified_checkpoint_generations(
    checkpoint_dir: Path,
    *,
    protected: Path,
    protected_sha256: str,
) -> None:
    verified: list[Path] = []
    for path in _checkpoint_generation_candidates(checkpoint_dir):
        if path == protected:
            matches = path.is_file() and sha256_file(path) == protected_sha256
        else:
            matches = _checkpoint_file_hash_matches(path)
        if matches:
            verified.append(path)
    verified.sort(
        key=lambda path: (
            _checkpoint_generation_step(path),
            path == protected,
            path.name,
        ),
        reverse=True,
    )
    removed = False
    for path in verified[CHECKPOINT_RETAINED_GENERATIONS:]:
        _sidecar_path(path).unlink()
        path.unlink()
        removed = True
    if removed:
        fsync_directory(checkpoint_dir)


def write_verified_checkpoint(
    *,
    run_dir: Path,
    payload: Mapping[str, Any],
    contract_identity: str,
    HSTasNet: Any,
    learning_rate: float,
) -> dict[str, Any]:
    step = payload.get("step")
    require(isinstance(step, int) and not isinstance(step, bool) and step >= 0, "checkpoint step is invalid")
    require(step < 10**12, "checkpoint step exceeds the generation filename range")
    checkpoint_dir = run_dir / CHECKPOINT_DIRECTORY_NAME
    checkpoint_dir_created = not checkpoint_dir.exists()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    require(checkpoint_dir.is_dir(), f"checkpoint path is not a directory: {checkpoint_dir}")
    if checkpoint_dir_created:
        fsync_directory(run_dir)
    generation = checkpoint_dir / f"step-{step:012d}-{uuid.uuid4().hex}.pt"
    temporary = checkpoint_dir / f".{generation.name}.{os.getpid()}.tmp"
    temporary_sidecar = _sidecar_path(temporary)
    _write_torch_file(temporary, payload)
    file_hash = sha256_file(temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(loaded, contract_identity=contract_identity, HSTasNet=HSTasNet, learning_rate=learning_rate)
    require(loaded["step"] == step, "checkpoint read-back changed the global step")
    require(loaded["model_state_sha256"] == payload["model_state_sha256"], "checkpoint read-back changed the model state")
    atomic_write_bytes(temporary_sidecar, f"{file_hash}  {generation.name}\n".encode("ascii"))
    os.replace(temporary, generation)
    os.replace(temporary_sidecar, _sidecar_path(generation))
    fsync_directory(checkpoint_dir)
    _prune_verified_checkpoint_generations(
        checkpoint_dir,
        protected=generation,
        protected_sha256=file_hash,
    )
    pointer = checkpoint_dir / CHECKPOINT_POINTER_NAME
    atomic_write_json(
        pointer,
        {
            "schema_version": 1,
            "kind": "hs_tasnet_c91_checkpoint_pointer",
            "updated_at_utc": utc_now(),
            "generation": generation.name,
            "path": str(generation.relative_to(run_dir)),
            "sha256": file_hash,
            "step": step,
            "model_state_sha256": payload["model_state_sha256"],
        },
    )
    return {
        "path": str(generation),
        "sha256": file_hash,
        "step": step,
        "model_state_sha256": payload["model_state_sha256"],
        "pointer": str(pointer),
    }


def load_verified_checkpoint_file(
    path: Path,
    *,
    contract_identity: str,
    HSTasNet: Any,
    learning_rate: float,
) -> dict[str, Any]:
    sidecar = _sidecar_path(path)
    require(path.is_file() and sidecar.is_file(), f"checkpoint or SHA sidecar is missing: {path}")
    expected = _read_checkpoint_sidecar(path)
    actual = sha256_file(path)
    require(actual == expected, f"checkpoint file SHA-256 mismatch for {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload, contract_identity=contract_identity, HSTasNet=HSTasNet, learning_rate=learning_rate)
    return payload


def recover_checkpoint(
    run_dir: Path,
    *,
    contract_identity: str,
    HSTasNet: Any,
    learning_rate: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    verified: list[tuple[int, str, str, dict[str, Any]]] = []
    for path in _checkpoint_candidates(run_dir):
        try:
            payload = load_verified_checkpoint_file(
                path,
                contract_identity=contract_identity,
                HSTasNet=HSTasNet,
                learning_rate=learning_rate,
            )
            filename_step = _checkpoint_generation_step(path)
            require(
                filename_step is None or payload["step"] == filename_step,
                f"checkpoint filename step disagrees with payload: {path}",
            )
            verified.append(
                (
                    int(payload["step"]),
                    str(payload.get("saved_at_utc", "")),
                    path.name,
                    payload,
                )
            )
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
    if not verified:
        return None, errors
    verified.sort(key=lambda item: item[:3], reverse=True)
    return verified[0][3], errors


def configure_c91_schedule(experiment: Any, config: Mapping[str, Any]) -> None:
    schedule = config["schedule"]
    require(
        schedule.get("endpoint_semantics")
        == "exclusive_optimizer_steps_0_through_total_steps_minus_1",
        "production schedule must declare its zero-based exclusive endpoint semantics",
    )
    total = int(schedule["total_steps"])
    values = (
        int(schedule["projection_ramp_start"]),
        int(schedule["projection_ramp_end"]),
        int(schedule["lr_decay_start"]),
        int(schedule["lr_decay_end"]),
    )
    require(0 <= values[0] < values[1] <= values[2] < values[3] == total, f"invalid c91 production schedule: {values}, total={total}")
    experiment.DERANGED_PROJECTION_RAMP_START_STEP = values[0]
    experiment.DERANGED_PROJECTION_RAMP_END_STEP = values[1]
    experiment.LEARNING_RATE_DECAY_START_STEP = values[2]
    experiment.LEARNING_RATE_DECAY_END_STEP = values[3]
    finalization = config["finalization"]
    require(finalization.get("bake_decoder_hann") is True, "c91 deployment requires decoder Hann baking")
    require(
        tuple(float(value) for value in finalization.get("source_gains", ()))
        == tuple(float(value) for value in experiment.FINAL_SOURCE_CALIBRATION),
        "production source gains differ from the frozen c91 calibration",
    )


def build_model_optimizer(
    *,
    experiment: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, Any, Any, Any]:
    model_cfg = config["model"]
    experiment_config = experiment.ExperimentConfig(
        batch_size=int(model_cfg["batch_size"]),
        crop_seconds=float(model_cfg["crop_samples"]) / 44_100,
        precision=str(model_cfg["precision"]),
        learning_rate=float(model_cfg["learning_rate"]),
        grad_clip_norm=float(model_cfg["grad_clip_norm"]),
        log_every=int(config["schedule"]["log_every"]),
    )
    model = experiment.build_model(config=experiment_config, smoke=False)
    require(model.num_parameters == int(model_cfg["expected_parameters"]), f"c91 parameter count changed: {model.num_parameters}")
    require(model.sample_rate == int(model_cfg["sample_rate"]) == 44_100, "c91 sample-rate contract changed")
    require(model.audio_channels == int(model_cfg["channels"]) == 2, "c91 channel contract changed")
    require(model.num_sources == len(model_cfg["sources"]) == 4 and tuple(model_cfg["sources"]) == SOURCE_NAMES, "c91 source contract changed")
    require(model.segment_len == 1024 and model.overlap_len == 512, "c91 latency contract changed")
    require(model_is_raw(model), "fresh c91 model is unexpectedly finalized")
    model.to(device)
    optimizer = experiment.build_optimizer(model=model, config=experiment_config)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scaler, experiment_config


def restore_training_state(
    *,
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, list[float]]:
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    optimizer._autoresearch_step = int(payload["optimizer_meta"]["autoresearch_step"])
    optimizer._autoresearch_base_lrs = tuple(float(value) for value in payload["optimizer_meta"]["base_lrs"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng_state"])
    require(model_is_raw(model), "restored training model is not raw")
    return int(payload["step"]), [float(value) for value in payload.get("recent_losses", [])]


def save_raw_model_artifact(
    *,
    model: torch.nn.Module,
    path: Path,
    HSTasNet: Any,
) -> dict[str, Any]:
    require(model_is_raw(model), "raw model artifact requested from a finalized model")
    state = recursive_to_cpu(model.state_dict())
    package = {"model": state, "config": model._config}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_torch_file(temporary, package)
    digest = sha256_file(temporary)
    with preserve_rng():
        loaded = HSTasNet.init_and_load_from(temporary, strict=True)
    require(model_is_raw(loaded), "raw model artifact reloaded as finalized")
    state_hash = tensor_state_sha256(state)
    require(tensor_state_sha256(loaded.state_dict()) == state_hash, "raw model artifact state changed after reload")
    os.replace(temporary, path)
    atomic_write_bytes(_sidecar_path(path), f"{digest}  {path.name}\n".encode("ascii"))
    fsync_directory(path.parent)
    return {"path": str(path), "sha256": digest, "model_state_sha256": state_hash}


def finalize_deployment_copy(
    *,
    model: torch.nn.Module,
    path: Path,
    gains: Sequence[float],
    HSTasNet: Any,
    streaming_callbacks: int,
) -> dict[str, Any]:
    require(model_is_raw(model), "deployment finalization requires a raw training model")
    with preserve_rng():
        raw_state = recursive_to_cpu(model.state_dict())
        deployment = HSTasNet(**pickle.loads(model._config))
        deployment.load_state_dict(raw_state, strict=True)
    require(model_is_raw(deployment), "deployment clone was not raw before finalization")
    deployment.eval()
    probe = torch.linspace(-0.2, 0.2, 2048, dtype=torch.float32).repeat(2, 1).unsqueeze(0)
    with torch.inference_mode():
        raw_output = deployment(probe)[0].float()
    gain_tensor = torch.tensor(gains, dtype=torch.float32)
    deployment.set_output_source_gains(gain_tensor)
    with torch.inference_mode():
        calibrated_output = deployment(probe)[0].float()
    torch.testing.assert_close(calibrated_output, raw_output * gain_tensor[None, :, None, None], rtol=2e-5, atol=2e-6)
    deployment.bake_decoder_hann_window_()
    with torch.inference_mode():
        baked_output = deployment(probe)[0].float()
    torch.testing.assert_close(baked_output, calibrated_output, rtol=2e-5, atol=2e-6)
    expected_scales = 0.5 * gain_tensor
    require(torch.equal(deployment.output_source_scales, expected_scales), "deployment source scales are wrong")
    require(bool(pickle.loads(deployment._config).get("decoder_hann_baked")), "deployment config did not persist decoder baking")
    require(bool(deployment.conv_decode.hann_window_baked), "deployment module did not persist decoder baking")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_torch_file(temporary, {"model": recursive_to_cpu(deployment.state_dict()), "config": deployment._config})
    file_hash = sha256_file(temporary)
    with preserve_rng():
        loaded = HSTasNet.init_and_load_from(temporary, strict=True)
    loaded.eval()
    state_hash = tensor_state_sha256(deployment.state_dict())
    require(tensor_state_sha256(loaded.state_dict()) == state_hash, "deployment state changed after strict reload")
    require(torch.equal(loaded.output_source_scales, expected_scales), "reloaded deployment source scales changed")
    with torch.inference_mode():
        loaded_output = loaded(probe)[0].float()
    torch.testing.assert_close(loaded_output, baked_output, rtol=0, atol=0)

    transform_a = deployment.init_stateful_transform_fn(device="cpu")
    transform_b = loaded.init_stateful_transform_fn(device="cpu")
    stream_peak = 0.0
    for index in range(streaming_callbacks):
        chunk = torch.sin(torch.arange(512, dtype=torch.float32)[None, :] * (0.001 + index * 0.0001)).repeat(2, 1) * 0.1
        output_a = transform_a(chunk)
        output_b = transform_b(chunk)
        require(tuple(output_a.shape) == (4, 2, 512), f"unexpected streaming shape: {tuple(output_a.shape)}")
        require(bool(torch.isfinite(output_a).all()), "non-finite deployment streaming output")
        torch.testing.assert_close(output_b, output_a, rtol=0, atol=0)
        stream_peak = max(stream_peak, float(output_a.abs().max()))

    os.replace(temporary, path)
    atomic_write_bytes(_sidecar_path(path), f"{file_hash}  {path.name}\n".encode("ascii"))
    metadata = {
        "schema_version": 1,
        "path": str(path),
        "sha256": file_hash,
        "model_state_sha256": state_hash,
        "source_gains": [float(value) for value in gains],
        "persistent_output_source_scales": [float(value) for value in loaded.output_source_scales],
        "decoder_hann_baked": True,
        "bake_output_parity": True,
        "strict_reload_output_parity": True,
        "streaming_callbacks": streaming_callbacks,
        "streaming_shape": [4, 2, 512],
        "streaming_peak": stream_peak,
        "created_at_utc": utc_now(),
    }
    atomic_write_json(path.with_suffix(path.suffix + ".json"), metadata)
    fsync_directory(path.parent)
    return metadata


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def install_signal_handlers() -> None:
    def request_stop(signum: int, _frame: Any) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True
        print(json.dumps({"event": "signal_stop_requested", "signal": signum, "time": utc_now()}), flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--verify-audio-hashes", action="store_true")
    parser.add_argument("--stop-after-step", type=int, help="test/maintenance stop; schedule identity is unchanged")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    config_path = args.config.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"

    config, config_hash = load_json(config_path)
    require(config.get("schema_version") == 1, "training config schema_version must be 1")
    repo, experiment, HSTasNet = resolve_source(config)
    configure_c91_schedule(experiment, config)
    manifest, tracks, manifest_file_hash, manifest_content_hash = load_corpus_manifest(
        manifest_path,
        expected_file_sha256=args.manifest_sha256,
        config=config,
    )
    full_audio_hash_required = bool(
        config["safety"].get("require_full_audio_hash_verification", False)
    )
    inventory = validate_audio_inventory(
        tracks,
        full_hash=bool(args.verify_audio_hashes or full_audio_hash_required),
    )
    require(args.torch_threads > 0, "torch thread count must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    seed = int(config["seed"])
    configure_determinism(seed)
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA was requested but is unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    static_contract = build_static_contract(
        config=config,
        config_path=config_path,
        config_sha256=config_hash,
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_hash,
        manifest_content_sha256=manifest_content_hash,
        repo=repo,
        device=device,
        requested_torch_threads=args.torch_threads,
    )
    if args.resume == "required":
        require((run_dir / "run_contract.json").is_file(), "--resume required but this run directory has no contract")
    contract, contract_identity, contract_created = open_run_contract(run_dir, static_contract)
    if args.resume == "never":
        require(contract_created, "--resume never requires a new run directory")
    append_jsonl(events_path, {"event": "inventory_verified", "time": utc_now(), **inventory})
    append_jsonl(events_path, {
        "event": "run_opened",
        "time": utc_now(),
        "run_uuid": contract["run_uuid"],
        "contract_identity_sha256": contract_identity,
        "manifest_track_count": len(tracks),
        "device": str(device),
    })

    model, optimizer, scaler, experiment_config = build_model_optimizer(experiment=experiment, config=config, device=device)
    learning_rate = float(config["model"]["learning_rate"])
    payload, recovery_errors = recover_checkpoint(
        run_dir,
        contract_identity=contract_identity,
        HSTasNet=HSTasNet,
        learning_rate=learning_rate,
    )
    if recovery_errors:
        append_jsonl(events_path, {"event": "resume_candidates_rejected", "time": utc_now(), "errors": recovery_errors})
    if args.resume == "never":
        require(payload is None, "--resume never refuses an existing checkpoint")
    if args.resume == "required":
        require(payload is not None, "--resume required but no verified checkpoint was found")
    if payload is None:
        require(not recovery_errors, "all existing resume checkpoints were invalid")
        require(contract_created, "existing run contract has no verified resume checkpoint; refusing a silent step-zero restart")
        step = 0
        recent_losses: list[float] = []
    else:
        step, recent_losses = restore_training_state(payload=payload, model=model, optimizer=optimizer, scaler=scaler, device=device)
        append_jsonl(events_path, {"event": "resumed", "time": utc_now(), "step": step, "model_state_sha256": payload["model_state_sha256"]})
        del payload

    total_steps = int(config["schedule"]["total_steps"])
    require(0 <= step <= total_steps, f"resume step {step} is outside schedule")
    batch_size = int(config["model"]["batch_size"])
    crop_samples = int(config["model"]["crop_samples"])
    if step == 0 and contract_created:
        initial = make_checkpoint_payload(model=model, optimizer=optimizer, scaler=scaler, step=0, contract_identity=contract_identity, recent_losses=[])
        checkpoint = write_verified_checkpoint(run_dir=run_dir, payload=initial, contract_identity=contract_identity, HSTasNet=HSTasNet, learning_rate=learning_rate)
        restore_rng_state(initial["rng_state"])
        append_jsonl(events_path, {"event": "initial_checkpoint_verified", "time": utc_now(), **checkpoint})
        del initial

    stop_step = total_steps if args.stop_after_step is None else min(total_steps, int(args.stop_after_step))
    require(stop_step >= step, f"--stop-after-step {stop_step} precedes resume step {step}")
    sampling = config["sampling"]
    dataset = CounterAddressedCropDataset(
        tracks,
        root_weights=sampling["root_weights"],
        seed=seed,
        crop_samples=crop_samples,
        vocal_active_probability=float(sampling["vocal_active_probability"]),
        final_sample_index=stop_step * batch_size,
    )
    sampler = AbsoluteIndexSampler(step * batch_size, stop_step * batch_size)
    workers = int(sampling["num_workers"])
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "drop_last": True,
        "num_workers": workers,
        "pin_memory": bool(sampling["pin_memory"]) and device.type == "cuda",
        "worker_init_fn": worker_init,
        "generator": torch.Generator().manual_seed(seed + 99173),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(sampling["prefetch_factor"])
        # CUDA is initialized before resume restoration; spawning avoids the
        # undefined CUDA-after-fork state inherited by ordinary Linux workers.
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(**loader_kwargs)

    checkpoint_every = int(config["schedule"]["checkpoint_every"])
    snapshot_every = int(config["schedule"]["snapshot_every"])
    log_every = int(config["schedule"]["log_every"])
    min_free_checkpoint = float(config["safety"]["minimum_free_gb_checkpoint"])
    min_free_continue = float(config["safety"]["minimum_free_gb_continue"])
    gains = tuple(float(value) for value in config["finalization"]["source_gains"])
    started = time.monotonic()
    session_start_step = step
    model.train()
    install_signal_handlers()
    last_checkpoint_step = step
    inside_train_step = False

    def persist_committed_checkpoint(event: str) -> dict[str, Any] | None:
        nonlocal last_checkpoint_step
        if last_checkpoint_step == step:
            return None
        require(not inside_train_step, "refusing to checkpoint inside an incomplete optimizer step")
        require(int(optimizer._autoresearch_step) == step, "refusing to checkpoint divergent optimizer/global steps")
        require(model_is_raw(model), "refusing to checkpoint a finalized training model")
        require(free_gb(run_dir) >= min_free_checkpoint, "insufficient disk space for a verified resume checkpoint")
        checkpoint_rng = capture_rng_state()
        checkpoint_payload = make_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            contract_identity=contract_identity,
            recent_losses=recent_losses,
        )
        checkpoint = write_verified_checkpoint(
            run_dir=run_dir,
            payload=checkpoint_payload,
            contract_identity=contract_identity,
            HSTasNet=HSTasNet,
            learning_rate=learning_rate,
        )
        restore_rng_state(checkpoint_rng)
        last_checkpoint_step = step
        append_jsonl(events_path, {"event": event, "time": utc_now(), **checkpoint})
        return checkpoint

    try:
        for mixture, targets in loader:
            if STOP_REQUESTED:
                break
            mixture = mixture.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            inside_train_step = True
            try:
                loss = experiment.train_step(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    mixture=mixture,
                    targets=targets,
                    config=experiment_config,
                    amp_dtype=torch.bfloat16 if device.type == "cuda" else None,
                )
            except BaseException:
                raise
            else:
                inside_train_step = False
            step += 1
            require(int(optimizer._autoresearch_step) == step, "optimizer and global steps diverged")
            recent_losses.append(float(loss))
            if len(recent_losses) > 1000:
                del recent_losses[:-1000]

            if step % log_every == 0 or step == stop_step:
                elapsed = time.monotonic() - started
                rate = (step - session_start_step) / max(elapsed, 1e-9)
                eta_hours = (total_steps - step) / max(rate, 1e-9) / 3600
                peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                record = {
                    "event": "train_progress",
                    "time": utc_now(),
                    "step": step,
                    "total_steps": total_steps,
                    "loss": float(loss),
                    "loss_mean_recent": float(sum(recent_losses[-log_every:]) / min(log_every, len(recent_losses))),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "steps_per_second": rate,
                    "eta_hours": eta_hours,
                    "peak_vram_bytes": int(peak_vram),
                    "free_disk_gb": free_gb(run_dir),
                }
                append_jsonl(events_path, record)
                print(json.dumps(record, sort_keys=True), flush=True)

            should_checkpoint = step % checkpoint_every == 0 or step == stop_step or STOP_REQUESTED
            if should_checkpoint:
                persist_committed_checkpoint("checkpoint_verified")

            if step < total_steps and step % snapshot_every == 0:
                snapshot_path = run_dir / "snapshots" / f"c91-step-{step:06d}-deploy.pt"
                metadata = finalize_deployment_copy(
                    model=model,
                    path=snapshot_path,
                    gains=gains,
                    HSTasNet=HSTasNet,
                    streaming_callbacks=2,
                )
                append_jsonl(events_path, {"event": "snapshot_verified", "time": utc_now(), "step": step, **metadata})

            require(free_gb(run_dir) >= min_free_continue, "free disk dropped below the continuation floor")
            if STOP_REQUESTED or step >= stop_step:
                break
    except BaseException as error:
        append_jsonl(events_path, {"event": "training_exception", "time": utc_now(), "step": step, "error": repr(error), "traceback": traceback.format_exc()})
        if (
            not inside_train_step
            and model_is_raw(model)
            and int(optimizer._autoresearch_step) == step
            and free_gb(run_dir) >= min_free_checkpoint
        ):
            persist_committed_checkpoint("emergency_checkpoint_verified")
        raise

    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if STOP_REQUESTED or step < total_steps:
        # Covers a signal arriving after the in-loop checkpoint decision.
        persist_committed_checkpoint("pause_checkpoint_verified")
        result = {
            "status": "paused",
            "step": step,
            "total_steps": total_steps,
            "peak_vram_bytes": int(peak_vram),
            "contract_identity_sha256": contract_identity,
        }
        atomic_write_json(run_dir / "status.json", result)
        append_jsonl(events_path, {"event": "run_paused", "time": utc_now(), **result})
        return result

    require(step == total_steps, "training loop ended at the wrong final step")
    require(model_is_raw(model), "training model was finalized before optimization completed")
    raw_artifact = save_raw_model_artifact(model=model, path=run_dir / "final-training-raw.pt", HSTasNet=HSTasNet)
    deployment = finalize_deployment_copy(
        model=model,
        path=run_dir / "final-deployment.pt",
        gains=gains,
        HSTasNet=HSTasNet,
        streaming_callbacks=16,
    )
    result = {
        "status": "complete",
        "step": step,
        "total_steps": total_steps,
        "peak_vram_bytes": int(peak_vram),
        "maximum_peak_vram_bytes": int(float(config["safety"]["maximum_peak_vram_gb"]) * 1e9),
        "peak_vram_pass": peak_vram <= float(config["safety"]["maximum_peak_vram_gb"]) * 1e9,
        "contract_identity_sha256": contract_identity,
        "manifest_file_sha256": manifest_file_hash,
        "manifest_content_sha256": manifest_content_hash,
        "raw_artifact": raw_artifact,
        "deployment": deployment,
        "completed_at_utc": utc_now(),
    }
    require(result["peak_vram_pass"], f"peak VRAM exceeded the configured ceiling: {peak_vram}")
    atomic_write_json(run_dir / "final_report.json", result)
    atomic_write_json(run_dir / "status.json", result)
    append_jsonl(events_path, {"event": "run_complete", "time": utc_now(), **result})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with exclusive_run_lock(args.run_dir.expanduser().resolve()):
        result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
