#!/usr/bin/env python3
"""Freeze and validate the HS-TasNet c91 full-training corpus.

This program is intentionally standalone and writes only below its own
production directory. It never imports either HS-TasNet checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "corpus_config.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "manifests"
FILE_ORDER = ("mixture", "drums", "bass", "vocals", "other")
ALLOWED_EXTENSIONS = ("wav", "flac")
ALLOWED_FORMATS = {"WAV", "FLAC"}
HASH_CHUNK_BYTES = 8 << 20
CHROMAPRINT_ALGORITHM = "ffmpeg chromaprint raw, first 120 seconds"


@dataclass(frozen=True)
class TrackSpec:
    root_id: str
    root: Path
    name: str
    metadata: Mapping[str, Any]
    prior_validation: Mapping[str, Any] | None = None

    @property
    def stable_id(self) -> str:
        return f"{self.root_id}:{self.name}"

    @property
    def directory(self) -> Path:
        return self.root / self.name


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload_without_hash)).hexdigest()


def seal_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    require("content_sha256" not in payload, "payload is already sealed")
    sealed = dict(payload)
    sealed["content_sha256"] = payload_sha256(payload)
    return sealed


def validate_payload_hash(payload: Mapping[str, Any], label: str) -> str:
    recorded = payload.get("content_sha256")
    require(
        isinstance(recorded, str) and len(recorded) == 64,
        f"{label}: missing content_sha256",
    )
    unhashed = dict(payload)
    unhashed.pop("content_sha256")
    actual = payload_sha256(unhashed)
    require(recorded == actual, f"{label}: content hash mismatch {recorded} != {actual}")
    return actual


def render_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, content: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def provenance_record(path: Path, *, schema: Any = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if schema is not None:
        record["schema"] = schema
    return record


def normalize_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def read_holdout_ids(path: Path) -> list[str]:
    values = sorted(path.read_text(encoding="utf-8").split())
    require(values and len(values) == len(set(values)), "invalid duplicate/empty holdout IDs")
    return values


def validate_config(config: Mapping[str, Any]) -> None:
    require(config.get("schema_version") == 1, "config schema_version must be 1")
    require(config.get("sample_rate") == 44_100, "config sample_rate must be 44100")
    require(config.get("channels") == 2, "config channels must be 2")
    require(
        tuple(config.get("source_order", ())) == ("drums", "bass", "vocals", "other"),
        "config source_order changed",
    )
    roots = config.get("roots")
    require(isinstance(roots, list) and len(roots) == 3, "config must define three roots")
    root_ids = [root.get("root_id") for root in roots if isinstance(root, dict)]
    require(len(root_ids) == 3 and len(set(root_ids)) == 3, "root IDs must be unique")
    weights = [root.get("sample_weight") for root in roots]
    require(
        all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in weights)
        and math.isclose(sum(float(value) for value in weights), 1.0, abs_tol=1e-12),
        "root sample weights must sum to one",
    )
    activity = config.get("vocal_activity")
    require(isinstance(activity, dict), "config vocal_activity must be an object")
    require(activity.get("block_frames") == 44_100, "activity block must be one second")
    require(
        activity.get("mean_square_floor_int16") == 1_073_742,
        "activity floor must remain 1073742",
    )
    require(activity.get("comparison") == ">", "activity comparison must remain strict >")


def load_context(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = load_json(config_path, "corpus config")
    validate_config(config)

    docs: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for root in config["roots"]:
        selection = root["selection"]
        kind = selection["kind"]
        if kind == "names_from_manifest":
            paths["musdb_train_manifest"] = Path(selection["manifest"]).resolve()
        elif kind == "recordpool_selection":
            paths["recordpool_selection"] = Path(selection["manifest"]).resolve()
            paths["recordpool_validation"] = Path(selection["validation"]).resolve()
        elif kind != "all_directories":
            raise RuntimeError(f"unsupported selection kind: {kind}")

    exclusions = config["exclusion_evidence"]
    paths["musdb_validation_manifest"] = Path(
        exclusions["musdb_validation_manifest"]
    ).resolve()
    paths["musdb_test_manifest"] = Path(exclusions["musdb_test_manifest"]).resolve()
    paths["recordpool_holdout_ids"] = Path(exclusions["recordpool_holdout_ids"]).resolve()
    paths["reference_fingerprints"] = Path(config["fingerprint_evidence"]["path"]).resolve()

    for label, path in paths.items():
        require(path.is_file(), f"missing provenance input {label}: {path}")
        if label != "recordpool_holdout_ids":
            docs[label] = load_json(path, label)

    for label in (
        "musdb_train_manifest",
        "musdb_validation_manifest",
        "musdb_test_manifest",
    ):
        doc = docs[label]
        validate_payload_hash(doc, label)
        require(doc.get("schema_version") == 1, f"{label}: unsupported schema")
        require(doc.get("dataset") == "MUSDB18-HQ", f"{label}: wrong dataset")

    selection_doc = docs["recordpool_selection"]
    validation_doc = docs["recordpool_validation"]
    require(
        selection_doc.get("schema") == "hs-tasnet-recordpool-best200-selection-v1",
        "unexpected RecordPool selection schema",
    )
    require(
        validation_doc.get("schema") == "hs-tasnet-recordpool-best200-validation-v1",
        "unexpected RecordPool validation schema",
    )
    selection_hash = sha256_file(paths["recordpool_selection"])
    require(
        validation_doc.get("selection_sha256") == selection_hash,
        "RecordPool validation is not bound to the selected track file",
    )
    reference_doc = docs["reference_fingerprints"]
    require(
        reference_doc.get("schema") == "hs-tasnet-evaluation-reference-fingerprints-v1",
        "unexpected reference fingerprint schema",
    )

    provenance: dict[str, Any] = {
        "builder": provenance_record(Path(__file__).resolve()),
        "config": provenance_record(config_path, schema=config.get("schema_version")),
        "inputs": {},
    }
    for label, path in sorted(paths.items()):
        schema = docs.get(label, {}).get("schema") if label in docs else None
        if schema is None and label in docs:
            schema = docs[label].get("schema_version")
        provenance["inputs"][label] = provenance_record(path, schema=schema)
        if label in docs and "content_sha256" in docs[label]:
            provenance["inputs"][label]["content_sha256"] = docs[label][
                "content_sha256"
            ]

    return {
        "config": config,
        "config_path": config_path,
        "paths": paths,
        "docs": docs,
        "provenance": provenance,
        "holdout_ids": read_holdout_ids(paths["recordpool_holdout_ids"]),
    }


def track_names(manifest: Mapping[str, Any], label: str) -> list[str]:
    raw_tracks = manifest.get("tracks")
    require(isinstance(raw_tracks, list), f"{label}: tracks must be a list")
    names = sorted(track.get("name") for track in raw_tracks if isinstance(track, dict))
    require(
        len(names) == len(raw_tracks)
        and all(isinstance(name, str) and name for name in names)
        and len(names) == len(set(names)),
        f"{label}: invalid or duplicate track names",
    )
    require(manifest.get("track_count") == len(names), f"{label}: track_count mismatch")
    return names


def select_tracks(context: Mapping[str, Any]) -> tuple[list[TrackSpec], dict[str, Any]]:
    config = context["config"]
    docs = context["docs"]
    holdout_ids = set(context["holdout_ids"])
    train_names = track_names(docs["musdb_train_manifest"], "MUSDB train manifest")
    valid_names = track_names(
        docs["musdb_validation_manifest"], "MUSDB validation manifest"
    )
    test_names = track_names(docs["musdb_test_manifest"], "MUSDB test manifest")
    require(len(train_names) == 83, f"expected 83 canonical MUSDB train tracks, got {len(train_names)}")
    require(len(valid_names) == 14, f"expected 14 MUSDB validation tracks, got {len(valid_names)}")
    require(len(test_names) == 46, f"expected 46 local MUSDB test tracks, got {len(test_names)}")
    require(not set(train_names) & set(valid_names), "MUSDB train/validation overlap")
    require(not set(train_names) & set(test_names), "MUSDB train/test overlap")

    recordpool_rows = docs["recordpool_selection"].get("tracks")
    validation_records = docs["recordpool_validation"].get("records")
    require(isinstance(recordpool_rows, list), "RecordPool selection tracks must be a list")
    require(isinstance(validation_records, dict), "RecordPool validation records must be an object")
    recordpool_by_uid = {
        row["uid"]: row
        for row in recordpool_rows
        if isinstance(row, dict) and isinstance(row.get("uid"), str)
    }
    require(
        len(recordpool_rows) == 200 and len(recordpool_by_uid) == 200,
        "RecordPool selection must contain 200 unique UIDs",
    )
    require(not set(recordpool_by_uid) & holdout_ids, "RecordPool selection contains a holdout UID")
    require(
        set(recordpool_by_uid) == set(validation_records),
        "RecordPool validation inventory differs from selection",
    )

    specs: list[TrackSpec] = []
    root_inventory: dict[str, Any] = {}
    for root_config in config["roots"]:
        root_id = root_config["root_id"]
        root = Path(root_config["path"]).expanduser().resolve()
        require(root.is_dir(), f"root does not exist: {root}")
        actual_directories = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
        kind = root_config["selection"]["kind"]
        if kind == "names_from_manifest":
            selected_names = train_names
            require(
                set(actual_directories) == set(train_names) | set(valid_names),
                "MUSDB on-disk train directory is not exactly canonical train + validation",
            )
            metadata_by_name = {
                name: {"dataset": "MUSDB18-HQ", "canonical_split": "train"}
                for name in selected_names
            }
            prior_by_name: dict[str, Mapping[str, Any] | None] = {}
        elif kind == "all_directories":
            selected_names = actual_directories
            metadata_by_name = {
                name: {"dataset": "MoisesDB", "canonical_split": "train"}
                for name in selected_names
            }
            prior_by_name = {}
        elif kind == "recordpool_selection":
            selected_names = sorted(recordpool_by_uid)
            require(
                set(actual_directories) == set(selected_names),
                "RecordPool training root differs from the frozen 200-track selection",
            )
            metadata_by_name = {}
            prior_by_name = {}
            for uid in selected_names:
                row = recordpool_by_uid[uid]
                meta = row.get("metadata", {})
                metadata_by_name[uid] = {
                    "dataset": "RecordPool BS-RoFormer teacher stems",
                    "selection_role": row.get("selection_role"),
                    "artist": meta.get("artist"),
                    "title": meta.get("title"),
                    "artist_key": row.get("artist_key"),
                    "title_key": row.get("title_key"),
                    "source_container_sha256": row.get("source_sha256"),
                    "mixture_chromaprint_sha256": row.get(
                        "mixture_chromaprint", {}
                    ).get("sha256"),
                }
                prior_by_name[uid] = validation_records[uid]
                require(
                    not validation_records[uid].get("problems"),
                    f"RecordPool prior validation contains problems for {uid}",
                )
        else:
            raise RuntimeError(f"unsupported root selection kind: {kind}")

        input_track_count = len(selected_names)
        expected_input_count = root_config.get(
            "expected_input_track_count", root_config["expected_track_count"]
        )
        require(
            input_track_count == expected_input_count,
            f"{root_id}: expected {expected_input_count} input tracks, "
            f"found {input_track_count}",
        )

        duplicate_exclusions = root_config.get("exact_duplicate_exclusions", [])
        require(
            isinstance(duplicate_exclusions, list),
            f"{root_id}: exact_duplicate_exclusions must be a list",
        )
        excluded_names: set[str] = set()
        validated_exclusions: list[dict[str, Any]] = []
        for exclusion in duplicate_exclusions:
            require(isinstance(exclusion, dict), f"{root_id}: malformed duplicate exclusion")
            excluded_name = exclusion.get("excluded_name")
            retained_name = exclusion.get("retained_name")
            expected_hashes = exclusion.get("file_sha256")
            require(
                isinstance(excluded_name, str)
                and isinstance(retained_name, str)
                and excluded_name != retained_name,
                f"{root_id}: invalid duplicate exclusion names",
            )
            require(
                excluded_name in selected_names and retained_name in selected_names,
                f"{root_id}: duplicate exclusion names are not both selected",
            )
            require(
                excluded_name not in excluded_names,
                f"{root_id}: duplicate track excluded more than once: {excluded_name}",
            )
            require(
                isinstance(expected_hashes, dict)
                and set(expected_hashes) == set(FILE_ORDER)
                and all(
                    isinstance(expected_hashes[source], str)
                    and len(expected_hashes[source]) == 64
                    for source in FILE_ORDER
                ),
                f"{root_id}: duplicate exclusion must freeze all five SHA-256 values",
            )
            pair_specs = [
                TrackSpec(root_id, root, name, metadata_by_name[name], prior_by_name.get(name))
                for name in (retained_name, excluded_name)
            ]
            pair_files = [resolve_track_files(spec) for spec in pair_specs]
            observed_hashes: dict[str, str] = {}
            for source in FILE_ORDER:
                retained_hash = sha256_file(pair_files[0][source])
                excluded_hash = sha256_file(pair_files[1][source])
                require(
                    retained_hash == excluded_hash,
                    f"{root_id}: configured duplicate differs for {source}: "
                    f"{retained_name} != {excluded_name}",
                )
                require(
                    retained_hash == expected_hashes[source],
                    f"{root_id}: configured duplicate hash changed for {source}",
                )
                observed_hashes[source] = retained_hash
            excluded_names.add(excluded_name)
            validated_exclusions.append(
                {
                    "excluded_name": excluded_name,
                    "excluded_id": f"{root_id}:{excluded_name}",
                    "retained_name": retained_name,
                    "retained_id": f"{root_id}:{retained_name}",
                    "reason": exclusion.get("reason"),
                    "file_sha256": observed_hashes,
                }
            )
        selected_names = [name for name in selected_names if name not in excluded_names]

        expected_count = root_config["expected_track_count"]
        require(
            len(selected_names) == expected_count,
            f"{root_id}: expected {expected_count} tracks, found {len(selected_names)}",
        )
        missing = [name for name in selected_names if not (root / name).is_dir()]
        require(not missing, f"{root_id}: selected track directories missing: {missing[:5]}")
        root_inventory[root_id] = {
            "root_id": root_id,
            "path": str(root),
            "sample_weight": float(root_config["sample_weight"]),
            "selection_kind": kind,
            "input_track_count": input_track_count,
            "track_count": len(selected_names),
            "exact_duplicate_exclusions": validated_exclusions,
        }
        specs.extend(
            TrackSpec(
                root_id=root_id,
                root=root,
                name=name,
                metadata=metadata_by_name[name],
                prior_validation=prior_by_name.get(name),
            )
            for name in selected_names
        )

    stable_ids = [spec.stable_id for spec in specs]
    require(len(stable_ids) == len(set(stable_ids)), "duplicate stable training track IDs")
    require(len(specs) == 501, f"combined corpus must contain 501 tracks, got {len(specs)}")
    return specs, root_inventory


def resolve_track_files(spec: TrackSpec) -> dict[str, Path]:
    directory = spec.directory
    require(directory.is_dir(), f"missing track directory: {directory}")
    root_resolved = spec.root.resolve()
    resolved: dict[str, Path] = {}
    for source in FILE_ORDER:
        matches = [
            directory / f"{source}.{extension}"
            for extension in ALLOWED_EXTENSIONS
            if (directory / f"{source}.{extension}").is_file()
        ]
        require(
            len(matches) == 1,
            f"{spec.stable_id}: expected exactly one WAV/FLAC for {source}, got {matches}",
        )
        path = matches[0]
        require(
            path.resolve().is_relative_to(root_resolved),
            f"{spec.stable_id}: audio path escapes root: {path}",
        )
        resolved[source] = path

    recognized_audio = {
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
    }
    require(
        recognized_audio == set(resolved.values()),
        f"{spec.stable_id}: unexpected WAV/FLAC files: "
        f"{sorted(str(path) for path in recognized_audio - set(resolved.values()))}",
    )
    return resolved


def file_info(path: Path, expected_sample_rate: int, expected_channels: int) -> dict[str, Any]:
    info = sf.info(str(path))
    require(info.format in ALLOWED_FORMATS, f"unsupported audio format {info.format}: {path}")
    require(info.samplerate == expected_sample_rate, f"sample rate differs: {path}")
    require(info.channels == expected_channels, f"channel count differs: {path}")
    require(info.frames > 0, f"empty audio file: {path}")
    expected_format = "WAV" if path.suffix.lower() == ".wav" else "FLAC"
    require(info.format == expected_format, f"extension/format mismatch: {path}")
    return {
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def vocal_active_seconds(
    path: Path,
    *,
    effective_frames: int,
    sample_rate: int,
    channels: int,
    mean_square_floor: int,
) -> list[int]:
    full_seconds = effective_frames // sample_rate
    active: list[int] = []
    with sf.SoundFile(str(path), mode="r") as handle:
        require(handle.samplerate == sample_rate, f"vocal sample rate changed: {path}")
        require(handle.channels == channels, f"vocal channel count changed: {path}")
        for second in range(full_seconds):
            block = handle.read(sample_rate, dtype="int16", always_2d=True)
            require(
                block.shape == (sample_rate, channels),
                f"short vocal activity block {second}: {path}",
            )
            integers = block.astype(np.int64, copy=False)
            square_sum = int(np.multiply(integers, integers).sum(dtype=np.int64))
            threshold_sum = mean_square_floor * sample_rate * channels
            if square_sum > threshold_sum:
                active.append(second)
    return active


def validate_prior_recordpool(
    spec: TrackSpec,
    source: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if spec.prior_validation is None:
        return {}
    prior_hashes = spec.prior_validation.get("file_sha256")
    prior_info = spec.prior_validation.get("basic_info")
    require(isinstance(prior_hashes, dict), f"{spec.stable_id}: no prior file hashes")
    require(isinstance(prior_info, dict), f"{spec.stable_id}: no prior basic_info")
    expected_hash = prior_hashes.get(source)
    expected_info = prior_info.get(source)
    require(isinstance(expected_hash, str), f"{spec.stable_id}: missing prior {source} hash")
    require(isinstance(expected_info, dict), f"{spec.stable_id}: missing prior {source} info")
    require(record["sha256"] == expected_hash, f"{spec.stable_id}: {source} hash changed")
    comparisons = {
        "frames": "frames",
        "sample_rate": "sample_rate",
        "channels": "channels",
        "format": "format",
        "subtype": "subtype",
        "size_bytes": "size",
    }
    for current_key, old_key in comparisons.items():
        require(
            record[current_key] == expected_info.get(old_key),
            f"{spec.stable_id}: prior {source} {current_key} differs",
        )
    return {
        "prior_validation_match": True,
        "prior_validation_sha256": expected_hash,
    }


def process_track(
    spec: TrackSpec,
    *,
    sample_rate: int,
    channels: int,
    mean_square_floor: int,
) -> dict[str, Any]:
    paths = resolve_track_files(spec)
    records: dict[str, dict[str, Any]] = {}
    for source in FILE_ORDER:
        path = paths[source]
        info = file_info(path, sample_rate, channels)
        record: dict[str, Any] = {
            "relative_path": path.relative_to(spec.root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            **info,
        }
        record.update(validate_prior_recordpool(spec, source, record))
        records[source] = record

    mixture_frames = records["mixture"]["frames"]
    shorter_sources = [
        source
        for source in FILE_ORDER[1:]
        if records[source]["frames"] < mixture_frames
    ]
    require(
        not shorter_sources,
        f"{spec.stable_id}: sources shorter than mixture: {shorter_sources}",
    )
    effective_frames = min(record["frames"] for record in records.values())
    tail_frames = {
        source: record["frames"] - effective_frames for source, record in records.items()
    }
    active = vocal_active_seconds(
        paths["vocals"],
        effective_frames=effective_frames,
        sample_rate=sample_rate,
        channels=channels,
        mean_square_floor=mean_square_floor,
    )
    return {
        "id": spec.stable_id,
        "root_id": spec.root_id,
        "name": spec.name,
        "effective_frames": effective_frames,
        "effective_duration_seconds": effective_frames / sample_rate,
        "tail_frames": tail_frames,
        "longer_tail_sources": [
            source for source in FILE_ORDER if tail_frames[source] > 0
        ],
        "vocal_active_seconds": active,
        "vocal_active_full_second_count": effective_frames // sample_rate,
        "vocal_active_fraction": (
            len(active) / (effective_frames // sample_rate)
            if effective_frames // sample_rate
            else 0.0
        ),
        "files": records,
        "metadata": dict(spec.metadata),
    }


def process_tracks(
    specs: Sequence[TrackSpec], context: Mapping[str, Any], workers: int
) -> list[dict[str, Any]]:
    activity = context["config"]["vocal_activity"]

    def one(spec: TrackSpec) -> dict[str, Any]:
        return process_track(
            spec,
            sample_rate=context["config"]["sample_rate"],
            channels=context["config"]["channels"],
            mean_square_floor=activity["mean_square_floor_int16"],
        )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(one, specs), 1):
            results.append(result)
            if index % 25 == 0 or index == len(specs):
                print(f"validated {index}/{len(specs)} tracks", flush=True)
    return results


def excluded_identities(context: Mapping[str, Any]) -> dict[str, Any]:
    docs = context["docs"]
    provenance = context["provenance"]["inputs"]
    return {
        "musdb_validation": {
            "identity_type": "exact_track_name",
            "values": track_names(docs["musdb_validation_manifest"], "MUSDB validation"),
            "provenance_sha256": provenance["musdb_validation_manifest"]["sha256"],
        },
        "musdb_test": {
            "identity_type": "exact_track_name",
            "values": track_names(docs["musdb_test_manifest"], "MUSDB test"),
            "provenance_sha256": provenance["musdb_test_manifest"]["sha256"],
        },
        "recordpool_holdout": {
            "identity_type": "exact_uuid",
            "values": list(context["holdout_ids"]),
            "provenance_sha256": provenance["recordpool_holdout_ids"]["sha256"],
        },
    }


def exact_identity_collisions(
    tracks: Sequence[Mapping[str, Any]], exclusions: Mapping[str, Any]
) -> list[dict[str, Any]]:
    protected_names: dict[str, list[str]] = {}
    for group in ("musdb_validation", "musdb_test"):
        for value in exclusions[group]["values"]:
            protected_names.setdefault(normalize_identity(value), []).append(f"{group}:{value}")
    protected_uids = set(exclusions["recordpool_holdout"]["values"])
    collisions: list[dict[str, Any]] = []
    for track in tracks:
        keys = {normalize_identity(str(track["name"]))}
        metadata = track.get("metadata", {})
        artist, title = metadata.get("artist"), metadata.get("title")
        if isinstance(artist, str) and isinstance(title, str):
            keys.add(normalize_identity(f"{artist} - {title}"))
        matched = sorted({item for key in keys for item in protected_names.get(key, [])})
        if matched:
            collisions.append({"training_id": track["id"], "protected_identities": matched})
        if track["root_id"] == "recordpool_best200_v1" and track["name"] in protected_uids:
            collisions.append(
                {
                    "training_id": track["id"],
                    "protected_identities": [f"recordpool_holdout:{track['name']}"],
                }
            )
    return collisions


def chromaprint(path: Path) -> dict[str, Any]:
    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-t",
                "120",
                "-f",
                "chromaprint",
                "-fp_format",
                "raw",
                "-",
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        stderr = getattr(error, "stderr", b"") or b""
        raise RuntimeError(
            f"Chromaprint failed for {path}: {stderr.decode('utf-8', errors='replace')[-500:]}"
        ) from error
    require(
        len(process.stdout) >= 100,
        f"implausibly short Chromaprint for {path}: {len(process.stdout)} bytes",
    )
    return {
        "algorithm": CHROMAPRINT_ALGORITHM,
        "sha256": hashlib.sha256(process.stdout).hexdigest(),
        "bytes": len(process.stdout),
    }


def build_fingerprint_report(
    context: Mapping[str, Any],
    tracks: Sequence[Mapping[str, Any]],
    exclusions: Mapping[str, Any],
    root_inventory: Mapping[str, Any],
    workers: int,
) -> dict[str, Any]:
    config = context["config"]
    reference = context["docs"]["reference_fingerprints"]
    selection_rows = context["docs"]["recordpool_selection"]["tracks"]
    identities = reference.get("identities")
    records = reference.get("records")
    require(isinstance(identities, dict), "reference fingerprints lack identities")
    require(isinstance(records, dict), "reference fingerprints lack records")

    prefixes = config["fingerprint_evidence"]["reference_prefixes"]
    chosen_reference_keys: list[str] = []
    reference_counts: dict[str, int] = {}
    for prefix, expected_count in prefixes.items():
        keys = sorted(key for key in records if key.startswith(prefix))
        require(
            len(keys) == expected_count,
            f"fingerprint prefix {prefix}: expected {expected_count}, got {len(keys)}",
        )
        reference_counts[prefix] = len(keys)
        chosen_reference_keys.extend(keys)
    require(
        len(chosen_reference_keys) == len(set(chosen_reference_keys)),
        "fingerprint prefix groups overlap",
    )

    identity_failures: list[dict[str, Any]] = []
    reference_fingerprints: dict[str, str] = {}
    for key in chosen_reference_keys:
        identity = identities.get(key)
        record = records.get(key)
        if not isinstance(identity, dict) or not isinstance(record, dict):
            identity_failures.append({"reference": key, "reason": "missing identity/record"})
            continue
        path = Path(str(identity.get("path", "")))
        if not path.is_file():
            identity_failures.append({"reference": key, "reason": "path missing", "path": str(path)})
            continue
        stat = path.stat()
        observed = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        expected = {
            "path": identity.get("path"),
            "size": identity.get("size"),
            "mtime_ns": identity.get("mtime_ns"),
        }
        if observed != expected:
            identity_failures.append(
                {"reference": key, "reason": "identity changed", "expected": expected, "observed": observed}
            )
            continue
        require(
            all(record.get(field) == identity.get(field) for field in ("path", "size", "mtime_ns")),
            f"reference record is not bound to identity for {key}",
        )
        fingerprint = record.get("chromaprint")
        require(isinstance(fingerprint, dict), f"missing Chromaprint for {key}")
        fingerprint_hash = fingerprint.get("sha256")
        require(
            isinstance(fingerprint_hash, str) and len(fingerprint_hash) == 64,
            f"invalid Chromaprint hash for {key}",
        )
        require(
            fingerprint.get("algorithm") == CHROMAPRINT_ALGORITHM,
            f"unexpected fingerprint algorithm for {key}",
        )
        reference_fingerprints[key] = fingerprint_hash
    require(not identity_failures, f"stale fingerprint evidence: {identity_failures[:3]}")

    training_records: dict[str, dict[str, Any]] = {}
    for row in selection_rows:
        uid = row["uid"]
        fingerprint = row.get("mixture_chromaprint")
        require(isinstance(fingerprint, dict), f"selection lacks fingerprint for {uid}")
        value = fingerprint.get("sha256")
        require(isinstance(value, str) and len(value) == 64, f"bad training fingerprint: {uid}")
        training_records[f"recordpool_best200_v1:{uid}"] = {
            "algorithm": CHROMAPRINT_ALGORITHM,
            "sha256": value,
            "bytes": fingerprint.get("bytes"),
            "evidence": "reused frozen RecordPool selection fingerprint",
        }
    require(len(training_records) == 200, "expected 200 RecordPool fingerprints")

    root_paths = {
        root["root_id"]: Path(root["path"]).expanduser().resolve()
        for root in config["roots"]
    }
    fresh_tasks: list[tuple[str, Path]] = []
    for track in tracks:
        if track["root_id"] == "recordpool_best200_v1":
            continue
        path = root_paths[track["root_id"]] / track["files"]["mixture"]["relative_path"]
        require(path.is_file(), f"missing training mixture for fingerprint: {path}")
        fresh_tasks.append((track["id"], path))
    for root_id, inventory in root_inventory.items():
        for exclusion in inventory["exact_duplicate_exclusions"]:
            path = root_paths[root_id] / exclusion["excluded_name"] / "mixture.wav"
            require(path.is_file(), f"missing excluded duplicate mixture: {path}")
            fresh_tasks.append((exclusion["excluded_id"], path))

    require(
        len(fresh_tasks) == len({track_id for track_id, _ in fresh_tasks}),
        "duplicate fresh fingerprint task IDs",
    )

    def fingerprint_one(task: tuple[str, Path]) -> tuple[str, dict[str, Any]]:
        track_id, path = task
        return track_id, {
            **chromaprint(path),
            "evidence": "freshly computed from frozen input mixture",
        }

    print(f"fingerprinting {len(fresh_tasks)} MUSDB/Moises input mixtures", flush=True)
    with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
        for index, (track_id, record) in enumerate(
            pool.map(fingerprint_one, sorted(fresh_tasks)), 1
        ):
            training_records[track_id] = record
            if index % 25 == 0 or index == len(fresh_tasks):
                print(f"fingerprinted {index}/{len(fresh_tasks)} mixtures", flush=True)

    expected_input_count = sum(
        inventory["input_track_count"] for inventory in root_inventory.values()
    )
    require(expected_input_count == 502, f"expected 502 corpus inputs, got {expected_input_count}")
    require(
        len(training_records) == expected_input_count,
        f"incomplete acoustic coverage: {len(training_records)}/{expected_input_count}",
    )
    training_fingerprints = {
        track_id: record["sha256"] for track_id, record in training_records.items()
    }

    by_training_hash: dict[str, list[str]] = {}
    for track_id, value in sorted(training_fingerprints.items()):
        by_training_hash.setdefault(value, []).append(track_id)
    internal_collisions = [
        {"sha256": value, "training_ids": ids}
        for value, ids in sorted(by_training_hash.items())
        if len(ids) > 1
    ]
    declared_pairs = {
        frozenset((exclusion["retained_id"], exclusion["excluded_id"]))
        for inventory in root_inventory.values()
        for exclusion in inventory["exact_duplicate_exclusions"]
    }
    declared_internal_collisions = [
        collision
        for collision in internal_collisions
        if frozenset(collision["training_ids"]) in declared_pairs
    ]
    unexpected_internal_collisions = [
        collision
        for collision in internal_collisions
        if frozenset(collision["training_ids"]) not in declared_pairs
    ]
    observed_declared_pairs = {
        frozenset(collision["training_ids"])
        for collision in declared_internal_collisions
    }
    require(
        observed_declared_pairs == declared_pairs,
        "configured exact duplicate exclusion is absent from acoustic fingerprints",
    )
    by_reference_hash: dict[str, list[str]] = {}
    for reference_id, value in sorted(reference_fingerprints.items()):
        by_reference_hash.setdefault(value, []).append(reference_id)
    evaluation_collisions = []
    for value in sorted(set(by_training_hash) & set(by_reference_hash)):
        evaluation_collisions.append(
            {
                "sha256": value,
                "training_ids": by_training_hash[value],
                "reference_ids": by_reference_hash[value],
            }
        )

    identity_collisions = exact_identity_collisions(tracks, exclusions)
    status = (
        "pass"
        if not unexpected_internal_collisions
        and not evaluation_collisions
        and not identity_collisions
        else "collision"
    )
    report = {
        "schema_version": 1,
        "kind": "hs_tasnet_acoustic_collision_report",
        "corpus_id": config["corpus_id"],
        "status": status,
        "algorithm": "SHA-256 of ffmpeg Chromaprint raw output over first 120 seconds",
        "evidence": {
            "recordpool_selection_sha256": context["provenance"]["inputs"][
                "recordpool_selection"
            ]["sha256"],
            "reference_fingerprints_sha256": context["provenance"]["inputs"][
                "reference_fingerprints"
            ]["sha256"],
            "reference_identity_validation": "pass",
            "reference_counts": reference_counts,
        },
        "training_coverage": {
            "input_tracks": expected_input_count,
            "included_tracks": len(tracks),
            "fingerprinted_input_tracks": len(training_fingerprints),
            "fingerprinted_root_ids": sorted(root_paths),
            "unfingerprinted_root_ids": [],
            "input_coverage_fraction": len(training_fingerprints) / expected_input_count,
            "status": "pass",
        },
        "training_fingerprint_records": dict(sorted(training_records.items())),
        "training_fingerprints": dict(sorted(training_fingerprints.items())),
        "reference_fingerprints": dict(sorted(reference_fingerprints.items())),
        "internal_training_collisions": internal_collisions,
        "declared_internal_collisions": declared_internal_collisions,
        "unexpected_internal_collisions": unexpected_internal_collisions,
        "evaluation_reference_collisions": evaluation_collisions,
        "exact_excluded_identity_collisions": identity_collisions,
        "limitations": [
            "A first-120-second exact Chromaprint hash does not detect shifted, cropped, remastered, or near-duplicate audio.",
            "Reference fingerprints were reused only after exact path, size, and mtime_ns identity validation.",
            "RecordPool training fingerprints were reused from the frozen selection; MUSDB and Moises input fingerprints were freshly computed.",
        ],
    }
    require(status == "pass", "training/evaluation collision evidence is not clean")
    return seal_payload(report)


def mixture_hash_collisions(tracks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for track in tracks:
        grouped.setdefault(track["files"]["mixture"]["sha256"], []).append(track["id"])
    return [
        {"sha256": value, "training_ids": ids}
        for value, ids in sorted(grouped.items())
        if len(ids) > 1
    ]


def root_exclusions(root_id: str, exclusions: Mapping[str, Any]) -> dict[str, Any]:
    if root_id == "musdb18hq_train":
        return {
            "musdb_validation": exclusions["musdb_validation"],
            "musdb_test": exclusions["musdb_test"],
        }
    if root_id == "recordpool_best200_v1":
        return {"recordpool_holdout": exclusions["recordpool_holdout"]}
    return {}


def build_manifests(
    context: Mapping[str, Any],
    tracks: Sequence[dict[str, Any]],
    root_inventory: Mapping[str, Any],
    fingerprint_report: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    config = context["config"]
    exclusions = excluded_identities(context)
    collisions = mixture_hash_collisions(tracks)
    require(not collisions, f"exact duplicate training mixtures: {collisions[:3]}")

    common = {
        "schema_version": 1,
        "corpus_id": config["corpus_id"],
        "sample_rate": config["sample_rate"],
        "channels": config["channels"],
        "source_order": config["source_order"],
        "file_order": list(FILE_ORDER),
        "validation_policy": {
            "allowed_formats": sorted(ALLOWED_FORMATS),
            "effective_frames": "minimum of all five files",
            "source_length": "every stem must be at least as long as mixture; longer tails are recorded",
            "file_hash": "fresh SHA-256 of every complete WAV/FLAC file",
            "vocal_activity": config["vocal_activity"],
        },
        "provenance": context["provenance"],
    }

    root_payloads: dict[str, dict[str, Any]] = {}
    for root_config in config["roots"]:
        root_id = root_config["root_id"]
        selected = [track for track in tracks if track["root_id"] == root_id]
        inventory = root_inventory[root_id]
        root_entry = {
            "root_id": root_id,
            "path": inventory["path"],
            "sample_weight": inventory["sample_weight"],
            "track_count": len(selected),
            "input_track_count": inventory["input_track_count"],
            "exact_duplicate_exclusions": inventory["exact_duplicate_exclusions"],
        }
        payload = {
            **common,
            "kind": "hs_tasnet_training_root_manifest",
            "roots": [root_entry],
            "track_count": len(selected),
            "total_effective_frames": sum(track["effective_frames"] for track in selected),
            "total_effective_hours": sum(track["effective_frames"] for track in selected)
            / config["sample_rate"]
            / 3600,
            "excluded_identities": root_exclusions(root_id, exclusions),
            "tracks": selected,
        }
        root_payloads[root_id] = seal_payload(payload)

    root_manifest_names = {
        root_id: f"{root_id}.manifest.json" for root_id in root_payloads
    }
    combined_roots = []
    for root_config in config["roots"]:
        root_id = root_config["root_id"]
        inventory = root_inventory[root_id]
        combined_roots.append(
            {
                "root_id": root_id,
                "path": inventory["path"],
                "sample_weight": inventory["sample_weight"],
                "track_count": inventory["track_count"],
                "input_track_count": inventory["input_track_count"],
                "exact_duplicate_exclusions": inventory["exact_duplicate_exclusions"],
                "root_manifest": root_manifest_names[root_id],
                "root_manifest_content_sha256": root_payloads[root_id]["content_sha256"],
            }
        )
    combined = seal_payload(
        {
            **common,
            "kind": "hs_tasnet_training_corpus_manifest",
            "roots": combined_roots,
            "sampling_policy": {
                "kind": "choose_root_then_uniform_track",
                "root_probabilities": {
                    root["root_id"]: float(root["sample_weight"])
                    for root in config["roots"]
                },
            },
            "track_count": len(tracks),
            "total_effective_frames": sum(track["effective_frames"] for track in tracks),
            "total_effective_hours": sum(track["effective_frames"] for track in tracks)
            / config["sample_rate"]
            / 3600,
            "excluded_identities": exclusions,
            "integrity": {
                "exact_mixture_file_sha256_collisions": collisions,
                "configured_exact_duplicate_exclusions": [
                    exclusion
                    for root_id in root_inventory
                    for exclusion in root_inventory[root_id]["exact_duplicate_exclusions"]
                ],
                "recordpool_prior_validation_hashes_matched": True,
                "acoustic_collision_report": {
                    "artifact": "acoustic_collision_report.json",
                    "content_sha256": fingerprint_report["content_sha256"],
                    "status": fingerprint_report["status"],
                    "fingerprinted_input_tracks": fingerprint_report[
                        "training_coverage"
                    ]["fingerprinted_input_tracks"],
                    "input_tracks": fingerprint_report["training_coverage"][
                        "input_tracks"
                    ],
                    "coverage_status": fingerprint_report["training_coverage"][
                        "status"
                    ],
                },
            },
            "tracks": list(tracks),
        }
    )
    return root_payloads, combined


def write_artifact_set(
    output_dir: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    rendered = {name: render_json(payload) for name, payload in artifacts.items()}
    artifact_records = {
        name: {
            "file_sha256": hashlib.sha256(content).hexdigest(),
            "content_sha256": artifacts[name]["content_sha256"],
            "size_bytes": len(content),
        }
        for name, content in sorted(rendered.items())
    }
    inventory = seal_payload(
        {
            "schema_version": 1,
            "kind": "hs_tasnet_corpus_artifact_inventory",
            "corpus_id": context["config"]["corpus_id"],
            "builder_sha256": context["provenance"]["builder"]["sha256"],
            "config_sha256": context["provenance"]["config"]["sha256"],
            "artifacts": artifact_records,
        }
    )
    inventory_name = "artifact_inventory.json"
    inventory_bytes = render_json(inventory)

    all_targets = [output_dir / name for name in rendered]
    all_targets.append(output_dir / inventory_name)
    all_targets.extend(
        target.with_name(target.name + ".sha256") for target in list(all_targets)
    )
    if not overwrite:
        existing = [str(path) for path in all_targets if path.exists()]
        if existing:
            raise FileExistsError(
                f"refusing partial overwrite; pass --overwrite: {existing[:5]}"
            )

    for name in sorted(rendered):
        path = output_dir / name
        content = rendered[name]
        file_hash = hashlib.sha256(content).hexdigest()
        atomic_write(path, content, overwrite=overwrite)
        atomic_write(
            path.with_name(path.name + ".sha256"),
            f"{file_hash}  {path.name}\n".encode("ascii"),
            overwrite=overwrite,
        )

    inventory_path = output_dir / inventory_name
    inventory_file_hash = hashlib.sha256(inventory_bytes).hexdigest()
    atomic_write(inventory_path, inventory_bytes, overwrite=overwrite)
    atomic_write(
        inventory_path.with_name(inventory_path.name + ".sha256"),
        f"{inventory_file_hash}  {inventory_path.name}\n".encode("ascii"),
        overwrite=overwrite,
    )
    return inventory


def validate_sidecar(path: Path, expected_hash: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing SHA sidecar: {sidecar}")
    expected_line = f"{expected_hash}  {path.name}\n"
    require(sidecar.read_text(encoding="ascii") == expected_line, f"bad SHA sidecar: {sidecar}")


def validate_existing(output_dir: Path) -> None:
    output_dir = output_dir.expanduser().resolve()
    inventory_path = output_dir / "artifact_inventory.json"
    inventory = load_json(inventory_path, "artifact inventory")
    validate_payload_hash(inventory, "artifact inventory")
    inventory_file_hash = sha256_file(inventory_path)
    validate_sidecar(inventory_path, inventory_file_hash)
    artifacts = inventory.get("artifacts")
    require(isinstance(artifacts, dict) and artifacts, "artifact inventory is empty")
    for name, expected in sorted(artifacts.items()):
        path = output_dir / name
        require(path.is_file(), f"missing frozen artifact: {path}")
        actual_file_hash = sha256_file(path)
        require(
            actual_file_hash == expected.get("file_sha256"),
            f"artifact file hash differs: {path}",
        )
        validate_sidecar(path, actual_file_hash)
        payload = load_json(path, name)
        content_hash = validate_payload_hash(payload, name)
        require(
            content_hash == expected.get("content_sha256"),
            f"artifact content hash differs from inventory: {path}",
        )
        require(path.stat().st_size == expected.get("size_bytes"), f"artifact size differs: {path}")
    print(
        json.dumps(
            {
                "status": "pass",
                "artifact_count": len(artifacts),
                "inventory_content_sha256": inventory["content_sha256"],
            },
            sort_keys=True,
        )
    )


def quick_audit(context: Mapping[str, Any], workers: int) -> None:
    specs, root_inventory = select_tracks(context)
    sample_specs = []
    for root_id in root_inventory:
        sample_specs.append(next(spec for spec in specs if spec.root_id == root_id))
    tracks = process_tracks(sample_specs, context, min(workers, len(sample_specs)))
    report = {
        "status": "pass",
        "mode": "quick",
        "full_inventory_counts": {
            root_id: {
                "input": value["input_track_count"],
                "included": value["track_count"],
                "excluded_exact_duplicates": len(value["exact_duplicate_exclusions"]),
            }
            for root_id, value in root_inventory.items()
        },
        "sampled_track_ids": [track["id"] for track in tracks],
        "note": "Quick mode does not write or freeze manifests.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary_name:
        root = Path(temporary_name) / "root"
        track = root / "Synthetic - Tail"
        track.mkdir(parents=True)
        sample_rate = 44_100
        mixture_frames = sample_rate * 2
        quiet = np.zeros((mixture_frames, 2), dtype=np.int16)
        for source in ("mixture", "drums", "bass", "other"):
            sf.write(track / f"{source}.wav", quiet, sample_rate, subtype="PCM_16")
        vocals = np.empty((mixture_frames + 17, 2), dtype=np.int16)
        vocals[:sample_rate] = 2000
        vocals[sample_rate:mixture_frames] = 500
        vocals[mixture_frames:] = 0
        sf.write(track / "vocals.flac", vocals, sample_rate, subtype="PCM_16")
        spec = TrackSpec(
            root_id="synthetic",
            root=root,
            name=track.name,
            metadata={"dataset": "self-test"},
        )
        result = process_track(
            spec,
            sample_rate=sample_rate,
            channels=2,
            mean_square_floor=1_073_742,
        )
        require(result["effective_frames"] == mixture_frames, "self-test effective frames")
        require(result["tail_frames"]["vocals"] == 17, "self-test vocal tail")
        require(result["vocal_active_seconds"] == [0], "self-test activity gate")
        payload = seal_payload({"schema_version": 1, "result": result})
        path = Path(temporary_name) / "self-test.json"
        atomic_write(path, render_json(payload), overwrite=False)
        loaded = load_json(path, "self-test output")
        validate_payload_hash(loaded, "self-test output")
    print(json.dumps({"status": "pass", "mode": "self-test"}, sort_keys=True))


def full_build(
    context: Mapping[str, Any], output_dir: Path, workers: int, *, overwrite: bool
) -> None:
    specs, root_inventory = select_tracks(context)
    tracks = process_tracks(specs, context, workers)
    exclusions = excluded_identities(context)
    fingerprint_report = build_fingerprint_report(
        context, tracks, exclusions, root_inventory, workers
    )
    root_payloads, combined = build_manifests(
        context, tracks, root_inventory, fingerprint_report
    )
    artifacts: dict[str, Mapping[str, Any]] = {
        f"{root_id}.manifest.json": payload
        for root_id, payload in root_payloads.items()
    }
    artifacts["combined.manifest.json"] = combined
    artifacts["acoustic_collision_report.json"] = fingerprint_report
    inventory = write_artifact_set(
        output_dir, artifacts, context, overwrite=overwrite
    )
    print("---", flush=True)
    print(f"track_count: {combined['track_count']}")
    print(f"total_effective_hours: {combined['total_effective_hours']:.6f}")
    print(f"combined_content_sha256: {combined['content_sha256']}")
    print(f"collision_report_status: {fingerprint_report['status']}")
    print(f"artifact_inventory_content_sha256: {inventory['content_sha256']}")
    print(f"output_dir: {output_dir.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.validate_only:
        validate_existing(args.output_dir)
        return 0
    context = load_context(args.config)
    if args.quick:
        quick_audit(context, args.workers)
        return 0
    full_build(context, args.output_dir, args.workers, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
