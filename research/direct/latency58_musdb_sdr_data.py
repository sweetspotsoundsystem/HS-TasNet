"""Select recorded training stems without modifying the sealed source corpus."""
from __future__ import annotations

from research.direct.run_latency58_quality import ROOT, read, require, sha
from research.direct.train_latency58 import PRODUCTION


def selection_contract():
    manifest_path = PRODUCTION / "manifests/combined.manifest.json"
    validation_path = ROOT / "research/manifests/valid.json"
    manifest, validation = read(manifest_path), read(validation_path)
    selected = [t for t in manifest["tracks"] if t["root_id"] == "musdb18hq_train"]
    validation_names = sorted(t["name"] for t in validation["tracks"])
    exclusions = manifest["excluded_identities"]
    require(len(selected) == 83 and len(validation_names) == 14
            and validation_names == sorted(exclusions["musdb_validation"]["values"])
            and all(t["metadata"] == {"canonical_split": "train", "dataset": "MUSDB18-HQ"} for t in selected)
            and len({t["id"] for t in selected}) == len(selected), "Unexpected recorded training subset")
    blocked = set(validation_names) | set(exclusions["musdb_test"]["values"])
    require(not blocked.intersection(t["name"] for t in selected), "Held-out music entered the training selection")
    return {"version": "musdb83-recorded-ordinary-training-only-v1",
            "source_manifest_sha256": sha(manifest_path), "validation_manifest_sha256": sha(validation_path),
            "track_ids": sorted(t["id"] for t in selected), "track_count": 83,
            "validation_overlap": 0, "test_overlap": 0,
            "target_origin": "recorded MUSDB18-HQ training stems",
            "ordinary_recorded_mixture": True, "teacher_generated_targets_in_current_stage": False,
            "parent_pretraining_corpus_changed": False}


def select_tracks(tracks, contract):
    require(contract == selection_contract(), "Training selection changed after preparation")
    selected = [t for t in tracks if t.root_id == "musdb18hq_train"]
    require(sorted(t.track_id for t in selected) == contract["track_ids"] and len(selected) == 83,
            "Decoded corpus differs from the selected training tracks")
    return selected
