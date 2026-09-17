"""Select recorded training stems without modifying the sealed source corpus."""
from __future__ import annotations

from research.direct.run_latency58_quality import ROOT, read, require, sha
from research.direct.train_latency58 import PRODUCTION

ROOT_WEIGHTS = {"musdb18hq_train": .5, "moisesdb_train": .5}
EXPECTED = {"musdb18hq_train": (83, "MUSDB18-HQ"), "moisesdb_train": (218, "MoisesDB")}


def selection_contract():
    manifest_path = PRODUCTION / "manifests/combined.manifest.json"
    validation_path = ROOT / "research/manifests/valid.json"
    manifest, validation = read(manifest_path), read(validation_path)
    selected = [t for t in manifest["tracks"] if t["root_id"] in ROOT_WEIGHTS]
    validation_names = sorted(t["name"] for t in validation["tracks"])
    exclusions = manifest["excluded_identities"]
    require(len(selected) == 301 and len(validation_names) == 14
            and validation_names == sorted(exclusions["musdb_validation"]["values"])
            and all(sum(t["root_id"] == root for t in selected) == count for root, (count, _) in EXPECTED.items())
            and all(t["metadata"] == {"canonical_split": "train", "dataset": EXPECTED[t["root_id"]][1]} for t in selected)
            and len({t["id"] for t in selected}) == len(selected), "Unexpected recorded training subset")
    blocked = set(validation_names) | set(exclusions["musdb_test"]["values"])
    require(not blocked.intersection(t["name"] for t in selected), "Held-out music entered the training selection")
    return {"version": "recorded301-musdb-moises-training-only-v1",
            "source_manifest_sha256": sha(manifest_path), "validation_manifest_sha256": sha(validation_path),
            "track_ids": sorted(t["id"] for t in selected), "track_count": 301, "root_weights": ROOT_WEIGHTS, "root_counts": {k: v[0] for k, v in EXPECTED.items()},
            "validation_overlap": 0, "test_overlap": 0,
            "target_origin": "recorded MUSDB18-HQ and MoisesDB training stems",
            "ordinary_recorded_mixture": True, "teacher_generated_targets_in_current_stage": False,
            "parent_pretraining_corpus_changed": False}


def select_tracks(tracks, contract):
    require(contract == selection_contract(), "Training selection changed after preparation")
    selected = [t for t in tracks if t.root_id in ROOT_WEIGHTS]
    require(sorted(t.track_id for t in selected) == contract["track_ids"] and len(selected) == 301,
            "Decoded corpus differs from the selected training tracks")
    return selected
