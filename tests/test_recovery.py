"""Scientific replay rejects drift while permitting measurement differences."""
import hashlib
import json

import pytest

from stemgenrt.recovery import ReplayVerifier


def journal(tmp_path, rows, tail=b""):
    path = tmp_path / "previous.jsonl"
    contents = b"".join((json.dumps(row) + "\n").encode() for row in rows) + tail
    path.write_bytes(contents)
    return path, hashlib.sha256(contents).hexdigest()


def row(step):
    return {"step": step, "loss": .25, "raw_model_state_sha256": "a" * 64,
            "groups": {"ordinary": {"active": [1, 2, 3, 4]}}, "elapsed_seconds": 12.}


def test_repeats_unsaved_prefix_then_advances_without_reference(tmp_path):
    path, digest = journal(tmp_path, [row(1), row(2), row(3)], tail=b'{"step":4')
    verifier = ReplayVerifier(path, sha256=digest, resume_step=1, schedule_steps=5)
    verifier.compare({**row(2), "elapsed_seconds": 100.})
    verifier.compare({**row(3), "elapsed_seconds": 200.})
    verifier.compare(row(4))
    verifier.verify_source()
    assert verifier.report()["verified_steps"] == [2, 3]
    assert verifier.report()["incomplete_tail_ignored"] is True
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("field", ["loss", "raw_model_state_sha256", "groups", "missing", "extra"])
def test_replay_rejects_changed_or_missing_scientific_fields(tmp_path, field):
    path, digest = journal(tmp_path, [row(1), row(2)])
    verifier = ReplayVerifier(path, sha256=digest, resume_step=1, schedule_steps=3)
    actual = row(2)
    if field == "missing": del actual["groups"]
    elif field == "extra": actual["extra"] = 1
    else: actual[field] = "changed"
    with pytest.raises(ValueError, match="scientific update differs"):
        verifier.compare(actual)
    assert verifier.report()["verified_steps"] == [] and verifier.next_step == 2


@pytest.mark.parametrize("steps", [[1, 3], [1, 2, 2], [2, 1], [True, 2], [1], [3, 4], [1, 2, 4]])
def test_rejects_incomplete_or_ambiguous_reference(tmp_path, steps):
    path, digest = journal(tmp_path, [row(step) for step in steps])
    with pytest.raises(ValueError):
        ReplayVerifier(path, sha256=digest, resume_step=1, schedule_steps=3)


def test_reference_digest_and_live_mutation_are_rejected(tmp_path):
    path, digest = journal(tmp_path, [row(1), row(2)])
    with pytest.raises(ValueError, match="SHA-256"):
        ReplayVerifier(path, sha256="0" * 64, resume_step=1, schedule_steps=3)
    verifier = ReplayVerifier(path, sha256=digest, resume_step=1, schedule_steps=3)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed during continuation"):
        verifier.verify_source()


def test_verifier_cannot_skip_a_recorded_update(tmp_path):
    path, digest = journal(tmp_path, [row(1), row(2), row(3)])
    verifier = ReplayVerifier(path, sha256=digest, resume_step=1, schedule_steps=3)
    with pytest.raises(ValueError, match="skipped or reordered"):
        verifier.compare(row(3))


def test_complete_malformed_or_nonfinite_rows_are_not_discarded(tmp_path):
    path = tmp_path / 'invalid.jsonl'
    for contents in (b'{"step":1}\n{broken}\n', b'{"step":1}\n{"step":2,"loss":NaN}\n',
                     b'{"step":1}\n{"step":2,"loss":1e999}\n'):
        path.write_bytes(contents)
        with pytest.raises(ValueError):
            ReplayVerifier(path, sha256=hashlib.sha256(contents).hexdigest(), resume_step=1, schedule_steps=3)
