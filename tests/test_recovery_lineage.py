"""Reject relabelled, missing and corrupt receipts across resumed runs."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.direct.latency58_four_second_recovery_lineage import verify_receipts


def fixture_segment(start=0, end=100, plan_sha="first-plan"):
    journal = b"".join((json.dumps({"step": step}) + "\n").encode() for step in range(1, end + 1))
    lines = journal.splitlines(keepends=True)
    receipts, previous = [], None
    for step in range(start + 50, end + 1, 50):
        row = {"step": step, "sha256": f"checkpoint-{step}", "bytes": 123,
               "plan_sha256": plan_sha, "previous": previous, "planned_stop_step": 2000,
               "journal_sha256": hashlib.sha256(b"".join(lines[:step])).hexdigest()}
        receipts.append(row)
        previous = {key: row[key] for key in ("step", "sha256", "bytes")}
    return receipts, dict(start=start, end=end, plan_sha=plan_sha, journal=journal,
                         checkpoint_sha=f"checkpoint-{end}", planned_stop=2000)


def test_resumed_receipts_start_a_new_chain_with_full_journal_prefix():
    for start, end, plan in ((0, 750, "original"), (750, 1900, "recovery1"), (1900, 2000, "recovery2")):
        receipts, arguments = fixture_segment(start, end, plan)
        assert receipts[0]["previous"] is None
        assert verify_receipts(receipts, **arguments) == list(range(start + 50, end + 1, 50))


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "plan", "previous", "journal", "endpoint", "short"])
def test_corrupt_receipt_segment_is_rejected(corruption):
    receipts, arguments = fixture_segment(1900, 2000, "recovery2")
    if corruption == "missing":
        receipts.pop(0)
    elif corruption == "duplicate":
        receipts.append(copy.deepcopy(receipts[-1]))
    elif corruption == "plan":
        receipts[0]["plan_sha256"] = "recovery1"
    elif corruption == "previous":
        receipts[0]["previous"] = {"step": 1900, "sha256": "old", "bytes": 123}
    elif corruption == "journal":
        arguments["journal"] = arguments["journal"].replace(b'"step": 1}', b'"step": 0}', 1)
    elif corruption == "endpoint":
        arguments["checkpoint_sha"] = "different-save"
    else:
        arguments["journal"] = b"".join(arguments["journal"].splitlines(keepends=True)[:1950])
    with pytest.raises(RuntimeError):
        verify_receipts(receipts, **arguments)


def test_recovery_lineage_fix_preserves_scoring_and_decisions():
    directory = Path(__file__).resolve().parents[1] / "research/direct"
    trees = [ast.parse((directory / name).read_text()) for name in
             ("run_latency58_four_second_recovery_quality.py", "run_latency58_four_second_recovery_quality_v2.py")]
    for name in ("transport_proof", "run_stage", "evaluate", "review", "main"):
        functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
                     for tree in trees]
        assert ast.dump(functions[0], include_attributes=False) == ast.dump(functions[1], include_attributes=False)
