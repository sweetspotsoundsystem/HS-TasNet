"""The publication comparison must observe working files and reject stale plans."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    "source_comparison", Path(__file__).resolve().parents[1] / "scripts/sync_research.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_comparison_includes_uncommitted_sources_and_detects_changes(tmp_path):
    source, helpers, public = (tmp_path / name for name in ("live", "helpers", "public"))
    (source / "research/direct").mkdir(parents=True)
    helpers.mkdir(); public.mkdir()
    subprocess.run(["git", "init", "-q", str(public)], check=True)
    model = source / "research/direct/model.py"
    model.write_text("VALUE = 1\n")
    for name in ("full_config.json", "corpus_config.json"):
        (helpers / name).write_text("{}\n")
    (public / "model.py").write_text("VALUE = 1\n")
    plan = source / "plan.json"
    plan.write_text(json.dumps({"source_bindings": {str(model): comparison.digest(model)}}))
    original = comparison.comparison(source, helpers, plan, public=public)
    assert "model.py" in original["public_files"]
    assert original["research_files"]["repository/research/direct/model.py"]["bound_by_active_plan"]
    extra = source / "research/direct/uncommitted.py"
    extra.write_text("NEW = True\n")
    changed = comparison.comparison(source, helpers, plan, public=public)
    assert changed != original and "repository/research/direct/uncommitted.py" in changed["research_files"]
    extra.unlink()
    (public / "model.py").write_text("VALUE = 2\n")
    assert comparison.comparison(source, helpers, plan, public=public) != original
    model.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Active plan source changed"):
        comparison.comparison(source, helpers, plan, public=public)


def test_comparison_rejects_a_plan_outside_the_live_checkout(tmp_path):
    with pytest.raises(ValueError, match="separate public checkout"):
        comparison.comparison(tmp_path / "live", tmp_path / "helpers", tmp_path / "wrong.json",
                              public=tmp_path / "public")


def test_selected_nested_sources_are_recorded_without_changing_the_active_plan(tmp_path):
    source, helpers, public = (tmp_path / name for name in ("live", "helpers", "public"))
    (source / "research/direct").mkdir(parents=True)
    helpers.mkdir(); public.mkdir()
    subprocess.run(["git", "init", "-q", str(public)], check=True)
    model = source / "research/direct/model.py"
    model.write_text("VALUE = 1\n")
    for name in ("full_config.json", "corpus_config.json"):
        (helpers / name).write_text("{}\n")
    plan = source / "plan.json"
    plan.write_text(json.dumps({"source_bindings": {str(model): comparison.digest(model)}}))
    plan_bytes = plan.read_bytes()
    extra = source / "research/evaluation/quality.py"
    extra.parent.mkdir()
    extra.write_text("METRIC = 'unchanged'\n")
    relative = extra.relative_to(source)
    baseline = comparison.comparison(source, helpers, plan, public=public)
    recorded = comparison.comparison(source, helpers, plan, public=public, extra_sources=[relative])
    key = "repository/" + relative.as_posix()
    assert key not in baseline["research_files"]
    assert recorded["research_files"][key] == {
        "sha256": comparison.digest(extra), "bytes": extra.stat().st_size, "bound_by_active_plan": False}
    assert comparison.comparison(source, helpers, plan, public=public, extra_sources=[extra]) == recorded
    extra.write_text("METRIC = 'changed'\n")
    assert comparison.comparison(source, helpers, plan, public=public, extra_sources=[relative]) != recorded
    assert plan.read_bytes() == plan_bytes
    outside = tmp_path / "unbound.py"
    outside.write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="unbound source lies outside"):
        comparison.comparison(source, helpers, plan, public=public, extra_sources=[outside])
    model.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Active plan source changed"):
        comparison.comparison(source, helpers, plan, public=public, extra_sources=[model])
