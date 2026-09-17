#!/usr/bin/env python3
"""Publish and verify the complete current streaming research source snapshot.

Only explicit source files are copied; datasets, weights and run output are not.
The scientific source bytes are preserved, including historical path bindings.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research/source-manifest.json"
SOURCE_SUFFIXES = {".py", ".cpp", ".ps1", ".sh"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(source, bound_sources=()):
    """Include every direct-research source, then close imports and file loads."""
    paths = {p for p in (source / "research/direct").iterdir()
             if p.is_file() and p.suffix in SOURCE_SUFFIXES}
    paths.update(source / p for p in (
        "export_onnx.py", "tests/test_export_onnx.py", "hs_tasnet/hs_tasnet.py",
        "hs_tasnet/alternative_rnns.py", "hs_tasnet/trainer.py"))
    paths.update(source / p for p in bound_sources)
    paths.update((source / "research/direct").glob("*.json"))
    paths.update(source / p for p in ("research/eval_config.json", "research/manifests/valid.json"))
    pending, visited = set(paths), set()
    while pending:
        path = pending.pop()
        visited.add(path)
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        candidates = set()
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module, *(node.module + "." + alias.name for alias in node.names)]
            for module in modules:
                if module.split(".")[0] in {"research", "hs_tasnet"}:
                    candidates.add(source.joinpath(*module.split(".")).with_suffix(".py"))
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".py"):
                name = Path(node.value).name
                candidates.update((source / node.value, path.parent / node.value,
                                   source / "research/direct" / name, source / "research" / name))
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_relative_to(source) and candidate.is_file() and candidate not in visited:
                paths.add(candidate)
                pending.add(candidate)
    return {p.relative_to(source).as_posix(): p for p in sorted(paths)}


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_snapshot(args):
    if not all((args.source_root, args.production_root, args.active_plan)):
        raise ValueError("--write requires --source-root, --production-root and --active-plan")
    source, production = args.source_root.resolve(), args.production_root.resolve()
    if source == ROOT or source.is_relative_to(ROOT):
        raise ValueError("Use an independent research checkout as the source")
    plan = args.active_plan.resolve()
    frozen_plan = json.loads(plan.read_text())
    bound_sources = sorted(Path(p).relative_to(source).as_posix()
                           for p in frozen_plan["source_bindings"]
                           if Path(p).suffix == ".py" and Path(p).is_relative_to(source))
    files = {dest: ("repository", path.relative_to(source).as_posix(), path)
             for dest, path in inventory(source, bound_sources).items()}
    for path in sorted(production.glob("*.py")) + sorted((production / "tests").glob("*.py")):
        relative = path.relative_to(production).as_posix()
        files["research/production/" + relative] = ("production", relative, path)
    for name in ("full_config.json", "corpus_config.json"):
        files["research/production/" + name] = ("production", name, production / name)
    files["research/recipes/active-plan.json"] = ("repository", plan.relative_to(source).as_posix(), plan)
    rows = {}
    for relative, (origin, original, path) in sorted(files.items()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular source file: {path}")
        payload = path.read_bytes()
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        rows[relative] = {"origin": origin, "source_path": original,
                          "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    # Removed sources must be reviewed explicitly instead of silently retained.
    if MANIFEST.exists():
        stale = set(json.loads(MANIFEST.read_text())["files"]) - set(rows)
        if any((ROOT / relative).exists() for relative in stale):
            raise ValueError(f"Review and remove retired published sources first: {sorted(stale)}")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    release = json.loads((ROOT / "hs_tasnet/streaming_models.json").read_text())["current"]
    json_write(MANIFEST, {
        "schema": "hs-tasnet-research-source-v1", "source_commit": revision,
        "includes_working_tree_changes": True, "files": rows,
        "plan_bound_repository_sources": bound_sources,
        "release_onnx_sha256": release["sha256"],
        "active_trainer": frozen_plan["trainer_module"],
        "active_plan": "research/recipes/active-plan.json",
        "scope": "All research/direct source files, repository dependencies, production helpers and active recipe",
    })


def verify(args):
    manifest = json.loads(MANIFEST.read_text())
    errors = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected["sha256"]:
            errors.append("Published source differs: " + relative)
        origin = args.source_root if expected["origin"] == "repository" else args.production_root
        if origin:
            original = origin / expected["source_path"]
            if not original.is_file() or original.is_symlink() or digest(original) != expected["sha256"]:
                errors.append("Research source differs: " + relative)
    included = set(manifest["files"])
    local_direct = {p.relative_to(ROOT).as_posix() for p in (ROOT / "research/direct").iterdir()
                    if p.is_file() and p.suffix in SOURCE_SUFFIXES}
    if local_direct != {p for p in included if Path(p).parent.as_posix() == "research/direct"
                        and Path(p).suffix in SOURCE_SUFFIXES}:
        errors.append("Direct research source inventory changed; update the source snapshot")
    if args.source_root:
        source_files = set(inventory(args.source_root.resolve(), manifest["plan_bound_repository_sources"]))
        published = {p for p, row in manifest["files"].items()
                     if row["origin"] == "repository" and p != manifest["active_plan"]}
        if source_files != published:
            errors.append("Research checkout inventory differs: " + repr(sorted(source_files ^ published)))
    if args.production_root:
        actual = {p.relative_to(args.production_root).as_posix()
                  for pattern in ("*.py", "tests/*.py") for p in args.production_root.glob(pattern)}
        expected = {row["source_path"] for row in manifest["files"].values()
                    if row["origin"] == "production" and row["source_path"].endswith(".py")}
        if actual != expected:
            errors.append("Production helper inventory changed")
    release = json.loads((ROOT / "hs_tasnet/streaming_models.json").read_text())["current"]
    if release["sha256"] != manifest["release_onnx_sha256"]:
        errors.append("Released model changed without updating the training source snapshot")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Verified {len(included)} research, training, recovery and export source/config files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--production-root", type=Path)
    parser.add_argument("--active-plan", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_snapshot(args)
    verify(args)


if __name__ == "__main__":
    main()
