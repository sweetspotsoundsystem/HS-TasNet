"""Qualify exact packed recovery and real atomic replacement on CPU."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import resource
import tempfile
import time

import numpy as np
import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs, disk_bytes
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.latency58_branch_memory_checkpoint import load_model, audit_live
from research.direct.latency58_branch_ema import BranchParameterEMA
from research.direct.check_latency58_grouped_vocal_restart import fingerprint
from research.direct.latency58_grouped_vocal_recovery import (
    policy as original_policy, make_snapshot, audit_snapshot, restore_training)
from research.direct.latency58_lossless_recovery_codec_v2 import (
    policy, pack_snapshot, unpack_snapshot, decode_tree, metadata_inventory,
    MAX_FILE_BYTES, MAX_METADATA_BYTES, MAX_METADATA_NODES, TENSOR_CODEC)
from research.direct.latency58_lossless_recovery_files_v2 import (
    CURRENT, PENDING, FINAL, RECEIPTS, identity, _publish_bytes, read_snapshot, finalize, load_inference)
from research.direct.profile_latency58_lossless_checkpoint import encode_tree, tensor_bytes


def equal_tree(original, restored):
    if isinstance(original, torch.Tensor):
        require(isinstance(restored, torch.Tensor) and original.dtype == restored.dtype
                and original.shape == restored.shape and tensor_bytes(original) == tensor_bytes(restored),
                "Packed tensor bytes differ")
        return 1
    require(type(original) is type(restored), "Packed metadata type differs")
    if type(original) is dict:
        require(list(original) == list(restored), "Packed dictionary keys or order differ")
        return sum(equal_tree(item, restored[key]) for key, item in original.items())
    if type(original) in (tuple, list):
        require(len(original) == len(restored), "Packed sequence length differs")
        return sum(equal_tree(a, b) for a, b in zip(original, restored, strict=True))
    require(original == restored, "Packed metadata value differs")
    return 0


def reject(cases, name, operation):
    try:
        operation()
    except (RuntimeError, ValueError, KeyError, TypeError, EOFError) as error:
        cases.append({"case": name, "exception": type(error).__name__, "reason": str(error)[:500]})
    else:
        raise RuntimeError("Invalid packed recovery accepted: " + name)


def primitive_checks():
    rejected = []
    bits = torch.tensor([0, -2147483648, 1, -2147483647, 2139095040, -8388608, 2143289345], dtype=torch.int32)
    values = {"ieee": bits.view(torch.float32), "empty": torch.empty(0, 2),
              "integer": torch.tensor([-2**63, 2**63 - 1], dtype=torch.int64),
              "byte": torch.arange(256, dtype=torch.uint8), "scalar": torch.tensor(-0.),
              "noncontiguous": torch.arange(24, dtype=torch.float32).reshape(4, 6).T,
              "metadata": [None, True, 7, .3, b"journal\n", ("tuple", {3: "value"})]}
    bases = {("ieee",): torch.arange(7, dtype=torch.float32)}
    encoded = encode_tree(values, bases=bases, rows=[])
    def decode(value, selected_bases=None):
        metadata = metadata_inventory(value, packed=True)
        return decode_tree(value, bases=bases if selected_bases is None else selected_bases,
                           budget={"tensor_bytes": 0, "packed_bytes": 0, "tensor_count": 0, **metadata})
    count = equal_tree(values, decode(encoded))
    descriptor = encoded["ieee"]
    for name, changed in (
        ("truncated_stream", {**descriptor, "data": descriptor["data"][:-1]}),
        ("trailing_compressed_bytes", {**descriptor, "data": torch.cat((descriptor["data"], torch.zeros(1, dtype=torch.uint8)))}),
        ("wrong_declared_shape", {**descriptor, "shape": [8]}),
        ("unsupported_dtype", {**descriptor, "dtype": "torch.float64"}),
        ("missing_xor_flag", {**descriptor, "xor_base": False}),
        ("oversize_shape", {**descriptor, "shape": [550_000_001]}),
        ("extra_descriptor_field", {**descriptor, "extra": 1}),
    ):
        reject(rejected, name, lambda changed=changed: decode({**encoded, "ieee": changed}))
    reject(rejected, "missing_xor_parent", lambda: decode(encoded, {}))
    reject(rejected, "metadata_value_size", lambda: metadata_inventory(b"x" * MAX_METADATA_BYTES))
    reject(rejected, "metadata_key_size", lambda: metadata_inventory({"x" * MAX_METADATA_BYTES: 0}))
    reject(rejected, "metadata_numeric_nodes", lambda: metadata_inventory([1] * (MAX_METADATA_NODES + 1)))
    nested = 1
    for _ in range(34):
        nested = [nested]
    reject(rejected, "metadata_depth", lambda: metadata_inventory(nested))
    return {"status": "pass", "edge_tensor_count": count, "all_edge_tensor_bytes_exact": True,
            "signed_zero_subnormal_infinity_nan_payloads_preserved": True, "rejected_cases": rejected}


def render_parity(first, second):
    audio = torch.linspace(-.1, .1, 2 * 8 * 128).reshape(1, 2, 8 * 128)
    with torch.inference_mode():
        a, b = first.render(audio), second.render(audio)
    require(all(tensor_bytes(getattr(a, name)) == tensor_bytes(getattr(b, name)) for name in
                ("raw", "deployed", "native_raw", "spectral", "waveform", "delayed_mixture"))
            and len(a.state) == len(b.state) == 8
            and all(tensor_bytes(x) == tensor_bytes(y) for x, y in zip(a.state, b.state, strict=True))
            and first.algorithmic_latency_samples == second.algorithmic_latency_samples == 256,
            "Packed native outputs, eight states or latency differ")
    closure = float((a.deployed.sum(1) - a.delayed_mixture).abs().max())
    require(closure < 1e-6, "Packed native mixture closure differs")
    return {"all_six_outputs_and_eight_states_bit_exact": True, "algorithmic_latency_samples": 256,
            "closure_max_abs": closure}


def exercise(parent, payload, source, out, progress):
    parent_sha = state_sha256(parent.state_dict())
    model = copy.deepcopy(parent).train().requires_grad_(True)
    model.training_precision = "fp32"
    model.provenance = {**model.provenance,
        "branch_memory_previous_provenance": copy.deepcopy(model.provenance),
        "branch_memory_parent_model_state_sha256": parent_sha,
        "branch_memory_current_stage_corpus": "Synthetic stochastic 1024-sample packed recovery fixture"}
    fixture = {**source, "config": {**source["config"], "steps": 3, "batch_size": 1, "data_start": 100,
                "lr": 6e-5}, "parent_training_updates": payload["provenance"]["training_updates"],
        "recovery_checkpoint": original_policy(), "packed_recovery": policy(),
        "parent_model_state_sha256": parent_sha, "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
        "precision_policy": "CPU FP32 synthetic recovery fixture", "qualification_only": True,
        "quality_measured": False}
    fixture_path = out / "fixture-plan.json"
    write(fixture_path, fixture); fixture_sha = sha(fixture_path)
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False)
    ema = BranchParameterEMA(model, decay=fixture["ema"]["decay"], base_state_sha256=parent_sha)
    journal, rejected, forecasts = [], [], []

    def update(candidate, adam, average, step):
        audio = .03 * torch.randn(1, 2, 1024)
        target = .02 * torch.randn(1, 4, 2, 1024)
        audio = audio * (1 + .01 * random.random() + .01 * float(np.random.random()))
        adam.zero_grad(set_to_none=True)
        rendered = candidate.render(audio)
        loss = (rendered.deployed - target).square().mean() + .25 * (rendered.raw - target).square().mean()
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0
                    for p in candidate.parameters()), "Packed fixture did not reach all 40 parameters")
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5., error_if_nonfinite=True, foreach=False)
        adam.step(); audit_live(candidate, adam, step, frozen); average.update(candidate, step=step)
        return {"step": step, "loss": float(loss.detach()), "first_sample_index": 100 + step - 1,
                "next_sample_index": 100 + step, "raw_model_state_sha256": state_sha256(candidate.state_dict()),
                "ema_parameters_sha256": state_sha256(average.parameters)}

    def journal_bytes():
        return b"".join((json.dumps(row, allow_nan=False) + "\n").encode() for row in journal)

    def draws():
        return {"python": random.random(), "numpy": np.random.random(4).tolist(), "torch_cpu": torch.rand(4).tolist()}

    for step in (1, 2):
        journal.append(update(model, optimizer, ema, step)); progress("fixture_update", step=step)
    before = fingerprint(model, optimizer, ema)
    snapshot = make_snapshot(model, optimizer, ema, 2, fixture, fixture_sha, journal_bytes())
    packed, first_stats = pack_snapshot(snapshot, fixture, fixture_sha)
    loaded, stats = unpack_snapshot(packed, fixture, fixture_sha)
    count = equal_tree(snapshot, loaded)
    require(count == stats["tensor_count"] == first_stats["tensor_count"] == 217
            and fingerprint(model, optimizer, ema) == before, "Packed complete snapshot inventory differs")
    progress("full_snapshot_roundtrip", file_bytes=len(packed), tensor_count=count)
    del loaded

    with tempfile.TemporaryDirectory(prefix="packed-fixture-", dir=out) as temporary:
        directory = Path(temporary)
        def preflight(size):
            # These two synthetic files are temporary occupants of the existing
            # 600 MB save reserve. Charge any excess against real headroom;
            # preserve the other 800 MB and 250 MB standing allowances in full.
            budget = fixture["storage_budget"]
            actual = sum(disk_bytes(Path(p)) for p in budget["counted_roots"])
            actual += disk_bytes(Path(budget["external_git_common_directory"]))
            occupied = sum((directory / name).stat().st_size for name in (CURRENT, PENDING, FINAL)
                           if (directory / name).exists())
            peak = occupied + size
            require(peak <= 2 * MAX_FILE_BYTES, "Temporary packed fixture exceeds its two-file ceiling")
            forecast = (actual - occupied + budget["other_outside_allowance_bytes"]
                        + budget["diagnostic_artifact_allowance_bytes"]
                        + max(budget["live_training_save_reservation_bytes"], peak))
            require(forecast + 2_000_000 < budget["authorized_cap_bytes"], "Packed fixture exceeds all-inclusive storage cap")
            row = {"fixture_bytes_already_counted": occupied, "incoming_file_bytes": size,
                   "peak_fixture_file_bytes": peak, "standing_save_reserve_bytes": budget["live_training_save_reservation_bytes"],
                   "forecast_with_all_reserves": forecast, "authorized_cap_bytes": budget["authorized_cap_bytes"]}
            forecasts.append(row)
            return row

        def audit(data):
            return identity(unpack_snapshot(data, fixture, fixture_sha)[0])

        def publish(data, snap, before_replace=None):
            return _publish_bytes(data, directory, identity(snap), fixture_sha,
                                  preflight=preflight, audit=audit, before_replace=before_replace)

        binding = publish(packed, snapshot)
        first_receipt = read(binding["receipt"])
        loaded, audited = read_snapshot(binding, fixture, fixture_sha)
        require(equal_tree(snapshot, loaded) == 217 and fingerprint(model, optimizer, ema) == before,
                "Disk snapshot differs or changed live optimizer")
        parity = {"raw": render_parity(model, audited[0]),
                  "ema": render_parity(ema.inference_copy(model), audited[2])}
        reject(rejected, "wrong_file_hash", lambda: read_snapshot({**binding, "sha256": "0" * 64}, fixture, fixture_sha))
        reject(rejected, "wrong_plan_hash", lambda: unpack_snapshot(packed, fixture, "0" * 64))
        reject(rejected, "truncated_archive", lambda: unpack_snapshot(packed[:-100], fixture, fixture_sha))
        reject(rejected, "wrong_parent_binding", lambda: unpack_snapshot(packed, {**fixture,
            "parent_checkpoint": {**fixture["parent_checkpoint"], "sha256": "0" * 64}}, fixture_sha))
        reject(rejected, "wrong_parent_state", lambda: unpack_snapshot(packed, {**fixture,
            "parent_model_state_sha256": "0" * 64}, fixture_sha))
        reject(rejected, "changed_schedule", lambda: audit_snapshot(loaded,
            {**fixture, "config": {**fixture["config"], "steps": 2}}, fixture_sha))
        reject(rejected, "changed_cursor", lambda: audit_snapshot({**loaded,
            "resume": {**loaded["resume"], "next_sample_index": 999}}, fixture, fixture_sha))
        reject(rejected, "truncated_journal", lambda: audit_snapshot({**loaded, "journal": loaded["journal"][:-1]}, fixture, fixture_sha))
        reject(rejected, "ema_as_raw_owner", lambda: audit_snapshot({**loaded, "raw": loaded["average"]}, fixture, fixture_sha))
        reject(rejected, "wrong_numpy_rng", lambda: audit_snapshot({**loaded,
            "resume": {**loaded["resume"], "numpy_rng": []}}, fixture, fixture_sha))
        reject(rejected, "early_finalization", lambda: finalize(binding, fixture, fixture_sha))
        reject(rejected, "oversize_archive_before_write", lambda: publish(b"x" * (MAX_FILE_BYTES + 1), snapshot))
        require(not (directory / PENDING).exists() and sha(binding["path"]) == binding["sha256"],
                "Rejected publication changed filesystem")
        expected_draws = draws()
        continuous = update(model, optimizer, ema, 3)
        restored, adam, average = restore_training(loaded, audited, fixture, device="cpu", precision="fp32")
        require(draws() == expected_draws, "File-restored Python/NumPy/CPU RNG draws differ")
        replay = update(restored, adam, average, 3)
        require(continuous == replay and fingerprint(model, optimizer, ema) == fingerprint(restored, adam, average),
                "Stochastic third update differs after packed recovery")
        for a, b in zip(model.parameters(), restored.parameters(), strict=True):
            require(all(tensor_bytes(optimizer.state[a][key]) == tensor_bytes(adam.state[b][key])
                        for key in ("step", "exp_avg", "exp_avg_sq")), "A recovered Adam tensor differs")
        progress("stochastic_restart_exact", all_40_adam_states=True)
        journal.append(continuous)
        newer = make_snapshot(model, optimizer, ema, 3, fixture, fixture_sha, journal_bytes())
        del snapshot, loaded, audited, restored, adam, average, packed
        gc.collect()
        newer_packed, second_stats = pack_snapshot(newer, fixture, fixture_sha)
        def interrupt():
            raise RuntimeError("Simulated interruption immediately before atomic replacement")
        reject(rejected, "interrupted_publication", lambda: publish(newer_packed, newer, interrupt))
        require(sha(binding["path"]) == binding["sha256"] and (directory / PENDING).is_file(),
                "Interrupted publication destroyed the previous recovery")
        interrupted_receipt_path = directory / RECEIPTS / "step-000003.json"
        interrupted_receipt = read(interrupted_receipt_path)
        require(sha(directory / PENDING) == interrupted_receipt["sha256"], "Interrupted pending file is incomplete")
        old, old_audit = read_snapshot(binding, fixture, fixture_sha)
        require(old["step"] == 2, "Previous packed recovery is not loadable after interruption")
        del old, old_audit
        reject(rejected, "pending_file_preserved", lambda: publish(newer_packed, newer))
        progress("interrupted_file_preserved", previous_step=2, pending_step=3)
        # Delete only deliberate interruption files in this TemporaryDirectory.
        (directory / PENDING).unlink(); interrupted_receipt_path.unlink()
        replacement = publish(newer_packed, newer)
        second_receipt = read(replacement["receipt"])
        require(second_receipt["previous"]["sha256"] == first_receipt["sha256"]
                and len(list((directory / RECEIPTS).iterdir())) == 2, "Packed receipt history lost")
        reject(rejected, "stale_binding_after_replace", lambda: read_snapshot(binding, fixture, fixture_sha))
        before_inode = Path(replacement["path"]).stat().st_ino
        final = finalize(replacement, fixture, fixture_sha)
        require(Path(final["path"]).stat().st_ino == before_inode and not (directory / CURRENT).exists()
                and sha(final["path"]) == replacement["sha256"], "Finalization duplicated or changed tensors")
        reject(rejected, "publication_after_finalization", lambda: publish(newer_packed, newer))
        final_parity = {}
        for role, original in (("raw", model), ("ema", ema.inference_copy(model))):
            restored, _ = load_inference(final, fixture, fixture_sha, role=role)
            final_parity[role] = render_parity(original, restored)
            del restored
        progress("final_roles_loaded", file_bytes=second_stats["file_bytes"])
    require(state_sha256(parent.state_dict()) == parent_sha and sha(fixture_path) == fixture_sha,
            "Packed fixture changed its parent or schedule")
    return {"status": "pass", "tensor_count": count, "all_tensor_and_metadata_bytes_exact": True,
            "first_packing": first_stats, "second_packing": second_stats, "raw_and_ema_parity": parity,
            "final_raw_and_ema_parity": final_parity, "all_40_adam_states_bit_exact": True,
            "third_stochastic_update_raw_adam_ema_bit_exact": True, "all_cpu_rng_streams_replayed": True,
            "data_cursor_and_journal_exact": True, "interrupted_publication_retains_loadable_previous_file": True,
            "atomic_replacement_and_receipt_history": True, "finalization_reuses_same_inode_without_tensor_duplicate": True,
            "first_receipt": first_receipt, "interrupted_receipt": interrupted_receipt, "second_receipt": second_receipt,
            "storage_forecasts": forecasts, "rejected_cases": rejected, "ephemeral_tensor_files_removed": True,
            "production_budget_preflight_qualified": False,
            "fixture": "CPU stochastic short-context MSE; weighted grouped loss is separately qualified"}


def main():
    require(Path.cwd() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(os.environ.get(key) == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Require CUDA-hidden CPU1")
    out = PHASE / "lossless-recovery-cpu-002"
    require(not out.exists(), "Preserve previous packed recovery qualification")
    source_path = PHASE / "branch-grouped-vocal-013/plan.json"
    source = read(source_path)
    previous = PHASE / "weighted-vocal-quarter-cpu-002"
    old, result, execution = (read(previous / name) for name in ("plan.json", "result.json", "execution.json"))
    require(result["status"] == "pass" and execution["actual_exit_code"] == 0
            and result["third_update_raw_adam_ema_and_accounting_bit_exact"], "Weighted CPU qualification incomplete")
    bindings = dict(old["source_bindings"])
    failed = PHASE / "lossless-recovery-cpu-001"
    failed_plan = read(failed / "plan.json")
    verify_inputs(failed_plan); bindings.update(failed_plan["source_bindings"])
    paths = [Path(__file__).resolve(), source_path, *(previous / name for name in ("plan.json", "result.json", "execution.json"))]
    paths.extend(ROOT / "research/direct" / name for name in (
        "latency58_lossless_recovery_codec_v2.py", "latency58_lossless_recovery_files_v2.py",
        "profile_latency58_lossless_checkpoint.py", "latency58_grouped_vocal_recovery.py",
        "latency58_branch_ema_checkpoint.py", "latency58_branch_memory_checkpoint.py"))
    paths.extend([failed / "plan.json", failed / "metadata-failure-diagnostic.json",
                  PHASE / "lossless-recovery-cpu-001-stage/command.json",
                  PHASE / "lossless-recovery-cpu-001-stage/execution.json",
                  PHASE / "lossless-recovery-cpu-001-stage/evaluation-console.log"])
    bindings.update({str(path): sha(path) for path in paths})
    budget = budget_snapshot(source["storage_budget"])
    require(budget["headroom_bytes"] > 162_000_000, "Reserve two temporary packed archives within existing save reserve and headroom")
    plan = {"schema": "latency58-lossless-packed-recovery-cpu-qualification-v1", "source_bindings": bindings,
            "packed_recovery": policy(), "storage_budget": source["storage_budget"], "budget_before": budget,
            "metadata_node_limit_correction": "Existing endpoint traverses 204878 nodes; increase traversal ceiling to 400000 while retaining the 30 MB byte ceiling",
            "maximum_simultaneous_ephemeral_archive_bytes": 2 * MAX_FILE_BYTES,
            "ephemeral_storage_accounting": "Occupy existing 600 MB transient reserve; charge any excess to headroom, retaining every other allowance",
            "seed": 202611017, "maximum_observed_rss_bytes": 12_000_000_000,
            "production_checkpoint_created": False, "gpu_used": False, "quality_measured": False}
    verify_inputs(plan); out.mkdir(); write(out / "plan.json", plan)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    def progress(phase, **details):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        require(rss < plan["maximum_observed_rss_bytes"], "Packed CPU qualification exceeded its memory bound")
        print(json.dumps({"event": "packed_recovery_cpu_progress", "phase": phase,
                          "elapsed_seconds": time.monotonic() - began, "peak_rss_bytes": rss, **details}), flush=True)
    rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    torch.manual_seed(plan["seed"]); random.seed(plan["seed"]); np.random.seed(plan["seed"])
    try:
        primitive = primitive_checks(); write(out / "primitive-checks.json", primitive); progress("primitive_checks")
        parent, payload = load_model(source["parent_checkpoint"]); parent.training_precision = "fp32"
        recovered = exercise(parent, payload, source, out, progress)
        write(out / "recovery-checks.json", recovered)
    finally:
        torch.set_rng_state(rng); random.setstate(python_rng); np.random.set_state(numpy_rng)
    verify_inputs(plan)
    require(not torch.cuda.is_initialized() and not list(out.rglob("*.pt")), "CPU fixture retained tensor files or initialized CUDA")
    result = {"schema": plan["schema"], "status": "pass", "plan_sha256": sha(out / "plan.json"),
              "source_bindings_unchanged": True, "primitive_checks_sha256": sha(out / "primitive-checks.json"),
              "recovery_checks_sha256": sha(out / "recovery-checks.json"),
              "tensor_count": recovered["tensor_count"], "all_tensor_and_metadata_bytes_exact": True,
              "third_stochastic_update_raw_adam_ema_bit_exact": True,
              "interrupted_publication_retains_loadable_previous_file": True,
              "raw_and_ema_native_parity_and_256_sample_latency": True,
              "finalization_reuses_same_inode_without_tensor_duplicate": True,
              "parent_and_rng_preserved": True, "ephemeral_tensor_files_removed": True,
              "production_checkpoint_created": False, "gpu_used": False, "quality_measured": False,
              "production_budget_preflight_qualified": False, "gpu_recovery_qualified": False,
              "elapsed_seconds": time.monotonic() - began, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "completed_utc": datetime.now(timezone.utc).isoformat(), "budget_after": budget_snapshot(source["storage_budget"])}
    write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "tensor_count", "elapsed_seconds", "peak_rss_bytes")}), flush=True)


if __name__ == "__main__":
    main()
