"""Rehearse packed recovery with every RNG stream on the selected device."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from research.direct.run_latency58_quality import require, sha, write, read
from research.direct.train_latency58 import state_sha256
from research.direct.latency58_branch_ema import BranchParameterEMA
from research.direct.latency58_branch_memory_checkpoint import audit_live
from research.direct.latency58_grouped_vocal_recovery import make_snapshot, audit_snapshot, restore_training
from research.direct.latency58_lossless_recovery_codec_v3 import pack_snapshot, unpack_snapshot
from research.direct.latency58_four_second_recovery_files import publish_snapshot, read_snapshot, CURRENT, PENDING, RECEIPTS
from research.direct.check_latency58_grouped_vocal_restart import fingerprint
from research.direct.check_latency58_lossless_recovery_cpu_v3 import equal_tree


def exercise_recovery(parent, source, out, *, disk_directory=None, progress=None):
    device = next(parent.parameters()).device
    precision = parent.training_precision
    require((device.type, precision) in (("cpu", "fp32"), ("cuda", "bf16"))
            and len(list(parent.parameters())) == 40 and all(p.grad is None for p in parent.parameters()),
            "Require a clean selected-device parent for recovery")
    parent_sha = state_sha256(parent.state_dict())
    require(parent_sha == source["parent_model_state_sha256"], "Recovery parent differs from the planned parent")
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    began = time.monotonic()
    def event(phase, **details):
        row = {"event": "packed_device_recovery_progress", "phase": phase, "device": str(device),
               "elapsed_seconds": time.monotonic() - began, **details}
        if progress is not None:
            progress(row)
        else:
            print(json.dumps(row), flush=True)
    try:
        torch.manual_seed(202611018); random.seed(202611018); np.random.seed(202611018)
        model = copy.deepcopy(parent).train().requires_grad_(True)
        model.provenance = {**model.provenance,
            "branch_memory_previous_provenance": copy.deepcopy(model.provenance),
            "branch_memory_parent_model_state_sha256": parent_sha,
            "branch_memory_current_stage_corpus": "Synthetic stochastic 1024-sample selected-device packed recovery fixture"}
        fixture = {**source, "config": {**source["config"], "batch_size": 1, "data_start": 100, "lr": 6e-5},
            "precision_policy": precision + " synthetic packed recovery fixture",
            "qualification_only": True, "quality_measured": False}
        if disk_directory is not None:
            fixture["packed_publication_directory"] = str(Path(disk_directory).resolve())
            fixture["qualification_production_publication_directory"] = source["packed_publication_directory"]
        fixture_path = Path(out) / "fixture-plan.json"
        write(fixture_path, fixture); fixture_sha = sha(fixture_path)
        require(fixture["config"]["steps"] == source["config"]["steps"] == 2000,
                "Recovery fixture must retain the intended complete training schedule")
        optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False)
        ema = BranchParameterEMA(model, decay=source["ema"]["decay"], base_state_sha256=parent_sha)
        frozen = {name: value.clone() for name, value in model.named_buffers()}
        journal = []
        def update(candidate, adam, average, step):
            audio = .03 * torch.randn(1, 2, 1024, device=device)
            target = .02 * torch.randn(1, 4, 2, 1024, device=device)
            audio = audio * (1 + .01 * random.random() + .01 * float(np.random.random()))
            adam.zero_grad(set_to_none=True)
            rendered = candidate.render(audio)
            loss = (rendered.deployed - target).square().mean() + .25 * (rendered.raw - target).square().mean()
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0
                        for p in candidate.parameters()), "Recovery fixture did not reach all 40 parameters")
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5., error_if_nonfinite=True, foreach=False)
            adam.step(); audit_live(candidate, adam, step, frozen); average.update(candidate, step=step)
            return {"step": step, "loss": float(loss.detach()), "first_sample_index": 100 + step - 1,
                    "next_sample_index": 100 + step, "raw_model_state_sha256": state_sha256(candidate.state_dict()),
                    "ema_parameters_sha256": state_sha256(average.parameters)}
        def draws():
            return {"python": random.random(), "numpy": np.random.random(4).tolist(),
                    "torch_cpu": torch.rand(4).tolist(),
                    "torch_cuda": torch.rand(4, device=device).cpu().tolist() if device.type == "cuda" else []}
        for step in (1, 2):
            journal.append(update(model, optimizer, ema, step)); event("synthetic_update", step=step)
        journal_bytes = b"".join((json.dumps(row, allow_nan=False) + "\n").encode() for row in journal)
        before = fingerprint(model, optimizer, ema)
        snapshot = make_snapshot(model, optimizer, ema, 2, fixture, fixture_sha, journal_bytes)
        packing_began = time.monotonic()
        packed, packing = pack_snapshot(snapshot, fixture, fixture_sha)
        recovered, _ = unpack_snapshot(packed, fixture, fixture_sha)
        count = equal_tree(snapshot, recovered)
        audited = audit_snapshot(recovered, fixture, fixture_sha)
        complete_pack_and_audit_seconds = time.monotonic() - packing_began
        require(count == packing["tensor_count"] == 216 + (device.type == "cuda")
                and recovered["planned_stop_step"] == recovered["training_config"]["steps"] == 2000
                and recovered["resume"]["next_sample_index"] == 102
                and fingerprint(model, optimizer, ema) == before, "Packed device roundtrip differs")
        event("in_memory_roundtrip", file_bytes=len(packed), tensor_count=count,
              complete_pack_and_audit_seconds=complete_pack_and_audit_seconds)
        disk_result = None
        if disk_directory is not None:
            directory = Path(disk_directory)
            require(not directory.exists(), "Never reuse an existing production directory for a CPU fixture")
            directory.mkdir()
            disk_began = time.monotonic()
            binding = publish_snapshot(snapshot, directory, fixture, fixture_sha)
            disk_save_seconds = time.monotonic() - disk_began
            del recovered, audited
            recovered, audited = read_snapshot(binding, fixture, fixture_sha)
            require(equal_tree(snapshot, recovered) == count, "Production filesystem wrapper changed a tensor or metadata")
            receipt = read(binding["receipt"])
            write(Path(out) / "synthetic-disk-receipt.json", receipt)
            disk_result = {"complete_production_wrapper_save_seconds": disk_save_seconds,
                           "packing": binding["packing"], "storage_preflight": binding["storage_preflight"],
                           "receipt": receipt, "all_tensor_and_metadata_values_exact": True}
            require(sha(directory / CURRENT) == receipt["sha256"] and not (directory / PENDING).exists()
                    and set(directory.iterdir()) == {directory / CURRENT, directory / RECEIPTS}
                    and set((directory / RECEIPTS).iterdir()) == {Path(binding["receipt"])},
                    "Unexpected synthetic production-directory inventory")
            # This directory was absent at entry and contains only this fixture.
            # The copied receipt above records its complete, exact identity.
            (directory / CURRENT).unlink(); Path(binding["receipt"]).unlink()
            (directory / RECEIPTS).rmdir(); directory.rmdir()
            event("production_wrapper_disk_roundtrip", complete_save_seconds=disk_save_seconds)
        expected_draws = draws()
        continuous = update(model, optimizer, ema, 3)
        restored, adam, average = restore_training(recovered, audited, fixture, device=device, precision=precision)
        require(draws() == expected_draws, "Packed recovery failed to replay every device RNG stream")
        replay = update(restored, adam, average, 3)
        require(continuous == replay and fingerprint(model, optimizer, ema) == fingerprint(restored, adam, average),
                "Next selected-device Adam/EMA update differs after packed recovery")
        for a, b in zip(model.parameters(), restored.parameters(), strict=True):
            require(all(torch.equal(optimizer.state[a][key], adam.state[b][key])
                        for key in ("step", "exp_avg", "exp_avg_sq")), "A recovered selected-device Adam tensor differs")
        require(state_sha256(parent.state_dict()) == parent_sha and all(p.grad is None for p in parent.parameters())
                and sha(fixture_path) == fixture_sha, "Recovery qualification changed its parent or schedule")
        result = {"status": "pass", "device": str(device), "precision": precision,
                  "parent_model_state_sha256": parent_sha, "parent_weights_unchanged": True,
                  "tensor_count": count, "all_tensor_and_metadata_values_exact": True,
                  "all_40_adam_states_bit_exact": True, "third_stochastic_update_raw_adam_ema_bit_exact": True,
                  "all_rng_streams_replayed": True, "training_schedule_steps": 2000,
                  "saved_step": 2, "replayed_step": 3, "data_cursor_and_journal_exact": True,
                  "packed_file_bytes": len(packed), "packed_file_sha256": hashlib.sha256(packed).hexdigest(),
                  "complete_pack_and_audit_seconds": complete_pack_and_audit_seconds, "packing": packing,
                  "production_filesystem_wrapper": disk_result, "ephemeral_tensor_files_removed": True,
                  "quality_measured": False, "elapsed_seconds": time.monotonic() - began,
                  "scope": "Short stochastic recovery; full-context weighted gradients are separately qualified"}
        event("next_update_bit_exact", all_40_adam_states=True)
        del recovered, audited, snapshot, packed, model, optimizer, ema, restored, adam, average
        gc.collect()
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(cpu_rng)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
    require(torch.equal(cpu_rng, torch.get_rng_state())
            and (device.type != "cuda" or all(torch.equal(a, b) for a, b in
                zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))), "Recovery qualification changed production RNG")
    result["production_rng_restored"] = True
    return result
