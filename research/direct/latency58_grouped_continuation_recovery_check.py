"""Qualify complete rolling recovery against an explicitly planned continuation schedule."""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import random
import tempfile
import time

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.check_latency58_grouped_vocal_restart import fingerprint
from research.direct.latency58_grouped_vocal_recovery import (
    policy, make_snapshot, audit_snapshot, publish_snapshot, read_snapshot, restore_training)


def exercise_recovery(parent, source, out):
    """Short stochastic MSE fixture; grouped-objective proof remains separate."""
    import numpy as np
    import torch
    from research.direct.latency58_branch_ema import BranchParameterEMA
    from research.direct.latency58_branch_memory_checkpoint import audit_live
    device = next(parent.parameters()).device
    parent_sha = state_sha256(parent.state_dict())
    model = copy.deepcopy(parent).train().requires_grad_(True)
    precision = model.training_precision
    model.provenance = {**model.provenance,
        "branch_memory_previous_provenance": copy.deepcopy(model.provenance),
        "branch_memory_parent_model_state_sha256": parent_sha,
        "branch_memory_current_stage_corpus": "Synthetic stochastic 1024-sample recovery fixture"}
    fixture = {**source, "config": {**source["config"], "steps": source["config"]["steps"], "batch_size": 1, "data_start": 100,
                "lr": 6e-5}, "recovery_checkpoint": policy(),
        "parent_model_state_sha256": parent_sha, "fixed_buffers_sha256": state_sha256(dict(model.named_buffers())),
        "precision_policy": precision + " synthetic recovery fixture",
        "qualification_only": True, "quality_measured": False}
    fixture_path = out / "fixture-plan.json"
    write(fixture_path, fixture)
    fixture_sha = sha(fixture_path)
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=fixture["config"]["lr"], foreach=False)
    ema = BranchParameterEMA(model, decay=fixture["ema"]["decay"], base_state_sha256=parent_sha)
    journal = []
    constructions, publications = {}, []

    def timed_snapshot(*args, **kwargs):
        started = time.monotonic()
        value = make_snapshot(*args, **kwargs)
        constructions[value["step"]] = time.monotonic() - started
        return value

    def timed_publish(value, *args, **kwargs):
        started = time.monotonic()
        result = publish_snapshot(value, *args, **kwargs)
        publications.append({"step": value["step"], "construction_seconds": constructions[value["step"]],
                             "publication_seconds": time.monotonic() - started})
        return result

    def update(candidate, adam, average, step):
        # The third update depends on all RNG streams restored by recovery.
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
        adam.step()
        audit_live(candidate, adam, step, frozen)
        average.update(candidate, step=step)
        return {"step": step, "loss": float(loss.detach()), "first_sample_index": 100 + step - 1,
                "next_sample_index": 100 + step, "raw_model_state_sha256": state_sha256(candidate.state_dict()),
                "ema_parameters_sha256": state_sha256(average.parameters)}

    def journal_bytes(rows):
        return b"".join((json.dumps(row, allow_nan=False) + "\n").encode() for row in rows)

    def draws():
        return {"python": random.random(), "numpy": np.random.random(4).tolist(),
                "torch_cpu": torch.rand(4).tolist(),
                "torch_cuda": torch.rand(4, device=device).cpu().tolist() if device.type == "cuda" else []}

    rejected = []
    def reject(name, operation):
        try:
            operation()
        except (RuntimeError, ValueError) as error:
            rejected.append({"case": name, "reason": str(error)})
        else:
            raise RuntimeError("Invalid recovery accepted: " + name)

    for step in (1, 2):
        journal.append(update(model, optimizer, ema, step))
    original_endpoint = fingerprint(model, optimizer, ema)
    snapshot = timed_snapshot(model, optimizer, ema, 2, fixture, fixture_sha, journal_bytes(journal))
    require(snapshot["planned_stop_step"] == snapshot["training_config"]["steps"] == source["config"]["steps"]
            and type(source["config"]["steps"]) is int and source["config"]["steps"] > 3 and sha(fixture_path) == fixture_sha,
            "Intermediate snapshot changed the planned full schedule")
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="recovery-fixture-", dir=out) as temporary:
        directory = Path(temporary)
        binding = timed_publish(snapshot, directory, fixture, fixture_sha)
        first_receipt = read(binding["receipt"])
        loaded, audited = read_snapshot(binding, fixture, fixture_sha)
        require(fingerprint(model, optimizer, ema) == original_endpoint, "Saving mutated live optimizer state")
        expected_draws = draws()
        continuous = update(model, optimizer, ema, 3)
        restored, adam, average = restore_training(loaded, audited, fixture, device=device, precision=precision)
        require(draws() == expected_draws, "File-restored Python/NumPy/CPU/CUDA RNG draws differ")
        replay = update(restored, adam, average, 3)
        require(continuous == replay and fingerprint(model, optimizer, ema) == fingerprint(restored, adam, average),
                "Stochastic third update differs after rolling-file recovery")
        for a, b in zip(model.parameters(), restored.parameters(), strict=True):
            require(all(torch.equal(optimizer.state[a][key], adam.state[b][key])
                        for key in ("step", "exp_avg", "exp_avg_sq")), "A recovered Adam tensor differs")
        reject("wrong_file_hash", lambda: read_snapshot({**binding, "sha256": "0" * 64}, fixture, fixture_sha))
        reject("wrong_plan_hash", lambda: read_snapshot(binding, fixture, "0" * 64))
        reject("changed_full_schedule", lambda: audit_snapshot(loaded,
            {**fixture, "config": {**fixture["config"], "steps": 3}}, fixture_sha))
        reject("changed_cursor", lambda: audit_snapshot({**loaded,
            "resume": {**loaded["resume"], "next_sample_index": 999}}, fixture, fixture_sha))
        reject("truncated_journal", lambda: audit_snapshot({**loaded, "journal": loaded["journal"][:-1]}, fixture, fixture_sha))
        reject("EMA_used_as_raw_optimizer_owner", lambda: audit_snapshot({**loaded, "raw": loaded["average"]}, fixture, fixture_sha))
        reject("wrong_numpy_rng", lambda: audit_snapshot({**loaded,
            "resume": {**loaded["resume"], "numpy_rng": []}}, fixture, fixture_sha))
        journal.append(continuous)
        newer = timed_snapshot(model, optimizer, ema, 3, fixture, fixture_sha, journal_bytes(journal))
        del snapshot, loaded, audited, restored, adam, average
        gc.collect()
        def interrupt():
            raise RuntimeError("Simulated interruption immediately before atomic replacement")
        reject("interrupted_publication", lambda: publish_snapshot(newer, directory, fixture, fixture_sha,
                                                                   before_replace=interrupt))
        require(sha(binding["path"]) == binding["sha256"] and (directory / "recovery.pending.pt").is_file(),
                "Interrupted publication destroyed the previous recovery")
        interrupted_receipt = read(directory / "recovery-receipts/step-000003.json")
        require(sha(directory / "recovery.pending.pt") == interrupted_receipt["sha256"], "Interrupted file is incomplete")
        recovered_old, old_audit = read_snapshot(binding, fixture, fixture_sha)
        require(recovered_old["step"] == 2, "Previous recovery cannot be loaded after interruption")
        del recovered_old, old_audit
        reject("pending_file_preserved", lambda: publish_snapshot(newer, directory, fixture, fixture_sha))
        # Remove only deliberately created synthetic interruption files in this
        # TemporaryDirectory; no experiment checkpoint or baseline is touched.
        (directory / "recovery.pending.pt").unlink()
        (directory / "recovery-receipts/step-000003.json").unlink()
        replacement = timed_publish(newer, directory, fixture, fixture_sha)
        second_receipt = read(replacement["receipt"])
        replaced, replacement_audit = read_snapshot(replacement, fixture, fixture_sha)
        require(replaced["step"] == 3 and len(list((directory / "recovery-receipts").iterdir())) == 2
                and second_receipt["previous"]["sha256"] == first_receipt["sha256"],
                "Successful replacement lost receipt history")
        reject("stale_binding_after_replacement", lambda: read_snapshot(binding, fixture, fixture_sha))
        require(sha(replacement["path"]) == replacement["sha256"], "Negative checks changed the current recovery")
        del replaced, replacement_audit, newer
    require(state_sha256(parent.state_dict()) == parent_sha, "Recovery fixture changed its parent")
    require([row["step"] for row in publications] == [2, 3], "Wrong successful publication inventory")
    complete_times = [row["construction_seconds"] + row["publication_seconds"] for row in publications]
    require(max(complete_times) < 10, "Full recovery save exceeds the unchanged progress margin")
    return {"status": "pass", "complete_successful_save_seconds": complete_times, "complete_save_timing": publications,
        "device": str(device), "precision": precision,
        "fixture": "Short stochastic MSE; grouped loss and full context remain separately qualified",
        "parent_weights_unchanged": True, "all_40_adam_states_bit_exact": True,
        "third_stochastic_update_raw_adam_ema_bit_exact": True, "all_rng_streams_replayed": True,
        "interrupted_publication_retains_loadable_previous_file": True,
        "successful_atomic_replacement_and_receipt_history": True, "training_schedule_steps": source["config"]["steps"],
        "first_receipt": first_receipt, "interrupted_receipt": interrupted_receipt, "second_receipt": second_receipt,
        "first_publication_seconds": binding["publication_seconds"],
        "replacement_publication_seconds": replacement["publication_seconds"], "rejected_cases": rejected,
        "ephemeral_tensor_files_removed": True, "quality_measured": False, "elapsed_seconds": time.monotonic() - began}

