"""Qualify selected-geometry weighted updates and exact packed disk restart for the 2000-step plan.

The update fixture has 512 warmup and 44160 scored samples. Full four-second
neural gradients are separately qualified. No production optimizer is changed.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time
from functools import partial
from research.direct.latency58_four_second_geometry_restart import exercise as geometry_exercise

import numpy as np
import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT, POLICY, snapshot as budget_snapshot
from research.direct.latency58_branch_memory_checkpoint import load_model, audit_live
from research.direct.latency58_branch_ema import BranchParameterEMA
from research.direct.check_latency58_grouped_vocal_restart import fingerprint, RequestedStop
from research.direct.latency58_weighted_vocal_canonical import grouped_update
from research.direct.latency58_grouped_vocal_recovery import (
    make_snapshot, audit_snapshot, restore_training, policy as recovery_policy)
from research.direct.latency58_lossless_recovery_codec_v3 import (
    policy as packed_policy, pack_snapshot, unpack_snapshot, decode_tree, metadata_inventory,
    MAX_FILE_BYTES, MAX_METADATA_BYTES, MAX_METADATA_NODES, TENSOR_CODEC)
from research.direct.latency58_four_second_recovery_files import (
    CURRENT, PENDING, FINAL, RECEIPTS, publish_snapshot, read_snapshot, finalize, load_inference,
    production_preflight)
from research.direct.check_latency58_lossless_recovery_cpu_v3 import equal_tree, reject, render_parity
from research.direct.profile_latency58_lossless_checkpoint import encode_tree, tensor_bytes

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


def run_fixture(parent, plan, out, progress):
    decision = read(plan['scientific_decision']['path'])
    ordinary_microbatch = decision['proposed_training_config']['microbatch_size']
    auxiliary_microbatch = decision['proposed_training_config']['auxiliary_microbatch_size']
    require(ordinary_microbatch in (2, 4, 8, 16) and auxiliary_microbatch == 2, 'Unsupported selected geometry')
    exercise = partial(geometry_exercise, ordinary_microbatch=ordinary_microbatch,
        auxiliary_microbatch=auxiliary_microbatch, update_impl=grouped_update)
    parent_sha = state_sha256(parent.state_dict())
    require(parent_sha == decision['parent_model_state_sha256'], 'Restart parent changed')
    model = copy.deepcopy(parent).train().requires_grad_(True)
    model.training_precision = 'fp32'
    model.provenance = {**model.provenance,
        'branch_memory_previous_provenance': copy.deepcopy(model.provenance),
        'branch_memory_parent_model_state_sha256': parent_sha,
        'branch_memory_current_stage_corpus': 'Synthetic B16 quarter-weight source-view restart fixture; no recorded audio',
        'branch_memory_current_stage_training_context': {'warmup_samples': 512, 'scored_samples': 44160,
            'ordinary_microbatch': ordinary_microbatch, 'auxiliary_microbatch': auxiliary_microbatch}}
    directory = out / 'disk-fixture'
    require(not directory.exists(), 'Preserve any previous fixture directory')
    directory.mkdir()
    fixture = {'schema': 'latency58-four-second-proposal-short-cpu-restart-fixture-v1',
        'config': {**decision['proposed_training_config'], 'device': 'cpu', 'precision': 'fp32',
            'crop_samples': 44672, 'batch_size': 16, 'data_start': 100,
            'augmentation': 'Synthetic stochastic fixture only; production data is separately qualified'},
        **{key: decision[key] for key in ('parent_checkpoint', 'parent_model_state_sha256',
            'parent_training_updates', 'ema', 'objective_version')},
        'fixed_buffers_sha256': state_sha256(dict(model.named_buffers())),
        'recovery_checkpoint': recovery_policy(), 'packed_recovery': packed_policy(),
        'precision_policy': 'CPU FP32 synthetic weighted selected-geometry restart fixture',
        'qualification_only': True, 'quality_measured': False,
        'warmup_samples': 512, 'scored_samples': 44160,
        'scientific_decision': plan['scientific_decision'],
        'new_storage_policy_binding': {'path': str(POLICY), 'sha256': sha(POLICY)},
        'packed_publication_directory': str(directory)}
    require(fixture['config']['steps'] == 2000 and fixture['config']['microbatch_size'] == ordinary_microbatch
            and fixture['config']['auxiliary_microbatch_size'] == auxiliary_microbatch, 'Retain the selected complete schedule and grouping')
    fixture_path = out / 'fixture-plan.json'; write(fixture_path, fixture); fixture_sha = sha(fixture_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=decision['proposed_training_config']['lr'], foreach=False)
    ema = BranchParameterEMA(model, decay=decision['ema']['decay'], base_state_sha256=parent_sha)
    frozen = {name: value.clone() for name, value in model.named_buffers()}
    rejected, journal = [], []

    def make_inputs():
        truth = .02 * torch.randn(16, 4, 2, 44672)
        truth[:3, 2] = 0; truth[4:6, 1] = 0; truth[8:9, 3] = 0
        truth *= 1 + .01 * random.random() + .01 * float(np.random.random())
        return truth.sum(1), truth

    def update(candidate, adam, average, step):
        inputs = make_inputs()
        row = exercise(candidate, adam, average, inputs, step=step)
        row.update(first_sample_index=100 + (step - 1) * 16,
                   next_sample_index=100 + step * 16,
                   synthetic_input_sha256=state_sha256({'mixture': inputs[0], 'targets': inputs[1]}))
        audit_live(candidate, adam, step, frozen)
        progress('weighted_update_complete', step=step)
        return row

    def journal_bytes():
        return b''.join((json.dumps(row, allow_nan=False) + '\n').encode() for row in journal)

    def draws():
        return {'python': random.random(), 'numpy': np.random.random(4).tolist(), 'torch_cpu': torch.rand(4).tolist()}

    inputs = make_inputs(); initial = fingerprint(model, optimizer, ema)
    reject(rejected, 'noncontiguous_step', lambda: grouped_update(model, optimizer, ema, *inputs,
        step=2, warmup_samples=512, ordinary_microbatch=ordinary_microbatch, auxiliary_microbatch=auxiliary_microbatch))
    require(fingerprint(model, optimizer, ema) == initial and all(p.grad is None for p in model.parameters()),
            'Invalid update changed its endpoint')
    interrupted_groups = [exercise(model, optimizer, ema, inputs, step=1, stop_after=group)
                          for group in ('ordinary', 'auxiliary')]
    del inputs
    for step in (1, 2):
        journal.append(update(model, optimizer, ema, step))
    before = fingerprint(model, optimizer, ema)
    saved = make_snapshot(model, optimizer, ema, 2, fixture, fixture_sha, journal_bytes())
    packed, stats = pack_snapshot(saved, fixture, fixture_sha)
    decoded, _ = unpack_snapshot(packed, fixture, fixture_sha)
    tensor_count = equal_tree(saved, decoded)
    require(tensor_count == 216 and decoded['planned_stop_step'] == 2000, 'CPU packed inventory or schedule changed')
    del decoded
    reject(rejected, 'wrong_publication_directory', lambda: production_preflight(out, fixture, len(packed)))
    reject(rejected, 'wrong_policy_binding', lambda: production_preflight(directory,
        {**fixture, 'new_storage_policy_binding': {'path': str(POLICY), 'sha256': '0' * 64}}, len(packed)))
    reject(rejected, 'oversize_preflight', lambda: production_preflight(directory, fixture, MAX_FILE_BYTES + 1))
    binding = publish_snapshot(saved, directory, fixture, fixture_sha)
    first_receipt = read(binding['receipt'])
    loaded, audited = read_snapshot(binding, fixture, fixture_sha)
    require(equal_tree(saved, loaded) == tensor_count and fingerprint(model, optimizer, ema) == before,
            'Production publication changed the snapshot or live optimizer')
    parity = {'raw': render_parity(model, audited[0]),
              'ema': render_parity(ema.inference_copy(model), audited[2])}
    reject(rejected, 'wrong_file_hash', lambda: read_snapshot({**binding, 'sha256': '0' * 64}, fixture, fixture_sha))
    reject(rejected, 'wrong_plan_hash', lambda: unpack_snapshot(packed, fixture, '0' * 64))
    reject(rejected, 'changed_schedule', lambda: audit_snapshot(loaded,
        {**fixture, 'config': {**fixture['config'], 'steps': 3}}, fixture_sha))
    reject(rejected, 'changed_cursor', lambda: audit_snapshot({**loaded,
        'resume': {**loaded['resume'], 'next_sample_index': 999}}, fixture, fixture_sha))
    reject(rejected, 'truncated_journal', lambda: audit_snapshot({**loaded, 'journal': loaded['journal'][:-1]}, fixture, fixture_sha))
    reject(rejected, 'ema_as_raw_owner', lambda: audit_snapshot({**loaded, 'raw': loaded['average']}, fixture, fixture_sha))
    reject(rejected, 'wrong_numpy_rng', lambda: audit_snapshot({**loaded,
        'resume': {**loaded['resume'], 'numpy_rng': []}}, fixture, fixture_sha))
    reject(rejected, 'early_finalization', lambda: finalize(binding, fixture, fixture_sha))
    expected_draws = draws()
    continuous = update(model, optimizer, ema, 3)
    restored, adam, average = restore_training(loaded, audited, fixture, device='cpu', precision='fp32')
    require(draws() == expected_draws, 'Disk restore changed Python, NumPy or CPU RNG draws')
    replay = update(restored, adam, average, 3)
    require(continuous == replay and fingerprint(model, optimizer, ema) == fingerprint(restored, adam, average),
            'Weighted selected-geometry update or raw/Adam/EMA endpoint differs after disk restart')
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        require(all(tensor_bytes(optimizer.state[a][key]) == tensor_bytes(adam.state[b][key])
                    for key in ('step', 'exp_avg', 'exp_avg_sq')), 'An Adam tensor changed on restart')
    progress('weighted_packed_restart_exact', all_40_adam_states=True)
    journal.append(continuous)
    newer = make_snapshot(model, optimizer, ema, 3, fixture, fixture_sha, journal_bytes())
    del loaded, audited, restored, adam, average, saved, packed
    gc.collect()

    def interrupt():
        raise RuntimeError('Simulated interruption immediately before atomic replacement')

    reject(rejected, 'interrupted_publication', lambda: publish_snapshot(newer, directory, fixture, fixture_sha,
                                                                       before_replace=interrupt))
    interrupted_receipt_path = directory / RECEIPTS / 'step-000003.json'
    interrupted_receipt = read(interrupted_receipt_path)
    require(sha(binding['path']) == binding['sha256'] and (directory / PENDING).is_file()
            and sha(directory / PENDING) == interrupted_receipt['sha256'], 'Interruption damaged a complete generation')
    old, old_audit = read_snapshot(binding, fixture, fixture_sha)
    require(old['step'] == 2 and old['planned_stop_step'] == 2000, 'Previous generation no longer loads')
    del old, old_audit
    reject(rejected, 'pending_file_preserved', lambda: publish_snapshot(newer, directory, fixture, fixture_sha))
    # Only deliberate interruption artifacts inside this newly created fixture.
    (directory / PENDING).unlink(); interrupted_receipt_path.unlink()
    replacement = publish_snapshot(newer, directory, fixture, fixture_sha)
    second_receipt = read(replacement['receipt'])
    require(second_receipt['previous']['sha256'] == first_receipt['sha256'], 'Receipt history changed')
    reject(rejected, 'stale_binding', lambda: read_snapshot(binding, fixture, fixture_sha))
    reject(rejected, 'incomplete_2000_schedule_finalization', lambda: finalize(replacement, fixture, fixture_sha))
    write(out / 'disk-receipts.json', {'first': first_receipt, 'interrupted': interrupted_receipt, 'replacement': second_receipt})
    require(set(directory.iterdir()) == {directory / CURRENT, directory / RECEIPTS}
            and {p.name for p in (directory / RECEIPTS).iterdir()} == {'step-000002.json', 'step-000003.json'},
            'Unexpected fixture inventory before cleanup')
    (directory / CURRENT).unlink()
    for name in ('step-000002.json', 'step-000003.json'):
        (directory / RECEIPTS / name).unlink()
    (directory / RECEIPTS).rmdir(); directory.rmdir()
    del newer
    gc.collect()

    # A separate, explicitly three-update fixture tests finalization without
    # pretending that the selected 2000-update production schedule is complete.
    final_dir = out / 'finalization-fixture'; final_dir.mkdir()
    final_plan = {**fixture, 'config': {**fixture['config'], 'steps': 3},
                  'packed_publication_directory': str(final_dir), 'fixture_purpose': 'Three-update finalization only'}
    final_plan_path = out / 'finalization-fixture-plan.json'; write(final_plan_path, final_plan)
    final_plan_sha = sha(final_plan_path)
    endpoint = make_snapshot(model, optimizer, ema, 3, final_plan, final_plan_sha, journal_bytes())
    final_binding = publish_snapshot(endpoint, final_dir, final_plan, final_plan_sha)
    before_inode = Path(final_binding['path']).stat().st_ino
    final = finalize(final_binding, final_plan, final_plan_sha)
    require(Path(final['path']).stat().st_ino == before_inode and final['sha256'] == final_binding['sha256']
            and not (final_dir / CURRENT).exists(), 'Finalization duplicated or changed a tensor archive')
    reject(rejected, 'publication_after_finalization', lambda: publish_snapshot(endpoint, final_dir, final_plan, final_plan_sha))
    final_parity = {}
    for role, original in (('raw', model), ('ema', ema.inference_copy(model))):
        recovered, _ = load_inference(final, final_plan, final_plan_sha, role=role)
        final_parity[role] = render_parity(original, recovered)
        del recovered
    write(out / 'finalization-receipt.json', read(final['receipt']))
    require(set(final_dir.iterdir()) == {final_dir / FINAL, final_dir / RECEIPTS}
            and {p.name for p in (final_dir / RECEIPTS).iterdir()} == {'step-000003.json'}, 'Unexpected finalized inventory')
    (final_dir / FINAL).unlink(); (final_dir / RECEIPTS / 'step-000003.json').unlink()
    (final_dir / RECEIPTS).rmdir(); final_dir.rmdir()
    require(state_sha256(parent.state_dict()) == parent_sha and sha(fixture_path) == fixture_sha,
            'Restart qualification changed its retained parent or fixture plan')
    return {'status': 'pass', 'parent_model_state_sha256': parent_sha, 'ordinary_microbatch': ordinary_microbatch,
        'auxiliary_microbatch': auxiliary_microbatch, 'logical_batch_size': 16, 'warmup_samples': 512, 'scored_samples': 44160,
        'saved_step': 2, 'replayed_step': 3, 'planned_stop_step': 2000, 'finalization_fixture_stop_step': 3,
        'tensor_count': tensor_count, 'packing': stats, 'interrupted_group_checks': interrupted_groups,
        'all_tensor_and_metadata_bytes_exact': True, 'third_weighted_selected_update_and_accounting_bit_exact': True,
        'all_40_adam_states_bit_exact': True, 'all_cpu_rng_streams_replayed': True,
        'interrupted_publication_retains_loadable_previous_generation': True,
        'finalization_reuses_inode': True, 'raw_and_ema_parity': parity, 'final_raw_and_ema_parity': final_parity,
        'production_budget_preflight_qualified': True, 'ephemeral_tensor_files_removed': True,
        'rejected_cases': rejected, 'quality_measured': False, 'parent_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and args.plan.resolve().is_relative_to(ARTIFACT_ROOT)
            and sha(args.plan) == args.plan_sha256 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'Require a frozen CPU1 restart qualification')
    plan = read(args.plan); verify_inputs(plan)
    require(torch.__version__ == plan['torch_version'] and not (args.plan.parent / 'result.json').exists(),
            'Changed runtime or existing result')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    began = time.monotonic(); budget_before = budget_snapshot()
    old_python, old_numpy, old_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def progress(phase, **detail):
        print(json.dumps({'event': 'four_second_restart_progress', 'phase': phase,
            'elapsed_seconds': time.monotonic() - began, **detail}), flush=True)
    try:
        torch.manual_seed(20261104); random.seed(20261104); np.random.seed(20261104)
        primitives = primitive_checks(); progress('bounded_codec_primitive_checks_pass')
        parent, _ = load_model(plan['fixture_checkpoint'])
        require(state_sha256(parent.state_dict()) == plan['fixture_model_state_sha256'], 'Changed retained parent')
        report = run_fixture(parent, plan, args.plan.parent, progress)
    finally:
        random.setstate(old_python); np.random.set_state(old_numpy); torch.set_rng_state(old_torch)
    require(not torch.cuda.is_initialized() and torch.equal(old_torch, torch.get_rng_state()), 'CPU check initialized CUDA or changed RNG')
    verify_inputs(plan)
    result = {'status': 'pass', 'plan_sha256': args.plan_sha256, 'source_bindings_unchanged': True,
        'bounded_codec_primitive_checks': primitives, 'restart': report, 'global_rng_restored': True,
        'gpu_used': False, 'quality_measured': False, 'production_training_started': False,
        'elapsed_seconds': time.monotonic() - began,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'budget_before': budget_before, 'budget_after': budget_snapshot(),
        'limits': ['CPU FP32 short weighted fixture; full four-second CPU neural gradients qualified separately.',
                   'No CUDA BF16 resource or production-throughput qualification.']}
    write(args.plan.parent / 'result.json', result)
    print(json.dumps({'status': 'pass', 'elapsed_seconds': result['elapsed_seconds'],
                      'peak_rss_bytes': result['peak_rss_bytes']}), flush=True)


if __name__ == '__main__':
    main()
