"""Measure saved GRU tensors and qualify exact sharing on short CUDA fixtures."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, verify_inputs, state_sha256, load_source, continuity
from research.direct.latency58_branch_memory_checkpoint import load_model
from research.direct.latency58_four_second_storage import snapshot


class SavedInventory:
    def __init__(self, module, *, share):
        self.share = share
        self.candidates = [(name, p.detach().t().to(torch.bfloat16))
                           for name, p in module.named_parameters() if p.ndim == 2]
        self.rows = {}
        self.hits = {name: 0 for name, _ in self.candidates}
        self.shared_bytes = 0

    def pack(self, tensor):
        key = (str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()))
        row = self.rows.setdefault(key, {'count': 0, 'storages': {}, 'matches': {}})
        row['count'] += 1
        storage = tensor.untyped_storage()
        row['storages'][storage.data_ptr()] = storage.nbytes()
        for name, value in self.candidates:
            if tensor.dtype == value.dtype and tensor.shape == value.shape and torch.equal(tensor, value):
                row['matches'][name] = row['matches'].get(name, 0) + 1
                self.hits[name] += 1
                if self.share:
                    self.shared_bytes += tensor.numel() * tensor.element_size()
                    return value
                break
        return tensor.detach()

    @staticmethod
    def unpack(value):
        return value

    def report(self):
        return {'share': self.share, 'matched_parameter_transpose_counts': self.hits,
            'matched_values_checked_exactly_at_every_save': True, 'shared_logical_bytes': self.shared_bytes,
            'saved_tensors': [{'dtype': key[0], 'shape': key[1], 'stride': key[2], 'count': row['count'],
                'unique_storage_pointers': len(row['storages']),
                'sum_unique_storage_bytes': sum(row['storages'].values()), 'parameter_matches': row['matches']}
                for key, row in self.rows.items()]}


def capture(module, x, h, mode):
    module.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    audio, hidden = x.clone().requires_grad_(), h.clone().requires_grad_()
    inventory = None if mode == 'plain' else SavedInventory(module, share=mode == 'shared')
    context = nullcontext() if inventory is None else torch.autograd.graph.saved_tensors_hooks(inventory.pack, inventory.unpack)
    with context:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output, state = module(audio.to(torch.bfloat16), hidden.to(torch.bfloat16))
        loss = output.float().square().mean() + state.float().square().mean()
        allocated_after_forward = torch.cuda.memory_allocated()
        loss.backward()
    torch.cuda.synchronize()
    values = {'output': output.detach().cpu(), 'state': state.detach().cpu(),
        'input_gradient': audio.grad.detach().cpu(), 'hidden_gradient': hidden.grad.detach().cpu(),
        **{name: p.grad.detach().cpu().clone() for name, p in module.named_parameters()}}
    require(all(bool(torch.isfinite(v).all()) and torch.count_nonzero(v) > 0 for v in values.values()),
            'Unexercised or nonfinite GRU fixture')
    report = {'mode': mode, 'seconds': time.monotonic() - began, 'loss': float(loss.detach()),
        'allocated_after_forward_bytes': allocated_after_forward,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'saved_inventory': None if inventory is None else inventory.report()}
    return values, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(); plan = read(args.plan)
    require(Path.cwd() == ROOT and sha(args.plan) == args.plan_sha256
            and all(os.environ.get(k) == v for k, v in plan['environment'].items()), 'Changed diagnostic environment')
    verify_inputs(plan)
    out = args.plan.parent
    require(not (out/'result.json').exists() and not (out/'metrics.jsonl').exists(), 'Preserve diagnostic evidence')
    monitor = load_source('saved_gru_watchdog', plan['watchdog_source'])
    _, events = continuity(plan, monitor); write(out/'event-continuity.json', events)
    before = snapshot()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    model, _ = load_model(plan['parent_checkpoint'])
    parent = state_sha256(model.state_dict())
    require(parent == plan['parent_model_state_sha256'] and not torch.cuda.is_initialized(), 'Changed CPU parent')
    torch.cuda.set_per_process_memory_fraction(.75)
    sys.path.insert(0, str(PRODUCTION)); import train_production as production
    production.configure_determinism(20261105)
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all()
    model.cuda().train().requires_grad_(True); model.training_precision = 'bf16'
    results = []
    generator = torch.Generator(device='cuda').manual_seed(202611051)
    with torch.random.fork_rng(devices=[0]):
        for name in ('fusion_branch', 'spec_memory', 'waveform_memory'):
            module = getattr(model, name)
            for frames in (32, 96):
                x = .03 * torch.randn(1, frames, module.input_size, device='cuda', generator=generator)
                h = .03 * torch.randn(module.num_layers, 1, module.hidden_size, device='cuda', generator=generator)
                expected, baseline = capture(module, x, h, 'plain')
                comparisons = []
                for mode in ('inventory', 'shared'):
                    values, report = capture(module, x, h, mode)
                    errors = {key: float((values[key] - value).abs().max()) for key, value in expected.items()}
                    exact = all(torch.equal(values[key], value) for key, value in expected.items())
                    report.update(all_outputs_input_hidden_and_parameter_gradients_bit_exact=exact,
                        maximum_errors=errors)
                    require(exact and report['loss'] == baseline['loss'], 'Saved-value sharing changed GRU arithmetic')
                    comparisons.append(report); del values
                results.append({'module': name, 'frames': frames, 'parameter_count': len(list(module.parameters())),
                    'plain': baseline, 'comparisons': comparisons})
                del expected, x, h
                module.zero_grad(set_to_none=True); gc.collect()
                with (out/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step': len(results), 'kind': 'diagnostic_fixture',
                        'training_optimizer_updates': 0, 'module': name, 'frames': frames}) + '\n')
                print(json.dumps({'event':'fixture_pass','module':name,'frames':frames,
                    'plain_peak':baseline['peak_allocated_bytes'], 'shared_peak':comparisons[-1]['peak_allocated_bytes']}),flush=True)
    require(state_sha256(model.state_dict()) == parent and torch.equal(cpu_rng, torch.get_rng_state())
            and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)),
            'Diagnostic changed parent weights or RNG')
    verify_inputs(plan)
    write(out/'result.json', {'status':'pass','plan_sha256':args.plan_sha256,'source_bindings_unchanged':True,
        'parent_model_state_sha256':parent,'parent_weights_unchanged':True,'rng_unchanged':True,
        'gpu_used':True,'training_optimizer_updates':0,'checkpoint_files_written':False,'quality_measured':False,
        'fixtures':results,'budget_before':before,'budget_after':snapshot(),
        'limits':['Short isolated GRUs only; no full-context or training qualification.']})
    print(json.dumps({'status':'pass','fixtures':len(results)}),flush=True)


if __name__ == '__main__': main()
