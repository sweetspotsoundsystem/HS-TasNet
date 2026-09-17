"""Bind the selected four-batch prefix, including complete source-view reductions."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT, POLICY, snapshot
from research.direct.latency58_four_second_data import dataset, policy, CROP_SAMPLES, WARMUP_SAMPLES
from research.direct.latency58_grouped_vocal_auxiliary import source_views, prepare_groups
from research.direct.latency58_remix_augmentation import augment
from research.direct.check_latency58_four_second_data import TracedCrops, audio_sha, production
from research.direct.latency58_recorded301_data import select_tracks


def main():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only prefix preparation')
    out = ARTIFACT_ROOT / 'selected-data-prefix-001'
    require(not out.exists(), 'Preserve selected prefix')
    decision_path = PHASE / 'weighted-vocal-review-014/next-experiment-decision.json'
    decision = read(decision_path)
    require(sha(decision_path) == '2d518bb6067914779e8adcd1903f53c6d52a361b44352093cf890f618a4e295c',
            'Scientific decision changed')
    config = decision['proposed_training_config']
    previous = PHASE / 'four-second-data-cpu-001'
    inputs, data_result = read(previous / 'inputs.json'), read(previous / 'result.json')
    execution = read(PHASE / 'four-second-data-cpu-stage-001/root-execution.json')
    require(data_result['status'] == 'pass' and data_result['inputs_sha256'] == sha(previous / 'inputs.json')
            and execution['actual_exit_code'] == 0 and config['augmentation'] == policy()
            and config['data_start'] == inputs['first_sample_index'] == 4132000
            and config['seed'] == inputs['augmentation_seed'] == 20261102,
            'Selected recipe differs from the qualified four-second prefix')
    bindings = dict(inputs['source_bindings'])
    paths = [Path(__file__).resolve(), decision_path, previous / 'inputs.json', previous / 'result.json',
             PHASE / 'four-second-data-cpu-stage-001/root-execution.json', POLICY]
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if isinstance(value, str):
            path = Path(value)
            if path.is_file() and path.suffix == '.py' and path.resolve().is_relative_to(ROOT):
                paths.append(path.resolve())
    for path in paths:
        digest = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == digest, 'Frozen input conflict')
        bindings[str(path)] = digest
    plan = {'schema': 'latency58-four-second-selected-prefix-v1', 'source_bindings': bindings,
            'decision': {'path': str(decision_path), 'sha256': sha(decision_path)},
            'config': config, 'first_index': config['data_start'], 'stop_index': config['data_start'] + 64,
            'workers': 2, 'ordinary_window_count_per_stem': 64, 'auxiliary_window_count_per_stem': 8,
            'budget_before': snapshot(), 'model_loaded': False, 'gpu_used': False}
    verify_inputs(plan)
    out.mkdir(); write(out / 'plan.json', plan)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    production_config = read(PRODUCTION / 'full_config.json')
    _, tracks, _, _ = production.load_corpus_manifest(PRODUCTION / 'manifests/combined.manifest.json',
        expected_file_sha256=inputs['selection']['source_manifest_sha256'], config=production_config)
    tracks = select_tracks(tracks, inputs['selection'])
    data = dataset(production, tracks, config, plan['stop_index'], dataset_class=TracedCrops)
    expected = data_result['augmented_loaders'][1]['batches']
    loader = DataLoader(data, batch_size=16, num_workers=2,
        sampler=production.AbsoluteIndexSampler(plan['first_index'], plan['stop_index']),
        worker_init_fn=production.worker_init, multiprocessing_context='spawn', prefetch_factor=2,
        generator=torch.Generator().manual_seed(config['seed'] + 1))
    rows = []
    for index, (mixture, targets) in enumerate(loader):
        require(index < 4 and mixture.shape == (16, 2, CROP_SAMPLES), 'Wrong prefix geometry')
        first = plan['first_index'] + 16 * index
        before = audio_sha(mixture, targets)
        mixture, targets = augment(mixture, targets, seed=config['seed'], first_sample_index=first)
        after = audio_sha(mixture, targets)
        require({'first_index': first, 'input_sha256': before, 'after_remix_sha256': after} == expected[index],
                'Selected prefix does not reproduce the prior worker-qualified data')
        audio, truth = source_views(mixture, targets)
        groups = prepare_groups(targets[..., WARMUP_SAMPLES:], truth[..., WARMUP_SAMPLES:])
        row = {'first_index': first, 'input_sha256': before, 'after_remix_sha256': after,
               'auxiliary_full_context_sha256': audio_sha(audio, truth)}
        for group, count in [('ordinary', 64), ('auxiliary', 8)]:
            reduction = getattr(groups, group)
            require(torch.equal(reduction.active + reduction.absent, torch.full((4,), count)),
                    'The prefix lost a complete scored window')
            row[group + '_active_counts'] = reduction.active.tolist()
            row[group + '_absent_counts'] = reduction.absent.tolist()
        rows.append(row)
        print(json.dumps({'event': 'selected_prefix_batch', **row}), flush=True)
    require(len(rows) == 4 and not torch.cuda.is_initialized(), 'Incomplete selected CPU prefix')
    verify_inputs(plan)
    result = {'status': 'pass', 'plan_sha256': sha(out / 'plan.json'), 'source_bindings_unchanged': True,
              'qualified_data_prefix': rows, 'ordinary_complete_windows_per_stem': 64,
              'auxiliary_complete_windows_per_stem': 8, 'reproduces_prior_two_worker_proof': True,
              'all_285_previously_decoded_training_files_authenticated': True,
              'model_loaded': False, 'gpu_used': False, 'quality_measured': False,
              'elapsed_seconds': time.monotonic() - began, 'budget_after': snapshot(),
              'created_utc': datetime.now(timezone.utc).isoformat()}
    write(out / 'result.json', result)
    print(json.dumps({'status': 'pass', 'batches': len(rows), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
