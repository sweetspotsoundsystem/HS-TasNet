"""Inventory the retained training chain and limits of existing data-overlap evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256

PRODUCTION = Path('/home/axel/autoresearch/production/hs-tasnet-c91-full-v1')
DATA = Path('/home/axel/autoresearch/datasets/hs-tasnet-recordpool-best200-v1')
OUT = PHASE / 'hare-data-lineage-audit-001'
R11 = ROOT / 'research/direct/runs/latency11'
EXPECTED_MANIFEST = '300b0bfbd835e2ce40c8832d10219a35941d6453392fa50688b236499375061a'


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not OUT.exists(), 'Use CPU and a fresh inventory')
    evidence = {}
    def bind(path):
        path = Path(path).resolve()
        evidence[str(path)] = sha(path)
        return {'path': str(path), 'sha256': evidence[str(path)]}
    def document(path):
        bind(path)
        return read(path)
    def content_hash(value):
        original = dict(value)
        expected = original.pop('content_sha256')
        actual = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':'),
                                            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        require(expected == actual, 'Metadata payload checksum failed')
        return actual
    bind(__file__)
    corpus_path = PRODUCTION / 'manifests/combined.manifest.json'
    corpus = document(corpus_path)
    require(sha(corpus_path) == EXPECTED_MANIFEST, 'Final-stage corpus changed')
    content_hash(corpus)
    collision = document(PRODUCTION / 'manifests/acoustic_collision_report.json')
    require(content_hash(collision) == corpus['integrity']['acoustic_collision_report']['content_sha256'],
            'Collision report does not belong to the corpus')
    splits = {name: document(ROOT / f'research/manifests/{name}.json') for name in ('train', 'valid', 'test')}
    for name, split in splits.items():
        source_name = {'train': 'musdb_train_manifest', 'valid': 'musdb_validation_manifest', 'test': 'musdb_test_manifest'}[name]
        require(corpus['provenance']['inputs'][source_name]['sha256'] == sha(ROOT / f'research/manifests/{name}.json'),
                'Corpus exclusion inventory differs')
    normalize = lambda name: re.sub(r'[^\w]+', ' ', name.casefold()).strip()
    train_names = {normalize(track['name']) for track in corpus['tracks']}
    overlap = {name: sorted(track['name'] for track in splits[name]['tracks']
                           if normalize(track['name']) in train_names) for name in ('valid', 'test')}
    require(not any(overlap.values()), 'Normalized training/evaluation name collision')
    require(collision['status'] == 'pass' and not collision['evaluation_reference_collisions']
            and not collision['exact_excluded_identity_collisions'], 'Stored exact-collision evidence failed')

    positive_path = PHASE / 'leader-cleanup-production-prep-001/training-plan.json'
    positive = document(positive_path)
    positive_checkpoint = Path(positive['run_dir']) / 'checkpoints/step-000250/model.pt'
    receipt = document(positive_checkpoint.parent / 'receipt.json')
    require(bind(positive_checkpoint)['sha256'] == receipt['files']['model.pt']['sha256'], 'C204 bytes changed')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = torch.load(positive_checkpoint, map_location='cpu', weights_only=True)
    require(state_sha256(payload['model']) == receipt['model_state_sha256'], 'C204 tensor identity differs')
    provenance = payload['provenance']
    del payload
    stage_specs = [
        ('C91 transfer, 256-sample hop', R11 / 'smoke/cropped1024-ola-prep/training-plan-released-001.json', 2000),
        ('Raw four-head control', R11 / 'smoke/cropped1024-relative-mse-prep/training-raw4_control-plan-released-001.json', 250),
        ('128-sample hop transfer', PHASE / 'pilot-prep/training-plan.json', 2000),
        ('Asymmetric analysis transfer', PHASE / 'asymmetric-prep/training-plan.json', 500),
        ('11.6 ms ancestor teacher', PHASE / 'teacher-prep/half-plan.json', 250),
        ('C91 teacher', PHASE / 'sdr-teacher-prep-001/c91-plan.json', 1000),
        ('Carried training context', PHASE / 'sdr-context-training-prep-001/warm-plan.json', 500),
        ('Accumulated batch 16', PHASE / 'sdr-accum-prep-001/training-plan.json', 1000),
        ('Drum-weighted objective', PHASE / 'sdr-drum-accum-prep-001/training-plan.json', 500),
        ('Controlled deployed-output supervision', positive_path, 250),
    ]
    # The nested provenance records later retained stages. Earlier crop-transfer
    # ancestry is read from its explicitly named retained checkpoint below.
    plan_digests = set()
    node = provenance
    while isinstance(node, dict):
        if node.get('training_plan_sha256'):
            plan_digests.add(node['training_plan_sha256'])
        node = node.get('parent_provenance')
    crop_path = R11 / 'cropped1024-matched-raw4_control-b4-bf16-lr3e-5/checkpoints/step-000250/model.pt'
    bind(crop_path)
    crop = torch.load(crop_path, map_location='cpu', weights_only=True)
    crop_provenance = crop['provenance']
    plan_digests.add(crop['plan_sha256'])
    plan_digests.add(crop_provenance['parent_training_plan']['sha256'])
    require(crop['step'] == 250 and crop_provenance['parent_training_updates'] == 2000,
            'Earlier cropped-model endpoint differs')
    del crop
    stages = []
    for label, path, used_steps in stage_specs:
        plan = document(path)
        config = plan['config']
        require(sha(path) in plan_digests and plan['manifest_sha256'] == EXPECTED_MANIFEST
                and used_steps <= config['steps'], 'Retained training plan is outside the actual lineage')
        stages.append({'stage': label, 'training_plan': bind(path), 'used_updates': used_steps,
                       'planned_schedule_updates': config['steps'], 'batch_size': config['batch_size'],
                       'crop_samples': config['crop_samples'], 'seed': config['seed'],
                       'data_start': config['data_start'], 'data_stop': config['data_start'] + used_steps * config['batch_size'],
                       'manifest_sha256': plan['manifest_sha256']})
    require(sum(stage['used_updates'] for stage in stages) == provenance['training_updates'] == 8250,
            'Retained student update accounting differs')
    teacher_selection = document(ROOT / 'research/direct/runs/c91-refined-v1/selection.json')
    teacher_refine = document(ROOT / 'research/direct/runs/c91-50k-lr3e-5-projection-off/config.json')
    teacher_binding = positive['teacher']
    require(bind(teacher_binding['path'])['sha256'] == teacher_binding['sha256'] == teacher_selection['checkpoint_sha256'],
            'Refined C91 teacher differs from selection metadata')
    require(bind(teacher_refine['parent'])['sha256'] == teacher_refine['parent_sha256'], 'C91 50k parent changed')
    snapshot = document(Path(teacher_refine['parent'] + '.json'))
    contract = document(PRODUCTION / 'runs/c91-full-best200-v1-seed60/run_contract.json')
    full_config = document(PRODUCTION / 'full_config.json')
    require(snapshot['sha256'] == teacher_refine['parent_sha256']
            and teacher_refine['manifest_sha256'] == contract['static']['manifest_file_sha256'] == EXPECTED_MANIFEST
            and contract['static']['config'] == full_config
            and contract['static']['config_file_sha256'] == sha(PRODUCTION / 'full_config.json'),
            'C91 production/refinement corpus or configuration differs')
    label_inventory = document(DATA / 'selection.json')
    bind(DATA / 'README.md')
    bind(DATA / 'validation.json')
    require(len(label_inventory['tracks']) == 200, 'RecordPool selection count changed')

    OUT.mkdir()
    names_path = OUT / 'test-identity-search.txt'
    names_path.write_text('\n'.join(track['name'] for track in splits['test']['tracks']) + '\n')
    search_roots = [ROOT / 'research', PRODUCTION, Path('/home/axel/autoresearch/claude/HS-TasNet/research')]
    command = ['rg', '-l', '-F', '--hidden', '--no-ignore', '-g', '*.json', '-g', '!**/node_modules/**',
               '-g', '!**/build*/**', '-g', '!**/dependencies/**', '-f', str(names_path), *map(str, search_roots)]
    search = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    require(search.returncode in (0, 1), 'Prior-use metadata search failed')
    matches = sorted(set(search.stdout.splitlines()))
    for path in matches:
        bind(path)
    # Preserve candidates verbatim as paths for review; a name match in a corpus
    # exclusion list is not an evaluation, and no match cannot prove non-use.
    result = {'schema': 'hare-training-lineage-inventory-v1', 'status': 'partial_metadata_inventory_complete',
              'source_bindings': evidence, 'source_bindings_unchanged': all(sha(p) == s for p, s in evidence.items()),
              'c204_model_state_sha256': receipt['model_state_sha256'], 'retained_student_stages': stages,
              'student_updates_after_C91_transfer': 8250,
              'c91_teacher': {'production_checkpoint_step': 50000, 'additional_updates': teacher_selection['additional_updates'],
                              'planned_refinement_updates': teacher_refine['steps'], 'refinement_config': teacher_refine,
                              'source_gains': teacher_selection['deployment_source_gains'],
                              'same_manifest_as_student': True},
              'corpus': {'tracks': corpus['track_count'], 'effective_hours': corpus['total_effective_hours'],
                         'root_counts': {root['root_id']: root['track_count'] for root in corpus['roots']},
                         'normalized_name_overlaps': overlap, 'local_test_tracks': len(splits['test']['tracks']),
                         'test_missing_from_official_count': splits['test']['missing_from_official_count']},
              'existing_fingerprint_check': {key: collision[key] for key in ('algorithm', 'status', 'limitations', 'evidence')},
              'recordpool_label_provenance': {'source_container_format': 'five-stream .stem.m4a, imported as FLAC',
                    'selection_roles': label_inventory['counts'],
                    'curation_model_hashes': label_inventory['input_hashes'],
                    'original_stem_generator_identified': False,
                    'note': 'C91 is recorded as a semantic-purity selector, not established as the original label generator.'},
              'prior_test_use_search': {'roots': list(map(str, search_roots)), 'file_filter': '*.json',
                                       'identity_count': len(splits['test']['tracks']), 'matching_paths': matches,
                                       'returncode': search.returncode, 'exhaustive_non_use_proven': False},
              'new_model_inference_on_test_audio': False, 'test_set_cleared_for_unseen_claim': False,
              'cuda_initialized': torch.cuda.is_initialized(),
              'remaining_work': ['Review all matched prior-use metadata; search external experiment records before declaring a test set untouched.',
                   'Resolve the original RecordPool stem-generator provenance and describe any unavailable training history.',
                   'Check near duplicates, including shifted/cropped audio; the existing exact first-120-second fingerprint test is insufficient.',
                   'Revalidate intended evaluation audio identities and freeze a full-song protocol before inference.',
                   'Describe this as a 46-song local subset unless the four missing official test songs are acquired and audited.']}
    require(result['source_bindings_unchanged'] and not result['cuda_initialized'], 'Inventory inputs changed or CUDA initialized')
    write(OUT / 'result.json', result)
    print(json.dumps({'status': result['status'], 'student_stages': len(stages), 'student_updates': 8250,
                      'same_manifest_as_teacher': True, 'prior_use_candidate_paths': matches,
                      'output': str(OUT / 'result.json')}), flush=True)


if __name__ == '__main__':
    main()
