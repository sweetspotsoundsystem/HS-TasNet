"""Freeze the selected four-second experiment after current CPU evidence closes."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import sys

import torch

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import state_sha256, verify_inputs
from research.direct.train_latency58_four_second import validate_recipe, runtime_policy, budget_snapshot
from research.direct.run_latency58_four_second import require_cpu_evidence, require_monitor_closed, binding
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT, POLICY
from research.direct.latency58_lossless_recovery_codec_v3 import policy as packed_policy
from research.direct.latency58_branch_memory_checkpoint import load_model
from research.direct.run_latency58_four_second_quality import transport_proof, load_references
from research.direct.recover_latency58_nvml_guard_idle import require_monitor_qualification

OUT = ARTIFACT_ROOT / 'branch-four-second-015'
DECISION = PHASE / 'weighted-vocal-review-014/next-experiment-decision.json'
SOURCE = PHASE / 'branch-weighted-vocal-014'


def scientific_loop_proof():
    paths = [ROOT / 'research/direct' / name for name in
             ('train_latency58_weighted_vocal.py', 'train_latency58_four_second.py')]
    loops = []
    for path in paths:
        found = [n for n in ast.walk(ast.parse(path.read_text()))
                 if isinstance(n, ast.For) and ast.unparse(n.target) == '(mixture_cpu, truth_cpu)']
        require(len(found) == 1, 'Ambiguous scientific training loop')
        loops.append(ast.dump(found[0], include_attributes=False))
    require(loops[0] == loops[1], 'Scientific training loop changed beyond its bound configuration/imports')
    return {'status': 'pass', 'reference': binding(paths[0]), 'candidate': binding(paths[1]),
            'complete_training_loop_ast_identical': True,
            'scope': 'Data addressing, remix calls, schedule computation, grouped update, journals and rolling save call sites.',
            'selected_changes': 'Bound crop geometry, configuration, ordinary microbatch size and separately qualified storage adapters.'}


def main():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'Require CUDA-hidden CPU1 preparation')
    require(not OUT.exists(), 'Preserve existing four-second training plans')
    decision, source = read(DECISION), read(SOURCE / 'plan.json')
    require(sha(DECISION) == '2d518bb6067914779e8adcd1903f53c6d52a361b44352093cf890f618a4e295c', 'Scientific decision changed')
    completed = read(SOURCE / 'quality-root-execution.json')
    require(completed['actual_exit_code'] == 0 and completed['all_six_child_processes_closed']
            and completed['result_sha256'] == sha(SOURCE / 'result.json'), 'Finish prior quality review before a new trial')
    verify_inputs(decision); verify_inputs(source)
    bindings = dict(source['source_bindings'])
    def merge(incoming):
        for path, digest in incoming.items():
            require(path not in bindings or bindings[path] == digest, 'Conflicting frozen input: ' + path)
            bindings[path] = digest
    merge(decision['source_bindings'])
    paths = [Path(__file__).resolve(), DECISION, SOURCE / 'plan.json', SOURCE / 'quality-root-execution.json',
             SOURCE / 'production-stage/execution.json', SOURCE / 'production-root-execution.json', POLICY,
             ARTIFACT_ROOT / 'evaluation-adapter-source-proof.json']
    qualifications = [(PHASE / 'four-second-model-cpu-002', 'four_second_model_cpu_qualification'),
                      (ARTIFACT_ROOT / 'restart-cpu-001', 'four_second_restart_cpu_qualification'),
                      (ARTIFACT_ROOT / 'selected-data-prefix-002', 'four_second_prefix_qualification')]
    for directory, key in qualifications:
        prepared = read(directory / 'plan.json'); verify_inputs(prepared)
        merge(prepared['source_bindings'])
        paths.extend(directory / name for name in ('plan.json', 'result.json', 'root-execution.json', 'root-command.json'))
    prefix = read(ARTIFACT_ROOT / 'selected-data-prefix-002/result.json')['qualified_data_prefix']
    paths.extend(ROOT / 'research/direct' / name for name in (
        'latency58_four_second_data.py', 'latency58_four_second_storage.py', 'latency58_four_second_recovery_files.py',
        'latency58_lossless_recovery_codec_v3.py', 'check_latency58_four_second_device.py',
        'latency58_four_second_device_recovery.py', 'train_latency58_four_second.py', 'run_latency58_four_second.py',
        'latency58_four_second_evaluation.py', 'run_latency58_four_second_quality.py',
        'evaluate_latency58_four_second_memory.py', 'evaluate_latency58_four_second_vocal_views.py',
        'check_latency58_four_second_vocal_views.py'))
    evaluation_proof = transport_proof()
    require(evaluation_proof == read(ARTIFACT_ROOT / 'evaluation-adapter-source-proof.json'),
            'Evaluation adapters changed after source proof')
    template_path = PHASE / 'paired-vocal-grouped-013/plan.json'
    paths.append(template_path)
    reference_bindings = {}
    references = load_references(read(template_path), source, reference_bindings)
    merge(reference_bindings)
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if isinstance(value, str):
            path = Path(value)
            if path.is_file() and path.suffix == '.py' and path.resolve().is_relative_to(ROOT):
                paths.append(path.resolve())
    merge({str(path): sha(path) for path in paths})
    require_monitor_qualification()
    previous = read(SOURCE / 'production-stage/execution.json')
    monitor = read(previous['monitor_result'])
    require_monitor_closed(previous, monitor, final_step=1000)
    merge({previous['monitor_result']: sha(previous['monitor_result'])})
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    model, payload = load_model(decision['parent_checkpoint'])
    require(not torch.cuda.is_initialized() and state_sha256(model.state_dict()) == decision['parent_model_state_sha256']
            and model.architecture_metadata == decision['inference_architecture']
            and list(dict(model.named_parameters())) == decision['trainable_parameter_names']
            and payload['provenance']['training_updates'] == decision['parent_training_updates']
            and state_sha256(dict(model.named_buffers())) == source['fixed_buffers_sha256'],
            'Selected saved parent, architecture, parameter set or buffers differ')
    proof = scientific_loop_proof()
    plan = copy.deepcopy(source)
    for key in ('weighted_storage', 'weighted_storage_forecast', 'weighted_training_prefix_ast_proof',
                'controlled_reference_training_plan', 'storage_allocation_qualification', 'generated_cache_retirement',
                'packed_integration_cpu_qualification', 'weighted_cpu_qualification', 'packed_cpu_qualification'):
        plan.pop(key, None)
    plan.update(name=OUT.name, output_directory=str(OUT), config=decision['proposed_training_config'],
        source_bindings=bindings, scientific_decision=binding(DECISION),
        parent_checkpoint=decision['parent_checkpoint'], parent_model_state_sha256=decision['parent_model_state_sha256'],
        initialized_model_state_sha256=decision['parent_model_state_sha256'],
        parent_training_updates=decision['parent_training_updates'], parent_weight_role='ema',
        optimizer_initialization='fresh_adam', warmup_samples=88064, scored_samples=176512, accumulation_steps=8,
        qualified_data_prefix=prefix, packed_recovery=packed_policy(), runtime_allowance=runtime_policy(),
        new_storage_policy_binding=binding(POLICY), packed_publication_directory=str(OUT / 'production-run'),
        previous_execution_for_resource=str(SOURCE / 'production-stage/execution.json'),
        continuation_kind='four_second_scored_context_with_more_optimization_on_new_addresses',
        scientific_training_loop_ast_proof=proof,
        evaluation_adapter_source_proof=binding(ARTIFACT_ROOT / 'evaluation-adapter-source-proof.json'),
        evaluation_reference_models={key: value['views']['model'] for key, value in references.items()},
        event_continuity_scope='Continue the completed 014 production monitor into the new four-second resource rehearsal.',
        qualified_data_prefix_scope='Current four-batch, two-worker proof includes full auxiliary histories and 64/8 per-stem windows.',
        historical_cpu_qualification_scope='Inherited component identities remain historical; current full-context, B2 restart and data proofs qualify this trial.',
        quality_selected=False, overall_goal_complete=False,
        post_training_requirements=['Audit both packed raw/EMA roles and all optimizer/journal state from the saved endpoint.',
            'Run the unchanged 14-track/28-excerpt full-band protocol and both continuous source views for raw and EMA.',
            'Review all 56 track/stem cells and 840 source-view windows against 012 EMA, retained 006 EMA, both 014 endpoints and the released graph.',
            'Qualify the exact export and physical M4 playback before release; retain all rollback baselines.'])
    for directory, key in qualifications:
        plan[key] = binding(directory / 'result.json')
    # The new auditor retains the complete prior root/Git/reserve accounting,
    # the whole 4 GB M4 allocation and this disjoint whole 2.5 GB allocation.
    before = budget_snapshot(plan)
    plan['storage_budget'] = {'authorized_cap_bytes': 100_000_000_000,
        'accounting_policy': binding(POLICY), 'complete_forecast': before,
        'additional_research_root': str(ARTIFACT_ROOT), 'additional_reserved_peak_bytes': 2_500_000_000}
    plan['counted_roots'] = [*source['counted_roots'],
        *(v['root'] for v in before['prior_training_and_m4_forecast']['additional_allocations']), str(ARTIFACT_ROOT)]
    plan['budget_before'] = before
    validate_recipe(plan); require_cpu_evidence(plan); verify_inputs(plan)
    OUT.mkdir(); write(OUT / 'plan.json', plan)
    write(OUT / 'preparation-result.json', {'status': 'prepared', 'plan_sha256': sha(OUT / 'plan.json'),
        'source_bindings_unchanged': True, 'verified_input_count': len(bindings),
        'scientific_training_loop_ast_proof': proof, 'all_current_cpu_evidence_pass': True,
        'parent_model_state_sha256': decision['parent_model_state_sha256'],
        'budget_after': budget_snapshot(plan), 'gpu_workload_started': False, 'quality_measured': False})
    print(json.dumps({'status': 'prepared', 'plan_sha256': sha(OUT / 'plan.json'),
        'updates': 2000, 'verified_input_count': len(bindings), 'combined_peak_bytes': before['combined_peak_bytes']}), flush=True)


if __name__ == '__main__':
    main()
