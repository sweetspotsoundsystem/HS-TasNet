"""Prepare a fresh grouped continuation from a completely reviewed saved pair."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PHASE, read, require, sha, write
from research.direct.train_latency58 import verify_inputs, state_sha256
from research.direct.run_latency58_paired_vocal_serial import load_endpoint, merge_bindings, binding
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.train_latency58_grouped_continuation import validate_recipe
from research.direct.recover_latency58_nvml_guard_idle import require_monitor_qualification, WATCHDOG_SHA

SOURCE = PHASE / 'branch-grouped-vocal-012'
VIEWS = PHASE / 'paired-vocal-grouped-012'
REVIEW = PHASE / 'grouped-vocal-parent-review-012'
PREFIX = PHASE / 'grouped-vocal-next-prefix-013'
OUT = PHASE / 'branch-grouped-vocal-013'


def prepare(selection_path):
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'Require CUDA-hidden CPU1 preparation')
    require(not OUT.exists(), 'Preserve previous training plans')
    selection = read(selection_path)
    report, report_execution, report_plan = (read(REVIEW / name) for name in ('result.json', 'execution.json', 'plan.json'))
    paired, paired_execution, paired_plan = (read(VIEWS / name) for name in ('result.json', 'root-execution.json', 'plan.json'))
    require(report['status'] == paired['status'] == 'pass'
            and report['source_bindings_unchanged'] and paired['source_bindings_unchanged']
            and report_execution['actual_exit_code'] == paired_execution['actual_exit_code'] == 0
            and report_execution['source_bindings_unchanged'] and paired_execution['source_bindings_unchanged']
            and not report_execution['timed_out'] and not paired_execution['timed_out']
            and report_execution['result_sha256'] == sha(REVIEW / 'result.json')
            and paired_execution['result_sha256'] == sha(VIEWS / 'result.json')
            and report['plan_sha256'] == report_execution['plan_sha256'] == sha(REVIEW / 'plan.json')
            and paired['plan_sha256'] == paired_execution['plan_sha256'] == sha(VIEWS / 'plan.json'),
            'Complete both source views, actual root exit and retained-parent review first')
    require(selection['schema'] == 'latency58-grouped-continuation-selection-v1'
            and selection['status'] == 'selected_for_research_continuation'
            and selection['parent_comparison'] == binding(REVIEW / 'result.json')
            and selection['parent_comparison_execution'] == binding(REVIEW / 'execution.json')
            and selection['paired_source_views'] == binding(VIEWS / 'result.json')
            and selection['all_track_stem_and_worst_windows_reviewed']
            and selection['complete_parent_and_candidate_views_reviewed']
            and not selection['overall_goal_complete'] and not selection['plugin_replaced'],
            'Selection must describe this completed comparison and all remaining goals')
    records, models, bindings = load_endpoint(SOURCE)
    source = records['source']
    require(paired['models'] == models, 'Paired views and saved generation differ')
    role = selection['selected_role']
    require(role in ('raw', 'ema') and selection['selected_checkpoint'] == models[role]['checkpoint']
            and selection['selected_model_state_sha256'] == models[role]['model_state_sha256'], 'Wrong selected parent')
    schedule = selection['schedule']
    require(set(schedule) == {'steps', 'warmup', 'lr', 'min_lr'}
            and type(schedule['steps']) is int and schedule['steps'] in (500, 1000)
            and type(schedule['warmup']) is int and 0 < schedule['warmup'] <= 100
            and schedule['warmup'] < schedule['steps']
            and 0 < schedule['min_lr'] <= schedule['lr'] <= 3e-5,
            'Keep the reviewed schedule within the bounded continuation range')
    require(selection['objective_version'] == source['objective_version']
            and selection['grouped_vocal_loss'] == source['grouped_vocal_loss'],
            'A changed objective requires separate CPU qualification')
    prefix_plan, prefix, prefix_execution = (read(PREFIX / name) for name in ('plan.json', 'result.json', 'execution.json'))
    require(prefix['status'] == 'pass' and prefix['source_bindings_unchanged']
            and prefix_execution['actual_exit_code'] == 0 and not prefix_execution['timed_out']
            and prefix_execution['source_bindings_unchanged']
            and prefix_execution['result_sha256'] == sha(PREFIX / 'result.json')
            and prefix_execution['plan_sha256'] == prefix['plan_sha256'] == sha(PREFIX / 'plan.json')
            and prefix['independent_ordinary_references_match'] and prefix['zero_and_two_worker_replay_exact']
            and prefix['ordinary_samples_and_rng_unchanged_by_source_views']
            and prefix['source_views_exact_through_warmup_and_scored_suffix']
            and prefix['first_sample_index'] == source['config']['data_start'] + source['config']['steps'] * 16
            and prefix['stop_sample_index'] == prefix['first_sample_index'] + 64
            and prefix_plan['augmentation_seed'] == source['config']['seed']
            and prefix_plan['source_training_plan'] == binding(SOURCE / 'plan.json'), 'New addressed prefix is not qualified')
    for record in (source, paired_plan, report_plan, prefix_plan, read(PREFIX / 'inputs.json')):
        merge_bindings(bindings, record['source_bindings'])
    paths = [selection_path, Path(__file__).resolve(), SOURCE / 'root-command.json']
    for directory, names in ((REVIEW, ('plan.json','result.json','execution.json')),
                             (VIEWS, ('plan.json','result.json','root-execution.json','root-command.json')),
                             (PREFIX, ('plan.json','result.json','execution.json','inputs.json','data-config.json','ordinary-reference.json'))):
        paths.extend(directory / name for name in names)
    paths.extend(ROOT / 'research/direct' / name for name in ('run_latency58_grouped_continuation.py',
        'train_latency58_grouped_vocal_recovery.py', 'train_latency58_grouped_continuation.py',
        'latency58_grouped_continuation_recovery_check.py', 'qualify_latency58_grouped_continuation_recovery.py',
        'run_latency58_grouped_vocal_nvml_guard.py',
        'recover_latency58_nvml_guard_idle.py', 'qualify_latency58_grouped_vocal_prefix.py'))
    retired = PHASE / 'grouped-continuation-build-cache-retirement-001'
    receipt, retired_execution = read(retired / 'receipt.json'), read(retired / 'execution.json')
    require(receipt['status'] == 'complete' and receipt['source_bindings_unchanged']
            and receipt['all_preserved_files_unchanged'] and not receipt['checkpoints_or_optimizers_removed']
            and retired_execution['actual_exit_code'] == 0
            and retired_execution['receipt_sha256'] == sha(retired / 'receipt.json'), 'Storage retirement is incomplete')
    schedule_cpu = PHASE / 'grouped-continuation-recovery-cpu-013'
    cpu_result, cpu_execution = read(schedule_cpu / 'result.json'), read(schedule_cpu / 'execution.json')
    from research.direct.run_latency58_grouped_continuation import require_recovery_check
    require_recovery_check(cpu_result, expected_steps=schedule['steps'])
    require(cpu_result['device'] == 'cpu' and not cpu_result['gpu_used']
            and cpu_result['parent_checkpoint'] == models[role]['checkpoint']
            and cpu_result['source_bindings_unchanged'] and cpu_execution['actual_exit_code'] == 0
            and cpu_execution['source_bindings_unchanged'] and not cpu_execution['timed_out']
            and cpu_execution['result_sha256'] == sha(schedule_cpu / 'result.json')
            and cpu_execution['plan_sha256'] == cpu_result['plan_sha256'] == sha(schedule_cpu / 'plan.json'),
            'Actual planned-schedule CPU recovery qualification is incomplete')
    merge_bindings(bindings, read(schedule_cpu / 'plan.json')['source_bindings'])
    paths.extend(schedule_cpu / name for name in ('plan.json','result.json','execution.json','fixture-plan.json'))
    paths.extend(retired / name for name in ('inventory.json','intent.json','receipt.json','execution.json'))
    merge_bindings(bindings, {str(path): sha(path) for path in paths})
    require_monitor_qualification()
    monitor = records['monitor']
    require(monitor['status'] == monitor['supervisor_health'] == 'pass'
            and monitor['source_sha256'] == sha(source['watchdog_source']) == WATCHDOG_SHA
            and monitor['child_exit_code'] == 0 and monitor['post_exit_quiet_completed'], 'Prior GPU monitor is incomplete')
    for worker in ('event_worker_close','gpu_worker_close'):
        require(monitor[worker]['closed'] and monitor[worker]['actual_exit_code'] == 0
                and not monitor[worker]['forced'], 'Prior monitor worker did not close normally')
    verify_inputs({'source_bindings': bindings})
    import torch
    from research.direct.latency58_branch_memory_checkpoint import load_model
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    model, _ = load_model(models[role]['checkpoint'])
    require(not torch.cuda.is_initialized()
            and state_sha256(model.state_dict()) == models[role]['model_state_sha256']
            and state_sha256(dict(model.named_buffers())) == source['fixed_buffers_sha256']
            and model.architecture_metadata == source['inference_architecture']
            and list(dict(model.named_parameters())) == source['parameter_names']
            and model.provenance['training_updates'] == source['parent_training_updates'] + source['config']['steps'],
            'Saved initialization, buffers, parameters or inference architecture changed')
    before = budget_snapshot(source['storage_budget'])
    require(before['headroom_bytes'] > 1_250_000_000, 'Reserve both persistent generations and metadata')
    outside = before['external_git_common_bytes'] + source['storage_budget']['other_outside_allowance_bytes'] + source['storage_budget']['diagnostic_artifact_allowance_bytes']
    plan = copy.deepcopy(source)
    historical_keys = ('retry_of','retry_reason','original_resource_failure','numerical_diagnostic',
        'original_failed_resource_preserved','original_resource_updates','prior_controlled_stop','prior_successful_resource',
        'prior_unsaved_training_updates','replay_from_saved_selected_parent','prior_failed_training','recovered_host_idle',
        'resume_checkpoint','resume_prefix_qualification','replay_reference','post_alert_recovery_audit','parent_recovery_result')
    for key in historical_keys:
        plan.pop(key, None)
    plan.update(name=OUT.name, output_directory=str(OUT), source_bindings=bindings,
        config={**source['config'], **schedule, 'checkpoint_every':schedule['steps'], 'data_start':prefix['first_sample_index']},
        parent_checkpoint=models[role]['checkpoint'], parent_weight_role=role,
        parent_model_state_sha256=models[role]['model_state_sha256'], initialized_model_state_sha256=models[role]['model_state_sha256'],
        parent_training_updates=model.provenance['training_updates'], parent_full_sdr_db=records['terminal']['full_sdr_db'][role],
        reference_result=models[role]['original_full_mixture_report']['path'],
        parent_selection_review=binding(selection_path), continuation_review=selection,
        continuation_kind='fresh_adam_from_completed_grouped_source_model', original_parent_training_monitor_successful=True,
        retained_best_research_checkpoint=source['parent_checkpoint'], retained_best_research_reference_result=source['reference_result'],
        retained_best_research_full_sdr_db=source['parent_full_sdr_db'],
        quality_endpoints=[schedule['steps']], qualified_data_prefix=prefix['batches'], budget_before=before,
        outside_roots_reservation_bytes=outside, stop_counted_bytes=90_000_000_000-outside,
        previous_execution_for_resource=str(SOURCE / 'production-stage/execution.json'),
        event_continuity_scope='Continue the successful attempt-012 production monitor through a fresh resource stage.',
        schedule_recovery_cpu_qualification=binding(schedule_cpu / 'result.json'),
        canonical_cpu_control_model_state_sha256=read(PHASE / 'grouped-vocal-canonical-cpu-001/plan.json')['fixture_model_state_sha256'],
        canonical_cpu_control_scope='Existing unchanged-objective and recovery algorithm qualification; fresh GPU qualification checks the selected new saved parent.',
        continuation_storage_forecast={'new_rolling_generation_bytes':600_000_000,'new_final_generation_bytes':600_000_000,
            'additional_logs_and_metadata_bytes':50_000_000, 'standing_diagnostic_reserve_bytes':250_000_000,
            'projected_total_including_standing_reserves':before['conservative_total_with_reservations']+1_250_000_000,
            'all_existing_checkpoints_and_optimizers_preserved':True})
    validate_recipe(plan)
    require(plan['config']['augmentation'] == source['config']['augmentation']
            and plan['objective_version'] == source['objective_version']
            and plan['accumulation_policy'] == source['accumulation_policy']
            and plan['grouped_vocal_loss'] == source['grouped_vocal_loss']
            and plan['supervision'] == source['supervision']
            and not plan['inference_architecture_changed'] and not plan['automatic_plugin_replacement'],
            'Continuation changed its qualified scientific or runtime contract')
    verify_inputs(plan)
    OUT.mkdir(); write(OUT / 'plan.json', plan)
    print(json.dumps({'status':'prepared','plan_sha256':sha(OUT / 'plan.json'),'parent_role':role,
        'schedule':schedule,'first_sample_index':prefix['first_sample_index'],'gpu_workload_started':False,
        'forecast_bytes':plan['continuation_storage_forecast']['projected_total_including_standing_reserves']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-review',type=Path,required=True)
    prepare(parser.parse_args().selection_review.resolve(strict=True))
