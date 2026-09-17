"""Authenticate completed CPU and CUDA proofs for selected B16/B2 execution."""
from pathlib import Path
from research.direct.run_latency58_quality import read,require,sha
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT
from research.direct.latency58_four_second_monitor import require_monitor_closed
from research.direct.run_latency58_four_second_single import require_cpu_evidence as require_reference_cpu_evidence


def binding(path):
    return {'path':str(path),'sha256':sha(path)}


def completed(root,plan,key):
    prepared,result,enclosing=(read(root/name) for name in ('plan.json','result.json','root-execution.json'))
    require(plan[key]==binding(root/'result.json') and result['status']=='pass'
        and result['source_bindings_unchanged'] and enclosing['source_bindings_unchanged']
        and type(enclosing['actual_exit_code']) is int and enclosing['actual_exit_code']==0
        and enclosing['timed_out'] is False
        and enclosing['result_sha256']==sha(root/'result.json')
        and enclosing['plan_sha256']==result['plan_sha256']==sha(root/'plan.json'),
        'Incomplete selected execution qualification')
    verify_inputs(prepared)
    return prepared,result,enclosing


def require_cpu_evidence(plan):
    # These frozen full-context reference and codec proofs remain relevant to
    # the same parent, complete objective, context and selected data prefix.
    require_reference_cpu_evidence(plan)
    prepared,result,enclosing=completed(ARTIFACT_ROOT/'geometry-restart-cpu-001',plan,
        'selected_geometry_restart_cpu_qualification')
    r=result['restart']
    require(prepared['scientific_decision']==plan['scientific_decision']
        and r['ordinary_microbatch']==plan['config']['microbatch_size']==16
        and r['auxiliary_microbatch']==plan['config']['auxiliary_microbatch_size']==2
        and r['logical_batch_size']==16 and r['planned_stop_step']==2000
        and r['parent_model_state_sha256']==plan['parent_model_state_sha256']
        and not result['gpu_used'] and result['global_rng_restored']
        and r['tensor_count']==216 and r['all_tensor_and_metadata_bytes_exact']
        and r['third_weighted_selected_update_and_accounting_bit_exact']
        and r['all_40_adam_states_bit_exact'] and r['all_cpu_rng_streams_replayed']
        and r['interrupted_publication_retains_loadable_previous_generation']
        and r['production_budget_preflight_qualified'] and r['finalization_reuses_inode'] and r['parent_unchanged']
        and all(v['algorithmic_latency_samples']==256 and v['all_six_outputs_and_eight_states_bit_exact']
                for v in r['final_raw_and_ema_parity'].values()),'Selected CPU update and packed recovery differs')


def require_gpu_evidence(plan):
    root=ARTIFACT_ROOT/'geometry-gpu-001'
    prepared,result,enclosing=completed(root,plan,'selected_geometry_gpu_qualification')
    monitor_path=Path(enclosing['monitor_result'])
    require(enclosing['monitor_result_sha256']==sha(monitor_path) and enclosing['all_owned_processes_closed'],
        'Selected CUDA proof lacks completed telemetry')
    require_monitor_closed(enclosing,read(monitor_path),final_step=12)
    gradients,restart=result['whole_group_gradients'],result['weighted_restart']
    require(prepared['scientific_decision']==result['scientific_decision']==plan['scientific_decision']
        and result['ordinary_microbatch']==gradients['ordinary_microbatch']==restart['ordinary_microbatch']==16
        and result['auxiliary_microbatch']==gradients['auxiliary_microbatch']==restart['auxiliary_microbatch']==2
        and result['logical_batch_size']==16 and result['parent_weights_unchanged'] and result['rng_unchanged']
        and result['training_optimizer_updates']==0
        and result['parent_model_state_sha256']==gradients['model_state_sha256']==restart['parent_model_state_sha256']
            ==plan['parent_model_state_sha256']
        and gradients['status']=='pass' and gradients['precision']=='bf16' and gradients['device'].startswith('cuda')
        and gradients['warmup_samples']==88064 and gradients['scored_samples']==176512
        and gradients['absolute_gradient_tolerance']==1e-7 and gradients['relative_gradient_tolerance']==1e-4
        and gradients['relative_l2_tolerance']==5e-5 and gradients['loss_absolute_tolerance']==3e-6
        and len(gradients['all_40_gradients'])==40
        and all(v['relative_l2_error']<5e-5 and v['reference_norm']>0 for v in gradients['all_40_gradients'].values())
        and gradients['ordinary_all_40_gradients_bit_exact_against_unmodified_reference']
        and gradients['canonical_replay_outputs_bit_exact']
        and restart['status']=='pass' and restart['precision']=='bf16' and restart['device'].startswith('cuda')
        and restart['all_40_adam_states_checked'] and restart['third_update_raw_adam_ema_and_accounting_bit_exact']
        and restart['noncontiguous_step_rejected_before_gradients'] and restart['parent_weights_unchanged']
        and len(restart['interrupted_accumulations'])==2
        and all(v['weights_adam_ema_unchanged'] for v in restart['interrupted_accumulations']),
        'Selected complete CUDA gradients or weighted restart differs')
    auxiliary=result['auxiliary_context']
    require(auxiliary['views']==['instrumental','vocals_only'] and auxiliary['microbatch_size']==2
        and auxiliary['status']=='pass' and auxiliary['scored_samples']==176512
        and len(auxiliary['all_40_gradients'])==40
        and all(v['maximum_error']==0 and v['reference_norm']>0 for v in auxiliary['all_40_gradients'].values()),
        'Selected auxiliary CUDA context differs')
    return {'schema':'latency58-shared-qualified-geometry-reference-v1','status':'pass',
        'qualification':binding(root/'result.json'),'actual_root_execution':binding(root/'root-execution.json'),
        'parent_model_state_sha256':result['parent_model_state_sha256'],
        'ordinary_microbatch':16,'auxiliary_microbatch':2,'training_optimizer_updates':0}
