"""Launch batch profiling only after actual full-model GPU qualification closes."""
from datetime import datetime, timezone
import os
from pathlib import Path

from research.direct.run_latency58_quality import ROOT, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT, snapshot
from research.direct.latency58_four_second_monitor import WATCHDOG, WATCHDOG_SHA, require_monitor_qualification, require_monitor_closed


def main():
    require(Path.cwd()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
        'Require a CPU-only enclosing preparation')
    previous_root=ARTIFACT_ROOT/'shared-four-second-gpu-001'
    previous=read(previous_root/'root-execution.json');monitor_path=Path(previous['monitor_result']);monitor=read(monitor_path)
    require_monitor_closed(previous,monitor,final_step=76)
    result=read(previous_root/'result.json');source=read(previous_root/'plan.json')
    require(previous['actual_session_id']==83436 and previous['all_owned_processes_closed']
        and previous['result_sha256']==sha(previous_root/'result.json')
        and previous['plan_sha256']==result['plan_sha256']==sha(previous_root/'plan.json')
        and previous['monitor_result_sha256']==sha(monitor_path)
        and result['status']=='pass' and result['source_bindings_unchanged']
        and result['parent_weights_unchanged'] and result['rng_unchanged']
        and result['training_optimizer_updates']==0,'Incomplete actual full-model closure')
    require(result['short_original_vs_shared']['all_40_parameter_gradients_and_outputs_bit_exact']
        and result['short_original_vs_shared']['modified_saved_value_rejected']
        and len(result['full_contexts'])==3
        and all(len(c['all_40_gradients'])==40 and c['scored_samples']==176512
                and all(v['maximum_error']==0 and v['reference_norm']>0 for v in c['all_40_gradients'].values())
                for c in result['full_contexts'])
        and result['whole_group_gradients']['ordinary_all_40_gradients_bit_exact_against_unmodified_reference']
        and len(result['whole_group_gradients']['all_40_gradients'])==40
        and all(v['relative_l2_error']<5e-5 and v['reference_norm']>0
                for v in result['whole_group_gradients']['all_40_gradients'].values()),'Full-model gradient proof differs')
    verify_inputs(source)
    out=ARTIFACT_ROOT/'shared-batch-profile-001';require(not out.exists(),'Preserve earlier batch-profile evidence')
    bindings=dict(source['source_bindings']);bindings.update(require_monitor_qualification())
    paths=[Path(__file__).resolve(),ROOT/'research/direct/profile_latency58_four_second_shared_batches.py',
        monitor_path,*[previous_root/name for name in ('plan.json','result.json','root-execution.json','root-command.json')]]
    for path in paths:
        digest=sha(path)
        require(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting frozen input')
        bindings[str(path)]=digest
    plan={'schema':'latency58-four-second-shared-batch-profile-v1',
        'parent_checkpoint':source['parent_checkpoint'],'parent_model_state_sha256':source['parent_model_state_sha256'],
        'environment':source['environment'],'source_bindings':bindings,
        'watchdog_source':str(WATCHDOG),'watchdog_sha256':WATCHDOG_SHA,
        'previous_event_record_id':monitor['last_event_record_id'],
        'previous_monitor':{'path':str(monitor_path),'sha256':sha(monitor_path)},
        'microbatches':[2,4,8,16],'warmup_samples':88064,'scored_samples':176512,
        'expected_progress_stages':4,'maximum_seconds':1200,'training_optimizer_updates':0,
        'quality_claimed':False,'production_qualified':False,'storage_before':snapshot(),
        'created_utc':datetime.now(timezone.utc).isoformat()}
    verify_inputs(plan);out.mkdir();write(out/'plan.json',plan)
    child=[PYTHON,'-u','-m','research.direct.profile_latency58_four_second_shared_batches',
        '--plan',str(out/'plan.json'),'--plan-sha256',sha(out/'plan.json')]
    spec={'schema':'gpu-watchdog-launch-finalization-v1','expected_final_step':4,'cwd':str(ROOT),
        'environment':plan['environment'],'progress_path':str(out/'metrics.jsonl'),'argv':child}
    write(out/'watchdog-spec.json',spec)
    monitor_out=ARTIFACT_ROOT/'monitors/shared-batch-profile-001'
    argv=[PYTHON,str(WATCHDOG),'--launch-spec',str(out/'watchdog-spec.json'),
        '--launch-spec-sha256',sha(out/'watchdog-spec.json'),'--output-dir',str(monitor_out),
        '--max-runtime-seconds','1200','--poll-seconds','2','--query-timeout-seconds','10',
        '--startup-grace-seconds','240','--progress-timeout-seconds','180','--finalization-timeout-seconds','180',
        '--stop-grace-seconds','15','--post-exit-quiet-seconds','10','--max-temperature-c','80','--memory-headroom-mib','4096']
    write(out/'root-command.json',{'argv':argv,'cwd':str(ROOT),
        'environment':{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},'plan_sha256':sha(out/'plan.json'),
        'source_bindings':{str(p):sha(p) for p in (out/'plan.json',out/'watchdog-spec.json')},
        'monitor_directory':str(monitor_out),'storage_before':snapshot()})
    print({'event':'shared_batch_profile_prepared','bound_inputs':len(bindings),
        'plan_sha256':sha(out/'plan.json')},flush=True)
    os.execvpe(PYTHON,argv,os.environ.copy())


if __name__=='__main__':main()
