"""Check real disk recovery for the selected parent and proposed full schedule."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
from research.direct.run_latency58_quality import ROOT,PHASE,read,require,sha,write
from research.direct.train_latency58 import verify_inputs,state_sha256
from research.direct.run_latency58_deployed_vocal_views import budget_snapshot
from research.direct.run_latency58_paired_vocal_serial import binding,merge_bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-review',type=Path,required=True)
    args=parser.parse_args()
    require(Path.cwd()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==''
            and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
            'Require CUDA-hidden CPU1')
    selection_path=args.selection_review.resolve(strict=True);selection=read(selection_path)
    source_path=PHASE/'branch-grouped-vocal-012/plan.json';source=read(source_path)
    paired_root=PHASE/'paired-vocal-grouped-012';paired=read(paired_root/'result.json');executed=read(paired_root/'root-execution.json')
    review_root=PHASE/'grouped-vocal-parent-review-012';review=read(review_root/'result.json');review_execution=read(review_root/'execution.json')
    require(paired['status']==review['status']=='pass' and executed['actual_exit_code']==review_execution['actual_exit_code']==0
            and executed['source_bindings_unchanged'] and review_execution['source_bindings_unchanged']
            and executed['result_sha256']==sha(paired_root/'result.json')
            and review_execution['result_sha256']==sha(review_root/'result.json')
            and selection['schema']=='latency58-grouped-continuation-selection-v1'
            and selection['status']=='selected_for_research_continuation'
            and selection['complete_parent_and_candidate_views_reviewed']
            and selection['parent_comparison']==binding(review_root/'result.json')
            and selection['parent_comparison_execution']==binding(review_root/'execution.json')
            and selection['paired_source_views']==binding(paired_root/'result.json')
            and not selection['plugin_replaced'] and not selection['overall_goal_complete'],
            'Complete the paired review and selection before qualification')
    role=selection['selected_role'];model_identity=paired['models'][role]
    require(selection['selected_checkpoint']==model_identity['checkpoint']
            and selection['selected_model_state_sha256']==model_identity['model_state_sha256']
            and selection['schedule']['steps'] in (500,1000), 'Selected model or schedule differs')
    out=PHASE/'grouped-continuation-recovery-cpu-013';require(not out.exists(),'Preserve earlier CPU qualification')
    bindings=dict(source['source_bindings'])
    paths=[Path(__file__).resolve(),source_path,selection_path,Path(model_identity['checkpoint']['path'])]
    paths.extend(ROOT/'research/direct'/name for name in ('latency58_grouped_continuation_recovery_check.py',
        'latency58_grouped_vocal_recovery.py','latency58_branch_ema_checkpoint.py','latency58_branch_memory_checkpoint.py',
        'latency58_branch_ema.py','check_latency58_grouped_vocal_restart.py','run_latency58_deployed_vocal_views.py'))
    paths.extend(paired_root/name for name in ('plan.json','result.json','root-execution.json'))
    paths.extend(review_root/name for name in ('plan.json','result.json','execution.json'))
    merge_bindings(bindings,{str(path):sha(path) for path in paths})
    verify_inputs({'source_bindings':bindings})
    before=budget_snapshot(source['storage_budget']);require(before['headroom_bytes']>1_200_000_000,'Reserve complete temporary recovery files')
    import torch,numpy as np,random
    from research.direct.latency58_branch_memory_checkpoint import load_model
    from research.direct.latency58_grouped_continuation_recovery_check import exercise_recovery
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(202610312);random.seed(202610312);np.random.seed(202610312)
    parent,_=load_model(model_identity['checkpoint']);parent.training_precision='fp32'
    require(state_sha256(parent.state_dict())==model_identity['model_state_sha256']
            and state_sha256(dict(parent.named_buffers()))==source['fixed_buffers_sha256']
            and not torch.cuda.is_initialized(), 'CPU selected parent differs')
    proposed=copy.deepcopy(source)
    proposed.update(parent_checkpoint=model_identity['checkpoint'],parent_model_state_sha256=model_identity['model_state_sha256'],
        initialized_model_state_sha256=model_identity['model_state_sha256'],parent_training_updates=parent.provenance['training_updates'],
        config={**source['config'],**selection['schedule']})
    proposed.pop('resume_checkpoint',None)
    plan={'schema':'latency58-grouped-continuation-recovery-cpu-v1','source_bindings':bindings,
        'parent_checkpoint':model_identity['checkpoint'],'parent_model_state_sha256':model_identity['model_state_sha256'],
        'planned_schedule':selection['schedule'],'selected_role':role,'budget_before':before,'gpu_used':False}
    out.mkdir();write(out/'plan.json',plan)
    result=exercise_recovery(parent,proposed,out)
    require(result['training_schedule_steps']==selection['schedule']['steps'] and result['device']=='cpu'
            and not torch.cuda.is_initialized(),'Recovery changed the full schedule or initialized CUDA')
    verify_inputs(plan)
    result.update(plan_sha256=sha(out/'plan.json'),parent_checkpoint=model_identity['checkpoint'],
        source_bindings_unchanged=True,gpu_used=False,budget_after=budget_snapshot(source['storage_budget']))
    write(out/'result.json',result)
    print(json.dumps({key:result[key] for key in ('status','device','training_schedule_steps','complete_successful_save_seconds','elapsed_seconds','gpu_used')}))


if __name__=='__main__':
    main()
