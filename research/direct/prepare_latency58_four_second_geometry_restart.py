"""Prepare CPU update/restart proof for the measured shared-storage geometry."""
import copy
import json
import os
from pathlib import Path
import sys

from research.direct.run_latency58_quality import ROOT,PYTHON,read,require,sha,write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT,snapshot
from research.direct.check_latency58_four_second_geometry_restart import run_fixture

OUT=ARTIFACT_ROOT/'geometry-restart-cpu-001'
PREVIOUS=ARTIFACT_ROOT/'single-restart-cpu-001'
PROFILE=ARTIFACT_ROOT/'shared-batch-profile-001'
DECISION=ARTIFACT_ROOT/'shared-microbatch-decision.json'


def main():
    require(Path.cwd()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
        'Require CPU-only preparation')
    require(not OUT.exists(),'Preserve selected-geometry restart evidence')
    previous=read(PREVIOUS/'plan.json');decision=read(DECISION)
    profile=read(PROFILE/'result.json');enclosing=read(PROFILE/'root-execution.json')
    require(profile['status']=='pass' and profile['source_bindings_unchanged']
        and enclosing['actual_exit_code']==0 and enclosing['all_owned_processes_closed']
        and enclosing['source_bindings_unchanged'] and enclosing['result_sha256']==sha(PROFILE/'result.json')
        and enclosing['plan_sha256']==profile['plan_sha256']==sha(PROFILE/'plan.json'),
        'Observe completed batch profiling before selected restart proof')
    ordinary=decision['proposed_training_config']['microbatch_size']
    auxiliary=decision['proposed_training_config']['auxiliary_microbatch_size']
    require(ordinary in (2,4,8,16) and auxiliary==2 and decision['logical_batch_size']==16
        and decision['ordinary_microbatch_size']==ordinary and decision['auxiliary_microbatch_size']==auxiliary
        and decision['parent_checkpoint']==previous['fixture_checkpoint']
        and decision['parent_model_state_sha256']==previous['fixture_model_state_sha256']
        and not decision['production_qualified'] and not decision['production_started'],
        'Selected geometry, parent or qualification state differs')
    selected=next(row for row in profile['microbatch_cases'] if row['microbatch']==ordinary)
    require(selected['status']=='pass' and selected['short']['all_40_parameter_gradients_and_outputs_bit_exact']
        and len(selected['full_context']['all_40_gradients'])==40
        and all(v['maximum_error']==0 and v['reference_norm']>0
                for v in selected['full_context']['all_40_gradients'].values()),'Selected GPU geometry did not pass')
    bindings={}
    def merge(incoming):
        for path,digest in incoming.items():
            require(path not in bindings or bindings[path]==digest,'Conflicting input: '+path)
            bindings[path]=digest
    for source in (previous,decision,read(PROFILE/'plan.json')):
        verify_inputs(source);merge(source['source_bindings'])
    paths=[Path(__file__).resolve(),DECISION,ROOT/'research/direct/check_latency58_four_second_geometry_restart.py',
        ROOT/'research/direct/latency58_four_second_geometry_restart.py',ARTIFACT_ROOT/'geometry-adapter-source-proof.json',
        *[PREVIOUS/name for name in ('plan.json','result.json','root-execution.json')],
        *[PROFILE/name for name in ('plan.json','result.json','root-execution.json','root-command.json')]]
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if isinstance(value,str):
            path=Path(value)
            if path.is_file() and path.suffix=='.py' and path.resolve().is_relative_to(ROOT):paths.append(path.resolve())
    merge({str(path):sha(path) for path in paths})
    plan=copy.deepcopy(previous)
    plan.update(schema='latency58-four-second-selected-geometry-packed-cpu-qualification-v1',source_bindings=bindings,
        scientific_decision={'path':str(DECISION),'sha256':sha(DECISION)},ordinary_microbatch=ordinary,
        auxiliary_microbatch=auxiliary,output_directory=str(OUT),budget_before=snapshot(),
        full_gpu_batch_profile={'path':str(PROFILE/'result.json'),'sha256':sha(PROFILE/'result.json')})
    require(plan['budget_before']['new_root_unused_reservation_bytes']>plan['maximum_new_artifact_bytes'],
        'Reserve the entire ephemeral disk fixture')
    OUT.mkdir();(OUT/'supervise.py').write_bytes((PREVIOUS/'supervise.py').read_bytes())
    bindings[str(OUT/'supervise.py')]=sha(OUT/'supervise.py')
    verify_inputs(plan);write(OUT/'plan.json',plan)
    write(OUT/'command.json',{'argv':[PYTHON,'-u','-m','research.direct.check_latency58_four_second_geometry_restart',
        '--plan',str(OUT/'plan.json'),'--plan-sha256',sha(OUT/'plan.json')],'cwd':str(ROOT),
        'environment':{**{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},'PYTHONPATH':str(ROOT)},
        'plan_sha256':sha(OUT/'plan.json'),'timeout_seconds':plan['timeout_seconds']})
    write(OUT/'preparation-result.json',{'status':'prepared','plan_sha256':sha(OUT/'plan.json'),
        'source_bindings_unchanged':True,'verified_input_count':len(bindings),'gpu_used':False,
        'ordinary_microbatch':ordinary,'auxiliary_microbatch':auxiliary,'logical_batch_size':16,'training_steps':2000})
    print(json.dumps({'status':'prepared','plan_sha256':sha(OUT/'plan.json'),'verified_input_count':len(bindings)}),flush=True)


if __name__=='__main__':main()
