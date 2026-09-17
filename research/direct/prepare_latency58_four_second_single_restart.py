"""Prepare selected B1 update and disk-restart checks after full-context CPU proof."""
import copy
import json
import os
from pathlib import Path
import sys

from research.direct.run_latency58_quality import ROOT,PYTHON,read,require,sha,write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT,snapshot
from research.direct.check_latency58_four_second_single_restart import exercise

OUT=ARTIFACT_ROOT/'single-restart-cpu-001'
PREVIOUS=ARTIFACT_ROOT/'restart-cpu-001'
MODEL=ARTIFACT_ROOT/'single-model-cpu-001'
DECISION=ARTIFACT_ROOT/'single-microbatch-decision.json'


def main():
    require(Path.cwd()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==''
            and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
            'Require CPU-only preparation')
    require(not OUT.exists(),'Preserve restart evidence')
    previous=read(PREVIOUS/'plan.json'); model_plan=read(MODEL/'plan.json')
    model=read(MODEL/'result.json'); completed=read(MODEL/'root-execution.json')
    gradients=model['whole_group_neural_gradient_comparison']
    require(model['status']=='pass' and model['source_bindings_unchanged']
            and type(completed['actual_exit_code']) is int and completed['actual_exit_code']==0
            and completed['timed_out'] is False and completed['source_bindings_unchanged']
            and completed['plan_sha256']==model['plan_sha256']==sha(MODEL/'plan.json')
            and completed['result_sha256']==sha(MODEL/'result.json')
            and model['ordinary_microbatch']==model['auxiliary_microbatch']==1
            and model['logical_batch_size']==16 and model['scored_samples']==176512
            and len(model['auxiliary_context_comparisons'])==2
            and len(gradients['all_40_gradients'])==40
            and all(row['bitwise_equal'] for row in gradients['all_40_gradients'].values()),
            'Complete B1 model qualification must pass before the restart check')
    decision=read(DECISION)
    require(sha(DECISION)=='a978ab2716f5764956b632a9decbb699895aca93dc309eb3f2d06d8c2355e6c2'
            and decision['proposed_training_config']['microbatch_size']==decision['proposed_training_config']['auxiliary_microbatch_size']==1,
            'Selected B1 revision changed')
    bindings={}
    def merge(incoming):
        for path,digest in incoming.items():
            require(path not in bindings or bindings[path]==digest,'Conflicting frozen input: '+path)
            bindings[path]=digest
    for source in (previous,model_plan,decision):
        verify_inputs(source); merge(source['source_bindings'])
    paths=[Path(__file__).resolve(),DECISION,ROOT/'research/direct/check_latency58_four_second_single_restart.py',
           PREVIOUS/'plan.json',PREVIOUS/'result.json',PREVIOUS/'root-execution.json',
           *[MODEL/name for name in ('plan.json','result.json','root-execution.json','root-command.json','qualification-execution.json')]]
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if isinstance(value,str):
            path=Path(value)
            if path.is_file() and path.suffix=='.py' and path.resolve().is_relative_to(ROOT):paths.append(path.resolve())
    merge({str(path):sha(path) for path in paths})
    plan=copy.deepcopy(previous)
    plan.update(schema='latency58-four-second-b1-and-packed-cpu-qualification-v1',source_bindings=bindings,
        scientific_decision={'path':str(DECISION),'sha256':sha(DECISION)},ordinary_microbatch=1,auxiliary_microbatch=1,
        output_directory=str(OUT),budget_before=snapshot(),
        full_model_cpu_qualification={'path':str(MODEL/'result.json'),'sha256':sha(MODEL/'result.json')})
    require(plan['budget_before']['new_root_unused_reservation_bytes']>plan['maximum_new_artifact_bytes'],
            'Reserve the entire ephemeral disk fixture')
    OUT.mkdir(); (OUT/'supervise.py').write_bytes((PREVIOUS/'supervise.py').read_bytes())
    bindings[str(OUT/'supervise.py')]=sha(OUT/'supervise.py')
    verify_inputs(plan); write(OUT/'plan.json',plan)
    write(OUT/'command.json',{'argv':[PYTHON,'-u','-m','research.direct.check_latency58_four_second_single_restart',
        '--plan',str(OUT/'plan.json'),'--plan-sha256',sha(OUT/'plan.json')],'cwd':str(ROOT),
        'environment':{**{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},'PYTHONPATH':str(ROOT)},
        'plan_sha256':sha(OUT/'plan.json'),'timeout_seconds':plan['timeout_seconds']})
    write(OUT/'preparation-result.json',{'status':'prepared','plan_sha256':sha(OUT/'plan.json'),
        'source_bindings_unchanged':True,'verified_input_count':len(bindings),'gpu_used':False,
        'ordinary_microbatch':1,'auxiliary_microbatch':1,'logical_batch_size':16,'training_steps':2000})
    print(json.dumps({'status':'prepared','plan_sha256':sha(OUT/'plan.json'),'verified_input_count':len(bindings)}),flush=True)


if __name__=='__main__':main()
