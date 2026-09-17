"""Freeze the complete B1 execution after both current CPU qualifications pass."""
import ast
import copy
import json
import os
from pathlib import Path
import sys

from research.direct.run_latency58_quality import ROOT,read,require,sha,write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT
from research.direct.latency58_four_second_monitor import require_monitor_qualification
from research.direct.train_latency58_four_second_single import validate_recipe,runtime_policy,budget_snapshot
from research.direct.run_latency58_four_second_single import require_cpu_evidence,require_failed_resource_closed,binding
from research.direct.run_latency58_four_second_single_quality import transport_proof

OUT=ARTIFACT_ROOT/'branch-four-second-015'
DECISION=ARTIFACT_ROOT/'single-microbatch-decision.json'


def main():
    require(Path.cwd()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==''
            and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
            'Require CPU-only preparation')
    target=OUT/'plan-single001.json';source_path=OUT/'plan-retry002.json'
    require(not target.exists() and not (OUT/'resource-stage-004').exists()
            and not (OUT/'resource-run-004').exists() and not (OUT/'production-run').exists(),
            'Preserve existing attempts')
    original=read(source_path);decision=read(DECISION)
    require(sha(DECISION)=='a978ab2716f5764956b632a9decbb699895aca93dc309eb3f2d06d8c2355e6c2','B1 decision changed')
    failed=read(OUT/'resource-root-execution-003.json');previous=read(OUT/'resource-stage-003/execution.json')
    monitor_path=Path(previous['monitor_result']);require_failed_resource_closed(previous,read(monitor_path))
    require(failed['actual_exit_code']==1 and failed['source_bindings_unchanged'] and failed['training_updates']==0
            and failed['all_owned_processes_closed'] and failed['plan_sha256']==sha(source_path)
            and failed['stage_execution_sha256']==sha(OUT/'resource-stage-003/execution.json')
            and failed['monitor_result_sha256']==sha(monitor_path),'Last capacity failure is incomplete')
    bindings={}
    def merge(incoming):
        for path,digest in incoming.items():
            require(path not in bindings or bindings[path]==digest,'Conflicting frozen input: '+path)
            bindings[path]=digest
    for prepared in (original,decision):verify_inputs(prepared);merge(prepared['source_bindings'])
    merge(require_monitor_qualification())
    paths=[Path(__file__).resolve(),source_path,DECISION,monitor_path,OUT/'resource-root-execution-003.json',
        OUT/'resource-stage-003/execution.json',OUT/'resource-root-command-003.json',
        ARTIFACT_ROOT/'evaluation-adapter-source-proof-single.json']
    qualifications=((ARTIFACT_ROOT/'single-model-cpu-001','four_second_model_cpu_qualification'),
                    (ARTIFACT_ROOT/'single-restart-cpu-001','four_second_restart_cpu_qualification'))
    for directory,key in qualifications:
        prepared=read(directory/'plan.json');verify_inputs(prepared);merge(prepared['source_bindings'])
        paths.extend(directory/name for name in ('plan.json','result.json','root-execution.json','root-command.json','qualification-execution.json'))
    paths.extend(ROOT/'research/direct'/name for name in (
        'train_latency58_four_second_single.py','run_latency58_four_second_single.py',
        'check_latency58_four_second_single_device.py','latency58_four_second_device_recovery.py',
        'latency58_four_second_single_evaluation.py','run_latency58_four_second_single_quality.py',
        'evaluate_latency58_four_second_single_memory.py','evaluate_latency58_four_second_single_vocal_views.py',
        'check_latency58_four_second_single_vocal_views.py'))
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if isinstance(value,str):
            path=Path(value)
            if path.is_file() and path.suffix=='.py' and path.resolve().is_relative_to(ROOT):paths.append(path.resolve())
    merge({str(path):sha(path) for path in paths})
    loop_paths=[ROOT/'research/direct'/name for name in ('train_latency58_weighted_vocal.py','train_latency58_four_second_single.py')]
    loops=[]
    for path in loop_paths:
        nodes=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.For) and ast.unparse(n.target)=='(mixture_cpu, truth_cpu)']
        require(len(nodes)==1,'Ambiguous full training loop');loops.append(ast.dump(nodes[0],include_attributes=False))
    require(loops[0]==loops[1],'Scientific training loop changed')
    require(transport_proof()==read(ARTIFACT_ROOT/'evaluation-adapter-source-proof-single.json'),'Scientific scoring adapters changed')
    plan=copy.deepcopy(original)
    plan.update(source_bindings=bindings,scientific_decision=binding(DECISION),config=decision['proposed_training_config'],
        accumulation_steps=16,runtime_allowance=runtime_policy(),
        previous_execution_for_resource=str(OUT/'resource-stage-003/execution.json'),
        resource_stage_directory=str(OUT/'resource-stage-004'),resource_run_directory=str(OUT/'resource-run-004'),
        resource_root_execution=str(OUT/'resource-root-execution-004.json'),
        controller_module='research.direct.run_latency58_four_second_single',
        trainer_module='research.direct.train_latency58_four_second_single',
        quality_controller_module='research.direct.run_latency58_four_second_single_quality',
        operational_retry_of=binding(source_path),
        operational_retry_reason='Use B1 renders within unchanged B16 and two-view complete-group losses after measured B2 capacity failures.',
        evaluation_adapter_source_proof=binding(ARTIFACT_ROOT/'evaluation-adapter-source-proof-single.json'),
        scientific_training_loop_ast_proof={'status':'pass','reference':binding(loop_paths[0]),'candidate':binding(loop_paths[1]),
            'complete_training_loop_ast_identical':True},
        qualified_accumulation_geometry={'ordinary_examples':16,'ordinary_microbatch':1,'ordinary_microbatches':16,
            'auxiliary_examples':2,'auxiliary_microbatch':1,'auxiliary_microbatches':2,
            'full_group_output_derivatives':True,'optimizer_and_ema_updates_per_complete_group_pair':1},
        progress_allowance_scope='Prospective 150-second update plus 60-second complete save inside 240 seconds; measure two real updates and a disk save before production. Health sampling/thresholds and .75 memory fraction unchanged.',
        historical_cpu_qualification_scope='Original B2 evidence remains historical; current B1 full-context gradients and restart evidence qualify this execution.')
    for directory,key in qualifications:plan[key]=binding(directory/'result.json')
    plan['allocator_retry']={**original['allocator_retry'],'scored_context_or_logical_batch_changed':False,
        'activation_microbatch_changed_to_one':True,'complete_scientific_update_loop_ast_identical':True}
    plan['allocator_retry'].pop('scored_context_or_batch_changed',None)
    require({k:v for k,v in plan['config'].items() if k not in ('microbatch_size','auxiliary_microbatch_size')}
            == {k:v for k,v in original['config'].items() if k not in ('microbatch_size','auxiliary_microbatch_size')},
            'Changed a scientific schedule, data or context field')
    validate_recipe(plan);require_cpu_evidence(plan);verify_inputs(plan)
    budget=budget_snapshot(plan);plan['budget_before']=budget
    plan['storage_budget']={**original['storage_budget'],'complete_forecast':budget}
    write(target,plan)
    write(OUT/'preparation-single001-result.json',{'status':'prepared','plan_sha256':sha(target),
        'source_bindings_unchanged':True,'verified_input_count':len(bindings),'all_current_cpu_qualifications_pass':True,
        'complete_training_loop_ast_identical':True,'scientific_score_functions_unchanged':True,
        'budget_after':budget_snapshot(plan),'gpu_workload_started':False,'quality_measured':False})
    print(json.dumps({'status':'prepared','plan_sha256':sha(target),'verified_input_count':len(bindings),
        'combined_peak_bytes':budget['combined_peak_bytes']}),flush=True)


if __name__=='__main__':main()
