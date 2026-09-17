"""Freeze selected B16/B2 training after completed CPU and CUDA qualification."""
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
from research.direct.train_latency58_four_second_shared import validate_recipe,runtime_policy,budget_snapshot
from research.direct.latency58_four_second_shared_qualification import require_cpu_evidence,require_gpu_evidence,binding
from research.direct.run_latency58_four_second_shared_quality import transport_proof

OUT=ARTIFACT_ROOT/'branch-four-second-015'
DECISION=ARTIFACT_ROOT/'shared-microbatch-decision.json'


def main():
    require(Path.cwd()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
        'Require CPU-only training preparation')
    target=OUT/'plan-shared001.json';source_path=OUT/'plan-single001.json'
    require(not target.exists() and not (OUT/'resource-stage-005').exists()
        and not (OUT/'resource-run-005').exists() and not (OUT/'production-run').exists(),
        'Preserve prior and current training attempts')
    original=read(source_path);decision=read(DECISION)
    require(sha(DECISION)=='4f4f32d9ad3c97ba208a72e78504df664d3ee2271f5f043a1dd690ccce4c0140',
        'Selected shared-storage execution changed')
    failed_path=OUT/'resource-root-execution-004.json';failed=read(failed_path)
    require(failed['actual_exit_code']==1 and failed['source_bindings_unchanged'] and failed['training_updates']==0
        and failed['all_owned_processes_closed'] and failed['plan_sha256']==sha(source_path),
        'Preserve complete B1 capacity-failure evidence')
    bindings={}
    def merge(incoming):
        for path,digest in incoming.items():
            require(path not in bindings or bindings[path]==digest,'Conflicting frozen input: '+path)
            bindings[path]=digest
    for source in (original,decision):verify_inputs(source);merge(source['source_bindings'])
    merge(require_monitor_qualification())
    paths=[Path(__file__).resolve(),source_path,DECISION,failed_path,
        ARTIFACT_ROOT/'evaluation-adapter-source-proof-shared.json',ARTIFACT_ROOT/'geometry-adapter-source-proof.json']
    qualifications=((ARTIFACT_ROOT/'geometry-restart-cpu-001','selected_geometry_restart_cpu_qualification'),
                    (ARTIFACT_ROOT/'geometry-gpu-001','selected_geometry_gpu_qualification'))
    for directory,key in qualifications:
        prepared=read(directory/'plan.json');verify_inputs(prepared);merge(prepared['source_bindings'])
        paths.extend(directory/name for name in ('plan.json','result.json','root-execution.json','root-command.json'))
    paths.extend(ROOT/'research/direct'/name for name in (
        'train_latency58_four_second_shared.py','run_latency58_four_second_shared.py',
        'latency58_four_second_shared_qualification.py','latency58_bf16_saved_gru_weights.py',
        'latency58_four_second_device_recovery.py','latency58_four_second_shared_evaluation.py',
        'run_latency58_four_second_shared_quality.py','evaluate_latency58_four_second_shared_memory.py',
        'evaluate_latency58_four_second_shared_vocal_views.py','check_latency58_four_second_shared_vocal_views.py'))
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if isinstance(value,str):
            path=Path(value)
            if path.is_file() and path.suffix=='.py' and path.resolve().is_relative_to(ROOT):paths.append(path.resolve())
    merge({str(path):sha(path) for path in paths})
    loop_paths=[ROOT/'research/direct'/name for name in ('train_latency58_weighted_vocal.py','train_latency58_four_second_shared.py')]
    loops=[]
    for path in loop_paths:
        nodes=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.For)
            and ast.unparse(n.target)=='(mixture_cpu, truth_cpu)']
        require(len(nodes)==1,'Ambiguous complete training loop');loops.append(ast.dump(nodes[0],include_attributes=False))
    require(loops[0]==loops[1],'Scientific training loop changed')
    require(transport_proof()==read(ARTIFACT_ROOT/'evaluation-adapter-source-proof-shared.json'),
        'Scientific scoring adapters changed')
    plan=copy.deepcopy(original)
    plan.update(source_bindings=bindings,scientific_decision=binding(DECISION),config=decision['proposed_training_config'],
        accumulation_steps=1,runtime_allowance=runtime_policy(),
        previous_execution_for_resource=str(ARTIFACT_ROOT/'geometry-gpu-001/root-execution.json'),
        resource_stage_directory=str(OUT/'resource-stage-005'),resource_run_directory=str(OUT/'resource-run-005'),
        resource_root_execution=str(OUT/'resource-root-execution-005.json'),
        controller_module='research.direct.run_latency58_four_second_shared',
        trainer_module='research.direct.train_latency58_four_second_shared',
        quality_controller_module='research.direct.run_latency58_four_second_shared_quality',
        operational_retry_of=binding(source_path),
        operational_retry_reason='Exact autograd saved-weight sharing removes measured per-frame BF16 weight duplication; selected B16/B2 execution improves measured throughput without changing the complete objective.',
        evaluation_adapter_source_proof=binding(ARTIFACT_ROOT/'evaluation-adapter-source-proof-shared.json'),
        saved_weight_storage=binding(ROOT/'research/direct/latency58_bf16_saved_gru_weights.py'),
        scientific_training_loop_ast_proof={'status':'pass','reference':binding(loop_paths[0]),
            'candidate':binding(loop_paths[1]),'complete_training_loop_ast_identical':True},
        qualified_accumulation_geometry=decision['selected_execution_geometry'],
        progress_allowance_scope='Prospective 100-second complete update and 60-second disk save inside 240 seconds; actual recorded-data rehearsal and disk-save measurement required before production. Health limits unchanged.',
        historical_cpu_qualification_scope='Existing B1/B2 full-context parent and codec references remain bound; selected B16/B2 CPU weighted disk restart and CUDA complete-group gradients/weighted restart add current geometry coverage.',
        reused_cuda_qualification_scope='The completed standalone B16/B2 proof is bound and the actual resource parent is authenticated against it; the resource stage adds selected-device packed recovery, EMA arithmetic and two actual recorded-data updates.')
    for directory,key in qualifications:plan[key]=binding(directory/'result.json')
    plan['allocator_retry']={**original['allocator_retry'],'activation_microbatch_changed_to_one':False,
        'ordinary_microbatch':16,'auxiliary_microbatch':2,'exact_saved_value_sharing':True,
        'complete_scientific_update_loop_ast_identical':True}
    require({k:v for k,v in plan['config'].items() if k not in ('microbatch_size','auxiliary_microbatch_size')}
        =={k:v for k,v in original['config'].items() if k not in ('microbatch_size','auxiliary_microbatch_size')},
        'Changed a scientific schedule, data or context field')
    validate_recipe(plan);require_cpu_evidence(plan);require_gpu_evidence(plan);verify_inputs(plan)
    budget=budget_snapshot(plan);runtime=runtime_policy()
    maximum_seconds=2000*runtime['production_seconds_per_update']+40*runtime['save_seconds_per_generation']+runtime['production_fixed_allowance_seconds']
    measured_monitor=ARTIFACT_ROOT/'monitors/shared-four-second-gpu-001/result.json'
    monitor=read(measured_monitor)
    observed_log_rate=monitor['artifacts']['watchdog_log']['bytes']/monitor['wall_seconds']
    require(observed_log_rate<5000,'Prepared production log-rate allowance is below observed full qualification')
    diagnostic_forecast=budget['new_root_actual_bytes']+maximum_seconds*5000+125_000_000+20_000_000
    require(diagnostic_forecast<=1_200_000_000,'Maximum-duration monitor and review forecast exceeds reserved diagnostics')
    plan['monitor_and_quality_forecast']={'maximum_production_seconds':maximum_seconds,
        'observed_complete_qualification_log_bytes_per_second':observed_log_rate,
        'prospective_log_bytes_per_second':5000,'additional_quality_and_metadata_bytes':125_000_000,
        'remaining_preproduction_diagnostic_reserve_bytes':20_000_000,
        'current_root_bytes':budget['new_root_actual_bytes'],'complete_diagnostic_forecast_bytes':diagnostic_forecast,
        'reserved_diagnostic_bytes':1_200_000_000,'monitor_reference':binding(measured_monitor)}
    plan['budget_before']=budget;plan['storage_budget']={**original['storage_budget'],'complete_forecast':budget}
    write(target,plan)
    write(OUT/'preparation-shared001-result.json',{'status':'prepared','plan_sha256':sha(target),
        'source_bindings_unchanged':True,'verified_input_count':len(bindings),
        'selected_cpu_and_cuda_qualifications_pass':True,'complete_training_loop_ast_identical':True,
        'scientific_score_functions_unchanged':True,'budget_after':budget_snapshot(plan),
        'gpu_workload_started':False,'quality_measured':False})
    print(json.dumps({'status':'prepared','plan_sha256':sha(target),'verified_input_count':len(bindings),
        'combined_peak_bytes':budget['combined_peak_bytes'],'maximum_production_seconds':maximum_seconds,
        'diagnostic_forecast_bytes':diagnostic_forecast}),flush=True)


if __name__=='__main__':main()
