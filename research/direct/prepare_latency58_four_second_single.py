"""Prepare full-context B1 CPU qualification after measured B2 GPU capacity failures."""
import copy
import json
import os
from pathlib import Path
import sys

from research.direct.run_latency58_quality import ROOT, PHASE, PYTHON, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT, snapshot
from research.direct.check_latency58_four_second_single_model import compare_group_gradients

OUT = ARTIFACT_ROOT / 'single-model-cpu-001'
PREVIOUS = PHASE / 'four-second-model-cpu-002'
TRAINING = ARTIFACT_ROOT / 'branch-four-second-015'


def main():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'Require CPU-only preparation')
    require(not OUT.exists(), 'Preserve prior qualifications')
    previous = read(PREVIOUS / 'plan.json'); verify_inputs(previous)
    failed_path = TRAINING / 'resource-root-execution-003.json'; failed = read(failed_path)
    require(failed['actual_session_id'] == 35373 and failed['actual_tool_chunk_id'] == '072243'
            and failed['actual_exit_code'] == 1 and failed['source_bindings_unchanged']
            and failed['training_updates'] == 0 and failed['all_owned_processes_closed']
            and failed['supervisor_health'] == 'pass' and failed['expandable_segments_observed'],
            'The measured B2 capacity failure is incomplete')
    bindings = dict(previous['source_bindings'])
    paths = [Path(__file__).resolve(), ROOT / 'research/direct/check_latency58_four_second_single_model.py',
        PREVIOUS / 'plan.json', PREVIOUS / 'result.json', PREVIOUS / 'root-execution.json',
        failed_path, TRAINING / 'resource-root-execution-002.json',
        TRAINING / 'resource-run-003/allocator-before-qualification.json',
        TRAINING / 'resource-run-003/allocator-qualification-oom.json', TRAINING / 'plan-retry002.json']
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if isinstance(value, str):
            path = Path(value)
            if path.is_file() and path.suffix == '.py' and path.resolve().is_relative_to(ROOT):
                paths.append(path.resolve())
    for path in paths:
        key, digest = str(path), sha(path)
        require(key not in bindings or bindings[key] == digest, 'Frozen source changed: ' + key)
        bindings[key] = digest
    budget = snapshot(); require(budget['new_root_unused_reservation_bytes'] > 5_000_000, 'Reserve complete CPU evidence')
    training = read(TRAINING / 'plan-retry002.json')
    config = {**training['config'], 'microbatch_size': 1, 'auxiliary_microbatch_size': 1}
    decision = {'schema': 'latency58-four-second-execution-microbatch-revision-v1',
        'source_bindings': {str(path): sha(path) for path in (TRAINING / 'plan-retry002.json', failed_path,
                            TRAINING / 'resource-root-execution-002.json')},
        'reason': 'B2 four-second reference forwards exceed the unchanged GPU process cap, including with observed expandable segments.',
        'proposed_training_config': config, 'logical_batch_size_unchanged': 16,
        'ordinary_microbatches_per_update': 16, 'auxiliary_microbatches_per_update': 2,
        'one_complete_ordinary_scalar_and_one_complete_weighted_auxiliary_scalar_per_update': True,
        'context_objective_parent_data_schedule_optimizer_and_inference_unchanged': True,
        'cpu_qualification_required': True, 'fresh_gpu_qualification_required': True,
        'production_plan_frozen': False, 'training_updates_run': 0, 'quality_measured': False}
    plan = copy.deepcopy(previous)
    plan.update(source_bindings=bindings, output_directory=str(OUT), ordinary_microbatch=1, auxiliary_microbatch=1,
        budget_before=budget, correction='Qualify B1 ordinary and auxiliary accumulation after measured B2 capacity failures; retain full B16 objective and four-second score.',
        execution_revision={'path':str(OUT/'execution-revision.json'), 'sha256':None},
        additional_scope='Both individual full-context auxiliary views are also compared against the unmodified renderer.')
    OUT.mkdir()
    (OUT / 'supervise.py').write_bytes((PREVIOUS / 'supervise.py').read_bytes())
    write(OUT / 'execution-revision.json', decision)
    plan['execution_revision']['sha256'] = sha(OUT / 'execution-revision.json')
    plan['source_bindings'][str(OUT / 'execution-revision.json')] = sha(OUT / 'execution-revision.json')
    plan['source_bindings'][str(OUT / 'supervise.py')] = sha(OUT / 'supervise.py')
    verify_inputs(plan)
    write(OUT / 'plan.json', plan)
    command = {'argv':[PYTHON,'-u','-m','research.direct.check_latency58_four_second_single_model',
        '--plan',str(OUT/'plan.json'),'--plan-sha256',sha(OUT/'plan.json')], 'cwd':str(ROOT),
        'environment':{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},
        'plan_sha256':sha(OUT/'plan.json'), 'timeout_seconds':plan['timeout_seconds'],
        'scope':'Full B16/four-second model-gradient and both auxiliary-context CPU qualification with B1; no optimizer update or quality claim.'}
    write(OUT/'command.json',command)
    write(OUT/'preparation-result.json',{'status':'prepared','plan_sha256':sha(OUT/'plan.json'),
        'source_bindings_unchanged':True,'verified_input_count':len(bindings),'gpu_used':False,
        'ordinary_microbatch':1,'auxiliary_microbatch':1,'logical_batch_size':16,'scored_samples':176512,
        'budget_after':snapshot()})
    print(json.dumps({'status':'prepared','plan_sha256':sha(OUT/'plan.json'),'verified_input_count':len(bindings)}),flush=True)


if __name__ == '__main__':
    main()
