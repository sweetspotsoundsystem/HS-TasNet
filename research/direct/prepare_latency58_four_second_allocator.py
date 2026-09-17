"""Retry the unchanged full four-second objective with expandable CUDA segments."""
import ast
import copy
import json
import os
from pathlib import Path
import sys

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT
from research.direct.latency58_four_second_monitor import require_monitor_qualification
from research.direct.train_latency58_four_second import validate_recipe, budget_snapshot
from research.direct.run_latency58_four_second_v3 import require_cpu_evidence, require_failed_resource_closed, binding
from research.direct.run_latency58_four_second_quality_v3 import transport_proof

OUT = ARTIFACT_ROOT / 'branch-four-second-015'


def main():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'Require CPU-only preparation')
    target = OUT / 'plan-retry002.json'; source_path = OUT / 'plan-retry001.json'
    require(not target.exists() and not (OUT / 'resource-run-003').exists()
            and not (OUT / 'resource-stage-003').exists() and not (OUT / 'production-run').exists(),
            'Preserve existing attempts')
    original = read(source_path)
    failed = read(OUT / 'resource-root-execution-002.json')
    stage_path = OUT / 'resource-stage-002/execution.json'; stage = read(stage_path)
    monitor_path = Path(stage['monitor_result']); monitor = read(monitor_path)
    require_failed_resource_closed(stage, monitor)
    require(type(failed['actual_exit_code']) is int and failed['actual_exit_code'] == 1
            and failed['source_bindings_unchanged'] and failed['actual_session_id'] == 85930
            and failed['actual_tool_chunk_id'] == '2bd4da' and failed['plan_sha256'] == sha(source_path)
            and failed['stage_execution_sha256'] == sha(stage_path)
            and failed['monitor_result_sha256'] == sha(monitor_path)
            and failed['all_owned_processes_closed'] and failed['training_updates'] == 0
            and not (OUT / 'resource-run/metrics.jsonl').exists(), 'OOM completion is not authenticated')
    verify_inputs(original)
    bindings = dict(original['source_bindings'])
    def merge(incoming):
        for path, digest in incoming.items():
            require(path not in bindings or bindings[path] == digest, 'Frozen input changed: ' + path)
            bindings[path] = digest
    merge(require_monitor_qualification())
    require(transport_proof() == read(original['evaluation_adapter_source_proof']['path']), 'Scientific evaluation changed')
    paths = [Path(__file__).resolve(), source_path, OUT / 'resource-root-execution-002.json',
             OUT / 'resource-root-command-002.json', stage_path, monitor_path, monitor_path.parent / 'child.log',
             ARTIFACT_ROOT / 'preparation-retry001-root-execution.json']
    paths.extend(ROOT / 'research/direct' / name for name in (
        'train_latency58_four_second_v2.py', 'run_latency58_four_second_v3.py',
        'run_latency58_four_second_quality_v3.py'))
    loops = []
    for name in ('train_latency58_four_second.py', 'train_latency58_four_second_v2.py'):
        tree = ast.parse((ROOT / 'research/direct' / name).read_text())
        found = [n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.target) == '(mixture_cpu, truth_cpu)']
        require(len(found) == 1, 'Ambiguous scientific update loop')
        loops.append(ast.dump(found[0], include_attributes=False))
    require(loops[0] == loops[1], 'Allocator observation changed the complete scientific update loop')
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if isinstance(value, str):
            path = Path(value)
            if path.is_file() and path.suffix == '.py' and path.resolve().is_relative_to(ROOT):
                paths.append(path.resolve())
    merge({str(path): sha(path) for path in paths})
    plan = copy.deepcopy(original)
    plan.update(source_bindings=bindings,
        environment={**original['environment'], 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
        operational_retry_of=binding(source_path),
        operational_retry_reason='Reference-forward CUDA OOM before updates with 3.32 GiB reserved but unused; test expandable segments without changing math or memory limits.',
        previous_execution_for_resource=str(stage_path),
        resource_stage_directory=str(OUT / 'resource-stage-003'),
        resource_run_directory=str(OUT / 'resource-run-003'),
        resource_root_execution=str(OUT / 'resource-root-execution-003.json'),
        controller_module='research.direct.run_latency58_four_second_v3',
        quality_controller_module='research.direct.run_latency58_four_second_quality_v3',
        trainer_module='research.direct.train_latency58_four_second_v2',
        allocator_retry={'setting': 'expandable_segments:True', 'backend': 'native', 'process_memory_fraction': .75,
            'observed_expandable_segment_required': True, 'complete_scientific_update_loop_ast_identical': True,
            'reference': 'https://docs.pytorch.org/docs/2.8/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf',
            'health_limits_changed': False, 'scored_context_or_batch_changed': False})
    allowed = {'source_bindings', 'environment', 'operational_retry_of', 'operational_retry_reason',
               'previous_execution_for_resource', 'resource_stage_directory', 'resource_root_execution',
               'controller_module', 'quality_controller_module'}
    require(all(plan[key] == original[key] for key in original if key not in allowed),
            'Allocator retry changed a scientific, schedule, storage or inference field')
    require({k:v for k,v in plan['environment'].items() if k != 'PYTORCH_CUDA_ALLOC_CONF'} == original['environment'],
            'Allocator retry changed another environment setting')
    validate_recipe(plan); require_cpu_evidence(plan); verify_inputs(plan)
    write(target, plan)
    write(OUT / 'preparation-retry002-result.json', {'status': 'prepared', 'plan_sha256': sha(target),
        'source_plan': binding(source_path), 'source_bindings_unchanged': True,
        'verified_input_count': len(bindings), 'all_scientific_fields_and_training_loop_unchanged': True,
        'allocator_retry': plan['allocator_retry'], 'budget_after': budget_snapshot(plan), 'gpu_workload_started': False})
    print(json.dumps({'status': 'prepared', 'plan_sha256': sha(target), 'verified_input_count': len(bindings),
                      'all_scientific_fields_and_training_loop_unchanged': True}), flush=True)


if __name__ == '__main__':
    main()
