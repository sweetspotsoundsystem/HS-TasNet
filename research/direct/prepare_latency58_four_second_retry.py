"""Prepare the same 015 scientific recipe after a prelaunch log-path rejection."""
import ast
import copy
import json
import os
from pathlib import Path
import sys

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import verify_inputs
from research.direct.latency58_four_second_storage import ARTIFACT_ROOT
from research.direct.latency58_four_second_monitor import WATCHDOG, WATCHDOG_SHA, require_monitor_qualification
from research.direct.train_latency58_four_second import validate_recipe, budget_snapshot
from research.direct.run_latency58_four_second_v2 import require_cpu_evidence, binding
from research.direct.run_latency58_four_second_quality_v2 import transport_proof

OUT = ARTIFACT_ROOT / 'branch-four-second-015'


def main():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'Require CPU-only preparation')
    target = OUT / 'plan-retry001.json'
    require(not target.exists() and not (OUT / 'resource-run').exists()
            and not (OUT / 'resource-stage-002').exists() and not (OUT / 'production-run').exists(),
            'Preserve existing attempts; only a prelaunch failure permits this operational retry')
    original = read(OUT / 'plan.json')
    completed = read(ARTIFACT_ROOT / 'preparation-root-execution.json')
    failed = read(OUT / 'resource-root-execution.json')
    stage = read(OUT / 'resource-stage/execution.json')
    require(type(completed['actual_exit_code']) is int and completed['actual_exit_code'] == 0
            and completed['source_bindings_unchanged'] and completed['plan_sha256'] == sha(OUT / 'plan.json')
            and failed['actual_exit_code'] == stage['actual_exit_code'] == 1
            and failed['source_bindings_unchanged'] and stage['source_bindings_unchanged']
            and failed['actual_session_id'] == 87824 and failed['actual_tool_chunk_id'] == '86dec1'
            and failed['plan_sha256'] == stage['plan_sha256'] == sha(OUT / 'plan.json')
            and failed['stage_execution_sha256'] == sha(OUT / 'resource-stage/execution.json')
            and failed['gpu_workload_started'] is False and failed['training_updates'] == 0
            and failed['monitor_workers_started'] is False and not Path(stage['monitor_result']).parent.exists(),
            'Original attempt was not an authenticated failure before telemetry and training')
    verify_inputs(original)
    bindings = dict(original['source_bindings'])
    def merge(incoming):
        for path, digest in incoming.items():
            require(path not in bindings or bindings[path] == digest, 'Frozen input changed: ' + path)
            bindings[path] = digest
    merge(require_monitor_qualification())
    proof = transport_proof()
    require(proof == read(original['evaluation_adapter_source_proof']['path']), 'Scientific score adapters changed')
    preserved = []
    for before_name, after_name, names in (
        ('run_latency58_four_second.py', 'run_latency58_four_second_v2.py',
         ('require_cpu_evidence', 'require_base_resource_result', 'require_resource_result')),
        ('run_latency58_weighted_vocal.py', 'latency58_four_second_monitor.py', ('require_monitor_closed',))):
        before, after = [ROOT / 'research/direct' / name for name in (before_name, after_name)]
        trees = [ast.parse(path.read_text()) for path in (before, after)]
        for name in names:
            nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name) for tree in trees]
            require(ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False),
                    'Resource or shutdown qualification changed: ' + name)
            preserved.append(name)
    paths = [Path(__file__).resolve(), OUT / 'plan.json', ARTIFACT_ROOT / 'preparation-root-execution.json',
             OUT / 'resource-root-execution.json', OUT / 'resource-root-command.json',
             OUT / 'resource-stage/execution.json', OUT / 'resource-stage/console.log',
             ARTIFACT_ROOT / 'monitor-path-root-command.json', WATCHDOG]
    paths.extend(ROOT / 'research/direct' / name for name in (
        'run_latency58_four_second_v2.py', 'run_latency58_four_second_quality_v2.py',
        'latency58_four_second_monitor.py', 'check_latency58_four_second_monitor.py'))
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if isinstance(value, str):
            path = Path(value)
            if path.is_file() and path.suffix == '.py' and path.resolve().is_relative_to(ROOT):
                paths.append(path.resolve())
    merge({str(path): sha(path) for path in paths})
    plan = copy.deepcopy(original)
    plan.update(source_bindings=bindings, watchdog_source=str(WATCHDOG),
        supervision={**original['supervision'], 'watchdog_sha256': WATCHDOG_SHA},
        operational_retry_of=binding(OUT / 'plan.json'),
        operational_retry_reason='The original guard rejected the allocated log root before any telemetry worker or GPU child launched.',
        monitor_path_cpu_qualification=binding(ARTIFACT_ROOT / 'monitor-path-cpu-001/result.json'),
        resource_stage_directory=str(OUT / 'resource-stage-002'),
        resource_root_execution=str(OUT / 'resource-root-execution-002.json'),
        controller_module='research.direct.run_latency58_four_second_v2',
        quality_controller_module='research.direct.run_latency58_four_second_quality_v2',
        unchanged_resource_and_shutdown_function_asts=preserved)
    require(all(plan[key] == original[key] for key in original if key not in ('source_bindings', 'watchdog_source', 'supervision')),
            'Operational retry changed a scientific, schedule, data, storage or inference field')
    validate_recipe(plan); require_cpu_evidence(plan); verify_inputs(plan)
    write(target, plan)
    write(OUT / 'preparation-retry001-result.json', {'status': 'prepared', 'plan_sha256': sha(target),
        'original_plan': binding(OUT / 'plan.json'), 'source_bindings_unchanged': True,
        'verified_input_count': len(bindings), 'all_original_scientific_fields_unchanged': True,
        'unchanged_resource_and_shutdown_function_asts': preserved,
        'monitor_path_cpu_qualification': plan['monitor_path_cpu_qualification'],
        'budget_after': budget_snapshot(plan), 'gpu_workload_started': False})
    print(json.dumps({'status': 'prepared', 'plan_sha256': sha(target), 'verified_input_count': len(bindings),
                      'all_original_scientific_fields_unchanged': True}), flush=True)


if __name__ == '__main__':
    main()
