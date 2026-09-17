"""Freeze and queue the complete CPU quality bundle for the live learning-rate trial."""
from __future__ import annotations
import json,os
from pathlib import Path
from research.direct.run_latency58_quality import ROOT,PHASE,read,require,sha,write
from research.direct.train_latency58 import verify_inputs,disk_bytes
from research.direct.report_latency58_sdr import load_completed
from research.direct.report_latency58_cleanup_rebound import REFERENCES,load_views,reference_views


def binding(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path)}


def main():
    require(Path.cwd()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation required')
    training_path=PHASE/'cleanup-rebound-prep-001/training-plan.json'
    resource_path=training_path.parent/'resource-plan.json'
    training=read(training_path);verify_inputs(training)
    require(training['schema']=='latency58-cleanup-rebound-training-v1' and not training['resource_only'],'Different production plan')
    identities=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv=[x.decode() for x in (proc/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
            if '-m' in argv and argv[argv.index('-m')+1]=='research.direct.run_latency58_cleanup_rebound':
                stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
                identities.append({'pid':int(proc.name),'start_ticks':int(stat[19]),'argv':argv})
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            pass
    require(len(identities)==1,'Require one live training supervisor')
    bindings={**training['source_bindings']}
    for p in [training_path,resource_path,*(ROOT/'research/direct').glob('*cleanup_rebound*.py'),training_path.parent/'arithmetic-proof.json']:
        bindings[str(p)]=sha(p)
    for label,prefix in REFERENCES.items():
        for mode in ('full14','actions60','probes'):
            load_completed(PHASE/(prefix+'-'+mode+'-001'),bindings,canonical_baseline=label=='working')
        load_views(reference_views(label),bindings)
    counted=sum(disk_bytes(Path(p)) for p in training['counted_roots'])
    outside=146342157+111344465+disk_bytes(ROOT/'.git/lfs')+disk_bytes(ROOT/'.git/objects')
    require(outside<500_000_000 and counted+400_000_000<79_500_000_000,'Combined cap or concurrent checkpoint reserve exceeded')
    out=PHASE/'cleanup-rebound-250-queue-001';require(not out.exists(),'Preserve existing queue');out.mkdir()
    reservation={'schema':'latency58-cleanup-rebound-views-reservation-v1',
        'evaluation_directories':[str(PHASE/'cleanup-rebound-250-views-001')],
        'new_artifact_allowance_bytes':10_000_000,'training_reserve_bytes':350_000_000,
        'counted_bytes_at_preparation':counted,'counted_roots':training['counted_roots'],
        'stop_counted_bytes':training['stop_counted_bytes'],'source_bindings':bindings,
        'extra_artifact_bytes_outside_counted_roots':outside,'combined_cap_bytes':80_000_000_000}
    write(out/'views-reservation.json',reservation)
    plan={'schema':'latency58-cleanup-rebound-queued-quality-plan-v1','maximum_wait_seconds':9000,
        'launch_next_training_arm':False,'output_directory':str(out),'stage_directory':str(PHASE/'cleanup-rebound-to-000250-001'),
        'training_plan':binding(training_path),'resource_plan':binding(resource_path),'quality_prefix':'cleanup-rebound-250',
        'supervisor':identities[0],'views_reservation':binding(out/'views-reservation.json'),
        'source_bindings':{**bindings,str(out/'views-reservation.json'):sha(out/'views-reservation.json')},
        'matched_comparison':'Learning-rate schedule only; complete matching is independently checked before reporting.',
        'confirmation_policy':'Full14 and existing examples are development measurements; no unseen confirmation claimed.'}
    verify_inputs(plan);write(out/'plan.json',plan)
    print(json.dumps({'status':'prepared','plan':binding(out/'plan.json'),'supervisor':identities[0]['pid'],'bindings':len(plan['source_bindings'])}),flush=True)


if __name__=='__main__':
    main()
