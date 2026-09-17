"""Qualify the selected complete B16/B2 gradients and shared-storage restart."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import torch

from research.direct.run_latency58_quality import ROOT,read,require,sha,write
from research.direct.train_latency58 import PRODUCTION,verify_inputs,state_sha256,load_source,continuity
from research.direct.latency58_branch_memory_checkpoint import load_model
from research.direct.latency58_four_second_storage import snapshot
from research.direct.latency58_four_second_data import CROP_SAMPLES,WARMUP_SAMPLES
from research.direct.latency58_grouped_vocal_auxiliary import source_views
from research.direct.latency58_bf16_saved_gru_weights import share_saved_gru_weights
from research.direct.check_latency58_branch_long_context_gpu_b4 import compare_context
from research.direct.latency58_four_second_geometry_gradients import compare_group_gradients
from research.direct.latency58_four_second_shared_device_restart import check_restart


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();plan=read(args.plan);out=args.plan.parent
    require(Path.cwd()==ROOT and sha(args.plan)==args.plan_sha256
        and all(os.environ.get(k)==v for k,v in plan['environment'].items()),'Changed selected GPU qualification')
    decision=read(plan['scientific_decision']['path']);config=decision['proposed_training_config']
    require(sha(plan['scientific_decision']['path'])==plan['scientific_decision']['sha256']
        and config['microbatch_size']==16 and config['auxiliary_microbatch_size']==2
        and config['batch_size']==16 and decision['scored_samples']==176512,'Changed selected geometry')
    verify_inputs(plan)
    require(not (out/'result.json').exists() and not (out/'metrics.jsonl').exists(),'Preserve selected qualification')
    monitor=load_source('selected_geometry_watchdog',plan['watchdog_source'])
    _,events=continuity(plan,monitor);write(out/'event-continuity.json',events)
    before=snapshot();began=time.monotonic()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,_=load_model(decision['parent_checkpoint']);parent=state_sha256(model.state_dict())
    require(parent==decision['parent_model_state_sha256'] and not torch.cuda.is_initialized(),'Changed retained parent')
    torch.cuda.set_per_process_memory_fraction(.75)
    sys.path.insert(0,str(PRODUCTION));import train_production as production
    production.configure_determinism(20261107)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    model.cuda().train().requires_grad_(True);model.training_precision='bf16'
    sequence=0
    def progress(phase,group,offset):
        nonlocal sequence
        sequence+=1
        row={'step':sequence,'kind':'selected_geometry_qualification','training_optimizer_updates':0,
            'phase':phase,'group':group,'offset':offset,'elapsed_seconds':time.monotonic()-began,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    with torch.random.fork_rng(devices=[0]):
        truth=.02*torch.randn(16,4,2,CROP_SAMPLES,generator=torch.Generator().manual_seed(202611042))
        truth[:3,2]=0;truth[4:6,1]=0;truth[8:9,3]=0
        mixes,targets=source_views(truth.sum(1),truth)
        with share_saved_gru_weights(model) as sharing:
            auxiliary=compare_context(model,mixes.cuda(),WARMUP_SAMPLES)
        auxiliary['sharing']=sharing.report();auxiliary['microbatch_size']=2
        auxiliary['views']=['instrumental','vocals_only']
        del mixes,targets,sharing
        progress('selected_auxiliary_context_pass','auxiliary',0)
        with share_saved_gru_weights(model) as sharing:
            gradients=compare_group_gradients(model,truth.sum(1),truth,warmup_samples=WARMUP_SAMPLES,
                ordinary_microbatch=16,auxiliary_microbatch=2,progress=progress)
        grouped_sharing=sharing.report();del truth,sharing
        gc.collect()
        restart=check_restart(model,decision,ordinary_microbatch=16,auxiliary_microbatch=2,progress=progress)
        progress('selected_weighted_restart_pass','both',3)
    require(sequence==12,'Unexpected selected qualification progress count')
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.synchronize()
    require(state_sha256(model.state_dict())==parent and torch.equal(cpu_rng,torch.get_rng_state())
        and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)),
        'Selected qualification changed retained weights or RNG')
    verify_inputs(plan)
    write(out/'result.json',{'status':'pass','plan_sha256':args.plan_sha256,'source_bindings_unchanged':True,
        'scientific_decision':plan['scientific_decision'],'parent_model_state_sha256':parent,
        'parent_weights_unchanged':True,'rng_unchanged':True,'gpu_used':True,'training_optimizer_updates':0,
        'private_restart_fixture_updates':3,'quality_measured':False,'checkpoint_written':False,
        'ordinary_microbatch':16,'auxiliary_microbatch':2,'logical_batch_size':16,
        'auxiliary_context':auxiliary,'whole_group_gradients':gradients,'grouped_sharing':grouped_sharing,
        'weighted_restart':restart,'elapsed_seconds':time.monotonic()-began,
        'budget_before':before,'budget_after':snapshot(),
        'limits':['Packed disk recovery and measured recorded-data updates remain required before production.']})
    print(json.dumps({'status':'pass','elapsed_seconds':time.monotonic()-began}),flush=True)


if __name__=='__main__':main()
