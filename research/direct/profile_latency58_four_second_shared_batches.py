"""Measure exact shared-storage four-second execution at larger microbatches."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import torch

from research.direct.run_latency58_quality import ROOT, read, require, sha, write
from research.direct.train_latency58 import PRODUCTION, verify_inputs, state_sha256, load_source, continuity
from research.direct.latency58_branch_memory_checkpoint import load_model
from research.direct.latency58_four_second_storage import snapshot
from research.direct.latency58_four_second_data import CROP_SAMPLES, WARMUP_SAMPLES
from research.direct.latency58_bf16_saved_gru_weights import share_saved_gru_weights
from research.direct.check_latency58_four_second_shared_gpu import short_model
from research.direct.check_latency58_branch_long_context_gpu_b4 import compare_context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();plan=read(args.plan);out=args.plan.parent
    require(Path.cwd()==ROOT and sha(args.plan)==args.plan_sha256
        and all(os.environ.get(k)==v for k,v in plan['environment'].items()),'Changed batch-profile environment')
    require(plan['microbatches']==[2,4,8,16] and plan['warmup_samples']==WARMUP_SAMPLES
        and plan['scored_samples']==176512 and plan['training_optimizer_updates']==0,'Changed profile scope')
    verify_inputs(plan)
    require(not (out/'result.json').exists() and not (out/'metrics.jsonl').exists(),'Preserve batch-profile evidence')
    monitor=load_source('shared_batch_watchdog',plan['watchdog_source'])
    _,events=continuity(plan,monitor);write(out/'event-continuity.json',events)
    before=snapshot();began=time.monotonic()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,_=load_model(plan['parent_checkpoint']);parent=state_sha256(model.state_dict())
    require(parent==plan['parent_model_state_sha256'] and not torch.cuda.is_initialized(),'Changed retained CPU parent')
    torch.cuda.set_per_process_memory_fraction(.75)
    sys.path.insert(0,str(PRODUCTION));import train_production as production
    production.configure_determinism(20261106)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    model.cuda().train().requires_grad_(True);model.training_precision='bf16'
    generator=torch.Generator(device='cuda').manual_seed(202611061)
    results=[]
    with torch.random.fork_rng(devices=[0]):
        for index,batch in enumerate(plan['microbatches']):
            print(json.dumps({'event':'batch_started','microbatch':batch}),flush=True)
            try:
                audio=.03*torch.randn(batch,2,512+12288,device='cuda',generator=generator)
                expected,baseline=short_model(model,audio,shared=False)
                actual,shared=short_model(model,audio,shared=True)
                require(all(torch.equal(v,actual[k]) for k,v in expected.items()) and baseline['loss']==shared['loss'],
                    'Shared storage changed short-context batch outputs or gradients')
                short={'all_40_parameter_gradients_and_outputs_bit_exact':True,'baseline':baseline,'shared':shared}
                del expected,actual,audio
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                audio=.03*torch.randn(batch,2,CROP_SAMPLES,device='cuda',generator=generator)
                torch.cuda.synchronize();case_began=time.monotonic()
                with share_saved_gru_weights(model) as sharing:
                    full=compare_context(model,audio,WARMUP_SAMPLES)
                torch.cuda.synchronize()
                row={'status':'pass','microbatch':batch,'short':short,'full_context':full,'sharing':sharing.report(),
                    'two_full_forward_backward_comparisons_seconds':time.monotonic()-case_began,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'training_optimizer_updates':0,'quality_measured':False}
                require(all(v>0 for v in row['sharing']['matched_parameter_transpose_counts'].values()),
                    'Unexercised recurrent matrix in full batch')
                del audio,sharing;model.zero_grad(set_to_none=True);gc.collect()
            except torch.OutOfMemoryError as error:
                write(out/f'batch-{batch:02d}-oom.json',{'status':'capacity_failure','microbatch':batch,
                    'message':str(error),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'training_optimizer_updates':0,'passed_smaller_microbatches':[r['microbatch'] for r in results]})
                raise
            results.append(row);write(out/f'batch-{batch:02d}.json',row)
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps({'step':index+1,'kind':'batch_geometry_fixture','microbatch':batch,
                    'training_optimizer_updates':0,'peak_allocated_bytes':row['peak_allocated_bytes'],
                    'two_full_forward_backward_comparisons_seconds':row['two_full_forward_backward_comparisons_seconds']})+'\n')
            print(json.dumps({'event':'batch_pass','microbatch':batch,'peak_allocated_bytes':row['peak_allocated_bytes'],
                'two_full_forward_backward_comparisons_seconds':row['two_full_forward_backward_comparisons_seconds']}),flush=True)
    require(state_sha256(model.state_dict())==parent and torch.equal(cpu_rng,torch.get_rng_state())
        and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)),
        'Batch profile changed weights or RNG')
    verify_inputs(plan)
    write(out/'result.json',{'status':'pass','plan_sha256':args.plan_sha256,'source_bindings_unchanged':True,
        'parent_model_state_sha256':parent,'parent_weights_unchanged':True,'rng_unchanged':True,
        'gpu_used':True,'training_optimizer_updates':0,'quality_measured':False,'checkpoint_written':False,
        'microbatch_cases':results,'elapsed_seconds':time.monotonic()-began,
        'budget_before':before,'budget_after':snapshot(),
        'limits':['Geometry and storage equivalence only; complete weighted updates and disk recovery remain required.']})
    print(json.dumps({'status':'pass','elapsed_seconds':time.monotonic()-began}),flush=True)


if __name__=='__main__':main()
