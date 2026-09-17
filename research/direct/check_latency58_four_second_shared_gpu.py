"""Qualify exact saved-value sharing and complete four-second BF16 gradients."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
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
from research.direct.latency58_grouped_vocal_auxiliary import source_views
from research.direct.latency58_bf16_saved_gru_weights import share_saved_gru_weights, SharedGRUSavedWeights
from research.direct.check_latency58_branch_long_context_gpu_b4 import compare_context
from research.direct.check_latency58_four_second_single_model import compare_group_gradients


def short_model(model, audio, *, shared):
    from research.direct.latency58_branch_memory_context import render_scored_context
    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    x = audio.clone().requires_grad_()
    context = share_saved_gru_weights(model) if shared else nullcontext(None)
    with context as sharing:
        result = render_scored_context(model, x, warmup_samples=512, carry_state=True)
        loss = result.raw.square().mean() + result.deployed.square().mean()
        loss.backward()
    values = {'raw':result.raw.detach().cpu(), 'deployed':result.deployed.detach().cpu(),
        'input_gradient':x.grad.detach().cpu(), **{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}}
    require(len(values)==43 and all(bool(torch.isfinite(v).all()) for v in values.values()), 'Incomplete model derivatives')
    return values, {'loss':float(loss.detach()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'sharing':None if sharing is None else sharing.report()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();plan=read(args.plan);out=args.plan.parent
    require(Path.cwd()==ROOT and sha(args.plan)==args.plan_sha256
        and all(os.environ.get(k)==v for k,v in plan['environment'].items()),'Changed qualification environment')
    verify_inputs(plan)
    require(not (out/'result.json').exists() and not (out/'metrics.jsonl').exists(),'Preserve qualification results')
    monitor=load_source('shared_gpu_watchdog',plan['watchdog_source'])
    _,events=continuity(plan,monitor);write(out/'event-continuity.json',events)
    before=snapshot();began=time.monotonic()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,_=load_model(plan['parent_checkpoint']);parent=state_sha256(model.state_dict())
    require(parent==plan['parent_model_state_sha256'] and not torch.cuda.is_initialized(),'Changed retained CPU parent')
    torch.cuda.set_per_process_memory_fraction(.75)
    sys.path.insert(0,str(PRODUCTION));import train_production as production
    production.configure_determinism(20261105)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    model.cuda().train().requires_grad_(True);model.training_precision='bf16'
    sequence=0
    def progress(phase,group,offset):
        nonlocal sequence
        sequence+=1
        row={'step':sequence,'kind':'qualification_fixture','training_optimizer_updates':0,
            'phase':phase,'group':group,'offset':offset,'elapsed_seconds':time.monotonic()-began,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    generator=torch.Generator(device='cuda').manual_seed(202611052)
    with torch.random.fork_rng(devices=[0]):
        # The unmodified full model fits on this short crop. Check storage-only
        # sharing against its actual CUDA arithmetic for every parameter.
        audio=.03*torch.randn(1,2,512+12288,device='cuda',generator=generator)
        expected,baseline=short_model(model,audio,shared=False)
        actual,shared=short_model(model,audio,shared=True)
        short_errors={k:float((v-actual[k]).abs().max()) for k,v in expected.items()}
        require(all(torch.equal(v,actual[k]) for k,v in expected.items()) and baseline['loss']==shared['loss'],
            'Sharing changed a full-model short-context output or gradient')
        require(all(v>0 for v in shared['sharing']['matched_parameter_transpose_counts'].values()),'Unexercised GRU sharing')
        del expected,actual,audio;model.zero_grad(set_to_none=True);gc.collect()
        progress('short_full_model_bit_exact','all',0)
        # Explicit negative control: saved-value mutation must be detected.
        probe=SharedGRUSavedWeights(model);candidate=next(iter(probe.candidates.values()))[0][1]
        token=probe.pack(candidate);candidate.add_(1)
        rejected=False
        try:probe.unpack(token)
        except RuntimeError:rejected=True
        require(rejected,'Modified shared value was accepted');del probe,candidate,token
        audio=.03*torch.randn(1,2,CROP_SAMPLES,device='cuda',generator=generator)
        contexts=[]
        with share_saved_gru_weights(model) as sharing:
            contexts.append(compare_context(model,audio,WARMUP_SAMPLES))
        contexts[-1]['sharing']=sharing.report();del audio,sharing
        progress('full_context_pass','ordinary',0)
        truth=.02*torch.randn(16,4,2,CROP_SAMPLES,generator=torch.Generator().manual_seed(202611042))
        truth[:3,2]=0;truth[4:6,1]=0;truth[8:9,3]=0
        mixes,targets=source_views(truth.sum(1),truth)
        for index,name in enumerate(('instrumental','vocals_only')):
            with share_saved_gru_weights(model) as sharing:
                contexts.append(compare_context(model,mixes[index:index+1].cuda(),WARMUP_SAMPLES))
            contexts[-1]['sharing']=sharing.report();del sharing
            progress('full_context_pass',name,index)
        del mixes,targets
        with share_saved_gru_weights(model) as sharing:
            gradients=compare_group_gradients(model,truth.sum(1),truth,warmup_samples=WARMUP_SAMPLES,progress=progress)
        grouped_sharing=sharing.report();del truth,sharing
    require(sequence==76,'Unexpected qualification progress count')
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.synchronize()
    require(state_sha256(model.state_dict())==parent and torch.equal(cpu_rng,torch.get_rng_state())
        and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)),
        'Qualification changed weights or RNG')
    verify_inputs(plan)
    write(out/'result.json',{'status':'pass','plan_sha256':args.plan_sha256,'source_bindings_unchanged':True,
        'parent_model_state_sha256':parent,'parent_weights_unchanged':True,'rng_unchanged':True,
        'gpu_used':True,'training_optimizer_updates':0,'quality_measured':False,'checkpoint_written':False,
        'short_original_vs_shared':{'all_40_parameter_gradients_and_outputs_bit_exact':True,
            'maximum_errors':short_errors,'baseline':baseline,'shared':shared,'modified_saved_value_rejected':rejected},
        'full_contexts':contexts,'whole_group_gradients':gradients,'grouped_sharing':grouped_sharing,
        'elapsed_seconds':time.monotonic()-began,'budget_before':before,'budget_after':snapshot(),
        'limits':['No training update, disk restart or production throughput qualified by this check.']})
    print(json.dumps({'status':'pass','elapsed_seconds':time.monotonic()-began}),flush=True)


if __name__=='__main__':main()
