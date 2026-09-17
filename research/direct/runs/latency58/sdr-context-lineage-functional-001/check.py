import copy, json, os
from pathlib import Path
from research.direct.run_latency58_quality import PHASE, read, require, sha, write
from research.direct.latency58_context_checkpoint import load_parent, expected_provenance
from research.direct.train_latency58 import state_sha256
require(os.environ.get("CUDA_VISIBLE_DEVICES")=="", "CPU-only parent check")
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
plan=read(PHASE/"sdr-context-lineage-functional-001/plan.json")
require(all(sha(p)==s for p,s in plan["source_bindings"].items()), "Inputs changed")
rows=[]
for name in ("working_baseline","c91","cropped11"):
    training=read(PHASE/"sdr-teacher-prep-001"/((name if name!="working_baseline" else "c91")+"-plan.json"))
    if name=="working_baseline":
        checkpoint=PHASE/"teacher-half-canonical-001/model.pt"
        parent={"kind":"working_baseline"}
    else:
        gen=PHASE/("sdr-teacher-"+name+"-b4-lr3e5-1000/checkpoints/step-001000")
        checkpoint=gen/"model.pt"
        binding=PHASE/"sdr-teacher-prep-001"/(name+"-plan.json")
        parent={"kind":"sdr_candidate","generation":str(gen),"training_plan":{"path":str(binding),"sha256":sha(binding)}}
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    parent.update(checkpoint={"path":str(checkpoint),"sha256":sha(checkpoint)},provenance=payload["provenance"],model_state_sha256=payload["model_state_sha256"])
    recipe={"parent":parent,"carry_state":False,"warmup_samples":88064,"scored_samples":88064,
            **{k:training[k] for k in ("teacher_kind","teacher_weight","teacher_model_state_sha256","precision_policy")}}
    model=load_parent(recipe)
    require(state_sha256(model.state_dict())==payload["model_state_sha256"] and all(torch.equal(v,payload["model"][k]) for k,v in model.state_dict().items()), "Parent load changed tensors")
    rejections=[]
    for field in ("model_state_sha256","provenance","checkpoint"):
        bad=copy.deepcopy(recipe)
        if field=="model_state_sha256":bad["parent"][field]="0"*64
        elif field=="provenance":bad["parent"][field]["training_updates"]-=1
        else:bad["parent"][field]["sha256"]="0"*64
        try:load_parent(bad)
        except RuntimeError:rejections.append(field)
        else:raise AssertionError("Accepted altered parent "+field)
    for carry in (False,True):
        recipe["carry_state"]=carry
        for step in (2,250,500):
            p=expected_provenance(recipe,step,"a"*64)
            require(p["parent_provenance"]==payload["provenance"] and p["parent_training_updates"]==payload["provenance"]["training_updates"]
                    and p["training_updates"]==payload["provenance"]["training_updates"]+step
                    and p["context_trial_updates"]==step and p["carry_state"] is carry and p["parent_checkpoint"]==parent["checkpoint"], "Context lineage counters differ")
    rows.append({"parent":name,"model_state_sha256":parent["model_state_sha256"],"cumulative_parent_updates":payload["provenance"]["training_updates"],"loaded_tensors_exact":True,"rejected_mutations":rejections,"context_provenance_cases":6})
    del model,payload
require(not torch.cuda.is_initialized() and all(sha(p)==s for p,s in plan["source_bindings"].items()), "CPU proof changed inputs or used CUDA")
write(PHASE/"sdr-context-lineage-functional-001/result.json",{"schema":"latency58-context-parent-proof-v1","status":"pass","source_bindings":plan["source_bindings"],"source_bindings_unchanged":True,"parents":rows,"cuda_initialized":False,"optimizer_updates":0,"saved_weights":False})
print(json.dumps({"status":"pass","parents":rows}))
