"""Exercise the unchanged weighted update at an explicit qualified geometry."""
import json
from unittest.mock import patch
import torch
from research.direct.run_latency58_quality import require
from research.direct.check_latency58_grouped_vocal_restart import fingerprint, RequestedStop

def exercise(model, optimizer, ema, inputs, *, step, ordinary_microbatch, auxiliary_microbatch, update_impl, stop_after=None):
    before = fingerprint(model, optimizer, ema)
    groups, gradients, optimizer_calls, stop = [], {}, [], False

    def check_continue():
        if stop:
            raise RequestedStop("Requested stop before committing grouped update")

    def after_group(group, row):
        nonlocal stop
        require(fingerprint(model, optimizer, ema) == before, "Endpoint advanced before both groups completed")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0
                    for p in model.parameters()), "A group did not reach all 40 parameters")
        if group == "ordinary":
            gradients.update({name: p.grad.detach().clone() for name, p in model.named_parameters()})
        else:
            require(all(float((p.grad - gradients[name]).norm()) > 0 for name, p in model.named_parameters()),
                    "Auxiliary group did not add to every parameter gradient")
            require(row["view_contribution_multipliers"] == [1., .25], "Restart lost the selected view weights")
            gradients.clear()
        groups.append(group); stop = group == stop_after
        print(json.dumps({"event": "weighted_restart_progress", "step": step, "group": group,
                          "stop_after": stop_after, "weighted_loss": row["weighted_loss"]}), flush=True)

    hook = optimizer.register_step_post_hook(lambda *args: optimizer_calls.append(1))
    try:
        with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clips, \
                patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zeroes:
            try:
                result = update_impl(model, optimizer, ema, *inputs, step=step, warmup_samples=512,
                                        ordinary_microbatch=ordinary_microbatch, auxiliary_microbatch=auxiliary_microbatch,
                                        check_continue=check_continue, after_group=after_group)
            except RequestedStop:
                require(stop_after is not None and fingerprint(model, optimizer, ema) == before
                        and not optimizer_calls and clips.call_count == 0 and zeroes.call_count == 1,
                        "Interrupted accumulation changed an optimizer endpoint")
                require(groups == (["ordinary"] if stop_after == "ordinary" else ["ordinary", "auxiliary"]),
                        "Wrong interruption boundary")
                return {"stop_after": stop_after, "groups_completed": groups,
                        "weights_adam_ema_unchanged": True, "optimizer_calls": 0, "clip_calls": 0}
            require(stop_after is None and groups == ["ordinary", "auxiliary"] and optimizer_calls == [1]
                    and clips.call_count == zeroes.call_count == 1 and ema.updates == before["ema_updates"] + 1,
                    "Weighted update was not committed exactly once")
            require(result["raw_model_state_sha256"] != before["model"]
                    and result["ema_parameters_sha256"] != before["ema"]
                    and len(result["groups"]["ordinary"]["microbatches"]) == 16 // ordinary_microbatch
                    and len(result["groups"]["auxiliary"]["microbatches"]) == 2 // auxiliary_microbatch,
                    "Weighted update or microbatch geometry differs")
            result.update(observed_optimizer_calls=len(optimizer_calls), observed_clip_calls=clips.call_count,
                          observed_zero_grad_calls=zeroes.call_count,
                          both_group_gradients_reach_all_40_parameters=True,
                          endpoint_unchanged_at_both_group_boundaries=True)
            return result
    finally:
        hook.remove()
