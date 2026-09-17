"""CPU training of the output head from frozen recurrent features and exact moments."""
from __future__ import annotations

import json
import math
import time

import torch

from research.direct.latency58_dynamic_mixer import make_head, loss_from_frames
from research.direct.run_latency58_quality import require


def fit_head(features, moments, plan, journal_path):
    head = make_head()
    optimizer = torch.optim.Adam(head.parameters(), lr=plan["maximum_lr"], foreach=False)
    generator = torch.Generator().manual_seed(plan["training_seed"])
    began = time.monotonic()
    with journal_path.open("x", buffering=1) as journal:
        for step in range(plan["optimization_steps"]):
            if step < plan["warmup_updates"]:
                lr = plan["maximum_lr"] * (step + 1) / plan["warmup_updates"]
            else:
                phase = (step - plan["warmup_updates"]) / (plan["optimization_steps"] - 1 - plan["warmup_updates"])
                lr = plan["minimum_lr"] + .5 * (plan["maximum_lr"] - plan["minimum_lr"]) * (1 + math.cos(math.pi * phase))
            optimizer.param_groups[0]["lr"] = lr
            indices = torch.randint(features.shape[0], (plan["fit_batch_size"],), generator=generator)
            batch = {name: tensor[indices] for name, tensor in moments.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, stems, absence = loss_from_frames(head, features[indices], batch)
            require(bool(torch.isfinite(loss)), "Nonfinite dynamic training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True, foreach=False)
            optimizer.step()
            row = {"step": step + 1, "lr": lr, "example_indices": indices.tolist(), "loss": float(loss.detach()),
                   "per_stem_training_sdr_db": stems.detach().tolist(), "absence": float(absence.detach()),
                   "grad_norm": float(norm), "elapsed_seconds": time.monotonic() - began}
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            if (step + 1) % 25 == 0 or step == 0:
                print(json.dumps({k: row[k] for k in ("step", "loss", "grad_norm", "elapsed_seconds")}), flush=True)
    require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), "Nonfinite fitted head")
    for state in optimizer.state.values():
        require(state["step"].item() == plan["optimization_steps"] and bool(torch.isfinite(state["exp_avg"]).all())
                and bool(torch.isfinite(state["exp_avg_sq"]).all()) and bool((state["exp_avg_sq"] >= 0).all()),
                "Invalid fitted head Adam state")
    return head.eval().requires_grad_(False), {"optimizer": optimizer.state_dict(), "sampler_rng": generator.get_state(),
                                            "steps": plan["optimization_steps"]}
