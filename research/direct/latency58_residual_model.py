"""C204 with a fixed instantaneous raw4 discrepancy correction.

The four recurrent states, neural weights and source gains are unchanged.
``raw`` means the corrected pre-residual estimates; ``native_raw`` retains
the original neural estimates. Spectral and waveform components are native.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch

from research.direct.latency58 import Latency58Output
from research.direct.latency58_asymmetric import Latency58AsymmetricModel
from research.direct.latency58_evaluate import model_state_sha256
from research.direct.run_latency58_quality import require, sha

VERSION = "latency58-c204-fixed-residual-sixteenth-v1"
BASE_STATE = "c204b0fcb9627ca7fecd287db42fb869a1ae6783a1bc24cf2d8864c3b4a565fb"
SHARE = 1 / 16


def corrected_estimates(raw, delayed_mixture, share):
    # Match the reference NumPy source-axis reduction order explicitly.
    discrepancy = delayed_mixture - (((raw[:, 0] + raw[:, 1]) + raw[:, 2]) + raw[:, 3])
    retained = raw[:, :3] + share * discrepancy.unsqueeze(1)
    corrected_raw = torch.cat((retained, raw[:, 3:4]), dim=1)
    other = delayed_mixture - ((retained[:, 0] + retained[:, 1]) + retained[:, 2])
    return corrected_raw, torch.cat((retained, other.unsqueeze(1)), dim=1)


@dataclass(frozen=True)
class ResidualShareOutput(Latency58Output):
    native_raw: torch.Tensor


class Latency58ResidualModel(Latency58AsymmetricModel):
    def __init__(self):
        super().__init__()
        self.register_buffer("fixed_residual_share", torch.tensor(SHARE, dtype=torch.float32))

    @property
    def architecture_metadata(self):
        return {**super().architecture_metadata,
                "output_policy_version": VERSION,
                "raw_output_semantics": "native DBV plus fixed discrepancy correction; native Other",
                "component_semantics": "spectral and waveform are before discrepancy correction",
                "fixed_residual_share": SHARE, "extra_stream_state_tensors": 0,
                "extra_audio_buffering_samples": 0}

    def _render_fp32(self, audio, state):
        native = super()._render_fp32(audio, state)
        raw, deployed = corrected_estimates(native.raw, native.delayed_mixture, self.fixed_residual_share)
        return ResidualShareOutput(raw, deployed, native.spectral, native.waveform,
                                   native.delayed_mixture, native.state, native.raw)

    @classmethod
    def from_c204(cls, parent):
        require(type(parent) is Latency58AsymmetricModel and model_state_sha256(parent) == BASE_STATE,
                "Correction requires the preserved C204 model")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = cls()
        values = dict(parent.state_dict(), fixed_residual_share=torch.tensor(SHARE, dtype=torch.float32))
        model.load_state_dict(values, strict=True)
        model.provenance = {**copy.deepcopy(parent.provenance),
                            "output_policy_version": VERSION, "fixed_residual_share": SHARE,
                            "neural_parent_model_state_sha256": BASE_STATE,
                            "additional_training_updates": 0,
                            "coefficient_selection": "Unchanged 1/16 from the earlier fixed experiment"}
        return model.eval().requires_grad_(False)


def load_checkpoint(path, expected_sha256):
    """Load a self-contained inference checkpoint without its training lineage."""
    path = Path(path)
    require(sha(path) == expected_sha256, "Residual checkpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["schema"] == VERSION and payload["source_bindings"][str(Path(__file__).resolve())] == sha(__file__),
            "Residual checkpoint schema or loader changed")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = Latency58ResidualModel()
    require(payload["architecture"] == model.architecture_metadata, "Different residual model architecture")
    model.load_state_dict(payload["model"], strict=True)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values())
            and model.fixed_residual_share.item() == SHARE
            and model_state_sha256(model) == payload["model_state_sha256"], "Residual model tensors changed")
    from research.direct.train_latency58 import state_sha256
    require(state_sha256({k: v for k, v in model.state_dict().items() if k != "fixed_residual_share"}) == BASE_STATE,
            "Residual checkpoint changed C204 neural tensors")
    model.provenance = payload["provenance"]
    require(model.provenance["neural_parent_model_state_sha256"] == BASE_STATE
            and model.provenance["output_policy_version"] == VERSION
            and model.provenance["additional_training_updates"] == 0, "Invalid residual provenance")
    return model.eval().requires_grad_(False), payload
