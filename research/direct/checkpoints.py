"""Load trainable C91 weights and export its fixed deployment calibration."""

import copy
import pickle
import tempfile
from pathlib import Path

import torch

from hs_tasnet.hs_tasnet import HSTasNet


DEPLOYMENT_GAINS = (1.0, 1.0, 0.8, 1.12)


def load_model(path, device="cpu", raw=False) -> HSTasNet:
    """Load a local checkpoint; ``raw`` removes calibration and decoder baking.

    Deployment checkpoints do not contain optimizer state. At Hann's zero
    endpoint the original decoder coefficient is unrecoverable but has no
    effect on the model output, so its trainable replacement is zero.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = pickle.loads(payload["config"])
    state = payload["model"]
    if raw:
        if config.get("decoder_hann_baked", False):
            window = state["conv_decode.window"]
            nonzero = window != 0
            weight = state["conv_decode.weight"]
            restored = torch.zeros_like(weight)
            restored[..., nonzero] = weight[..., nonzero] / window[nonzero]
            state["conv_decode.weight"] = restored
        config["decoder_hann_baked"] = False
        state["output_source_scales"].fill_(0.5)
    model = HSTasNet(**config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def save_deployment(model: HSTasNet, path) -> None:
    """Save calibrated, Hann-baked CPU weights without mutating the caller."""
    deployment = copy.deepcopy(model).cpu().eval()
    deployment.set_output_source_gains(DEPLOYMENT_GAINS)
    if not deployment.conv_decode.hann_window_baked:
        deployment.bake_decoder_hann_window_()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save({"model": deployment.state_dict(), "config": deployment._config}, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
