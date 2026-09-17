"""Explicit model identity for the prospective local-mask training arm."""
from __future__ import annotations

import copy

import torch

from research.direct.latency58_asymmetric import Latency58AsymmetricModel
from research.direct.latency58_local_mask_mixer import SpectralHeadWithLocalMixer, VERSION as MIXER_VERSION
from research.direct.run_latency58_quality import require
from research.direct.train_latency58 import state_sha256

VERSION = "latency58-asymmetric-local-mask-mixer-model-v1"


def parent_parameter_name(name):
    prefix = "to_spec_masks.head."
    return "to_spec_masks." + name[len(prefix):] if name.startswith(prefix) else name


class LocalMaskMixerModel(Latency58AsymmetricModel):
    def __init__(self):
        super().__init__()
        self.to_spec_masks = SpectralHeadWithLocalMixer(self.to_spec_masks)

    @property
    def architecture_metadata(self):
        value = super().architecture_metadata
        value.update(temporal_core_version=value["version"], version=VERSION,
                     spectral_mask_mixer={"version": MIXER_VERSION, "hidden_channels": 16,
                         "input_channels_per_stereo_side": 8, "frequency_kernel": 3,
                         "output_kernel": 1, "activation": "silu", "added_parameters": 536,
                         "stereo_weights_shared": True, "frames_independent": True,
                         "added_state_tensors": 0, "added_lookahead_samples": 0})
        return value

    @classmethod
    def from_parent(cls, parent, *, initialization_seed):
        """Copy every original tensor exactly; isolate new initialization RNG."""
        require(type(initialization_seed) is int and isinstance(parent, Latency58AsymmetricModel)
                and isinstance(parent.to_spec_masks, torch.nn.Linear)
                and all(value.device.type == "cpu" and value.dtype == torch.float32
                        for value in parent.state_dict().values())
                and len(list(parent.parameters())) == 21 and len(list(parent.buffers())) == 6
                and not torch.cuda.is_initialized(), "Require an unmodified CPU preparation parent")
        fingerprint = state_sha256(parent.state_dict())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            model = cls()
        original = parent.state_dict()
        initialized = model.state_dict()
        for name in initialized:
            if not name.startswith("to_spec_masks.mixer."):
                initialized[name] = original[parent_parameter_name(name)]
        model.load_state_dict(initialized, strict=True)
        inherited = {parent_parameter_name(name): value for name, value in model.state_dict().items()
                     if not name.startswith("to_spec_masks.mixer.")}
        require(state_sha256(inherited) == fingerprint == state_sha256(parent.state_dict())
                and len(list(model.parameters())) == 25 and len(list(model.buffers())) == 6
                and sum(p.numel() for p in model.to_spec_masks.mixer.parameters()) == 536
                and torch.count_nonzero(model.to_spec_masks.mixer.out.weight) == 0
                and torch.count_nonzero(model.to_spec_masks.mixer.out.bias) == 0,
                "Mixer initialization changed original tensors, size or zero-output contract")
        model.provenance = copy.deepcopy(parent.provenance)
        model.provenance.update(model_extension=VERSION,
                                extension_initialization_seed=initialization_seed,
                                extension_parent_model_state_sha256=fingerprint,
                                extension_training_updates=0)
        return model
