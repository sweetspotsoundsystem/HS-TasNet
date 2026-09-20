"""Share value-identical BF16 saved GRU weights without changing forward math."""
from contextlib import contextmanager
import torch
from ..losses import _require as require


class SharedGRUSavedWeights:
    def __init__(self, model):
        self.candidates = {}
        self.matched = {}
        self.unmatched = 0
        self.shared_logical_bytes = 0
        for prefix, module in model.named_modules():
            if type(module) is not torch.nn.GRU:
                continue
            for name, parameter in module.named_parameters(recurse=False):
                if parameter.ndim != 2:
                    continue
                require(parameter.is_cuda and parameter.dtype == torch.float32, 'Require CUDA FP32 GRU parameters')
                key = (prefix + '.' + name).lstrip('.')
                value = parameter.detach().t().to(torch.bfloat16)
                signature = (value.device, value.dtype, tuple(value.shape), tuple(value.stride()))
                self.candidates.setdefault(signature, []).append((key, value, parameter, parameter._version))
                self.matched[key] = 0
        require(self.matched, 'No CUDA GRU matrices to share')

    def pack(self, tensor):
        signature = (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))
        for name, value, parameter, version in self.candidates.get(signature, ()):
            require(parameter._version == version, 'GRU parameter changed inside a saved-weight scope')
            if torch.equal(tensor, value):
                self.matched[name] += 1
                self.shared_logical_bytes += tensor.numel() * tensor.element_size()
                return (value, value._version, parameter, version)
        self.unmatched += 1
        return (tensor.detach(), tensor._version, None, None)

    @staticmethod
    def unpack(packed):
        value, version, parameter, parameter_version = packed
        require(value._version == version, 'Saved tensor changed before backward')
        require(parameter is None or parameter._version == parameter_version,
                'GRU parameter changed before backward')
        return value

    def report(self):
        return {'matched_parameter_transpose_counts': dict(self.matched), 'unmatched_save_count': self.unmatched,
                'shared_logical_bytes': self.shared_logical_bytes,
                'every_shared_value_compared_exactly': True, 'forward_and_backward_operations_unchanged': True,
                'saved_value_and_parameter_versions_checked': True}


@contextmanager
def share_saved_gru_weights(model):
    sharing = SharedGRUSavedWeights(model)
    with torch.autograd.graph.saved_tensors_hooks(sharing.pack, sharing.unpack):
        yield sharing
