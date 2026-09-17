"""Source preparation for the untied full513/cropped512 family's ONNX graph.

Import is stdlib-only: no numerical imports, model construction, checkpoint
loading, export, verification, host work or artifact creation occurs on import.
The CLI requires --plan and --plan-sha256. ROOT must separately review the
selected checkpoint's quality evidence and the concrete execution plan before
any export. This source's presence is not export or native-host qualification.

Future plan contract (no plan or checkpoint is supplied by this source):
  schema: cropped1024-onnx-export-plan-v1
  release: root_reviewed_selected_cropped1024_cpu_export
  exporter_source, model_source, reusable_exporter_source, base_model_source,
  export_helpers_source, interpreter: exact {path, sha256} file bindings
  snapshot: {checkpoint: binding, model_state_sha256: digest,
             training_plan: binding, step: 2000 or 250,
             training_updates: 2000 or 2250, provenance: exact saved dictionary}
  reviewed_quality_evidence: nonempty list of actual file bindings
  output: fresh absolute .onnx path
  runtime_versions: exact torch/numpy/onnx/onnxruntime versions; also soundfile
                    when verify_audio is nonempty
  environment: exact CUDA-hidden CPU1 ENVIRONMENT below
  threads: 1; verify_hops: integer >= 8; verify_audio: list of file bindings

The graph uses five inputs/outputs: audio plus four family-specific states.
Its delay is 256 samples. The additional 256-sample host queue remains absent;
512 samples is the intended graph-plus-queue latency, not a qualified host PDC.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
FAMILY = 'ola-cropped1024-hann512-hop256-v1'
INITIALIZATION = 'authenticated_c91_full513_right_baked512_cropped1024_ola'
STATE_NAMES = ('audio_history', 'fusion_hidden', 'spectral_numerator_tail', 'waveform_tail')
STATE_SHAPES = ((1, 2, 768), (2, 1, 1000), (1, 4, 2, 256), (1, 4, 2, 256))
INPUT_NAMES = ('audio_chunk', *STATE_NAMES)
OUTPUT_NAMES = ('separated_chunk', *('next_' + name for name in STATE_NAMES))
INPUT_SHAPES = ((1, 2, 256), *STATE_SHAPES)
OUTPUT_SHAPES = ((1, 4, 2, 256), *STATE_SHAPES)
NATIVE_SOURCE_SCALES = (0.5, 0.5, 0.45, 0.56)
ENVIRONMENT = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'}
FROZEN_SOURCES = {
    'model_source': (ROOT / 'research/direct/runs/latency11/smoke/cropped1024-ola-prep/cropped1024_ola.py',
        '5d5359e1b25749a4d84db0a30671154378bbf4c91e0b2efaae50894b676d7b27'),
    'reusable_exporter_source': (ROOT / 'research/direct/latency_ola512_onnx.py',
        '1ae71047f07d143cb1513833cce124322f7a33d16301b309dc7876bd43300235'),
    'base_model_source': (ROOT / 'research/direct/latency_ola512.py',
        '90cbc93a3ab6c5e37b39442c066e82de7d1012b83b570b55acf57d026f775368'),
}
NUMERICAL_MODULES = ('torch', 'numpy', 'onnx', 'onnxruntime', 'soundfile')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Require an actual lowercase SHA256')
    return value


def metadata(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}
            and isinstance(value['path'], str) and Path(value['path']).is_absolute(), 'Require an absolute file binding')
    digest(value['sha256'])
    return Path(value['path'])


def bound(meta, bindings):
    path = metadata(meta)
    require(sha256(path) == meta['sha256'], 'Bound file changed: ' + str(path))
    require(str(path) not in bindings or bindings[str(path)] == meta['sha256'], 'Conflicting input identities')
    bindings[str(path)] = meta['sha256']
    return path


def artifact_paths(output):
    return {'graph': output, 'verification': output.with_suffix('.verification.json'),
            'failed_verification': output.with_suffix('.failed-verification.json'),
            'failed_graph': output.with_suffix('.failed.onnx'), 'hash': output.with_suffix('.onnx.sha256')}


def preflight(args):
    """Only stdlib checks of actual reviewed inputs; no directories are created."""
    require(not sys.flags.optimize and args.plan.is_absolute()
            and all(name not in sys.modules for name in NUMERICAL_MODULES), 'Use fresh ordinary Python and an absolute plan')
    sys.dont_write_bytecode = True
    bindings = {}
    plan_meta = {'path': str(args.plan), 'sha256': args.plan_sha256}
    plan = json.loads(bound(plan_meta, bindings).read_text())
    require(set(plan) == {'schema', 'release', 'exporter_source', 'model_source', 'reusable_exporter_source',
                         'base_model_source', 'export_helpers_source', 'interpreter', 'snapshot',
                         'reviewed_quality_evidence', 'output', 'runtime_versions', 'environment',
                         'threads', 'verify_hops', 'verify_audio'}
            and plan['schema'] == 'cropped1024-onnx-export-plan-v1'
            and plan['release'] == 'root_reviewed_selected_cropped1024_cpu_export', 'A separately reviewed export plan is required')
    require(bound(plan['exporter_source'], bindings) == Path(__file__).resolve(), 'Reviewed exporter source differs')
    for name, (path, expected) in FROZEN_SOURCES.items():
        require(plan[name] == {'path': str(path), 'sha256': expected}, 'Frozen source identity differs: ' + name)
        bound(plan[name], bindings)
    require(bound(plan['export_helpers_source'], bindings) == ROOT / 'export_onnx.py', 'Reviewed helper path differs')
    bound(plan['interpreter'], bindings)
    require(Path.cwd() == ROOT and Path(sys.executable).absolute() == Path(plan['interpreter']['path'])
            and plan['environment'] == ENVIRONMENT
            and all(os.environ.get(name) == value for name, value in ENVIRONMENT.items())
            and type(plan['threads']) is int and plan['threads'] == 1,
            'Use the exact reviewed interpreter/cwd and CUDA-hidden CPU1 environment')
    snapshot = plan['snapshot']
    require(set(snapshot) == {'checkpoint', 'model_state_sha256', 'training_plan', 'step', 'training_updates', 'provenance'}
            and type(snapshot['step']) is int and type(snapshot['training_updates']) is int
            and (snapshot['step'], snapshot['training_updates']) in ((2000, 2000), (250, 2250))
            and isinstance(snapshot['provenance'], dict), 'Expected parent2000 or local250/total2250 snapshot identity')
    digest(snapshot['model_state_sha256'])
    bound(snapshot['checkpoint'], bindings)  # Authentication of raw bytes only.
    bound(snapshot['training_plan'], bindings)
    require(isinstance(plan['reviewed_quality_evidence'], list) and plan['reviewed_quality_evidence'],
            'ROOT must bind the actual reviewed quality evidence before export')
    for meta in plan['reviewed_quality_evidence']:
        bound(meta, bindings)
    require(type(plan['verify_hops']) is int and plan['verify_hops'] >= 8
            and isinstance(plan['verify_audio'], list), 'Reviewed verification geometry differs')
    for meta in plan['verify_audio']:
        bound(meta, bindings)
    versions = set(NUMERICAL_MODULES if plan['verify_audio'] else NUMERICAL_MODULES[:-1])
    require(isinstance(plan['runtime_versions'], dict) and set(plan['runtime_versions']) == versions
            and all(isinstance(value, str) and value for value in plan['runtime_versions'].values()),
            'Pin every numerical package version used by this execution')
    output = Path(plan['output'])
    require(output.is_absolute() and output.suffix == '.onnx', 'Require an absolute .onnx output path')
    for path in artifact_paths(output).values():
        require(not path.exists() and not path.is_symlink() and str(path) not in bindings,
                'Preserve existing/input artifacts: ' + str(path))
    return plan, plan_meta, bindings, output


def load_source(meta, name):
    spec = importlib.util.spec_from_file_location(name, meta['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime(plan):
    """Called only by the reviewed CLI after the complete stdlib preflight."""
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort

    versions = {'numpy': np.__version__, 'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__}
    if plan['verify_audio']:
        import soundfile as sf
        versions['soundfile'] = sf.__version__
    require(versions == plan['runtime_versions'] and torch.get_default_dtype() == torch.float32
            and not torch.cuda.is_initialized(), 'Reviewed CPU runtime/precision differs')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reusable = load_source(plan['reusable_exporter_source'], 'cropped1024_frozen_ola_export_support')
    family = load_source(plan['model_source'], 'cropped1024_frozen_family_for_export')
    # The frozen cropped family deliberately imports OLA512State and inherits
    # these structural methods; it does not define a Cropped1024State class.
    # The verifier constructs fresh synthetic states in that exact container.
    # Shared structure does not authorize loading another family's saved states.
    require(family.OLA512State is reusable.ola.OLA512State
            and all(getattr(family.Cropped1024OLAModel, name) is getattr(reusable.ola.OLA512Model, name)
                    for name in ('initial_state', '_validate', 'forward_chunk', 'flush')),
            'Frozen cropped family must use the verifier\'s exact structural state container')
    require(Path(reusable.ola.__file__).resolve() == Path(plan['base_model_source']['path'])
            and Path(reusable.export_helpers.__file__).resolve() == Path(plan['export_helpers_source']['path'])
            and family.VERSION == FAMILY and family.INITIALIZATION == INITIALIZATION
            and reusable.STATE_NAMES == STATE_NAMES and reusable.STATE_SHAPES == STATE_SHAPES
            and reusable.INPUT_NAMES == INPUT_NAMES and reusable.OUTPUT_NAMES == OUTPUT_NAMES
            and reusable.INPUT_SHAPES == INPUT_SHAPES and reusable.OUTPUT_SHAPES == OUTPUT_SHAPES,
            'Loaded source/family or reusable verification ABI differs')
    return torch, onnx, reusable, family


def load_checkpoint(plan, torch, reusable, family):
    """Authenticate a saved same-family snapshot; never construct a C91 initializer."""
    expected = plan['snapshot']
    checkpoint = expected['checkpoint']
    require(sha256(checkpoint['path']) == checkpoint['sha256'], 'Checkpoint changed before loading')
    payload = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
    require(isinstance(payload, dict) and set(payload) == {
        'schema', 'step', 'model', 'model_state_sha256', 'provenance', 'architecture', 'plan_sha256'}
        and payload['schema'] == 'cropped1024-ola-inference-v1'
        and type(payload['step']) is int and payload['step'] == expected['step']
        and payload['model_state_sha256'] == expected['model_state_sha256']
        and payload['plan_sha256'] == expected['training_plan']['sha256']
        and payload['provenance'] == expected['provenance'], 'Saved snapshot schema/step/provenance/identity differs')
    provenance = payload['provenance']
    require(provenance.get('version') == FAMILY and provenance.get('initialization') == INITIALIZATION
            and provenance.get('reference_sha256') == reusable.ola.REFINED_C91_SHA256
            and provenance.get('training_updates') == expected['training_updates']
            and provenance.get('training_plan_sha256') == expected['training_plan']['sha256']
            and provenance.get('reference_model_modified') is False and provenance.get('trained_ola_state_used') is False
            and provenance.get('equivalence_claimed') is False, 'Cropped family/native initializer lineage differs')
    if expected['step'] == 250:
        require(provenance.get('parent_training_updates') == 2000 and provenance.get('pilot_updates') == 250
                and isinstance(provenance.get('training_objective'), str) and provenance['training_objective'],
                'Local pilot and total update counts must remain distinct')
        metadata(provenance['parent_checkpoint'])
        metadata(provenance['parent_training_plan'])
        digest(provenance['parent_model_state_sha256'])
    else:
        require(provenance.get('pilot_updates', 0) == 0, 'Parent checkpoint must not masquerade as a pilot')
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        model = family.Cropped1024OLAModel()
    require(payload['architecture'] == model.architecture_metadata, 'Saved cropped architecture differs')
    fixed = {name: value.clone() for name, value in model.named_buffers() if name != 'output_source_scales'}
    values = payload['model']
    shapes = {**family.PARAMETER_SHAPES, **family.BUFFER_SHAPES}
    require(isinstance(values, dict) and set(values) == set(shapes)
            and all(isinstance(value, torch.Tensor) and tuple(value.shape) == shapes[name]
                    and value.device.type == 'cpu' and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                    for name, value in values.items()), 'Require the exact finite CPU FP32 cropped tensor inventory')
    model.load_state_dict(values, strict=True)
    model.provenance = copy.deepcopy(provenance)
    model.eval().requires_grad_(False)
    require(len(tuple(model.parameters())) == 21 and len(tuple(model.buffers())) == 5 and len(model.state_dict()) == 26
            and all(torch.equal(value, fixed[name]) for name, value in model.named_buffers() if name in fixed)
            and torch.equal(model.output_source_scales, torch.tensor(NATIVE_SOURCE_SCALES, dtype=torch.float32))
            and reusable.state_sha256(model) == expected['model_state_sha256'], 'Loaded tensor identity or fixed native buffers differ')
    # The persistent denominator stays untouched; check the actual copied window formula.
    denominator = (model.analysis_window[512:768] * model.synthesis.window[:256]
                   + model.analysis_window[768:1024] * model.synthesis.window[256:])
    require(torch.equal(model.synthesis.spectral_denominator, denominator) and bool((denominator > 0).all())
            and sha256(checkpoint['path']) == checkpoint['sha256'], 'Original cropped denominator or checkpoint bytes changed')
    identity = {'family': FAMILY, 'state_family': FAMILY, 'checkpoint': checkpoint,
                'model_state_sha256': expected['model_state_sha256'], 'snapshot_step': expected['step'],
                'training_updates': expected['training_updates'], 'training_plan': expected['training_plan'],
                'native_output_source_scales': model.output_source_scales.tolist(),
                'architecture': payload['architecture'], 'provenance': provenance}
    return model, identity


def make_export_copy(model, torch, reusable):
    """Build only a separate export copy; the complete deployed residual is formed once."""
    class IRFFT1024(torch.autograd.Function):
        @staticmethod
        def forward(ctx, spectrum):
            value = torch.complex(spectrum[..., 0], spectrum[..., 1])
            return torch.fft.irfft(value, n=1024, dim=1)

        @staticmethod
        def symbolic(graph, spectrum):
            endpoints = torch.ones(513, 2, dtype=torch.float32)
            endpoints[0, 1] = 0.0
            endpoints[-1, 1] = 0.0
            values = graph.op('Mul', spectrum, graph.op('Constant', value_t=endpoints))
            indices = graph.op('Constant', value_t=torch.arange(511, 0, -1, dtype=torch.int64))
            reflected = graph.op('Gather', values, indices, axis_i=1)
            sign = graph.op('Constant', value_t=torch.tensor([1.0, -1.0], dtype=torch.float32))
            reflected = graph.op('Mul', reflected, sign)
            full = graph.op('Concat', values, reflected, axis_i=1)
            length = graph.op('Constant', value_t=torch.tensor(1024, dtype=torch.int64))
            inverse = graph.op('DFT', full, length, axis_i=1, inverse_i=1, onesided_i=0)
            real_index = graph.op('Constant', value_t=torch.tensor(0, dtype=torch.int64))
            result = graph.op('Gather', inverse, real_index, axis_i=2)
            return result.setType(spectrum.type().with_sizes([spectrum.type().sizes()[0], 1024]))

    class Cropped1024StreamingWrapper(torch.nn.Module):
        def __init__(self, copied):
            super().__init__()
            self.model = copied

        def forward(self, audio_chunk, audio_history, fusion_hidden, spectral_numerator_tail, waveform_tail):
            copied = self.model
            joined = torch.cat((audio_history, audio_chunk), dim=-1)
            feature_ri = reusable._RFFT1024.apply((joined * copied.analysis_window).reshape(2, 1024))
            feature_ri = feature_ri.reshape(1, 2, 1, 513, 2)
            packed = feature_ri.permute(0, 2, 1, 3, 4).reshape(1, 1, 2052)
            spec = copied.spec_encode(packed)
            to_relu, to_sigmoid = copied.conv_encode(joined).chunk(2, dim=1)
            basis = to_relu.relu() * to_sigmoid.sigmoid()
            waveform = copied.basis_to_embed(basis).transpose(1, 2)
            fusion_input = torch.cat((spec, waveform), dim=-1)
            recurrent, next_physical_hidden = copied.fusion_branch(
                fusion_input, fusion_hidden / reusable.ola.PUBLIC_FUSION_SCALE)
            fused = fusion_input + recurrent
            fused_spec, fused_waveform = fused.chunk(2, dim=-1)
            spec = fused_spec + spec
            waveform = fused_waveform + waveform

            logits = copied.to_spec_masks(copied.spec_norm(spec)).reshape(1, 1, 2, 513, 2, 4)
            masks = copied._residual_source_softmax(logits).permute(0, 2, 1, 3, 4, 5)
            masked_ri = feature_ri.unsqueeze(-1) * masks
            source_ri = masked_ri.permute(0, 5, 1, 2, 3, 4).contiguous().reshape(8, 513, 2)
            frames = IRFFT1024.apply(source_ri).reshape(1, 4, 2, 1024)[..., 512:1024]
            frames = frames * copied.synthesis.window
            spectral = (frames[..., :256] + spectral_numerator_tail) / copied.synthesis.spectral_denominator
            next_spectral_tail = frames[..., 256:].clone()

            waveform_logits = copied.to_waveform_masks(copied.waveform_norm(waveform))
            waveform_logits = waveform_logits.reshape(1, 1, 4, 1500).transpose(-1, -2)
            waveform_masks = copied._residual_source_softmax(waveform_logits)
            source_basis = basis.transpose(1, 2).unsqueeze(-1) * waveform_masks
            source_basis = source_basis.permute(0, 3, 1, 2)
            decoded = torch.nn.functional.linear(source_basis, copied.waveform_decoder_weight.flatten(1).t(), bias=None)
            decoded = decoded.reshape(1, 4, 1, 2, 512).permute(0, 1, 3, 2, 4)
            windowed = (decoded * copied.synthesis.window).reshape(1, 4, 2, 512)
            waveform_audio = windowed[..., :256] + waveform_tail
            next_waveform_tail = windowed[..., 256:].clone()

            raw = (spectral + waveform_audio) * copied.output_source_scales[None, :, None, None]
            delayed_mixture = audio_history[..., -256:]
            retained = raw[:, :3]
            deployed = torch.cat((retained, delayed_mixture.unsqueeze(1)
                                  - retained.sum(dim=1, keepdim=True)), dim=1)
            return (deployed, joined[..., -768:].clone(),
                    next_physical_hidden * reusable.ola.PUBLIC_FUSION_SCALE,
                    next_spectral_tail, next_waveform_tail)

    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        copied = copy.deepcopy(model).cpu().eval()
        copied.fusion_branch = reusable.export_helpers.OneFrameGRUForONNX(copied.fusion_branch)
        reusable.export_helpers.replace_rmsnorm_layers(copied)
        wrapper = Cropped1024StreamingWrapper(copied).eval()
    require(reusable.state_sha256(copied) == reusable.state_sha256(model), 'Export copy changed checkpoint tensor bytes')
    return wrapper


def graph_metadata(plan, identity):
    return {
        'hs_tasnet.kind': 'cropped1024_ola', 'hs_tasnet.mode': 'streaming',
        'hs_tasnet.architecture_version': FAMILY, 'hs_tasnet.state_family': FAMILY,
        'hs_tasnet.state_interchangeable_with_old_ola512': 'false',
        'hs_tasnet.sample_rate': '44100', 'hs_tasnet.hop_samples': '256',
        'hs_tasnet.analysis_fft_samples': '1024', 'hs_tasnet.carrier_fft_samples': '1024',
        'hs_tasnet.synthesis_fft_samples': '1024', 'hs_tasnet.spectral_mask_bins': '513',
        'hs_tasnet.spectral_output_crop': '[512,1024]', 'hs_tasnet.synthesis_frame_samples': '512',
        'hs_tasnet.waveform_decoder_samples': '512', 'hs_tasnet.analysis_history_samples': '768',
        'hs_tasnet.graph_output_delay_samples': '256', 'hs_tasnet.alignment_samples': '256',
        'hs_tasnet.future_context_samples': '256', 'hs_tasnet.future_callbacks_beyond_received_input': '0',
        'hs_tasnet.flush_required': 'true', 'hs_tasnet.flush_hops': '1',
        'hs_tasnet.initial_state': 'all_zeros', 'hs_tasnet.preroll': 'discard_first_output_hop_after_reset',
        'hs_tasnet.external_host_queue_implemented': 'false',
        'hs_tasnet.intended_external_host_queue_samples': '256', 'hs_tasnet.intended_total_latency_samples': '512',
        'hs_tasnet.graph_qualified': 'false', 'hs_tasnet.native_host_timing_qualified': 'false',
        'hs_tasnet.output_policy': 'complete deployed four stems; Other = previous physical mixture - sum(unchanged DBV), once',
        'hs_tasnet.source_order': 'drums,bass,vocals,other',
        'hs_tasnet.state_names': json.dumps(STATE_NAMES), 'hs_tasnet.state_shapes': json.dumps(STATE_SHAPES),
        'hs_tasnet.public_fusion_state_scale': str(2.0 ** -18),
        'hs_tasnet.output_source_scales': json.dumps(identity['native_output_source_scales']),
        'hs_tasnet.checkpoint_sha256': identity['checkpoint']['sha256'],
        'hs_tasnet.model_state_sha256': identity['model_state_sha256'],
        'hs_tasnet.training_plan_sha256': identity['training_plan']['sha256'],
        'hs_tasnet.snapshot_step': str(identity['snapshot_step']),
        'hs_tasnet.training_updates': str(identity['training_updates']),
        'hs_tasnet.exporter_sha256': plan['exporter_source']['sha256'],
        'hs_tasnet.model_source_sha256': plan['model_source']['sha256'],
        'hs_tasnet.base_model_source_sha256': plan['base_model_source']['sha256'],
        'hs_tasnet.reusable_exporter_sha256': plan['reusable_exporter_source']['sha256'],
        'hs_tasnet.export_helpers_sha256': plan['export_helpers_source']['sha256'],
        'hs_tasnet.fft_implementation': 'DFT1024; explicit513 Hermitian endpoints/interior; inverse1024 then crop512:1024',
        'hs_tasnet.external_data': 'false',
    }


def write_new_json(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with path.open('x') as stream:
        stream.write(encoded)


def execute_reviewed(plan, plan_meta, bindings, output):
    """Future controlled execution only. This function is never called on import."""
    torch, onnx, reusable, family = load_runtime(plan)
    model, identity = load_checkpoint(plan, torch, reusable, family)
    wrapper = make_export_copy(model, torch, reusable)
    states = model.initial_state(1, device='cpu')
    require(tuple(tuple(value.shape) for value in states) == STATE_SHAPES, 'Cropped state ABI differs')
    args = (torch.zeros(INPUT_SHAPES[0], dtype=torch.float32, device='cpu'), *states)
    props = graph_metadata(plan, identity)
    paths = artifact_paths(output)
    for path in paths.values():
        require(not path.exists() and not path.is_symlink(), 'Preserve existing output artifacts')
    receipt = {'schema': 'cropped1024-cpu-onnx-verification-v1', 'status': 'in_progress',
        'export_plan': plan_meta, 'identity': identity, 'output': str(output), 'metadata': props,
        'reviewed_quality_evidence': plan['reviewed_quality_evidence'], 'source_bindings_before': bindings,
        'graph_qualified': False, 'native_host_qualified': False, 'native_host_timing_qualified': False,
        'external_host_queue_implemented': False, 'quality_retention_decision': None,
        'training_updates_executed': 0, 'completed_process_exit_not_claimed': True}
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=output.stem + '.', suffix='.onnx', dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            examples = wrapper(*args)
            torch.onnx.export(wrapper, args, str(temporary), export_params=True, opset_version=17,
                do_constant_folding=True, input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                dynamo=False, external_data=False)
        graph = onnx.load(str(temporary), load_external_data=False)
        require(tuple(value.name for value in graph.graph.output) == OUTPUT_NAMES
                and not any(value.data_location == onnx.TensorProto.EXTERNAL for value in graph.graph.initializer),
                'Graph output order or self-contained storage differs')
        for value, example, expected in zip(graph.graph.output, examples, OUTPUT_SHAPES, strict=True):
            require(tuple(example.shape) == expected, 'Export-copy output shape differs')
            shape = value.type.tensor_type.shape
            shape.ClearField('dim')
            for size in expected:
                shape.dim.add().dim_value = size
        onnx.helper.set_model_props(graph, props)
        onnx.save(graph, str(temporary))
        onnx.checker.check_model(str(temporary), full_check=True)
        receipt['onnx_sha256'] = sha256(temporary)
        receipt['onnx_bytes'] = temporary.stat().st_size
        # The unchanged verifier maintains independent native/copy/ORT states,
        # checks each callback, physical-history closure, single flush and replay.
        receipt['verification'] = reusable.verify_onnx(model, wrapper, temporary, hops=plan['verify_hops'],
            audio_paths=tuple(meta['path'] for meta in plan['verify_audio']), threads=plan['threads'])
        after = {path: sha256(path) for path in bindings}
        receipt['source_bindings_after'] = after
        receipt['bound_inputs_verified_unchanged'] = after == bindings
        require(after == bindings and reusable.state_sha256(model) == reusable.state_sha256(wrapper.model)
                    == identity['model_state_sha256'] and not torch.cuda.is_initialized(),
                'Export/verification changed inputs, model tensors or CPU scope')
        require(receipt['verification']['passed'], 'Independent three-way CPU graph parity failed')
        receipt.update(status='passed_cpu_numerical_verification_only', elapsed_seconds=time.monotonic() - started)
        os.link(temporary, paths['graph'])
        write_new_json(paths['verification'], receipt)
        with paths['hash'].open('x') as stream:
            stream.write(f"{receipt['onnx_sha256']}  {output.name}\n")
        return receipt
    except Exception as error:
        receipt.update(status='failed_not_qualified', error={'type': type(error).__name__, 'message': str(error)},
                       elapsed_seconds=time.monotonic() - started)
        if temporary.is_file() and temporary.stat().st_size:
            os.link(temporary, paths['failed_graph'])
            receipt['failed_graph'] = {'path': str(paths['failed_graph']), 'sha256': sha256(temporary)}
        write_new_json(paths['failed_verification'], receipt)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    receipt = execute_reviewed(*preflight(args))
    print(json.dumps({'event': 'cropped1024_onnx_cpu_verification_completed', 'status': receipt['status'],
        'output': receipt['output'], 'onnx_sha256': receipt['onnx_sha256'],
        'verification': {'path': str(Path(receipt['output']).with_suffix('.verification.json')),
                         'sha256': sha256(Path(receipt['output']).with_suffix('.verification.json'))}}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
