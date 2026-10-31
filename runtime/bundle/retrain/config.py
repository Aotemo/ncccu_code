"""Validated configuration. Paths may move; scientific resume contracts may not."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re

BUNDLE = Path(__file__).resolve().parents[1]
MODELS = ('efficientnet_v2_s', 'convnext_tiny', 'dinov2_vits14', 'flair_resnet50', 'resnet50_imagenet', 'convnext_small', 'convnext_base', 'swin_base')
MODES = ('inspect', 'prepare', 'benchmark', 'train')


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value):
        raise ValueError('Run/job name must contain only letters, digits, underscore or hyphen')


def _unknown(obj, allowed, where):
    extra = set(obj) - set(allowed)
    if extra:
        raise ValueError(f'Unknown {where} keys: {sorted(extra)}')


def _number(value, key, minimum=0, integer=False, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{key} must be a finite number')
    if integer and not isinstance(value, int):
        raise ValueError(f'{key} must be an integer')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{key} outside supported range')


def default_job(name, model, kind='random'):
    pretrained = kind == 'imagenet'
    job = dict(name=name, model=model, seed=20261025, device='cuda:0', image_size=384,
               physical_batch=32, effective_batch=64, positive_per_batch=8,
               workers=4, eval_workers=0, prefetch_factor=1, precision='fp16',
               cycles=24 if pretrained else 60, warmup_cycles=2 if pretrained else 5,
               min_cycles=8 if pretrained else 20, patience=6 if pretrained else 10,
               min_delta=.0005, backbone_lr=2e-4 if pretrained else 3e-4,
               head_lr=1e-3 if pretrained else 3e-4, weight_decay=1e-4,
               dropout=.2, stochastic_depth=.1, checkpoint_seconds=300,
               checkpoint_updates=0, log_every=25,
               normalization='imagenet' if pretrained else 'unit', prepared_size=0,
               final_norm_eps=1e-6,head_lr_scope='classifier_legacy',age=None,mixup=None,attention_pool=None,
               stop_after_cycles=0,
               initialization={'kind': kind}, initial_checkpoint=None,
               benchmark_warmup=50, benchmark_updates=200, split={}, frozen_bn=False, experiment91=None)
    if model == 'resnet50_imagenet':
        job.update(image_size=512, prepared_size=512, precision='fp32',
                   stochastic_depth=0., frozen_bn=True, head_lr_scope='last_linear')
    return job


def validate_job(job):
    from .classification_loss import PAIRWISE_LOSS_KINDS, validate_loss
    validate_loss(job.get('loss'))
    if job.get('low_rank_policy') is not None:
        from .low_rank import validate_job as validate_low_rank_job
        validate_low_rank_job(job)
    if 'final_downsample_stride' in job and (type(job['final_downsample_stride']) is not int
            or job['final_downsample_stride'] != 1 or job['model'] != 'convnext_small'):
        raise ValueError('final downsample stride supports only explicit Small stride1')
    if 'freeze_features_before' in job and (type(job['freeze_features_before']) is not int
            or job['freeze_features_before'] != 4 or job['model'] != 'convnext_small'):
        raise ValueError('feature freezing supports only explicit Small features[:4]')
    if (job.get('loss') or {}).get('kind') in PAIRWISE_LOSS_KINDS:
        if job.get('mixup') is not None or job.get('training_views', 1) != 1:
            raise ValueError('global pairwise logistic requires single-view training without mixup')
    if 'training_views' in job:
        if type(job['training_views']) is not int or job['training_views'] not in (1, 2):
            raise ValueError('training_views must be 1 or 2')
        if job['training_views'] == 2 and job.get('mixup') is not None:
            raise ValueError('two training views do not support mixup')
    if 'training_scale' in job:
        scale = job['training_scale']
        if not isinstance(scale, (list, tuple)) or len(scale) != 2:
            raise ValueError('training_scale must contain lower and upper scale bounds')
        for value in scale:
            _number(value, 'training_scale', maximum=1)
        if not 0 < scale[0] <= scale[1]:
            raise ValueError('training_scale requires 0 < lower <= upper <= 1')
    if 'training_photometric' in job and job['training_photometric'] not in (None, 'identity-v1'):
        raise ValueError('training_photometric must be None or identity-v1')
    if 'training_rotation' in job and job['training_rotation'] not in (None, 'identity-v1'):
        raise ValueError('training_rotation must be None or identity-v1')
    if job.get('experiment91') is not None:
        from .experiment91 import validate_job as validate91
        validate91(job)
    _name(job['name'])
    if job.get('attention_pool') is not None:
        from .attention_pool import validate_spec
        validate_spec(job['attention_pool'])
        if job['model'] not in ('convnext_tiny', 'convnext_small') or job['precision'] != 'fp32':
            raise ValueError('content attention pooling requires a registered FP32 768-wide model')
    if job.get('stage3_residual') is not None:
        from .stage3_residual import _validate_spec
        _validate_spec(job['stage3_residual'])
        if job['model'] != 'convnext_small' or job['precision'] != 'fp32' or job.get('attention_pool') is not None:
            raise ValueError('stage3 residual requires the registered FP32 Small final-GAP model')
    if job.get('gradient_clip_policy') is not None:
        from .gradient_clip import ROLEWISE_CLIP_RULE
        if job['gradient_clip_policy'] != ROLEWISE_CLIP_RULE:
            raise ValueError('gradient clip policy differs from the frozen rolewise protocol')
        if (job['model'] != 'convnext_small' or job['precision'] != 'fp32'
                or job.get('attention_pool') is not None or job.get('stage3_residual') is not None
                or not (job.get('age') or {}).get('enabled')):
            raise ValueError('rolewise clip requires the registered FP32 Small age model')
    if job.get('head_warmup_policy') is not None:
        from .head_warmup import HEAD_WARMUP_RULE
        if job['head_warmup_policy'] != HEAD_WARMUP_RULE:
            raise ValueError('head warmup policy differs from the frozen protocol')
        if (job['model'] != 'convnext_small' or job['precision'] != 'fp32'
                or job.get('gradient_clip_policy') is not None
                or job.get('attention_pool') is not None or job.get('stage3_residual') is not None
                or not (job.get('age') or {}).get('enabled')):
            raise ValueError('head warmup requires the registered FP32 Small age model')
    if job.get('layerwise_lr_policy') is not None:
        from .layerwise_lr import LAYERWISE_LR_RULE
        if job['layerwise_lr_policy'] != LAYERWISE_LR_RULE:
            raise ValueError('layerwise LR policy differs from the frozen protocol')
        if (job['model'] != 'convnext_small' or job['precision'] != 'fp32'
                or any(job.get(key) is not None for key in
                       ('head_warmup_policy', 'gradient_clip_policy', 'attention_pool',
                        'stage3_residual', 'training_rotation', 'training_photometric',
                        'training_views'))
                or not (job.get('age') or {}).get('enabled')):
            raise ValueError('layerwise LR requires the registered FP32 Small age model')
    if job.get('mixup') is not None:
        from .mixup import validate_spec
        validate_spec(job['mixup'])
    if job.get('age') is not None:
        from .age import validate_spec
        validate_spec(job['age'])
    if job['model'] not in MODELS:
        raise ValueError(f'Unsupported model: {job["model"]}')
    for key in ('physical_batch', 'effective_batch', 'positive_per_batch', 'cycles',
                'image_size', 'patience', 'prefetch_factor', 'log_every'):
        _number(job[key], key, 1, integer=True)
    for key in ('workers', 'eval_workers'):
        _number(job[key], key, 0, integer=True, maximum=12)
    for key in ('seed', 'warmup_cycles', 'min_cycles', 'checkpoint_updates',
                'benchmark_warmup', 'benchmark_updates'):
        _number(job[key], key, 0, integer=True)
    for key in ('backbone_lr', 'head_lr', 'checkpoint_seconds'):
        _number(job[key], key, 1e-12)
    for key in ('weight_decay', 'min_delta'):
        _number(job[key], key)
    for key in ('dropout', 'stochastic_depth'):
        _number(job[key], key, maximum=.999)
    if job['image_size'] < 32:
        raise ValueError('image_size must be >=32')
    _number(job['prepared_size'], 'prepared_size', 0, integer=True, maximum=8192)
    if job['prepared_size'] and job['prepared_size'] < job['image_size']:
        raise ValueError('prepared_size must be >= image_size when enabled')
    _number(job['stop_after_cycles'], 'stop_after_cycles', 0, integer=True)
    if job['stop_after_cycles'] > job['cycles']:
        raise ValueError('stop_after_cycles cannot exceed the full schedule')
    effective, physical, positive = (job[k] for k in ('effective_batch', 'physical_batch', 'positive_per_batch'))
    if effective % physical or positive >= effective or (positive * physical) % effective:
        raise ValueError('Physical batches must evenly divide the effective batch and positive quota')
    if job['warmup_cycles'] > job['cycles']:
        raise ValueError('warmup_cycles cannot exceed cycles')
    if job['precision'] not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be fp32/fp16/bf16')
    if job['normalization'] not in ('unit', 'imagenet', 'flair_rgb01'):
        raise ValueError('unknown normalization')
    _number(job['final_norm_eps'],'final_norm_eps',1e-15,maximum=.001)
    if job['head_lr_scope'] not in ('classifier_legacy','last_linear'):
        raise ValueError('head_lr_scope must be explicit classifier_legacy/last_linear')
    init = job['initialization']
    _unknown(init, ('kind', 'path', 'sha256', 'provenance','format','expected_backbone_state_keys','original_sha256','visual_prefix_removed'), 'initialization')
    if init.get('kind') not in ('random', 'imagenet', 'self_supervised', 'retinal_vision_language'):
        raise ValueError('unknown initialization kind')
    if init['kind'] == 'imagenet':
        if job['normalization'] != 'imagenet':
            raise ValueError('ImageNet weights require the fixed ImageNet normalization')
        if not all(init.get(k) for k in ('path', 'sha256', 'provenance')):
            raise ValueError('ImageNet initialization needs local path, sha256 and provenance')
        if not re.fullmatch('[0-9a-f]{64}', init['sha256']):
            raise ValueError('Weight sha256 must be a full lowercase 64-character SHA256')
        if init.get('format','torchvision-1000-state-dict-v1') not in (
                'torchvision-1000-state-dict-v1', 'torchvision-resnet50-backbone-state-dict-v1',
                'torchvision-small22k-only-full-head-state-dict-v1'):
            raise ValueError('unsupported public initialization state format')
        if (init.get('format') == 'torchvision-small22k-only-full-head-state-dict-v1'
                and job.get('model') != 'convnext_small'):
            raise ValueError('Small 22K-only public format requires the registered Small factory')
        if (init.get('format') == 'torchvision-resnet50-backbone-state-dict-v1'
                and job.get('model') != 'resnet50_imagenet'):
            raise ValueError('DINO backbone format requires the registered R50 factory')
    elif init['kind'] == 'self_supervised':
        if (job['model'] != 'dinov2_vits14' or job['normalization'] != 'imagenet'
                or init.get('format') != 'dinov2-official-backbone-state-dict-v1'
                or init.get('sha256') != 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
                or not all(isinstance(init.get(k), str) and Path(init[k]).is_absolute()
                           and '://' not in init[k] for k in ('path', 'provenance'))):
            raise ValueError('N03 requires the fixed local DINOv2 SSL weight, identity and provenance')
    elif init['kind'] == 'retinal_vision_language':
        if (job['model']!='flair_resnet50' or job['normalization']!='flair_rgb01'
            or init.get('format')!='flair-visual-safetensors-exact-extraction-v1'
            or init.get('sha256')!='62dbdc8020ac2d0412092d01617ebff7ebfb499420629f03426c3b8ad75758c2'
            or init.get('original_sha256')!='050334cd934fa3126435c202c41f493ca222fc3fe3ea9c50cf21f67c792a1440'
            or init.get('expected_backbone_state_keys')!=318 or init.get('visual_prefix_removed')!='vision_model.model.'
            or not all(isinstance(init.get(k),str) and Path(init[k]).is_absolute() and '://' not in init[k] for k in ('path','provenance'))):
            raise ValueError('N05 requires exact local FLAIR visual provenance')
    elif any(init.get(k) for k in ('path', 'sha256', 'provenance','format')):
        raise ValueError('Random initialization cannot silently carry a weights path')
    if job['model'] == 'flair_resnet50':
        fixed={'image_size':512,'prepared_size':512,'precision':'fp32','normalization':'flair_rgb01',
               'seed':20261027,'physical_batch':32,'effective_batch':64,'positive_per_batch':8,
               'workers':8,'eval_workers':4,'prefetch_factor':2,'cycles':8,'warmup_cycles':2,
               'min_cycles':8,'patience':8,'stop_after_cycles':0,'backbone_lr':2e-6,'head_lr':1e-3,
               'weight_decay':1e-4,'dropout':.2,'stochastic_depth':0.,'head_lr_scope':'last_linear',
               'mixup':None,'attention_pool':None,'initial_checkpoint':None,'frozen_bn':True}
        if (any(job.get(k)!=v for k,v in fixed.items()) or init.get('kind')!='retinal_vision_language'
            or not job.get('age',{}).get('enabled')
            or job['age'].get('contract_sha256')!='1d04d7bac147479c69dc92d15bd4d10d58e7e529e44d392796995d5fa302324e'):
            raise ValueError('N05 fixed FLAIR contract differs')
    elif job['model'] == 'resnet50_imagenet':
        fixed = {'image_size': 512, 'precision': 'fp32', 'normalization': 'imagenet',
                 'dropout': .2, 'stochastic_depth': 0., 'head_lr_scope': 'last_linear',
                 'mixup': None, 'attention_pool': None, 'initial_checkpoint': None, 'frozen_bn': True}
        if (any(job.get(k) != v for k, v in fixed.items())
                or init.get('kind') != 'imagenet'
                or (init.get('sha256'), init.get('format', 'torchvision-1000-state-dict-v1')) not in (
                    ('11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca',
                     'torchvision-1000-state-dict-v1'),
                    ('7a228bfac4d2eb2332c91f93d5b0daaffd93bb2df4a1b8ad2918b2acc156cc68',
                     'torchvision-resnet50-backbone-state-dict-v1'))
                or job['prepared_size'] not in (0, 512)
                or job.get('age') is None or not job['age'].get('enabled')):
            raise ValueError('N08 ImageNet ResNet model/initialization/age contract differs')
    elif job.get('frozen_bn') or job['normalization']=='flair_rgb01' or init['kind']=='retinal_vision_language':
        raise ValueError('FLAIR-only options cannot alter other model contracts')
    if job['model'] == 'dinov2_vits14':
        fixed = {'image_size': 504, 'prepared_size': 512, 'precision': 'fp32',
                 'normalization': 'imagenet', 'seed': 20261027, 'physical_batch': 32,
                 'effective_batch': 64, 'positive_per_batch': 8, 'workers': 8,
                 'eval_workers': 4, 'prefetch_factor': 2, 'cycles': 8, 'warmup_cycles': 2,
                 'min_cycles': 8, 'patience': 8, 'stop_after_cycles': 0,
                 'backbone_lr': 2e-5, 'head_lr': 1e-3, 'weight_decay': 1e-4,
                 'dropout': .2, 'stochastic_depth': .1, 'head_lr_scope': 'last_linear',
                 'mixup': None, 'attention_pool': None, 'initial_checkpoint': None}
        if (any(job.get(k) != v for k, v in fixed.items())
                or init.get('kind') != 'self_supervised'
                or not job.get('age', {}).get('enabled')
                or job['age'].get('contract_sha256') != '1d04d7bac147479c69dc92d15bd4d10d58e7e529e44d392796995d5fa302324e'):
            raise ValueError('N03 model/optimization/age configuration differs from the frozen contract')


def load_config(path=None, environ=None, overrides=None):
    env = os.environ if environ is None else environ
    base = Path(path).resolve().parent if path else BUNDLE
    raw = json.loads(Path(path).read_text()) if path else {}
    _unknown(raw, ('schema_version', 'mode', 'resume', 'run_name', 'output_root', 'data', 'execution', 'jobs'), 'config')
    cfg = dict(schema_version=1, mode='inspect', resume=False, run_name='retrain-screening-20261024',
               output_root='/outputs', data=dict(root='/mnt/data', identity_path=str(BUNDLE/'assets/train_identity.json'),
               verify_images=True, expected_count=51396, expected_valid_count=5849,
               split=dict(kind='holdout', seed=20261024, valid_fraction=.2, folds=3, fold=0,
                          inner_fraction=.1, include_valid=False)),
               execution=dict(devices=['cuda:0', 'cuda:1'], cpu_cores_per_job=14,
                              grace_seconds=90, heartbeat_seconds=60, per_rank_microbatch=0,
                              evaluation_physical_batch=32,legacy_six_card_exception=False,technical_smoke=False))
    for key in ('schema_version', 'mode', 'resume', 'run_name', 'output_root'):
        if key in raw:
            cfg[key] = raw[key]
    for key in ('data', 'execution'):
        values = raw.get(key, {})
        _unknown(values, cfg[key], key)
        for k, value in values.items():
            if key == 'data' and k == 'split':
                _unknown(value, cfg['data']['split'], 'split')
                cfg['data']['split'].update(value)
            else:
                cfg[key][k] = value
    jobs = raw.get('jobs', [{'name': 'effnet-v2-s', 'model': MODELS[0]},
                            {'name': 'convnext-tiny', 'model': MODELS[1]}])
    cfg['jobs'] = []
    for item in jobs:
        if not isinstance(item, dict) or not {'name', 'model'} <= item.keys():
            raise ValueError('Every job requires name and model')
        job = default_job(item['name'], item['model'], item.get('initialization', {}).get('kind', 'random'))
        # Omitted optional fields remain absent in legacy checkpoint identities.
        _unknown(item, (*job, 'loss', 'training_scale', 'training_views', 'training_photometric', 'training_rotation', 'head_seed', 'bn_statistics', 'freeze_features_before', 'final_downsample_stride', 'stage3_residual', 'gradient_clip_policy', 'head_warmup_policy', 'layerwise_lr_policy', 'low_rank_policy'), 'job')
        job.update(item)
        if job.get('low_rank_policy') is None:
            job.pop('low_rank_policy', None)
        cfg['jobs'].append(job)
    for variable, key in [('MODE', 'mode'), ('RUN_NAME', 'run_name'), ('OUTPUT_ROOT', 'output_root')]:
        if variable in env:
            cfg[key] = env[variable]
    if 'DATA_ROOT' in env:
        cfg['data']['root'] = env['DATA_ROOT']
    if 'RESUME' in env:
        if env['RESUME'] not in ('0', '1'):
            raise ValueError('RESUME must be 0 or 1')
        cfg['resume'] = env['RESUME'] == '1'
    if 'DEVICE_IDS' in env:
        cfg['execution']['devices'] = parse_devices(env['DEVICE_IDS'])
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == 'data_root':
            cfg['data']['root'] = value
        elif key == 'devices':
            cfg['execution']['devices'] = parse_devices(value)
        elif key == 'workers':
            for job in cfg['jobs']:
                job['workers'] = value
        elif key in ('mode', 'resume', 'run_name', 'output_root'):
            cfg[key] = value
        else:
            raise ValueError(f'Unknown override {key}')
    def absolute(value):
        return str((base / value).resolve()) if value is not None else None
    cfg['output_root'] = absolute(cfg['output_root'])
    for key in ('root', 'identity_path'):
        cfg['data'][key] = absolute(cfg['data'][key])
    if cfg['schema_version'] != 1 or cfg['mode'] not in MODES or not isinstance(cfg['resume'], bool):
        raise ValueError('Invalid schema_version, mode or resume')
    _name(cfg['run_name'])
    if not cfg['jobs'] or len({j['name'] for j in cfg['jobs']}) != len(cfg['jobs']):
        raise ValueError('Jobs must be nonempty and names unique')
    if not isinstance(cfg['data']['verify_images'], bool):
        raise ValueError('verify_images must be boolean')
    for count_key in ('expected_count', 'expected_valid_count'):
        if cfg['data'][count_key] is not None:
            _number(cfg['data'][count_key], count_key, 1, integer=True)
    for job in cfg['jobs']:
        validate_job(job)
        job['initial_checkpoint'] = absolute(job['initial_checkpoint'])
        for key in ('path', 'provenance'):
            if job['initialization'].get(key):
                job['initialization'][key] = absolute(job['initialization'][key])
        split = dict(cfg['data']['split'])
        _unknown(job['split'], split, 'job.split')
        split.update(job['split'])
        validate_split(split)
    validate_split(cfg['data']['split'])
    if any(job['model'] == 'dinov2_vits14' for job in cfg['jobs']):
        if (len(cfg['jobs']) != 1 or cfg['mode'] != 'train' or cfg['resume']
                or cfg['data']['split']['kind'] != 'holdout'
                or cfg['data']['split']['include_valid'] or not cfg['data']['verify_images']
                or cfg['execution']['devices'] != ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
                or cfg['execution']['per_rank_microbatch'] != 0
                or cfg['execution']['evaluation_physical_batch'] != 32
                or cfg['execution']['technical_smoke']):
            raise ValueError('N03 forbids altered topology, resume, AMP, or validation-in-fit')
    devices = cfg['execution']['devices']
    if not isinstance(devices, list) or not devices or any(not isinstance(d, str) or not re.fullmatch(r'cuda:\d+|cpu', d) for d in devices):
        raise ValueError('devices must be a nonempty list of cuda:N or cpu')
    cuda = [d for d in devices if d != 'cpu']
    if len(cuda) != len(set(cuda)):
        raise ValueError('Accelerator slots must use distinct visible devices')
    for key in ('cpu_cores_per_job', 'grace_seconds', 'heartbeat_seconds'):
        _number(cfg['execution'][key], key, 1, integer=True)
    _number(cfg['execution']['per_rank_microbatch'],'per_rank_microbatch',0,integer=True)
    _number(cfg['execution']['evaluation_physical_batch'],'evaluation_physical_batch',1,integer=True)
    if not isinstance(cfg['execution']['legacy_six_card_exception'],bool):raise ValueError('legacy_six_card_exception must be boolean')
    if not isinstance(cfg['execution']['technical_smoke'],bool):raise ValueError('technical_smoke must be boolean')
    return cfg


def parse_devices(text):
    return [v.strip() if v.strip() == 'cpu' or v.strip().startswith('cuda:') else 'cuda:'+v.strip()
            for v in text.split(',')]


def validate_split(split):
    if split['kind'] not in ('holdout', 'cv', 'refit'):
        raise ValueError('split.kind must be holdout/cv/refit')
    for key in ('valid_fraction', 'inner_fraction'):
        _number(split[key], key, .0001, maximum=.4999)
    for key in ('seed', 'fold'):
        _number(split[key], key, 0, integer=True)
    _number(split['folds'], 'folds', 2, integer=True)
    if split['fold'] >= split['folds']:
        raise ValueError('fold must be less than folds')
    if not isinstance(split['include_valid'], bool) or (split['include_valid'] and split['kind'] != 'refit'):
        raise ValueError('Official valid can enter fitting only for an explicit refit')


def job_contract(job):
    result = copy.deepcopy(job)
    if result.get('low_rank_policy') is None:
        result.pop('low_rank_policy', None)
    for key in ('device', 'workers', 'eval_workers', 'prefetch_factor', 'checkpoint_seconds',
                'checkpoint_updates', 'log_every', 'split', 'initial_checkpoint'):
        result.pop(key, None)
    init = result['initialization']
    init.pop('path', None)
    init.pop('provenance', None)
    return result
