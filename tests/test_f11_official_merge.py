"""F11: official 5849 fully merged into fitting while internal dev stays the selector."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))

from retrain.config import validate_split  # noqa: E402
from retrain.data import build_manifest  # noqa: E402


def source(train_rows, valid_rows):
    rows = []
    for index, (label, group) in enumerate(train_rows):
        rows.append({'name': 't%03d' % index, 'label': label, 'group': group,
                     'age': '', 'label_name': str(label)})
    valid = []
    for index, (label, group) in enumerate(valid_rows):
        valid.append({'name': 'v%03d' % index, 'label': label, 'group': group,
                      'age': '', 'label_name': str(label)})
    return {'rows': rows, 'valid_rows': valid, 'identity': {'format': 'test'}}


def test_official_merge_keeps_internal_dev_and_trains_on_all_official():
    train = [(1, 'a%d' % i) for i in range(40)] + [(0, 'b%d' % i) for i in range(160)]
    valid = [(1, 'c%d' % i) for i in range(10)] + [(0, 'd%d' % i) for i in range(90)]
    split = {'kind': 'official_merge', 'seed': 20261024, 'valid_fraction': .1,
             'include_valid': True}
    manifest = build_manifest(source(train, valid), split)
    fit, dev = set(manifest['fit_indices']), set(manifest['dev_indices'])
    assert not fit & dev
    assert len(dev) + len(fit) == 300
    # internal dev comes only from the training rows; every official row is fitted
    assert all(index < 200 for index in dev)
    assert {index for index in fit if index >= 200} == set(range(200, 300))
    assert manifest['counts']['fit']['rows'] == 200 - len(dev) + 100
    assert manifest['counts']['dev']['positive'] > 0 and manifest['counts']['dev']['rows'] < 200


def test_official_merge_requires_include_valid():
    with pytest.raises(ValueError):
        validate_split({'kind': 'official_merge', 'seed': 1, 'valid_fraction': .1,
                        'inner_fraction': .1, 'fold': 0, 'folds': 3, 'include_valid': False})
    validate_split({'kind': 'official_merge', 'seed': 1, 'valid_fraction': .1,
                    'inner_fraction': .1, 'fold': 0, 'folds': 3, 'include_valid': True})


PAIR_SIGMOID = {'version': 'binary-pairwise-sigmoid-v1', 'kind': 'binary_pairwise_sigmoid',
                'temperature': 1.0, 'reduction': 'mean', 'dtype': 'float32', 'scope': 'global_batch'}


def _f11_job(version):
    import copy
    import json
    from retrain.config import F11_WARM_STARTS
    tensor, file_sha = F11_WARM_STARTS[version]
    age = json.loads((ROOT / 'configs/HP-AGE-CONTRACT.json').read_text())
    return {
        'name': 'convnext-small', 'model': 'convnext_small', 'seed': 20261205,
        'training_scale': [.85, 1.0], 'loss': copy.deepcopy(PAIR_SIGMOID),
        'image_size': 512, 'prepared_size': 512, 'physical_batch': 32, 'effective_batch': 64,
        'positive_per_batch': 8, 'precision': 'fp32', 'cycles': 15, 'warmup_cycles': 1,
        'min_cycles': 1, 'patience': 4, 'min_delta': 0.0, 'stop_after_cycles': 0,
        'backbone_lr': 1e-5, 'head_lr': 1e-4, 'weight_decay': 1e-4, 'dropout': .2,
        'stochastic_depth': .1, 'normalization': 'imagenet', 'final_norm_eps': 1e-6,
        'head_lr_scope': 'last_linear', 'mixup': None, 'attention_pool': None,
        'initial_checkpoint': None, 'frozen_bn': False, 'benchmark_warmup': 50,
        'benchmark_updates': 100, 'checkpoint_seconds': 300, 'checkpoint_updates': 0,
        'log_every': 25, 'prefetch_factor': 2, 'workers': 0, 'eval_workers': 0, 'split': {},
        'age': age,
        'initialization': {'kind': 'checkpoint_warm_start', 'format': 'exported-fp32-state-dict-v1',
                           'path': '/tmp/warmstart.pt', 'sha256': file_sha,
                           'tensor_sha256': tensor, 'source_model': '/tmp/src.pt'},
        'f11': {'version': version},
    }


def test_f11_warm_start_bindings_match_only_their_registered_pair():
    from retrain.config import validate_job

    r249 = 'f11-official-merge-r249-warmstart-v1'
    r269 = 'f11-official-merge-r269-warmstart-v1'
    # R249 keeps its historical behaviour.
    validate_job(_f11_job(r249))
    # R269 is accepted with its own pair.
    validate_job(_f11_job(r269))
    # Cross-pair identities are rejected in both directions.
    from retrain.config import F11_WARM_STARTS
    for job_version, other in ((r249, r269), (r269, r249)):
        job = _f11_job(job_version)
        tensor, file_sha = F11_WARM_STARTS[other]
        job['initialization']['tensor_sha256'] = tensor
        job['initialization']['sha256'] = file_sha
        with pytest.raises(ValueError):
            validate_job(job)
    # Unknown versions and extra keys are rejected.
    unknown = _f11_job(r249)
    unknown['f11'] = {'version': 'f11-official-merge-r999-warmstart-v1'}
    with pytest.raises(ValueError):
        validate_job(unknown)
    stray = _f11_job(r269)
    stray['f11'] = {'version': r269, 'note': 'x'}
    with pytest.raises(ValueError):
        validate_job(stray)
