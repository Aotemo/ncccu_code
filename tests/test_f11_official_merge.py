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
