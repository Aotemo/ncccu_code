"""New-seed MixStyle and R120 replicas preserve their registered science."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'runtime/bundle'))

SPECS = (
    ('R261', 'R134R1', 20261117, 171,
     'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261117-scale085-pairlog-mixstyle13-local8-v1',
     'r261-small-mixstyle-seed20261117-pairlog-24k-p4-9010-20261104',
     'r261-convnext-small-imagenet22k-to-1k-24k-p4-seed20261117-scale085-pairlog-mixstyle13-local8-9010'),
    ('R262', 'R120', 20261118, 172,
     'single-convnext-small-fe6b358a-fixed-9010-24k-earlystop-seed20261118-scale085-pairsigmoid-v1',
     'r262-small-seed20261118-pairsigmoid-24k-p4-9010-20261104',
     'r262-convnext-small-imagenet22k-to-1k-24k-p4-seed20261118-scale085-pairsigmoid-9010'),
)


def configured(ex, value, round_id):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'science.json'
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        site = {'work_root': folder, 'round_id': round_id,
                'science': {'configuration': {'path': str(path),
                           'sha256': hashlib.sha256(raw).hexdigest()}}}
        with mock.patch.object(ex, 'load_site', return_value=site):
            return ex.science_configuration()


def must_reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('modified fixed scientific identity was accepted')


def test_registered_seed_identity_and_policy():
    from retrain import experiment91 as ex
    for label, original, seed, order, profile, rid, candidate_id in SPECS:
        base = json.loads((ROOT / 'configs' / (original + '-SCIENCE.json')).read_text())
        value = copy.deepcopy(base)
        value.update(candidate_profile=profile, official_evaluation=ex.OFFICIAL_FINAL_ENSEMBLE_ONLY)
        value['job']['seed'] = seed
        value['candidate'].update(id=candidate_id, experiment_order=order)
        assert profile in ex.SUPPORTED_BINDINGS, label
        assert ex.candidate_from_configuration(value) == value['candidate']
        assert configured(ex, value, rid) == value
        assert {k: v for k, v in value['job'].items() if k != 'seed'} == {
            k: v for k, v in base['job'].items() if k != 'seed'}
        assert ex.expected_official_policy({'round_id': rid}, value) == ex.OFFICIAL_FINAL_ENSEMBLE_ONLY
        assert not ex.official_allowed_for_selected(ex.OFFICIAL_FINAL_ENSEMBLE_ONLY, {'auroc': 1.0})
        invalid = copy.deepcopy(value); invalid['job']['seed'] = 20261029
        must_reject(lambda: ex.candidate_from_configuration(invalid))
        invalid = copy.deepcopy(value)
        invalid['job']['loss'] = ex.PAIR_SIGMOID_LOSS if label == 'R261' else ex.PAIR_LOGISTIC_LOSS
        must_reject(lambda: configured(ex, invalid, rid))
        if label == 'R261':
            invalid = copy.deepcopy(value)
            invalid['job']['mixstyle_policy']['probability'] = 0.6
            must_reject(lambda: configured(ex, invalid, rid))
        invalid = copy.deepcopy(value); invalid['official_evaluation'] = ex.OFFICIAL_GATE_0996
        must_reject(lambda: configured(ex, invalid, rid))
        assert ex.candidate_from_configuration(base) == base['candidate']


if __name__ == '__main__':
    test_registered_seed_identity_and_policy()
    print('R261/R262 seeds, original recipes and final-only official gate passed', flush=True)
