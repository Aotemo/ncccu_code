"""Standalone F04 metadata must make strict loading independent of checkpoints."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / 'runtime/bundle'), str(ROOT / 'tools/diagnostics')]
spec = importlib.util.spec_from_file_location('f04_standalone_member', ROOT / 'tools/diagnostics/f04_standalone_member.py')
standalone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standalone)


def test_strict_checkpoint_free_export_and_tamper_rejection():
    from retrain.ema import tensor_digest
    from retrain.inference import _load_model
    from retrain.persistence import sha256_file
    source = torch.nn.Linear(3, 1).eval()
    state = {key: value.detach().half() for key, value in source.state_dict().items()}
    job = {'model': 'convnext_small', 'age': None}
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        torch.save(state, root / 'model.pt')
        metadata = standalone.export_metadata(job, sha256_file(root / 'model.pt'), tensor_digest(state))
        (root / 'model.json').write_text(json.dumps(metadata))
        model, loaded_job, _, origin, device = _load_model(
            root, device='cpu', variant='ema', model_factory=lambda _job, initialize: torch.nn.Linear(3, 1))
        assert loaded_job == job and origin['variant'] == 'ema' and str(device) == 'cpu'
        assert all(torch.equal(model.state_dict()[key], value.float()) for key, value in state.items())
        changed = dict(metadata); changed['model_sha256'] = '0' * 64
        (root / 'model.json').write_text(json.dumps(changed))
        try:
            _load_model(root, device='cpu', variant='ema', model_factory=lambda _job, initialize: torch.nn.Linear(3, 1))
        except ValueError:
            pass
        else:
            raise AssertionError('changed weight identity was accepted')


def test_online_selected_variant_remains_online_and_rejects_ema_alias():
    from retrain.ema import tensor_digest
    from retrain.inference import _load_model
    from retrain.persistence import sha256_file
    source = torch.nn.Linear(3, 1).eval()
    state = {key: value.detach().half() for key, value in source.state_dict().items()}
    job = {'model': 'convnext_small', 'age': None}
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        torch.save(state, root / 'model.pt')
        metadata = standalone.export_metadata(job, sha256_file(root / 'model.pt'), tensor_digest(state), variant='online')
        assert metadata['checkpoint']['variant'] == 'online'
        (root / 'model.json').write_text(json.dumps(metadata))
        _, _, _, origin, _ = _load_model(root, device='cpu', variant='online',
                                         model_factory=lambda _job, initialize: torch.nn.Linear(3, 1))
        assert origin['variant'] == 'online'
        try:
            _load_model(root, device='cpu', variant='ema',
                        model_factory=lambda _job, initialize: torch.nn.Linear(3, 1))
        except ValueError:
            pass
        else:
            raise AssertionError('online selected variant must not be presented as EMA')


if __name__ == '__main__':
    test_strict_checkpoint_free_export_and_tamper_rejection()
    test_online_selected_variant_remains_online_and_rejects_ema_alias()
    print('F04 standalone strict-load and tamper checks passed', flush=True)
