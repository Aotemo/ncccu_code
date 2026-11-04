"""Frozen historical and current datasets preserve their own evaluation defaults."""
import importlib.util
import os
from pathlib import Path

root = Path(os.environ.get('NCCCU_TEST_SOURCE', Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('f04_gpu_full_dev', root / 'tools/diagnostics/f04_gpu_full_dev.py')
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


class LegacyDataset:
    def __init__(self, root, rows, size, training, normalization='imagenet', prepared_size=0, age=None):
        self.received = (root, rows, size, training, normalization, prepared_size, age)


class CurrentDataset(LegacyDataset):
    def __init__(self, root, rows, size, training, normalization='imagenet', prepared_size=0,
                 age=None, input_preprocessing=None):
        super().__init__(root, rows, size, training, normalization, prepared_size, age)
        self.input_preprocessing = input_preprocessing


def test_historical_default_and_explicit_preprocessing_are_preserved():
    job = {'image_size': 512, 'prepared_size': 512, 'normalization': 'imagenet', 'age': None}
    legacy = gpu.evaluation_dataset(LegacyDataset, 'root', ['row'], job)
    assert legacy.received == ('root', ['row'], 512, False, 'imagenet', 512, None)
    assert gpu.evaluation_dataset(CurrentDataset, 'root', [], job).input_preprocessing is None
    configured = {**job, 'input_preprocessing': {'name': 'fixed'}}
    assert gpu.evaluation_dataset(CurrentDataset, 'root', [], configured).input_preprocessing == {'name': 'fixed'}
    try:
        gpu.evaluation_dataset(LegacyDataset, 'root', [], configured)
    except TypeError:
        pass
    else:
        raise AssertionError('legacy runtime silently dropped explicit preprocessing')


if __name__ == '__main__':
    test_historical_default_and_explicit_preprocessing_are_preserved()
    print('F04 historical and current dataset interfaces passed')
