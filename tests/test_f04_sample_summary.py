"""The fixed F04 prefix must be covered exactly once, without identity drift."""
import importlib.util
from pathlib import Path


here = Path(__file__).resolve()
module_path = here.with_name('F04-SAMPLE-TEST-SCRIPT.py')
if not module_path.exists():
    module_path = here.parents[1] / 'tools/diagnostics/f04_sample_summary.py'
spec = importlib.util.spec_from_file_location('f04_sample_summary', module_path)
sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample)


def test_chunks_cover_prefix_once_and_bind_identity():
    common = {'status': 'F04_paired_internal_dev_chunk_completed_no_fusion', 'round_id': 'r1',
              'inventory_sha256': 'inventory', 'export_sha256': 'export', 'git_commit': 'commit',
              'TTA': False, 'official_read': False, 'GPU_initialized': False,
              'execution': {'scope': 'compute_pod_cpu',
                            'gpu_visibility': {'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '',
                                               'ROCR_VISIBLE_DEVICES': ''}}}
    first, rest = dict(common, offset=0, count=16), dict(common, offset=16, count=48)
    assert [row['offset'] for row in sample.validate_chunk_plan([rest, first], 'r1', 'inventory', 'export', 'commit')] == [0, 16]
    for invalid in ([first, dict(rest, offset=17)], [first, dict(rest, round_id='r2')],
                    [first, dict(rest, TTA=True)], [first, dict(rest, official_read=True)],
                    [first, dict(rest, count=47)], [first, dict(rest, git_commit='other')]):
        try: sample.validate_chunk_plan(invalid, 'r1', 'inventory', 'export', 'commit')
        except ValueError: pass
        else: raise AssertionError('invalid prefix or identity was accepted')


if __name__ == '__main__':
    test_chunks_cover_prefix_once_and_bind_identity()
    print('F04 fixed-sample chunk identity and coverage checks passed', flush=True)
