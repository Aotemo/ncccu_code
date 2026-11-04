"""Run the frozen R120 test predictor once, with existing resource and model APIs."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'resource-source', 'resource-site', 'resource-site-sha256', 'safe-proof', 'ready'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--pod-uid'); parser.add_argument('--observed-at', type=float)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    preflight = read(args.preflight); root = Path(preflight['round_root'])
    assert preflight['status'] == 'R120_selected_checkpoint_actually_loaded_EMA_identity_and_real_test_coverage_verified'
    assert preflight['EMA_tensor_sha256'] == 'f21bffae88acbf7c216c4141c0ef0b9d990a624e3e92a1baf750a501c415fc85'
    assert preflight['test_count'] == 5855 and preflight['test_label_values_accessed'] is False
    assert sha(preflight['freeze']['path']) == preflight['freeze']['sha256']
    assert sha(preflight['model_identity_site']['path']) == preflight['model_identity_site']['sha256']
    source = Path(args.resource_source)
    resources = module(source / 'runtime/bundle/retrain/site_binding.py', 'submission_resources')
    execution_site = resources.load_site(args.resource_site, expected_sha256=args.resource_site_sha256)
    proof = read(args.safe_proof)
    assert proof['formal_root'] == execution_site['round_root'] and proof['phase'] == 'terminal'
    assert proof['done']['status'] == 'completed' and proof['safety']['safe'] is True
    bundle = Path(preflight['frozen_bundle']); frozen = read(preflight['freeze']['path'])
    for relative, digest in frozen['source_files'].items():
        assert sha(bundle / relative) == digest, relative
    if args.check_only:
        compile(Path(__file__).read_text(), __file__, 'exec')
        value = {'status': 'frozen_R120_source_existing_predictor_and_completed_predecessor_checked',
            'at': time.time(), 'script_sha256': sha(__file__), 'preflight_sha256': sha(args.preflight),
            'safe_proof_sha256': sha(args.safe_proof), 'resource_site_sha256': args.resource_site_sha256,
            'frozen_consumer_files_verified': len(frozen['source_files']), 'GPU_started': False,
            'model_reloaded_or_forward': False, 'old_actualsafe_repolled': False}
        write(args.ready, value); print(json.dumps(value, indent=2), flush=True); return
    ready = read(args.ready)
    assert ready['script_sha256'] == sha(__file__) and ready['preflight_sha256'] == sha(args.preflight)
    assert ready['safe_proof_sha256'] == sha(args.safe_proof)
    resources.verify_live(execution_site, pod_uid=args.pod_uid, ready=True, observed_at=args.observed_at)
    assert execution_site['cpu_execution']['owner_id'] == os.environ['NCCCU_EXECUTOR_ID']
    resources.prepare_native_entry(execution_site)
    mask = resources.bind_cpu_role(execution_site, 'owner', initial=True)
    observer = module(source / 'tools/observer/observe_round_once.py', 'submission_admission')
    sys.path.insert(0, str(bundle))
    os.environ.update(NCCCU_SITE_BINDING=preflight['model_identity_site']['path'],
        NCCCU_SITE_BINDING_SHA256=preflight['model_identity_site']['sha256'],
        TRAIN_IMAGE_CACHE_READONLY='1')
    import torch
    from retrain import inference
    from retrain.persistence import RunLock
    policy = execution_site['cpu_execution']['resource_policy']['value']
    torch.set_num_threads(policy['rank_threads']); torch.set_num_interop_threads(policy['interop_threads'])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time(); loaded_at = None; completed = 0
    original_load, original_forward = inference._load_model, inference._forward
    def load(*positional, **keywords):
        nonlocal loaded_at
        result = original_load(*positional, **keywords); model, job, metadata, origin, device = result
        assert origin['variant'] == 'ema' and origin['tensor_sha256'] == preflight['EMA_tensor_sha256']
        assert origin['sha256'] == preflight['checkpoint']['blob_sha256'] and origin['successful_updates'] == 4800
        assert job == preflight['job']
        loaded_at = time.time(); print(json.dumps({'event': 'R120_actual_model_strictly_loaded', 'at': loaded_at}), flush=True)
        return result
    def forward(*positional, **keywords):
        nonlocal completed
        values = original_forward(*positional, **keywords); completed += len(values)
        if completed == len(values) or completed % 512 == 0 or completed == 5855:
            elapsed = time.time() - loaded_at
            print(json.dumps({'event': 'production_test_progress', 'at': time.time(), 'completed': completed,
                'total': 5855, 'seconds_after_model_load': elapsed,
                'estimated_remaining_seconds_at_observed_rate': (5855-completed)*elapsed/completed}), flush=True)
        return values
    inference._load_model, inference._forward = load, forward
    with RunLock(resources.instance_lock(execution_site)):
        users, denied = observer.scan_device_users()
        assert not users and not denied, 'new submission admission found occupied or unreadable devices'
        write(output.parent / 'INFERENCE-START.json', {'at': started, 'pid': os.getpid(),
            'resource_site_sha256': args.resource_site_sha256, 'CPU_mask': mask,
            'model_identity_site': preflight['model_identity_site'], 'training_clock_or_token_used': False})
        report = inference.run_prediction(preflight['checkpoint']['path'], preflight['data_root'], output,
            device='cuda:0', variant='ema', expected_count=5855, threshold=.5, batch_size=32, precision='fp32')
        rows = inference._read_unlabeled_rows(preflight['data_root'], 'test', 5855)
        with output.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream); assert reader.fieldnames == ['Image_name', 'Probability', 'Pred_Label']
            values = list(reader)
        assert len(values) == len(rows) == completed == 5855
        assert [v['Image_name'] for v in values] == [r['name'] for r in rows]
        assert len({v['Image_name'].casefold() for v in values}) == 5855
        for value in values:
            p = float(value['Probability']); assert math.isfinite(p) and 0 <= p <= 1
            assert value['Pred_Label'] in ('0', '1') and int(value['Pred_Label']) == int(p >= .5)
        assert report['output_sha256'] == sha(output) and report['test_label_values_read'] is False
        receipt = {'status': 'completed', 'at': time.time(), 'output': str(output), 'output_sha256': sha(output),
            'metadata_sha256': sha(output.with_suffix('.metadata.json')), 'count': 5855, 'test_order_exact': True,
            'unique_coverage': True, 'finite_probability_in_unit_interval': True, 'threshold': .5,
            'header': ['Image_name', 'Probability', 'Pred_Label'], 'source': report['source'],
            'input_csv_sha256': report['input_csv_sha256'], 'wall_seconds': time.time()-started,
            'test_label_values_read': False, 'official_valid_or_dev_forward': False, 'training_clock_or_token_used': False}
        write(output.parent / 'VALIDATION.json', receipt); print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
