"""Pod-only frozen metadata verification and actual native sampler cross-check."""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / 'verification'
RESULT = HERE / 'result'
CHECKS = []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def load(path):
    return json.loads(path.read_bytes())


def check(name, condition, evidence=None):
    CHECKS.append({'name': name, 'passed': bool(condition), 'evidence': evidence})
    assert condition, name


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    import numpy as np
    check('designated_pod_and_python', socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
          and sys.executable == '/usr/local/python3.12/bin/python3.12')
    check('cpu_isolation', os.getpriority(os.PRIO_PROCESS, 0) == 19
          and all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
          and all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')))
    deployment = load(HERE / 'DEPLOYMENT.json')
    check('deployed_sources_unchanged', all(sha((HERE / x['relative_path']).read_bytes()) == x['sha256'] for x in deployment['files']))
    preflight = load(HERE / 'preflight/PREFLIGHT-RECEIPT.json')
    check('preflight_passed', preflight['status'] == 'passed' and preflight['exit_code'] == 0)
    freeze = load(RESULT / 'FREEZE.json')
    check('freeze_all_files_present_and_sha_match', freeze['status'] == 'SEALED'
          and all((RESULT / n).is_file() and (RESULT / n).stat().st_size == r['bytes']
                  and sha((RESULT / n).read_bytes()) == r['sha256'] for n, r in freeze['files'].items()),
          {'files': len(freeze['files']), 'freeze_sha256': sha((RESULT / 'FREEZE.json').read_bytes())})
    check('frozen_readonly', RESULT.stat().st_mode & 0o222 == 0
          and all(p.stat().st_mode & 0o222 == 0 for p in RESULT.iterdir()))
    check('generator_actual_exit_zero', (RESULT / 'EXIT-CODE.txt').read_text() == '0\n'
          and load(RESULT / 'EXECUTION.json')['exit_code'] == 0)
    manifest = load(RESULT / 'manifest.json')
    contract = load(RESULT / 'CONTRACT.json')
    fit, dev, rows = manifest['fit_indices'], manifest['dev_indices'], manifest['rows']
    check('seed_holdout_and_official_isolation', manifest['split'] == {'kind': 'holdout', 'seed': 20261024, 'valid_fraction': .1, 'include_valid': False}
          and not manifest['audit_indices'] and all(r['relative_path'].split('/')[0] == 'train' for r in rows))
    check('indices_exact_coverage_and_indivisible_groups', set(fit).isdisjoint(dev) and sorted(fit + dev) == list(range(len(rows)))
          and not ({rows[i]['group'] for i in fit} & {rows[i]['group'] for i in dev})
          and not ({rows[i]['sha256'] for i in fit} & {rows[i]['sha256'] for i in dev}))
    check('frozen_index_identity', load(RESULT / 'fit_indices.json') == fit and load(RESULT / 'dev_indices.json') == dev
          and sha(canonical(fit)) == contract['fit_indices_canonical_sha256']
          and sha(canonical(dev)) == contract['dev_indices_canonical_sha256'])
    check('manifest_identity', sha((RESULT / 'manifest.json').read_bytes()) == contract['new_manifest']['sha256']
          and sha(canonical(manifest)) == contract['new_manifest_canonical_sha256'])
    sampler_ns = extract(RESULT / 'SOURCE-data.py', ['_seed', 'BalancedBatchSampler'],
                         {'math': math, 'hashlib': hashlib, 'random': random})
    sampler = sampler_ns['BalancedBatchSampler'](rows, fit, physical_batch=16, effective_batch=64,
        positive_per_batch=8, seed=20261027, start_update=0, num_updates=1)
    schedule = load(RESULT / 'SCHEDULE.json')
    steps = sampler.steps_per_cycle
    native = {'fit_negative': len(sampler.pools[0]), 'fit_positive': len(sampler.pools[1]), 'steps_per_cycle': steps,
              'physical_batch': sampler.physical_batch, 'effective_batch': sampler.effective_batch,
              'positive_per_batch': sampler.positive_per_batch, 'microbatches': sampler.microbatches,
              'native_class': 'N06/retrain/data.py:BalancedBatchSampler', 'no_images_or_cache_read': True}
    check('actual_native_sampler_S_matches_new_schedule', steps == schedule['steps_per_cycle']
          and len(sampler.pools[0]) == schedule['fit_negative'] == contract['counts']['fit']['negative']
          and len(sampler.pools[1]) == contract['counts']['fit']['positive'], native)
    batches = list(sampler)
    check('native_G64_P8_sampler_metadata', len(batches) == 4 and all(len(batch) == 16 for batch in batches)
          and sum(rows[i]['label'] for batch in batches for i, seed in batch) == 8
          and all(i in set(fit) for batch in batches for i, seed in batch))
    check('dynamic_S_2S_8S', schedule['cycles'] == 8 and schedule['total_updates'] == 8 * steps
          and schedule['stop_after_updates'] == 8 * steps and schedule['warmup_updates'] == 2 * steps
          and schedule['ema_warmcopy_updates'] == 2 * steps and schedule['early_stopping_enabled'] is False
          and schedule['expected_online_ema_dev_vectors'] == 16)
    ema = load(RESULT / 'EMA-R50.json')
    check('R50_dynamic_warmcopy_and_frozen_bn_metadata', ema == {'version': f'ema-warmcopy{2*steps}-decay0999-frozenbn-v1',
          'warmcopy_updates': 2 * steps, 'decay': .999, 'dtype': 'torch.float32',
          'floating_buffers': 'same_as_parameters_except_frozen_bn', 'integer_buffers': 'copy', 'frozen_bn_buffers': 'copy'},
          {'first_blend_update': 2 * steps + 1, 'engine_integration_tested': False})
    age_ns = extract(RESULT / 'SOURCE-persistence.py', ['_json_bytes', 'canonical_digest'], {'json': json, 'hashlib': hashlib})
    age_ns.update(np=np, copy=copy, math=math, VERSION='age-linear-v1')
    extract(RESULT / 'SOURCE-age.py', ['parse_age', 'make_spec', 'validate_spec'], age_ns)
    reproduced_age = age_ns['make_spec'](manifest)
    age_ns['validate_spec'](reproduced_age)
    check('fit_only_age_reproduced_and_bound', reproduced_age == load(RESULT / 'AGE.json')
          and reproduced_age['contract']['fit_count'] == len(fit)
          and reproduced_age['contract']['fit_indices_sha256'] == sha(canonical(fit))
          and reproduced_age['contract_sha256'] == contract['age_contract_sha256'])
    source = contract['source_manifest']
    check('sealed_input_unchanged', sha(Path(source['path']).read_bytes()) == source['sha256'])
    schema = load(RESULT / 'SCHEMA.json')
    check('patient_schema_gate', not schema['patient_field_candidates'] and schema['identical_byte_sha_in_multiple_groups'] == 0,
          {'patient_ids_inferred_from_names': False, 'patient_or_bilateral_leakage_excluded': False})
    return {'status': 'passed', 'exit_code': 0, 'checks': CHECKS, 'sampler': native, 'counts': contract['counts'],
            'schedule': schedule, 'age_contract_sha256': contract['age_contract_sha256'],
            'freeze_path': str(RESULT / 'FREEZE.json'), 'freeze_sha256': sha((RESULT / 'FREEZE.json').read_bytes()),
            'deployment_sha256': sha((HERE / 'DEPLOYMENT.json').read_bytes()),
            'model_loaded': False, 'engine_checked': False, 'GPU_used': False, 'official_valid_read': False}


if __name__ == '__main__':
    OUT.mkdir()
    code = 1
    try:
        receipt = main()
        code = 0
    except Exception as error:
        receipt = {'status': 'failed', 'exit_code': 1, 'checks': CHECKS, 'error': repr(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    raw = (json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    path = OUT / 'STAGE1-VERIFICATION.json'
    path.write_bytes(raw)
    (OUT / 'EXIT-CODE.txt').write_text(str(code) + '\n')
    output_list = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
                   for root in (HERE / 'preflight', RESULT, OUT) if root.exists()
                   for p in sorted(root.rglob('*')) if p.is_file()]
    (OUT / 'OUTPUT-MANIFEST.json').write_text(json.dumps({'files': output_list, 'exit_code': code}, indent=2) + '\n')
    print(json.dumps({'path': str(path), 'sha256': sha(raw), 'status': receipt['status'], 'exit_code': code,
                      'checks': len(CHECKS), 'passed': sum(x['passed'] for x in CHECKS),
                      'counts': receipt.get('counts'), 'schedule': receipt.get('schedule')}), flush=True)
    sys.exit(code)
