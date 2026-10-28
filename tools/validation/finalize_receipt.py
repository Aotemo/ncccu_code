"""Audit completed R92 originals on Pod; no model construction or forward."""
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

F = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/R92-TINY1K-9010')
TOKEN = 'f1d4b2f895a84a15a3d81634b5153cee'
SOURCE = '57ef83209692eb79e682e04e2b3fe7a060225856f673505ab67396080216d51e'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def ident(old):
    p = Path('/proc') / str(old['pid'])
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        same = fields[19] == str(old['start_ticks'])
        encoded = int(fields[49]) if same and fields[0] in ('Z', 'X') else None
        return {'pid': old['pid'], 'start_ticks': fields[19], 'same_start': same, 'state': fields[0],
                'active': same and fields[0] not in ('Z', 'X'), 'encoded_exit': encoded,
                'actual_exit_code': os.waitstatus_to_exitcode(encoded) if encoded is not None else None}
    except FileNotFoundError:
        return {'pid': old['pid'], 'same_start': False, 'active': False, 'actual_exit_code': None}


def main():
    assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
    assert sys.executable == '/usr/local/python3.12/bin/python3.12' and os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
    out = Path(__file__).resolve().parent / 'result'
    out.mkdir(exist_ok=False)
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    clock, cfg = read(F / 'control/round-clock.json'), read(F / 'TRAIN_CONFIG.json')
    assert clock['token'] == TOKEN
    run = F / 'outputs' / cfg['run_name']
    model = run / cfg['jobs'][0]['name']
    done = read(F / 'control/done.json')
    owner = ident(read(F / 'control/owner.json')['owner'])
    rank_results = read(run / 'complete.json')
    complete = read(model / 'complete.json')
    components = {phase: read(F / 'control' / (phase + '-component-terminal.json')) for phase in ('train', 'freeze', 'official')}
    check('owner_terminal_success_and_OS_exit0', done['status'] == 'completed' and not owner['active'] and owner['actual_exit_code'] == 0)
    check('four_rank_real_exit0', rank_results['status'] == 'completed' and len(rank_results['exit_codes']) == 4 and all(code == 0 for code in rank_results['exit_codes'].values()))
    check('all_bounded_components_exit0', all(c['status'] == 'completed' and c['exit_code'] == 0 and not c['live_owned_ranks'] and c['hard_deadline_met'] for c in components.values()))
    check('complete_8cycles_6400_no_skip', complete['status'] == 'completed' and complete['cycle'] == complete['evaluated_cycle'] == 8 and complete['successful_updates'] == complete['attempted_updates'] == 6400 and complete['skipped_updates'] == 0)
    check('runtime_EMA_1600copy_4800blend', complete['ema']['copy_count'] == 1600 and complete['ema']['blend_count'] == 4800 and complete['ema']['update_count'] == 6400)
    initial = read(F / 'evidence/ACTUAL_FRESH_INITIAL.json')
    barrier = read(F / 'control/train-startup-barrier.json')
    check('real_fresh_initial_bound', initial['status'] == 'completed' and initial['rank_count'] == 4 and initial['optimizer_state_entries'] == 0 and initial['source_digest'] == SOURCE and barrier['actual_initial_receipt_sha256'] == sha(F / 'evidence/ACTUAL_FRESH_INITIAL.json') and initial['model_tensor_sha256'] == 'a968bb5c9bc892c966c850fe0c3d9c89c17f409eefe91ca66c66aefc67a1bb4b' and initial['parameter_tensors'] == 183)
    freeze_path = F / 'SINGLE-MODEL-FREEZE.json'
    frozen = read(freeze_path)
    entries = frozen['dev_entries']
    selected = min(entries, key=lambda x: (x['error_twice'], -x['ap'], x['successful_updates'], 0 if x['variant'] == 'ema' else 1))
    check('all16_unique_dev_only_selection', len(entries) == 16 and {(e['cycle'], e['variant']) for e in entries} == {(c, v) for c in range(1, 9) for v in ('online', 'ema')} and all(e['successful_updates'] == e['cycle'] * 800 for e in entries) and selected == frozen['selected'] and not frozen['official_metadata_used_for_selection'])
    check('production_source_variant_tensor', not frozen['cpu_fixture_only'] and frozen['source_digest'] == SOURCE and frozen['variant'] == selected['variant'] and frozen['tensor_sha256'] == selected['tensor_sha256'])
    check('same_stage1_source_age_manifest', frozen['experiment91'] == cfg['jobs'][0]['experiment91'])
    sys.path.insert(0, str(F / 'bundle'))
    from retrain.experiment91 import binding, validate_normalization
    actual_normalization = frozen['candidate_normalization']
    check('Tiny_candidate2_actual_noBN_LN_SD_contract', frozen['experiment_order'] == 2
        and frozen['experiment91'] == binding() and validate_normalization(actual_normalization)
        and actual_normalization == initial['candidate_normalization']
        and initial['batch_norm_modules'] == initial['buffer_count'] == 0
        and frozen['buffer_sha256'] == initial['buffer_sha256'] == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
    from retrain.persistence import canonical_digest
    actual_sources = {str(p.relative_to(F / 'bundle')): sha(p) for p in sorted([*(F / 'bundle').glob('*.py'), *(F / 'bundle/retrain').glob('*.py')])}
    check('selected_blob_and_frozen_source_bytes', sha(frozen['checkpoint']) == frozen['blob_sha256']
        and actual_sources == frozen['source_files'] and canonical_digest(actual_sources) == SOURCE)
    check('frozen_dev_vector_and_precision_bytes', all(sha(e['prediction_path']) == e['prediction_sha256'] and sha(e['precision_path']) == e['precision_sha256'] for e in entries))
    result = read(F / 'official/OFFICIAL-RESULT.json')
    reservation_path = model / 'OFFICIAL-ONCE.json'
    reservation = read(reservation_path)
    check('canonical_once_reservation_frozen_identity', result['reservation'] == reservation and reservation['freeze_sha256'] == sha(freeze_path) and reservation['blob_sha256'] == frozen['blob_sha256'] and reservation['tensor_sha256'] == frozen['tensor_sha256'] and reservation['variant'] == frozen['variant'])
    check('actual_official_strict_variant_tensor', result['source']['tensor_sha256'] == frozen['tensor_sha256'] and result['source']['variant'] == frozen['variant'] and result['source']['identity'] == frozen['identity'] and result['source']['sha256'] == frozen['blob_sha256'])
    assets = read(F / 'ASSETS.json')
    manifest = assets['manifests']['official_valid']
    check('official_bound_sealed_manifest', result['eval_manifest_sha256'] == manifest['sha256'] == sha(manifest['remote_path']))
    sealed = read(manifest['remote_path'])
    pred_path = F / 'official/official-raw.csv'
    with pred_path.open(newline='') as stream:
        raw = list(csv.DictReader(stream))
    vector = [{'name': r['name'], 'label': int(r['label']), 'logit': float(r['logit'])} for r in raw]
    check('official_full5849_order_labels', len(vector) == len(sealed['rows']) == 5849 and all(v['name'] == r['name'] and v['label'] == r['label'] for v, r in zip(vector, sealed['rows'])))
    positive = sum(v['label'] for v in vector)
    negative = len(vector) - positive
    check('official191P5658N', (positive, negative) == (191, 5658))
    sys.path.insert(0, str(F / 'bundle'))
    from retrain.n05_metrics import metrics
    exact = metrics(vector)
    check('official_raw_metrics_from_existing_vector', all(result['metrics'][k] == exact[k] for k in ('error_twice', 'ap', 'auroc')) and result['output_sha256'] == sha(pred_path))
    target = 1000 * exact['error_twice'] <= 2 * positive * negative
    check('target_exact_integer_predicate', result['target_auroc_at_least_0_999'] is target)
    lock = F.parents[2] / '.instance-owner-eec33bd6-b567-40ef-b866-633e272ea019.lock'
    inode = lock.stat().st_ino
    matches = [l for l in Path('/proc/locks').read_text().splitlines() if len(l.split()) > 5 and l.split()[5].endswith(':' + str(inode))]
    ranks = [ident(read(p)) for p in (run / 'rank-control').glob('rank-*-launch.json')]
    check('owner_ranks_inactive_original_lock_released', inode == 18030447997 and not matches and len(ranks) == 4 and not any(r['active'] for r in ranks) and not done['live_owned_ranks'])
    sys.path.insert(0, str(F))
    import run_round
    current_cards = run_round.card_snapshot(); current_device_users = run_round.device_users_before_start()
    check('terminal_four_cards_idle_no_device_users', not current_device_users and len(current_cards) == 4
        and all(c['busy_percent'] == 0 and c['vram_used_bytes'] < 64 * 1024**2 for c in current_cards))
    check('overall_deadline_met', done['ended_at'] <= clock['deadline_at'])
    value = {'status': 'experiment_evidence_verified' if all(c['passed'] for c in checks) else 'evidence_check_failed',
        'at': time.time(), 'token': TOKEN, 'source_digest': SOURCE, 'checks': checks,
        'owner': owner, 'ranks_now': ranks, 'rank_exit_codes': rank_results['exit_codes'], 'components': components,
        'done': done, 'clock': clock, 'dev_entries': entries, 'selected': selected,
        'freeze': {'path': str(freeze_path), 'sha256': sha(freeze_path), 'checkpoint': frozen['checkpoint'], 'blob_sha256': frozen['blob_sha256'], 'variant': frozen['variant'], 'tensor_sha256': frozen['tensor_sha256'], 'experiment91': frozen['experiment91'], 'buffer_sha256': frozen['buffer_sha256'], 'candidate_normalization': actual_normalization},
        'official': {'metrics': exact, 'rows': len(vector), 'positive': positive, 'negative': negative,
            'auroc_exact_fraction': str(Fraction(2 * positive * negative - exact['error_twice'], 2 * positive * negative)),
            'target_auroc_at_least_0_999': target, 'target_max_error_twice': (2 * positive * negative) // 1000,
            'prediction_path': str(pred_path), 'prediction_sha256': sha(pred_path),
            'result_sha256': sha(F / 'official/OFFICIAL-RESULT.json'), 'reservation_sha256': sha(reservation_path)},
        'original_lock_inode': inode, 'proc_locks': matches, 'terminal_cards': current_cards, 'terminal_device_users': current_device_users,
        'no_model_created_or_forward': True, 'official_invocations_added': 0,
        'script_sha256': sha(Path(__file__)), 'exit_code': 0 if all(c['passed'] for c in checks) else 1}
    (out / 'FINAL-RECEIPT.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    files = [F / 'RUNTIME_BINDING.json', freeze_path, F / 'official/OFFICIAL-RESULT.json', pred_path, reservation_path,
        F / 'control/done.json', F / 'evidence/ACTUAL_FRESH_INITIAL.json', run / 'complete.json', model / 'complete.json']
    files += [F / 'control' / (phase + '-component-terminal.json') for phase in components]
    output = {'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files],
              'receipt_path': str(out / 'FINAL-RECEIPT.json'), 'receipt_sha256': sha(out / 'FINAL-RECEIPT.json'), 'exit_code': value['exit_code']}
    (out / 'OUTPUT-MANIFEST.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'status': value['status'], 'exit_code': value['exit_code'], 'token': TOKEN,
        'checks': len(checks), 'passed': sum(c['passed'] for c in checks), 'selected': selected,
        'official': value['official'], 'owner': owner, 'rank_exit_codes': rank_results['exit_codes'],
        'components': {p: {k: c[k] for k in ('status', 'exit_code', 'component_wall_seconds')} for p, c in components.items()},
        'original_lock_inode': inode, 'proc_locks': matches, 'receipt': str(out / 'FINAL-RECEIPT.json'),
        'receipt_sha256': sha(out / 'FINAL-RECEIPT.json')}, indent=2), flush=True)
    print(json.dumps(output, indent=2), flush=True)
    raise SystemExit(value['exit_code'])


if __name__ == '__main__':
    main()
