"""One-shot, read-only round observer for an already guarded Pod.

The caller supplies every path and the mode; this module never allocates CPU,
contacts a host, runs a model, reads checkpoints, or keeps polling.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

TERMINAL = ('completed', 'failed', 'cancelled')
PROC = Path('/proc')
DRM = Path('/sys/class/drm')
LOCKS = Path('/proc/locks')


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def atomic_write(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.' + target.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def read_process(pid, proc_root=PROC):
    path = Path(proc_root) / str(pid) / 'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': pid, 'present': False, 'readable': True, 'state': None,
                'start_ticks': None, 'active': False}
    except PermissionError:
        return {'pid': pid, 'present': True, 'readable': False, 'state': None,
                'start_ticks': None, 'active': None}
    state = fields[0]
    return {'pid': pid, 'present': True, 'readable': True, 'state': state,
            'start_ticks': fields[19], 'active': state not in ('Z', 'X')}


def compare_identity(expected, actual):
    if expected is None:
        return 'unknown'
    if not actual['readable']:
        return 'unreadable'
    if not actual['present']:
        return 'missing'
    if str(actual['start_ticks']) != str(expected['start_ticks']):
        return 'mismatch'
    return 'active' if actual['active'] else 'exited'


def expected_identities(receipt):
    if not isinstance(receipt, dict):
        return None, None
    owner = receipt.get('owner')
    owner_expected = ({'pid': owner['pid'], 'start_ticks': str(owner['start_ticks'])}
                      if isinstance(owner, dict) and 'pid' in owner and 'start_ticks' in owner
                      else None)
    rows = receipt.get('rank_processes') or receipt.get('ranks') or []
    expected = [{'pid': row['pid'], 'start_ticks': str(row['start_ticks'])}
                for row in rows if isinstance(row, dict) and 'pid' in row and 'start_ticks' in row]
    return owner_expected, expected


def identity_snapshot(owner_expected, ranks_expected, proc_root=PROC):
    owner_actual = read_process(owner_expected['pid'], proc_root) if owner_expected else None
    ranks_actual = [read_process(row['pid'], proc_root) for row in ranks_expected or []]
    owner_status = compare_identity(owner_expected, owner_actual) if owner_expected else 'unknown'
    rank_status = [compare_identity(expected, actual)
                   for expected, actual in zip(ranks_expected or [], ranks_actual)]
    identity_match = [status in ('active', 'exited') for status in rank_status]
    return {'owner_expected': owner_expected, 'owner': owner_actual, 'owner_status': owner_status,
            'owner_match': owner_status in ('active', 'exited') if owner_expected else None,
            'ranks_expected': ranks_expected or [], 'ranks': ranks_actual,
            'rank_status': rank_status, 'rank_matches': identity_match,
            'rank_count': len(ranks_actual)}


def scan_device_users(proc_root=PROC):
    users, denied = [], 0
    try:
        entries = [name for name in os.listdir(proc_root) if name.isdigit()]
    except PermissionError:
        return [], 1
    for name in entries:
        pid = int(name)
        if pid == os.getpid():
            continue
        base = Path(proc_root) / name
        try:
            fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] in ('Z', 'X'):
                continue
            for descriptor in os.listdir(base / 'fd'):
                try:
                    device = os.readlink(base / 'fd' / descriptor)
                    if device == '/dev/kfd' or device.startswith('/dev/dri/renderD'):
                        users.append({'pid': pid, 'start_ticks': fields[19], 'device': device})
                        break
                except (FileNotFoundError, ProcessLookupError):
                    pass
                except PermissionError:
                    denied += 1
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            denied += 1
    return users, denied


def scan_cards(drm_root=DRM, expected=()):
    cards, readable = [], True
    for path in sorted(Path(drm_root).glob('card[0-9]*')):
        if not path.name[4:].isdigit():
            continue
        device = (path / 'device').resolve()
        if device.name not in expected:
            continue
        try:
            cards.append({'pci': device.name,
                          'busy_percent': int((device / 'gpu_busy_percent').read_text())})
        except (OSError, ValueError):
            readable = False
    return {'cards': cards, 'readable': readable,
            'all_expected_cards_seen': {card['pci'] for card in cards} == set(expected)}


def lock_evidence(work_root, pod_uid, locks_path=LOCKS):
    lock = Path(work_root) / 'control' / ('.instance-owner-' + str(pod_uid) + '.lock')
    if not lock.is_file():
        return {'present': False, 'readable': True, 'holders': []}
    try:
        inode = lock.stat().st_ino
        lines = Path(locks_path).read_text().splitlines()
    except OSError:
        return {'present': True, 'readable': False, 'holders': []}
    holders = [line for line in lines if len(line.split()) > 5
               and line.split()[5].endswith(':' + str(inode))]
    return {'present': True, 'readable': True, 'holders': holders}


def terminal_resources(formal_root, owner_expected, ranks_expected, proc_root=PROC,
                       drm_root=DRM, locks_path=LOCKS):
    site = read_json(Path(formal_root) / 'SITE_BINDING.json')
    if not isinstance(site, dict) or not site.get('pod') or not site.get('work_root'):
        return {'safe': None, 'reason': 'site_binding_unknown'}
    gpus = site.get('gpus')
    if not isinstance(gpus, list) or not gpus:
        return {'safe': None, 'reason': 'gpu_binding_unknown'}
    pcis, ordinals = [], []
    for card in gpus:
        if (not isinstance(card, dict) or not isinstance(card.get('pci'), str)
                or not card['pci'] or 'ordinal' not in card):
            return {'safe': None, 'reason': 'gpu_binding_incomplete'}
        pcis.append(card['pci'])
        ordinals.append(card['ordinal'])
    if (len(set(pcis)) != len(pcis) or len(set(ordinals)) != len(ordinals)
            or sorted(ordinals) != list(range(len(gpus)))):
        return {'safe': None, 'reason': 'gpu_binding_incomplete'}
    world = len(gpus)
    if not ranks_expected or len(ranks_expected) != world:
        return {'safe': None, 'reason': 'rank_binding_incomplete'}
    rank_pids = [row['pid'] for row in ranks_expected]
    if len(set(rank_pids)) != world:
        return {'safe': None, 'reason': 'rank_binding_duplicate'}
    identity = identity_snapshot(owner_expected, ranks_expected, proc_root)
    statuses = [identity['owner_status']] + identity['rank_status']
    if identity['owner_expected'] is None:
        return {'safe': None, 'reason': 'identity_unknown', 'identity': identity}
    if any(status in ('unreadable', 'unknown') for status in statuses):
        return {'safe': None, 'reason': 'identity_unreadable', 'identity': identity}
    if any(status == 'mismatch' for status in statuses):
        return {'safe': False, 'reason': 'identity_mismatch', 'identity': identity}
    if any(status == 'active' for status in statuses):
        return {'safe': False, 'reason': 'process_active', 'identity': identity}

    lock = lock_evidence(site['work_root'], site['pod'].get('uid'), locks_path)
    if not lock['readable']:
        return {'safe': None, 'reason': 'lock_unreadable', 'identity': identity, 'lock': lock}
    if lock['holders']:
        return {'safe': False, 'reason': 'lock_held', 'identity': identity, 'lock': lock}

    users, denied = scan_device_users(proc_root)
    if denied:
        return {'safe': None, 'reason': 'device_scan_denied', 'identity': identity,
                'lock': lock, 'device_users': live_users(users), 'device_scan_denied': denied}
    if users:
        return {'safe': False, 'reason': 'device_users', 'identity': identity, 'lock': lock,
                'device_users': live_users(users), 'device_scan_denied': denied}

    expected = set(pcis)
    cards = scan_cards(drm_root, expected)
    if not cards['readable']:
        return {'safe': None, 'reason': 'cards_unreadable', 'identity': identity, 'lock': lock,
                'device_users': [], 'device_scan_denied': 0, 'cards': cards['cards']}
    if not cards['all_expected_cards_seen']:
        return {'safe': None, 'reason': 'cards_missing', 'identity': identity, 'lock': lock,
                'device_users': [], 'device_scan_denied': 0, 'cards': cards['cards']}
    if any(card['busy_percent'] != 0 for card in cards['cards']):
        return {'safe': False, 'reason': 'cards_busy', 'identity': identity, 'lock': lock,
                'device_users': [], 'device_scan_denied': 0, 'cards': cards['cards']}
    return {'safe': True, 'reason': 'terminal_idle', 'identity': identity, 'lock': lock,
            'device_users': [], 'device_scan_denied': 0, 'cards': cards['cards'],
            'all_expected_cards_seen': True}


def live_users(users):
    return [{'pid': row['pid'], 'start_ticks': row['start_ticks'], 'device': row['device']}
            for row in users]


def fields(value, names):
    return {key: value.get(key) for key in names} if isinstance(value, dict) else None


def metric_summary(path):
    if not Path(path).is_file():
        return None, {}
    result, proofs = None, {}
    for line in Path(path).read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        result = value
        if (value.get('event') == 'head_warmup_phase_proof'
                and value.get('successful_updates') in (1, 1600, 1601)):
            records = value.get('rank_proofs') or []
            first = records[0] if records else None
            proofs[str(value['successful_updates'])] = {
                'at': value.get('at'), 'rank_count': len(records),
                'all_world_finite': len(records) == 8 and all(
                    row.get('world_finite') is True for row in records),
                'all_rank_proofs_equal': len(records) == 8 and all(
                    {key: row.get(key) for key in ('phase', 'active_roles', 'role_state')} ==
                    {key: first.get(key) for key in ('phase', 'active_roles', 'role_state')}
                    for row in records),
                'phase': first.get('phase') if first else None,
                'active_roles': first.get('active_roles') if first else None,
                'role_state': first.get('role_state') if first else None}
    return fields(result, ('event', 'at', 'successful_updates', 'attempted_updates',
                           'cycle', 'loss')), proofs


def selected_summary(freeze, round_result):
    source = None
    if isinstance(freeze, dict) and isinstance(freeze.get('selected'), dict):
        source = freeze['selected']
    elif isinstance(round_result, dict):
        source = round_result
    if not isinstance(source, dict):
        return None
    names = ('auroc', 'ap', 'variant', 'cycle', 'successful_updates', 'error_twice',
             'selected_internal_dev_auroc', 'selected_internal_dev_ap')
    result = {key: source[key] for key in names if key in source}
    return result or None


def status_best(status):
    """Whitelist the existing running best fields; never choose or recompute."""
    if not isinstance(status, dict):
        return None
    result = {}
    for name in ('best_metrics', 'ema_best_metrics'):
        value = status.get(name)
        if isinstance(value, dict):
            result[name] = {key: value[key] for key in
                            ('auroc', 'ap', 'variant', 'cycle', 'successful_updates',
                             'error_twice') if key in value}
    for name in ('lineage_best', 'selection'):
        value = status.get(name)
        if not isinstance(value, dict):
            continue
        row = {key: value[key] for key in ('variant', 'cycle', 'successful_updates')
               if key in value}
        metrics = value.get('metrics')
        if isinstance(metrics, dict):
            row['metrics'] = {key: metrics[key] for key in ('auroc', 'ap', 'error_twice')
                              if key in metrics}
        for key in ('auroc', 'ap'):
            if key in value:
                row[key] = value[key]
        result[name] = row
    return result or None


def terminal_bundle(formal, done):
    freeze = read_json(Path(formal) / 'SINGLE-MODEL-FREEZE.json')
    train = read_json(Path(formal) / 'evidence/TRAIN_RESULT.json')
    round_result = read_json(Path(formal) / 'evidence/ROUND_RESULT.json')
    official = read_json(Path(formal) / 'official/OFFICIAL-RESULT.json')
    return {
        'done': fields(done, ('status', 'phase', 'error', 'live_owned_ranks')),
        'train': fields(train, ('status', 'phase', 'successful_updates', 'cycles',
                                'early_stop', 'error')),
        'train_component': fields(train.get('component'), ('exit_code', 'signals'))
        if isinstance(train, dict) else None,
        'freeze': fields(freeze, ('status', 'selected', 'variant')),
        'round': fields(round_result, ('status', 'phase', 'successful_updates', 'cycles',
                                       'official_status', 'official_reason', 'official_evaluated',
                                       'selected_internal_dev_auroc', 'official_metrics',
                                       'target_auroc_at_least_0_999', 'error')),
        'official': fields(official, ('status', 'metrics', 'target_auroc_at_least_0_999'))}, \
        freeze, round_result, official


def base_result(formal, tag, mode, now):
    return {'tool': 'observe_round_once', 'tag': tag, 'mode': mode,
            'observed_at': now,
            'observed_at_beijing': datetime.fromtimestamp(
                now, ZoneInfo('Asia/Shanghai')).isoformat(),
            'formal_root': str(formal),
            'signals_sent': False, 'model_forward_executed': False, 'official_read': False}


def observe(formal_root, first_success_receipt, output, tag, mode,
            *, proc_root=PROC, drm_root=DRM, locks_path=LOCKS, clock=time.time):
    formal = Path(formal_root)
    now = float(clock())
    result = base_result(formal, tag, mode, now)
    done_path = formal / 'control' / 'done.json'

    if mode == 'completion':
        if not done_path.is_file():
            result.update({'done': {'present': False, 'status': None}, 'phase': 'pending',
                           'first_success_receipt_read': False})
            atomic_write(output, result)
            return result
        done = read_json(done_path)
        if done is None:
            result.update({'done': {'present': True, 'status': 'unknown'}, 'phase': 'unknown',
                           'done_error': 'invalid_marker',
                           'first_success_receipt_read': False})
            atomic_write(output, result)
            return result
        status = done.get('status')
        result['done'] = {'present': True, 'status': status}
        if status not in TERMINAL:
            result.update({'phase': 'marker_only', 'first_success_receipt_read': False})
            atomic_write(output, result)
            return result
        receipt = read_json(first_success_receipt)
        owner_expected, ranks_expected = expected_identities(receipt)
        bundle, freeze, round_result, official = terminal_bundle(formal, done)
        result.update({'phase': 'terminal', 'first_success_receipt_read': receipt is not None,
                       'identity': identity_snapshot(owner_expected, ranks_expected, proc_root),
                       'selected': selected_summary(freeze, round_result),
                       'terminal': bundle, 'official_read': official is not None,
                       'safety': terminal_resources(formal, owner_expected, ranks_expected,
                                                    proc_root, drm_root, locks_path)})
        atomic_write(output, result)
        return result

    done = read_json(done_path) if done_path.is_file() else None
    config = read_json(formal / 'TRAIN_CONFIG.json')
    receipt = read_json(first_success_receipt)
    owner_expected, ranks_expected = expected_identities(receipt)
    identity = identity_snapshot(owner_expected, ranks_expected, proc_root)
    run_name = config.get('run_name') if isinstance(config, dict) else None
    job_name = None
    if isinstance(config, dict) and isinstance(config.get('jobs'), list) and config['jobs']:
        job_name = config['jobs'][0].get('name')
    folder = formal / 'outputs' / run_name / job_name if run_name and job_name else None
    status = read_json(folder / 'status.json') if folder is not None else None
    last_metric, proofs = metric_summary(folder / 'metrics.jsonl') if folder is not None else (None, {})
    freeze = read_json(formal / 'SINGLE-MODEL-FREEZE.json')
    round_result = read_json(formal / 'evidence/ROUND_RESULT.json')
    done_status = done.get('status') if isinstance(done, dict) else None
    status_name = status.get('status') if isinstance(status, dict) else None
    terminal = done_status in TERMINAL or status_name in TERMINAL
    progress = None
    if isinstance(status, dict):
        progress = {key: status.get(key) for key in
                    ('status', 'phase', 'successful_updates', 'attempted_updates',
                     'skipped_updates', 'evaluated_cycle', 'stop_reason', 'termination_reason')}
        progress['early_bad_cycles'] = (status.get('early_stop') or {}).get('bad_cycles')
    result.update({'done': {'present': done is not None,
                            'status': done_status if isinstance(done, dict) else None},
                   'round': run_name, 'phase': 'terminal' if terminal else (status_name or 'unknown'),
                   'first_success_receipt_read': receipt is not None,
                   'identity': identity, 'progress': progress, 'metrics': last_metric,
                   'status_best': status_best(status),
                   'head_warmup_phase_proofs': proofs,
                   'selected': selected_summary(freeze, round_result)})
    if terminal:
        bundle, freeze, round_result, official = terminal_bundle(formal, done)
        result.update({'terminal': bundle, 'selected': selected_summary(freeze, round_result),
                       'official_read': official is not None,
                       'safety': terminal_resources(formal, owner_expected, ranks_expected,
                                                    proc_root, drm_root, locks_path)})
    else:
        result['safety'] = None
    atomic_write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description='one-shot read-only round observation')
    parser.add_argument('--formal-root', required=True)
    parser.add_argument('--first-success-receipt', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--mode', required=True, choices=('completion', 'progress'))
    arguments = parser.parse_args()
    result = observe(arguments.formal_root, arguments.first_success_receipt,
                     arguments.output, arguments.tag, arguments.mode)
    print(json.dumps({'output': arguments.output, 'tag': result['tag'],
                      'mode': result['mode'], 'phase': result['phase'],
                      'observed_at': result['observed_at']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
