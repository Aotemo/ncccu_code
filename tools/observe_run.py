"""Read-only R91 observation on the designated Pod; never dispatch or signal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

F = Path('/public/home/acfwvv9ejn/SothisAI/exp-v2/auroc999-20261027/rounds/R91-R50-IMAGENETV2-9010')
TOKEN = 'ffb6b56327af424bac31bd605df6259e'


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        encoded_exit = int(fields[49]) if len(fields) > 49 and fields[0] in ('Z', 'X') else None
        return {'pid': pid, 'state': fields[0], 'start_ticks': fields[19],
                'argv': [s.decode(errors='replace') for s in (p / 'cmdline').read_bytes().split(b'\0') if s],
                'exit_code_encoded': encoded_exit,
                'actual_exit_code': os.waitstatus_to_exitcode(encoded_exit) if encoded_exit is not None else None}
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    assert args.tag and all(c.isalnum() or c in '-_' for c in args.tag)
    assert socket.gethostname() == 'vcjob-2114373723310104577-acfwvv9ejn-90647-worker-0'
    assert sys.executable == '/usr/local/python3.12/bin/python3.12'
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert all(os.environ.get(k) == '' for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
    clock = read(F / 'control/round-clock.json')
    assert clock['token'] == TOKEN
    cfg = read(F / 'TRAIN_CONFIG.json')
    run = F / 'outputs' / cfg['run_name']
    model = run / cfg['jobs'][0]['name']
    out = Path(__file__).resolve().parent / 'snapshots' / args.tag
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    paths = list((F / 'control').glob('*.json'))
    paths += [F / 'evidence' / n for n in ('FORMAL_BINDING_RECEIPT.json', 'ACTUAL_FRESH_INITIAL.json', 'TRAIN_RESULT.json', 'R91_RESULT.json')]
    paths += [F / 'SINGLE-MODEL-FREEZE.json', F / 'official/OFFICIAL-RESULT.json', run / 'complete.json']
    paths += [model / n for n in ('status.json', 'complete.json', 'selection.json', 'initial-ranks.json', 'bn-initial-ranks.json')]
    paths += list((run / 'rank-control').glob('*.json'))
    manifest = []
    for p in sorted(set(paths)):
        if not p.is_file():
            continue
        data = p.read_bytes()
        value = json.loads(data)
        rel = p.relative_to(F)
        target = out / 'files' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append({'path': str(p), 'snapshot': str(target), 'bytes': len(data), 'sha256': sha(data)})
        records[str(rel)] = value
    metrics_path = model / 'metrics.jsonl'
    events = []
    if metrics_path.is_file():
        data = metrics_path.read_bytes()
        (out / 'metrics.jsonl').write_bytes(data)
        for line in data.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A writer may have an incomplete final line; never infer it.
    evaluations = [e for e in events if e.get('event') == 'evaluation']
    logs = {}
    for p in [F / 'evidence/controller.log', F / 'evidence/train-component.log', F / 'evidence/freeze-component.log', F / 'evidence/official-component.log', *run.glob('logs/*')]:
        if p.is_file():
            with p.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 6000))
                logs[str(p.relative_to(F))] = stream.read().decode(errors='replace')
    identities = []
    for key, value in records.items():
        if not isinstance(value, dict):
            continue
        expected = value.get('owner') or value.get('gate_identity') or (value if key.endswith('-launch.json') and 'pid' in value else None)
        if not isinstance(expected, dict) or 'pid' not in expected or 'start_ticks' not in expected:
            continue
        actual = identity(expected['pid'])
        identities.append({'record': key, 'expected': expected, 'actual': actual,
                           'same_start_live': bool(actual and actual['state'] not in ('Z', 'X') and actual['start_ticks'] == str(expected['start_ticks']))})
    lock = F.parents[2] / '.instance-owner-eec33bd6-b567-40ef-b866-633e272ea019.lock'
    inode = lock.stat().st_ino
    locks = [l for l in Path('/proc/locks').read_text().splitlines() if len(l.split()) > 5 and l.split()[5].endswith(':' + str(inode))]
    initial = records.get('evidence/ACTUAL_FRESH_INITIAL.json')
    barrier = records.get('control/train-startup-barrier.json')
    gate_verified = False
    if initial and barrier:
        initial_bytes = (F / 'evidence/ACTUAL_FRESH_INITIAL.json').read_bytes()
        assert initial['status'] == 'completed' and initial['rank_count'] == 4 and initial['optimizer_state_entries'] == 0
        assert initial['model_tensor_sha256'] == 'b88683d2cd788e74127e86225bed1cbbbec2fbe3baa56e3b172a33b8ef92ea96'
        assert initial['frozen_bn_sha256'] == '21b9ef124340ced1d13ffe0ea1ef8214f800c05ea0f7bdb9572f06d26475b210'
        assert barrier['actual_initial_receipt_sha256'] == sha(initial_bytes) and barrier['all_initials_passed']
        gate_verified = True
    result = {'status': 'read_only_observation', 'at': time.time(), 'tag': args.tag, 'token': TOKEN,
              'clock': clock, 'snapshot_root': str(out), 'lock_inode': inode, 'proc_locks': locks,
              'actual_initial_gate_verified': gate_verified,
              'latest_event': events[-1] if events else None,
              'evaluations': evaluations,
              'model_status': records.get(str((model / 'status.json').relative_to(F))),
              'done': records.get('control/done.json'), 'identities': identities,
              'records': records, 'log_tails': logs, 'manifest': manifest,
              'signals_sent': False, 'launch_attempted': False, 'project_model_loaded': False, 'exit_code': 0}
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    (out / 'OBSERVATION.json').write_text(payload)
    status = result['model_status'] or {}
    summary = {key: result[key] for key in ('status', 'at', 'tag', 'token', 'snapshot_root', 'lock_inode', 'proc_locks',
               'actual_initial_gate_verified', 'latest_event', 'evaluations', 'done', 'exit_code')}
    summary['model_status'] = {k: status.get(k) for k in ('status', 'successful_updates', 'attempted_updates', 'skipped_updates',
                'cycle', 'evaluated_cycle', 'stop_reason', 'ema', 'last_metrics', 'ema_last_metrics', 'timing', 'wall_seconds_this_process')}
    summary['identities'] = [{'record': x['record'], 'pid': x['expected']['pid'], 'start_ticks': x['expected']['start_ticks'],
                              'actual_state': x['actual']['state'] if x['actual'] else None,
                              'actual_exit_code': x['actual']['actual_exit_code'] if x['actual'] else None,
                              'same_start_live': x['same_start_live']} for x in identities]
    summary['terminals'] = {k: v for k, v in records.items() if k.endswith('terminal.json') or k in ('evidence/R91_RESULT.json', 'official/OFFICIAL-RESULT.json')}
    summary['errors_in_log_tails'] = {k: v for k, v in logs.items() if 'Traceback' in v or 'Error' in v}
    summary['observation_sha256'] = sha(payload.encode())
    summary['observer_sha256'] = sha(Path(__file__).read_bytes())
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
