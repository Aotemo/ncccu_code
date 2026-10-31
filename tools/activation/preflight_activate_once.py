"""R124R1 one-shot CPU preflight and activation, without a transport round trip."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROUND = 'r124r1-small22k-seed20261029-scale085-pairlog-contentpool-24k-p4-9010-20261031'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute_sequence(preflight_command, audit_prefix, preflight_path, observed_at, launch,
                     now=time.time):
    """Run one preflight, then pass that exact receipt into one audit call."""
    for command in (preflight_command, audit_prefix):
        position = command.index('--observed-at')
        if command[position + 1] != str(observed_at):
            raise ValueError('R124R1 snapshot timestamp was changed between calls')
    if audit_prefix[audit_prefix.index('--site-preflight') + 1] != str(preflight_path):
        raise ValueError('R124R1 audit does not consume this preflight receipt')
    if launch('preflight', preflight_command) != 0:
        raise ValueError('R124R1 preflight failed; activation not called')
    receipt = json.loads(Path(preflight_path).read_text())
    if (receipt.get('exit_code') != 0 or receipt.get('observed_at') != observed_at
            or receipt.get('round_id') != ROUND):
        raise ValueError('R124R1 preflight receipt identity differs; activation not called')
    # Reserve a few seconds for the existing, unchanged 120-second activation gate.
    if not 0 <= now() - observed_at <= 110:
        raise ValueError('R124R1 snapshot stale before activation; activation not called')
    digest = sha(preflight_path)
    if launch('audit', [*audit_prefix, '--site-preflight-sha256', digest]) != 0:
        raise ValueError('R124R1 unique activation failed; preserve its audit and never retry')
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--authorization-reference', required=True)
    args = parser.parse_args()
    root = args.work_root.resolve(strict=True)
    formal = root / 'formal-cpu'
    site = formal / 'package/SITE_BINDING.json'
    site_value = json.loads(site.read_text())
    if (site_value['round_id'] != ROUND or
            Path(site_value['round_root']).name != ROUND):
        raise ValueError('R124R1 must have its own round root')
    pod_live = root / 'POD-LIVE.json'
    observed_at = pod_live.stat().st_mtime  # Time the fresh kubectl snapshot was received.
    if not 0 <= time.time() - observed_at <= 110:
        raise ValueError('R124R1 snapshot is stale before preflight')
    pod = json.loads(pod_live.read_text())
    if (pod['metadata']['uid'] != os.environ['NCCCU_POD_UID'] or
            pod['status']['phase'] != 'Running' or not any(
                row['type'] == 'Ready' and row['status'] == 'True'
                for row in pod['status']['conditions'])):
        raise ValueError('R124R1 actual snapshot Pod identity or readiness differs')
    preflight = root / 'SITE-PREFLIGHT.json'
    audit_root = root / 'activation-audit'
    if preflight.exists() or audit_root.exists():
        raise ValueError('R124R1 one-shot preflight or activation already attempted')
    preflight_command = [sys.executable, '-B', str(formal / 'site_preflight.py'),
        '--prep-root', str(formal), '--site-binding', str(site),
        '--site-binding-sha256', sha(site), '--pod-live', str(pod_live),
        '--pod-live-sha256', sha(pod_live), '--observed-at', str(observed_at),
        '--output', str(preflight)]
    audit_prefix = [sys.executable, '-B', str(formal / 'activate_audit.py'),
        '--activate-formal', '--prep-root', str(formal),
        '--authorization-reference', args.authorization_reference,
        '--site-binding', str(site), '--site-binding-sha256', sha(site),
        '--package-manifest-sha256', sha(formal / 'PACKAGE-MANIFEST.json'),
        '--cpu-prep-receipt-sha256', sha(formal / 'CPU-PREP-RECEIPT.json'),
        '--deployment-receipt', str(root / 'DEPLOYMENT.json'),
        '--deployment-receipt-sha256', sha(root / 'DEPLOYMENT.json'),
        '--site-preflight', str(preflight), '--pod-uid', os.environ['NCCCU_POD_UID'],
        '--ready', 'True', '--observed-at', str(observed_at),
        '--audit-root', str(audit_root)]

    def launch(stage, command):
        with (root / ('CHAIN-' + stage.upper() + '.log')).open('x') as stream:
            return subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode

    receipt_sha = execute_sequence(preflight_command, audit_prefix, preflight,
                                   observed_at, launch)
    result = {'status': 'R124R1_unique_activation_exit0', 'round_id': ROUND,
              'observed_at': observed_at, 'preflight_sha256': receipt_sha,
              'activation_audit_sha256': sha(audit_root / 'FORMAL-BINDING-AUDIT.json'),
              'GPU_started': False}
    with (root / 'CHAIN-RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
