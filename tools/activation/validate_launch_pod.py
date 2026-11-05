"""Validate the post-activation Pod snapshot before launching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def validate_request_time(observed, request, now):
    if (type(observed) is not int or type(request) is not int or type(now) not in (int, float)
            or observed < request or not 0 <= now - observed <= 45):
        raise ValueError('launch snapshot predates activation or lacks launch time reserve')
    return True


def validate_snapshot(pod, site, *, observed_at, now=None, hostname=None, python=None):
    from retrain.site_binding import verify_live
    try:
        metadata, status, specification = pod['metadata'], pod['status'], pod['spec']
        expected = site['pod']
        if (metadata['name'] != expected['name'] or metadata['namespace'] != expected['namespace']
                or metadata.get('deletionTimestamp') or status['phase'] != 'Running'
                or expected['container'] not in {item['name'] for item in specification['containers']}):
            raise ValueError('launch snapshot name, namespace, container, phase or deletion differs')
        ready = any(item['type'] == 'Ready' and item['status'] == 'True'
                    for item in status['conditions'])
        verify_live(site, pod_uid=metadata['uid'], ready=ready, observed_at=observed_at,
                    now=now, hostname=hostname, python=python)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('launch snapshot is incomplete') from error
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('site-binding', 'pod-live', 'request', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--site-binding-sha256', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.site_binding.parent / 'bundle'))
    from retrain.site_binding import evidence_path, load_site, require_cpu_channel
    site = load_site(args.site_binding, expected_sha256=args.site_binding_sha256)
    require_cpu_channel(site)
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(key) != '' for key in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
            or site['pod']['uid'] != os.environ.get('NCCCU_POD_UID')):
        raise ValueError('guarded hidden-device current Pod required')
    for path in (args.pod_live, args.request, args.output):
        evidence_path(path, site['work_root'])
    if (args.pod_live.name != 'POD-LIVE-LAUNCH.json'
            or args.request != args.pod_live.parent / 'ASYNC-HANDOFF/LIVE-REQUEST.json'
            or args.output != args.pod_live.parent / 'ASYNC-HANDOFF/LIVE-VERIFIED.json'):
        raise ValueError('launch snapshot and receipt paths differ from this handoff')
    request = json.loads(args.request.read_text())
    before = args.pod_live.stat()
    raw = args.pod_live.read_bytes()
    after = args.pod_live.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if (args.pod_live.is_symlink() or any(getattr(before, k) != getattr(after, k) for k in fields)
            or before.st_size != len(raw) or request.get('round_id') != site['round_id']
            or type(request.get('request_at_epoch')) is not int):
        raise ValueError('launch snapshot or request changed')
    observed = int(before.st_mtime)
    validate_request_time(observed, request['request_at_epoch'], time.time())
    validate_snapshot(json.loads(raw), site, observed_at=observed)
    result = {'status': 'post_activation_live_pod_verified', 'round_id': site['round_id'],
              'pod_uid': site['pod']['uid'], 'site_sha256': args.site_binding_sha256,
              'snapshot_sha256': hashlib.sha256(raw).hexdigest(), 'observed_at': observed,
              'request_at_epoch': request['request_at_epoch'], 'at': time.time(),
              'GPU_initialized': False, 'official_read': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
