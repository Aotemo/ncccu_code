"""Explicit deployment identity; no host, Pod, interpreter or device defaults."""
import copy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_site(value):
    try:
        if value['schema_version'] != 1 or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', value['round_id']):
            raise ValueError('site schema/round identity differs')
        if not re.fullmatch(r'[0-9a-f]{40}', value['git_commit']):
            raise ValueError('an actual full Git commit is required')
        root = Path(value['work_root'])
        if not root.is_absolute() or root == Path('/') or '..' in root.parts:
            raise ValueError('explicit absolute work root required')
        root = root.resolve()
        for key in ('round_root', 'metadata_root', 'public_weight', 'public_provenance',
                    'data_root', 'train_identity', 'prepared_cache', 'official_manifest'):
            if key not in value and key in ('data_root', 'train_identity', 'prepared_cache', 'official_manifest'):
                continue
            path = Path(value[key])
            if (not path.is_absolute() or '..' in path.parts or path.is_symlink()
                    or not path.resolve().is_relative_to(root) or path.resolve() == root):
                raise ValueError('site path escapes the work root: ' + key)
        if not Path(value['round_root']).resolve().is_relative_to(root / 'rounds'):
            raise ValueError('formal round root must be inside work_root/rounds')
        pod = value['pod']
        if any(not isinstance(pod[key], str) or not pod[key].strip() for key in ('namespace', 'name', 'uid', 'container', 'python')):
            raise ValueError('complete explicit Pod and interpreter identity required')
        if str(uuid.UUID(pod['uid'])) != pod['uid'] or not Path(pod['python']).is_absolute():
            raise ValueError('Pod UID or interpreter path is invalid')
        cards = value['gpus']
        if (not isinstance(cards, list) or len(cards) != 8
                or any(type(card['ordinal']) is not int for card in cards)
                or sorted(card['ordinal'] for card in cards) != list(range(8))
                or len({card['pci'] for card in cards}) != 8
                or any(not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', card['pci'])
                       or type(card['vram_total_bytes']) is not int or card['vram_total_bytes'] <= 0 for card in cards)):
            raise ValueError('eight unique ordinal/PCI/memory identities are required')
        if not isinstance(value['science'], dict) or not isinstance(value['science']['candidate'], dict):
            raise ValueError('explicit candidate science required')
        if value.get('scope') not in (None, 'cpu_technical_prep', 'formal_exp_v3', 'technical_smoke'):
            raise ValueError('unknown deployment scope')
        if 'cpu_execution' in value:
            validate_cpu_execution(value['cpu_execution'], world=len(cards))
            evidence_path(value['cpu_execution']['allocation_receipt']['path'], root)
        if value.get('scope') == 'formal_exp_v3':
            if any(key not in value for key in ('data_root', 'train_identity', 'prepared_cache', 'official_manifest', 'cpu_execution')):
                raise ValueError('formal site requires complete assets and explicit GPU CPU budget')
            if not isinstance(value['science'].get('binding_version'), str) or not value['science']['binding_version']:
                raise ValueError('formal science binding version required')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('incomplete explicit site binding') from error
    return copy.deepcopy(value)


def load_site(path=None, *, expected_sha256=None):
    path = path if path is not None else os.environ.get('NCCCU_SITE_BINDING')
    if not path or not Path(path).is_absolute() or Path(path).is_symlink():
        raise ValueError('NCCCU_SITE_BINDING must name an explicit absolute regular file')
    expected = expected_sha256 if expected_sha256 is not None else os.environ.get('NCCCU_SITE_BINDING_SHA256')
    if expected is not None and sha256(path) != expected:
        raise ValueError('site binding SHA differs')
    return validate_site(json.loads(Path(path).read_text()))


def expected_pcis(site):
    return [card['pci'] for card in sorted(site['gpus'], key=lambda card: card['ordinal'])]


def verify_live(site, *, pod_uid, ready, observed_at, hostname=None, python=None, now=None):
    site = validate_site(site)
    current = time.time() if now is None else now
    hostname = socket.gethostname() if hostname is None else hostname
    python = sys.executable if python is None else python
    if (pod_uid != site['pod']['uid'] or ready is not True
            or type(observed_at) not in (int, float) or not math.isfinite(observed_at)
            or not 0 <= current - observed_at <= 120
            or hostname != site['pod']['name']
            or os.path.realpath(python) != os.path.realpath(site['pod']['python'])):
        raise ValueError('live Pod UID/Ready/hostname/interpreter/freshness differs')
    return site


def verify_rank_devices(proofs, site):
    site = validate_site(site)
    world = len(site['gpus'])
    if (not isinstance(proofs, list) or len(proofs) != world
            or any(type(proof.get('rank')) is not int for proof in proofs)
            or {proof['rank'] for proof in proofs} != set(range(world))):
        raise ValueError('actual rank coverage differs from the admitted world')
    ordered = sorted(proofs, key=lambda proof: proof['rank'])
    for rank, proof in enumerate(ordered):
        total = proof.get('total_memory_bytes')
        if (proof.get('world_size') != world or proof.get('device') != 'cuda:' + str(rank)
                or proof.get('physical_pci') != expected_pcis(site)[rank]
                or type(total) is not int or total != sorted(site['gpus'], key=lambda card: card['ordinal'])[rank]['vram_total_bytes']):
            raise ValueError('actual rank/device/ordinal-to-PCI identity differs')
        if 'cpu_execution' in site:
            mask = site['cpu_execution']['rank_masks'][rank]
            if proof.get('requested_cpu_ids') != mask or proof.get('actual_cpu_ids') != mask:
                raise ValueError('actual rank CPU mask differs from explicit allocation')
    return ordered


def verify_visible_devices(actual_pcis, site):
    if actual_pcis != expected_pcis(validate_site(site)):
        raise ValueError('visible ordinal-to-PCI mapping differs from the admitted site')


def instance_lock(site):
    site = validate_site(site)
    return Path(site['work_root']) / 'control' / ('.instance-owner-' + site['pod']['uid'] + '.lock')


def evidence_path(path, root):
    path = Path(path)
    if (not path.is_absolute() or '..' in path.parts or path.is_symlink()
            or not path.resolve().is_relative_to(Path(root).resolve()) or path.resolve() == Path(root).resolve()):
        raise ValueError('evidence path must remain inside the explicit work root')
    return path


def cpu_mask(value):
    if (not isinstance(value, list) or not value or any(type(cpu) is not int or cpu < 0 for cpu in value)
            or value != sorted(set(value))):
        raise ValueError('a nonempty sorted unique integer CPU mask is required')
    return value


def validate_cpu_execution(value, *, world):
    try:
        if (not isinstance(value['owner_id'], str) or not value['owner_id'].strip()
                or value['total_fraction_cap'] != '1/5'
                or type(value['effective_cpus']) not in (int, float)
                or not math.isfinite(value['effective_cpus']) or value['effective_cpus'] <= 0
                or value['rank_threads'] != 1 or type(value['rank_threads']) is not int
                or value['interop_threads'] != 1 or type(value['interop_threads']) is not int
                or type(value['loader_workers']) is not int or value['loader_workers'] not in (0, 1)):
            raise ValueError('explicit CPU owner/capacity/thread/loader contract differs')
        owner = cpu_mask(value['owner_mask']); control = cpu_mask(value['control_mask'])
        ranks = value['rank_masks']
        if not isinstance(ranks, list) or len(ranks) != world:
            raise ValueError('one explicit CPU mask per admitted rank is required')
        for mask in ranks:
            cpu_mask(mask)
        combined = control + [cpu for mask in ranks for cpu in mask]
        if (len(set(combined)) != len(combined) or sorted(combined) != owner
                or len(owner) > int(Fraction(str(value['effective_cpus'])) // 5)
                or len({len(mask) for mask in ranks}) != 1
                or any(len(mask) < value['rank_threads'] + value['loader_workers'] for mask in ranks)):
            raise ValueError('rank/control CPU masks overlap, escape owner or exceed the live 20% allocation')
        receipt = value['allocation_receipt']
        if not re.fullmatch(r'[0-9a-f]{64}', receipt['sha256']):
            raise ValueError('explicit allocation receipt SHA required')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('incomplete explicit GPU CPU allocation') from error
    return value


def _cpuset(text):
    cpus = set()
    for item in text.strip().split(','):
        ends = item.split('-')
        if len(ends) == 1:
            cpus.add(int(ends[0]))
        elif len(ends) == 2 and 0 <= int(ends[0]) <= int(ends[1]):
            cpus.update(range(int(ends[0]), int(ends[1]) + 1))
        else:
            raise ValueError('invalid live cgroup cpuset')
    return cpu_mask(sorted(cpus))


def live_cpu_resources():
    affinity = sorted(os.sched_getaffinity(0))
    cpuset = _cpuset(Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text())
    maximum = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if (len(maximum) != 2 or int(maximum[1]) <= 0
            or (maximum[0] != 'max' and int(maximum[0]) <= 0)):
        raise ValueError('invalid live cgroup CPU quota')
    return {'affinity': affinity, 'cpuset': cpuset, 'cpu_max': maximum}


def validate_gpu_allocation(site, receipt, live, *, initial=False, now=None):
    """Use the original admitted affinity after children inherit narrower masks."""
    policy = validate_cpu_execution(site['cpu_execution'], world=len(site['gpus']))
    now = time.time() if now is None else now
    expected = {key: site['pod'][field] for key, field in (
        ('namespace', 'namespace'), ('pod', 'name'), ('uid', 'uid'), ('container', 'container'), ('python', 'python'))}
    try:
        resources = receipt['resources']
        admitted_affinity = cpu_mask(resources['affinity'])
        original_cpuset = cpu_mask(resources['cpuset'])
        if (receipt['schema_version'] != 1 or receipt['stage'] != 'gpu-owner'
                or receipt['execution'] != expected or receipt['total_fraction_cap'] != '1/5'
                or receipt['owner_id'] != policy['owner_id'] or receipt['owner_mask'] != policy['owner_mask']
                or type(receipt['created_at']) not in (int, float) or not math.isfinite(receipt['created_at'])
                or type(receipt['valid_until']) not in (int, float) or not math.isfinite(receipt['valid_until'])
                or not receipt['created_at'] <= now <= receipt['valid_until']
                or resources['effective_cpus'] != policy['effective_cpus']):
            raise ValueError('GPU CPU receipt identity, capacity or validity differs')
        if initial and (live['affinity'] != admitted_affinity or live['cpuset'] != original_cpuset
                        or live['cpu_max'] != resources['cpu_max']):
            raise ValueError('first GPU entry must verify the original uncontracted live CPU resources')
        for cpuset, maximum in ((original_cpuset, resources['cpu_max']), (live['cpuset'], live['cpu_max'])):
            available = set(admitted_affinity) & set(cpu_mask(cpuset))
            if len(maximum) != 2 or int(maximum[1]) <= 0:
                raise ValueError('invalid admitted CPU quota')
            capacity = Fraction(len(available))
            if maximum[0] != 'max':
                capacity = min(capacity, Fraction(int(maximum[0]), int(maximum[1])))
            if (capacity <= 0
                    or len(policy['owner_mask']) > int(capacity // 5)
                    or not set(policy['owner_mask']) <= available):
                raise ValueError('owner CPU allocation exceeds current cpuset/quota 20% cap')
        # The recorded capacity must be an actual result, never an invented low/high default.
        original = Fraction(len(set(admitted_affinity) & set(original_cpuset)))
        quota, period = resources['cpu_max']
        if quota != 'max':
            original = min(original, Fraction(int(quota), int(period)))
        if original != Fraction(str(policy['effective_cpus'])):
            raise ValueError('recorded effective CPU capacity differs from measured allocation')
        releases = receipt['released_executors']
        if (not isinstance(releases, list) or not releases
                or len({item['executor_id'] for item in releases}) != len(releases)
                or any(item['executor_id'] == policy['owner_id'] for item in releases)):
            raise ValueError('other CPU executors must have explicit exit evidence')
    except (KeyError, TypeError, AttributeError, ZeroDivisionError) as error:
        raise ValueError('incomplete GPU CPU allocation receipt') from error
    return policy


def verify_gpu_cpu(site, *, initial=False):
    if site.get('scope') not in ('formal_exp_v3', 'technical_smoke'):
        raise ValueError('CPU technical preparation cannot dispatch a GPU stage')
    reference = site['cpu_execution']['allocation_receipt']
    path = evidence_path(reference['path'], site['work_root'])
    if sha256(path) != reference['sha256']:
        raise ValueError('GPU CPU allocation receipt SHA differs')
    receipt = json.loads(path.read_text())
    live = live_cpu_resources()
    policy = validate_gpu_allocation(site, receipt, live, initial=initial)
    for release in receipt['released_executors']:
        release_path = evidence_path(release['receipt_path'], site['work_root'])
        if sha256(release_path) != release['receipt_sha256']:
            raise ValueError('CPU executor exit receipt SHA differs')
        exited = json.loads(release_path.read_text())
        if (exited.get('executor_id') != release['executor_id'] or exited.get('pod_uid') != site['pod']['uid']
                or exited.get('exit_code') != 0 or exited.get('task_tree_exited') is not True
                or exited.get('live_pids') != [] or type(exited.get('ended_at')) not in (int, float)
                or not math.isfinite(exited['ended_at']) or exited['ended_at'] > receipt['created_at']):
            raise ValueError('CPU executor has no verified prior clean exit')
    return policy, live


def bind_cpu_role(site, role, *, rank=None, initial=False):
    policy, live = verify_gpu_cpu(site, initial=initial)
    if role == 'owner':
        mask = policy['owner_mask']
    elif role == 'control':
        mask = policy['control_mask']
    elif role == 'rank' and type(rank) is int and 0 <= rank < len(site['gpus']):
        mask = policy['rank_masks'][rank]
    else:
        raise ValueError('unknown explicit CPU role')
    if not initial and live['affinity'] not in (policy['owner_mask'], policy['control_mask'], mask):
        raise ValueError('inherited CPU mask is outside the admitted owner/control/rank envelope')
    os.sched_setaffinity(0, set(mask))
    if sorted(os.sched_getaffinity(0)) != mask:
        raise ValueError('failed to apply the explicit CPU role mask')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(OMP_MAX_ACTIVE_LEVELS='1', OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
    return mask


def require_cpu_channel(site=None):
    if site is not None:
        site = validate_site(site)
        if any(os.environ.get(key) != site['pod'][field] for key, field in (
                ('NCCCU_POD_NAMESPACE', 'namespace'), ('NCCCU_POD_NAME', 'name'),
                ('NCCCU_POD_UID', 'uid'), ('NCCCU_CONTAINER', 'container'))):
            raise ValueError('freshly checked CPU channel differs from site binding')
        if socket.gethostname() != site['pod']['name'] or os.path.realpath(sys.executable) != os.path.realpath(site['pod']['python']):
            raise ValueError('actual Pod/interpreter differs from CPU binding')
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError('CPU preparation requires nice 19')
    if any(os.environ.get(key) != '' for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')):
        raise ValueError('CPU preparation requires hidden accelerators')
    raw = os.environ.get('NCCCU_CPU_THREADS', '')
    mask = json.loads(os.environ.get('NCCCU_CPU_MASK', 'null'))
    if (not raw.isdecimal() or int(raw) <= 0 or not isinstance(mask, list)
            or any(type(cpu) is not int or cpu < 0 for cpu in mask)
            or len(set(mask)) != len(mask) or len(mask) != int(raw)
            or sorted(mask) != sorted(os.sched_getaffinity(0))
            or not os.environ.get('NCCCU_CPU_ALLOCATION') or not os.environ.get('NCCCU_EXECUTOR_ID')
            or any(os.environ.get(key) != raw for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('CPU preparation requires an explicit allocated mask and matching thread budget')
