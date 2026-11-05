"""Internal-dev F05 parameter soups from verified selected states only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class StateMean:
    """Accumulate one complete state at a time without holding every parent."""
    def __init__(self):
        self.count = 0
        self.values, self.spec = {}, {}

    def add(self, state):
        import torch
        if not isinstance(state, dict) or not state or (self.count and set(state) != set(self.spec)):
            raise ValueError('nonempty model states with identical keys are required')
        for key in sorted(state):
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.is_complex():
                raise ValueError('unsupported model tensor: ' + str(key))
            value = value.detach().cpu()
            identity = (value.shape, value.dtype, value.layout)
            if self.count and identity != self.spec[key]:
                raise ValueError('model state tensor shape, dtype or layout differs: ' + str(key))
            if value.is_floating_point():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError('model state contains nonfinite values: ' + str(key))
                if not self.count:
                    self.values[key] = torch.zeros(value.shape, dtype=torch.float64, device='cpu')
                self.values[key].add_(value.to(dtype=torch.float64))
            elif not self.count:
                self.values[key] = value.clone()
            elif not torch.equal(self.values[key], value):
                raise ValueError('nonfloating state differs: ' + str(key))
            if not self.count:
                self.spec[key] = identity
        self.count += 1

    def result(self):
        import torch
        if self.count < 2:
            raise ValueError('at least two selected model states are required')
        result = {}
        for key, value in self.values.items():
            dtype = self.spec[key][1]
            result[key] = (value.div(self.count).to(dtype=dtype) if value.is_floating_point() else value.clone())
            if result[key].is_floating_point() and not bool(torch.isfinite(result[key]).all()):
                raise ValueError('model state average is nonfinite: ' + str(key))
        return result


def average_states(states):
    """Uniformly average floating tensors; require identical nonfloating buffers."""
    accumulator = StateMean()
    for state in states:
        accumulator.add(state)
    return accumulator.result()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_cpu():
    if (os.getpriority(os.PRIO_PROCESS, 0) != 19 or any(os.environ.get(key) != '' for key in
            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))):
        raise ValueError('explicit hidden-device Pod CPU allocation required')


def pinned_json(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError('pinned input differs: ' + str(path))
    return json.loads(path.read_text())


def find_member(inventory, name):
    rows = [row for row in inventory['members'] if row.get('round_id', '').split('-')[0] == name]
    if len(rows) != 1:
        raise ValueError('unique selected soup parent required: ' + name)
    return rows[0]


def local_scratch(path):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.parent.resolve(strict=True).is_relative_to(Path('/tmp'))
            or not path.name.startswith('ncccu-f05-soup-')):
        raise ValueError('explicit isolated local soup scratch required')
    return path


def extract(args):
    require_cpu()
    from f04_export_member import construct_member
    import torch
    inventory = pinned_json(args.inventory, args.inventory_sha256)
    row = find_member(inventory, args.member)
    model, _, _, _, _ = construct_member(row)
    if torch.cuda.is_initialized():
        raise ValueError('GPU initialized during selected-state extraction')
    destination = local_scratch(args.output)
    with destination.open('xb') as stream:
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'status': 'selected_state_strictly_extracted', 'member': args.member,
                      'state_sha256': sha(destination), 'GPU_initialized': False,
                      'official_read': False}), flush=True)


def finalize(args):
    require_cpu()
    from f04_export_member import construct_member
    from f04_standalone_member import export_metadata
    import torch
    inventory = pinned_json(args.inventory, args.inventory_sha256)
    row = find_member(inventory, args.member)
    model, job, site, runtime, _ = construct_member(row)
    from retrain.age import forward
    from retrain.ema import tensor_digest
    from retrain.inference import _load_model
    mean = local_scratch(args.state)
    if mean.is_symlink() or sha(mean) != args.state_sha256:
        raise ValueError('sealed averaged state differs')
    state = torch.load(mean, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    variant = row['selected']['variant']
    if variant not in ('ema', 'online'):
        raise ValueError('base historical runtime variant differs')
    output = Path(args.output).resolve()
    work = Path(site['work_root']).resolve(strict=True)
    if output.exists() or not output.is_relative_to(work / 'technical-prep'):
        raise ValueError('fresh soup output in technical-prep required')
    output.mkdir(parents=True)
    exports = {}
    for precision in ('fp32', 'fp16'):
        folder = output / precision; folder.mkdir()
        values = {key: (value.to(torch.float16) if precision == 'fp16' and value.is_floating_point()
                        else value.clone()) for key, value in state.items()}
        if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError('soup export contains nonfinite tensors')
        model_path = folder / 'model.pt'
        with model_path.open('xb') as stream:
            torch.save(values, stream); stream.flush(); os.fsync(stream.fileno())
        tensor_sha = tensor_digest(values)
        metadata = export_metadata(job, sha(model_path), tensor_sha, variant=variant)
        with (folder / 'model.json').open('x') as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        model.load_state_dict(values, strict=True); model.eval()
        loaded, loaded_job, _, origin, device = _load_model(folder, device='cpu', variant=variant)
        if (loaded_job != job or str(device) != 'cpu' or origin['model_sha256'] != sha(model_path)
                or origin['tensor_sha256'] != tensor_sha
                or tensor_digest(loaded.state_dict()) != tensor_digest(model.state_dict())):
            raise ValueError('strict soup standalone reload differs')
        size = int(job['image_size'])
        images = torch.zeros((1, 3, size, size), dtype=torch.float32)
        age = torch.zeros((1, 2), dtype=torch.float32) if (job.get('age') or {}).get('enabled') else None
        loaded.eval()
        with torch.inference_mode():
            expected = forward(model, images, age, job).reshape(-1).float()
            actual = forward(loaded, images, age, job).reshape(-1).float()
        if (expected.shape != (1,) or not bool(torch.isfinite(expected).all())
                or not torch.allclose(expected, actual, rtol=1e-6, atol=1e-6)):
            raise ValueError('synthetic soup inference differs')
        exports[precision] = {'model': {'path': str(model_path), 'sha256': sha(model_path),
                                       'bytes': model_path.stat().st_size, 'tensor_sha256': tensor_sha},
                              'metadata': {'path': str(folder / 'model.json'), 'sha256': sha(folder / 'model.json')}}
    if torch.cuda.is_initialized():
        raise ValueError('GPU initialized during soup standalone validation')
    result = {'status': 'uniform_soup_standalone_strict_synthetic_passed', 'at': time.time(),
              'base_member': args.member, 'base_variant_for_loader': variant,
              'freeze': row['spec']['freeze'], 'science': row['science'],
              'site': {'path': str(Path(row['spec']['freeze']['path']).parent / 'SITE_BINDING.json'),
                       'sha256': runtime['site_binding_sha256']},
              'bundle': str(Path(row['spec']['freeze']['path']).parent / 'bundle'),
              'data_root': site['data_root'], 'cache': site['prepared_cache'],
              'state_sha256': args.state_sha256, 'exports': exports, 'synthetic_forward_only': True,
              'GPU_initialized': False, 'dev_read': False, 'official_read': False, 'TTA': False}
    with (output / 'STANDALONE.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'receipt_sha256': sha(output / 'STANDALONE.json')}), flush=True)


def create(args):
    require_cpu()
    import torch
    source = Path(__file__).resolve().parents[2]
    from f04_export_member import verify_published
    from f04_standalone_member import verify_own_blob
    verify_published(source, args.remote_metadata, args.git_commit)
    verify_own_blob(source, args.remote_metadata)
    remote_tree = json.loads((args.remote_metadata / 'REMOTE-TREE.json').read_text())
    blobs = {entry['path']: entry for entry in remote_tree['tree'] if entry.get('type') == 'blob'}
    for name in ('tools/diagnostics/f05_model_soup.py', 'configs/F05-MODEL-SOUPS.json'):
        path = source / name; raw = path.read_bytes()
        digest = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        if path.is_symlink() or name not in blobs or blobs[name]['sha'] != digest or blobs[name]['mode'] != mode:
            raise ValueError('soup code or config differs from the published tree: ' + name)
    config = json.loads((source / 'configs/F05-MODEL-SOUPS.json').read_text())
    if args.group not in config['groups']:
        raise ValueError('unregistered soup group')
    tree = pinned_json(args.tree_proof, args.tree_proof_sha256)
    proof_source = Path(tree.get('source_root', str(Path(args.tree_proof).parent / 'source-git'))).resolve()
    if (tree.get('status') != 'published_full_git_tree_verified_not_formal_deployment'
            or tree.get('commit') != args.git_commit or tree.get('tree') != remote_tree['sha'] or proof_source != source
            or tree.get('GPU_initialized') is not False):
        raise ValueError('exact published source tree proof required')
    inventory = pinned_json(args.inventory, config['inventory_sha256'])
    compatibility = pinned_json(args.compatibility, config['compatibility_sha256'])
    group = compatibility['groups'][args.group]
    names = config['groups'][args.group]
    if (group['parents'] != names or group['seed_only_compatible'] is not True
            or compatibility.get('official_read') is not False):
        raise ValueError('sealed compatibility evidence differs')
    rows = [find_member(inventory, name) for name in names]
    for name, row in zip(names, rows):
        public = row['public_source']
        if (row['science']['sha256'] != group['science_sha256'][name]
                or public.get('sha256') != config['public_weight_sha256']
                or public.get('revision') != config['public_revision']):
            raise ValueError('sealed parent source differs: ' + name)
    scratch = local_scratch(args.scratch)
    if scratch.exists() or Path(args.output).exists():
        raise ValueError('fresh soup scratch and output required')
    scratch.mkdir(mode=0o700)
    torch.set_num_threads(max(1, int(os.environ['NCCCU_CPU_THREADS']) - 2))
    torch.set_num_interop_threads(1)
    accumulator, parent_refs = StateMean(), []
    for index, (name, row) in enumerate(zip(names, rows)):
        part = scratch / ('ncccu-f05-soup-state-' + str(index) + '.pt')
        command = [sys.executable, '-B', str(Path(__file__).resolve()), 'extract', '--inventory', str(args.inventory),
                   '--inventory-sha256', config['inventory_sha256'], '--member', name, '--output', str(part)]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1200)
        if process.returncode != 0:
            raise RuntimeError('selected parent extraction failed: ' + name + ': ' + process.stderr[-500:])
        state_sha = sha(part)
        accumulator.add(torch.load(part, map_location='cpu', weights_only=True))
        part.unlink()
        parent_refs.append({'name': name, 'round_id': row['round_id'], 'science_sha256': row['science']['sha256'],
                            'selected_blob_sha256': row['selected_blob_sha256'],
                            'selected_tensor_sha256': row['selected']['tensor_sha256'],
                            'selected_variant': row['selected']['variant'], 'extracted_state_sha256': state_sha})
        print(json.dumps({'status': 'soup_parent_accumulated', 'group': args.group, 'parent': name,
                          'done': index + 1, 'total': len(names)}), flush=True)
    mean = accumulator.result()
    mean_path = scratch / 'ncccu-f05-soup-average.pt'
    with mean_path.open('xb') as stream:
        torch.save(mean, stream); stream.flush(); os.fsync(stream.fileno())
    state_sha = sha(mean_path)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), 'finalize', '--inventory', str(args.inventory),
               '--inventory-sha256', config['inventory_sha256'], '--member', names[0], '--state', str(mean_path),
               '--state-sha256', state_sha, '--output', str(args.output)]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1200)
    if process.returncode != 0:
        raise RuntimeError('strict soup finalization failed: ' + process.stderr[-500:])
    standalone = Path(args.output) / 'STANDALONE.json'
    result = {'status': 'uniform_soup_created_no_dev_evaluation', 'at': time.time(), 'group': args.group,
              'parents': parent_refs, 'method': config['method'], 'git_commit': args.git_commit,
              'tree_proof_sha256': args.tree_proof_sha256, 'inventory_sha256': config['inventory_sha256'],
              'compatibility_sha256': config['compatibility_sha256'], 'average_state_sha256': state_sha,
              'standalone': {'path': str(standalone), 'sha256': sha(standalone)},
              'GPU_initialized': False, 'dev_read': False, 'official_read': False, 'TTA': False}
    with (Path(args.output) / 'SOUP.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    mean_path.unlink()
    print(json.dumps({'status': result['status'], 'group': args.group,
                      'receipt_sha256': sha(Path(args.output) / 'SOUP.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    build = sub.add_parser('create')
    for name in ('inventory', 'compatibility', 'scratch', 'output', 'tree-proof', 'remote-metadata'):
        build.add_argument('--' + name, type=Path, required=True)
    for name in ('group', 'git-commit', 'tree-proof-sha256'):
        build.add_argument('--' + name, required=True)
    for name in ('extract', 'finalize'):
        command = sub.add_parser(name)
        for field in ('inventory', 'output'):
            command.add_argument('--' + field, type=Path, required=True)
        for field in ('inventory-sha256', 'member'):
            command.add_argument('--' + field, required=True)
        if name == 'finalize':
            command.add_argument('--state', type=Path, required=True)
            command.add_argument('--state-sha256', required=True)
    args = parser.parse_args()
    {'create': create, 'extract': extract, 'finalize': finalize}[args.operation](args)


if __name__ == '__main__':
    main()
