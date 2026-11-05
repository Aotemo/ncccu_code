"""Bind a CPU-built package to actual bytes from the verified remote Git tree."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_commit(value, commit_sha, tree_sha):
    require(re.fullmatch('[0-9a-f]{40}', commit_sha) and re.fullmatch('[0-9a-f]{40}', tree_sha)
            and value.get('sha') == commit_sha and value.get('tree', {}).get('sha') == tree_sha,
            'actual remote commit readback does not name this deployed Git tree')


def verify_tree(source, value, tree_sha):
    source = Path(source)
    require(re.fullmatch('[0-9a-f]{40}', tree_sha) and value.get('sha') == tree_sha
            and value.get('truncated') is False, 'complete pinned remote Git tree required')
    entries = {}
    for item in value['tree']:
        relative = Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and '.git' not in relative.parts,
                'remote tree has an unsafe path')
        require(item['type'] in ('blob', 'tree'), 'submodule or unsupported Git tree entry')
        if item['type'] == 'blob':
            require(item['mode'] in ('100644', '100755') and item['path'] not in entries
                    and re.fullmatch('[0-9a-f]{40}', item['sha']), 'invalid or duplicate Git blob entry')
            entries[item['path']] = item
    paths = list(source.rglob('*'))
    require(all(not path.is_symlink() and (path.is_dir() or path.is_file()) for path in paths),
            'source contains a symlink or nonregular file')
    actual = {str(path.relative_to(source)): path for path in paths if path.is_file()}
    require(entries and set(actual) == set(entries), 'actual deployment file set differs from the remote tree')
    files = {}
    for relative, path in actual.items():
        body = path.read_bytes(); row = entries[relative]
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        require(blob == row['sha'] and mode == row['mode'], 'actual Git blob bytes/mode differ: ' + relative)
        files[relative] = hashlib.sha256(body).hexdigest()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'remote-tree', 'remote-tree-sha256', 'tree-sha', 'git-commit',
                 'remote-commit', 'remote-commit-json', 'remote-commit-json-sha256',
                 'prep-root', 'package-manifest-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--local-source-proof')
    parser.add_argument('--local-source-proof-sha256')
    args = parser.parse_args()
    source = Path(args.source_root).resolve(strict=True); prep = Path(args.prep_root).resolve(strict=True)
    require(re.fullmatch('[0-9a-f]{40}', args.git_commit) and args.git_commit == args.remote_commit,
            'actual connector-verified remote commit required')
    require(sha(args.remote_commit_json) == args.remote_commit_json_sha256, 'remote commit metadata SHA differs')
    verify_commit(json.loads(Path(args.remote_commit_json).read_text()), args.git_commit, args.tree_sha)
    require(sha(args.remote_tree) == args.remote_tree_sha256, 'remote tree metadata SHA differs')
    files = verify_tree(source, json.loads(Path(args.remote_tree).read_text()), args.tree_sha)
    require(files.get('tools/activation/freeze_deployment.py') == sha(__file__),
            'deployment verifier itself is outside the remote commit')
    manifest_path = prep / 'PACKAGE-MANIFEST.json'
    require(sha(manifest_path) == args.package_manifest_sha256, 'CPU package manifest SHA differs')
    manifest = json.loads(manifest_path.read_text()); package = prep / 'package'
    require(manifest['git_commit'] == args.git_commit and manifest['package_root'] == str(package),
            'CPU package commit/root differs')
    actual = {str(path.relative_to(package)): sha(path) for path in package.rglob('*') if path.is_file()}
    require(all(not path.is_symlink() for path in package.rglob('*')) and actual == manifest['files'],
            'CPU package file set/bytes differ')
    for relative, digest in actual.items():
        if relative.endswith('.py'):
            require(files.get('runtime/' + relative) == digest, 'package runtime differs from Git: ' + relative)
    for name in ('activate.py', 'activate_audit.py', 'site_preflight.py'):
        require(files.get('tools/activation/' + name) == sha(prep / name), 'activation helper differs from Git')
    site = json.loads((package / 'SITE_BINDING.json').read_text())
    sys.path.insert(0, str(package / 'bundle'))
    from retrain.site_binding import validate_site, require_cpu_channel, evidence_path
    site = validate_site(site); require_cpu_channel(site)
    require(site.get('scope') == 'formal_exp_v3' and site['git_commit'] == args.git_commit
            and site.get('source_state', {}).get('kind') == 'remote_commit_frozen', 'formal remote source binding required')
    require(bool(args.local_source_proof) == bool(args.local_source_proof_sha256),
            'local source proof and SHA must be supplied together')
    for path in (prep, Path(args.remote_tree), Path(args.remote_commit_json), Path(args.output)):
        evidence_path(path, site['work_root'])
    if args.local_source_proof:
        proof_path = evidence_path(args.local_source_proof, Path(site['work_root']) / 'technical-prep')
        require(re.fullmatch('[0-9a-f]{64}', args.local_source_proof_sha256)
                and sha(proof_path) == args.local_source_proof_sha256, 'local source proof SHA differs')
        helper = source / 'tools/activation/local_snapshot.py'
        require(files.get('tools/activation/local_snapshot.py') == sha(helper), 'local source helper differs')
        spec = importlib.util.spec_from_file_location('verified_local_snapshot', helper)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.verify_local_receipt(source, site['pod']['uid'], args.git_commit, args.tree_sha,
                                    json.loads(proof_path.read_text()), files)
    else:
        evidence_path(source, site['work_root'])
    require('technical-prep' in source.parts and 'technical-prep' in Path(args.output).parts,
            'frozen source and deployment evidence must remain in technical-prep')
    result = {'schema_version': 1, 'status': 'frozen_from_remote_commit', 'scope': 'formal_exp_v3',
        'git_commit': args.git_commit, 'remote_commit': args.remote_commit, 'tree_sha': args.tree_sha,
        'remote_url': 'https://github.com/Aotemo/ncccu_code', 'pod_uid': site['pod']['uid'],
        'source_root': str(source), 'files': files, 'package_manifest_sha256': args.package_manifest_sha256,
        'remote_tree_metadata': {'path': args.remote_tree, 'sha256': args.remote_tree_sha256},
        'remote_commit_metadata': {'path': args.remote_commit_json, 'sha256': args.remote_commit_json_sha256},
        'GPU_started': False, 'formal_clock_created': False}
    if args.local_source_proof:
        result['local_source_proof'] = {'path': args.local_source_proof, 'sha256': args.local_source_proof_sha256}
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'receipt': args.output, 'sha256': sha(args.output),
                      'source_files': len(files), 'GPU_started': False}, indent=2))


if __name__ == '__main__':
    main()
