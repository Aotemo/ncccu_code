"""Pin the official DINO v1 ResNet-50 source metadata and, separately, its one weight file.

Metadata mode never downloads weights. The weight URL carries no code revision or
LFS SHA, so code revision and actual weight bytes remain separate identities.
"""
import argparse
import json
from pathlib import Path
import re

from source_chain import cpu_context, require, sha256, write_new_json
from small_source import fetch


REPO = 'facebookresearch/dino'
REPO_API = f'https://api.github.com/repos/{REPO}'
COMMIT_API = f'{REPO_API}/commits/main'
ISSUE_API = f'{REPO_API}/issues/108'
ISSUE_COMMENTS_API = f'{ISSUE_API}/comments'
WEIGHT_URL = 'https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth'
TRAINING_DATA = 'ImageNet self-supervised DINO v1'
RAW = 'https://raw.githubusercontent.com/{repo}/{revision}/{name}'


def metadata(assets, transport_dir=None):
    context = cpu_context()
    assets = Path(assets)
    require(assets.is_absolute() and '/exp-v3/' in str(assets), 'explicit exp-v3 asset root required')
    original = assets / 'original'
    original.mkdir(parents=True, exist_ok=True)
    transport = (lambda name: str(Path(transport_dir) / name)) if transport_dir else (lambda name: None)
    repo_record = fetch(REPO_API, original / 'repo.json', transport_file=transport('repo.json'))
    repo = json.loads((original / 'repo.json').read_text())
    require(repo.get('full_name') == REPO and repo.get('private') is False,
            'official repository identity differs')
    require(repo.get('license', {}).get('spdx_id') == 'Apache-2.0', 'repository license is not Apache-2.0')
    require(repo.get('default_branch') == 'main', 'official default branch differs')
    commit_record = fetch(COMMIT_API, original / 'commit.json', transport_file=transport('commit.json'))
    commit = json.loads((original / 'commit.json').read_text())
    revision = commit.get('sha')
    require(isinstance(revision, str) and len(revision) == 40
            and all(character in '0123456789abcdef' for character in revision),
            'single commit metadata read did not return one immutable revision')
    records = {'repo.json': repo_record, 'commit.json': commit_record}
    for name in ('hubconf.py', 'LICENSE', 'README.md'):
        records[name] = fetch(RAW.format(repo=REPO, revision=revision, name=name),
                              original / name, transport_file=transport(name))
    records['issue-108.json'] = fetch(ISSUE_API, original / 'issue-108.json',
                                      transport_file=transport('issue-108.json'))
    records['issue-108-comments.json'] = fetch(ISSUE_COMMENTS_API, original / 'issue-108-comments.json',
                                               transport_file=transport('issue-108-comments.json'))
    hubconf = (original / 'hubconf.py').read_text()
    compact = re.sub(r'\s+', '', hubconf)
    require('defdino_resnet50(' in compact and 'model.fc=torch.nn.Identity()' in compact
            and WEIGHT_URL in hubconf and 'ImageNet' in hubconf and 'pre-trained with DINO' in hubconf,
            'official dino_resnet50 hubconf identity differs')
    license_text = (original / 'LICENSE').read_text()
    require('Apache License' in license_text and 'Version 2.0, January 2004' in license_text
            and 'END OF TERMS AND CONDITIONS' in license_text, 'repository Apache-2.0 text incomplete')
    readme = (original / 'README.md').read_text().lower()
    require('imagenet' in readme and 'self-supervised' in readme and 'dino' in readme,
            'official README data category evidence differs')
    issue = json.loads((original / 'issue-108.json').read_text())
    question = issue.get('body') or ''
    require('license' in question.lower() and 'weight' in question.lower(),
            'issue 108 question evidence differs')
    comments = json.loads((original / 'issue-108-comments.json').read_text())
    require(isinstance(comments, list) and comments, 'issue 108 comments missing')
    qualifying = [comment for comment in comments
                  if comment.get('author_association') in ('OWNER', 'MEMBER', 'COLLABORATOR', 'CONTRIBUTOR')
                  and 'license' in (comment.get('body') or '').lower()
                  and 'weight' in (comment.get('body') or '').lower()
                  and ('cover' in (comment.get('body') or '').lower()
                       or 'apache' in (comment.get('body') or '').lower())]
    require(bool(qualifying), 'no author/maintainer statement that the license covers the weights')
    receipt = {'schema_version': 1, 'passed': True, 'scope': 'dino_resnet50_source_metadata',
               'execution': context, 'repo': REPO, 'revision': revision,
               'revision_source': 'single_commits_main_read_at_metadata_time',
               'repository': {'full_name': REPO, 'private': repo.get('private'),
                              'archived': repo.get('archived'), 'default_branch': repo.get('default_branch')},
               'hubconf': {'model': 'resnet50', 'fc': 'torch.nn.Identity',
                           'literal_strict_false_present': 'strict=False' in hubconf},
               'files': records,
               'weight': {'url': WEIGHT_URL, 'downloaded_in_metadata_step': False,
                          'url_has_code_revision': False, 'url_has_lfs_sha': False,
                          'code_revision_is_not_weight_identity': True},
               'license': {'identifier': 'Apache-2.0', 'repository_license_sha256': records['LICENSE']['sha256'],
                           'author_weight_statement': {'issue_url': ISSUE_API, 'comments_url': ISSUE_COMMENTS_API,
                                                       'comments_sha256': records['issue-108-comments.json']['sha256'],
                                                       'qualifying_comment_ids': [comment['id'] for comment in qualifying],
                                                       'author_associations': [comment['author_association'] for comment in qualifying],
                                                       'statements': [comment['body'] for comment in qualifying],
                                                       'mentions_license': True, 'mentions_weights': True},
                           'scope_note': 'repository code Apache-2.0 plus author statement about weights; '
                                         'not a competition-specific per-model approval'},
               'data_category': {'value': TRAINING_DATA,
                                 'evidence': ['README.md', 'hubconf.py dino_resnet50 docstring']},
               'selection': {'architecture': 'ResNet-50', 'head': 'fc=Identity per official hubconf',
                             'vit': False, 'dino_v2': False, 'medical': False, 'fallback': 'none'},
               'weight_downloads': 0, 'forward_calls': 0, 'datasets_read': False, 'GPU_imported': False,
               'validation_source_sha256': {'dino_resnet_source.py': sha256(__file__),
                                            'source_chain.py': sha256(Path(__file__).with_name('source_chain.py'))}}
    write_new_json(original / 'source-metadata.json', receipt)
    receipt['metadata_receipt'] = {'path': str(original / 'source-metadata.json'),
                                   'sha256': sha256(original / 'source-metadata.json')}
    return receipt


def weights(assets, source_receipt, source_receipt_sha, transport_file=None):
    context = cpu_context()
    assets = Path(assets)
    require(assets.is_absolute() and '/exp-v3/' in str(assets), 'explicit exp-v3 asset root required')
    pinned = json.loads(Path(source_receipt).read_text())
    require(sha256(source_receipt) == source_receipt_sha, 'source metadata receipt SHA differs')
    require(pinned.get('passed') is True and pinned.get('repo') == REPO
            and pinned.get('weight', {}).get('url') == WEIGHT_URL, 'source metadata identity differs')
    target = assets / 'original' / 'dino_resnet50_pretrain.pth'
    record = fetch(WEIGHT_URL, target, transport_file=transport_file)
    require(record['bytes'] > 20 * 1024 * 1024, 'weight file too small to be the official checkpoint')
    receipt = {'schema_version': 1, 'passed': True, 'scope': 'dino_resnet50_weight_bytes',
               'execution': context, 'repo': REPO, 'revision': pinned['revision'],
               'source_metadata': {'path': source_receipt, 'sha256': source_receipt_sha},
               'weight': record, 'url_has_code_revision': False,
               'code_revision_is_not_weight_identity': True,
               'weight_bytes_and_sha_are_the_identity': True,
               'parsed': False, 'forward_calls': 0, 'datasets_read': False, 'GPU_imported': False}
    write_new_json(assets / 'original' / 'weight-receipt.json', receipt)
    receipt['weight_receipt'] = {'path': str(assets / 'original' / 'weight-receipt.json'),
                                 'sha256': sha256(assets / 'original' / 'weight-receipt.json')}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('metadata', 'weights'):
        item = sub.add_parser(name)
        item.add_argument('--assets', required=True)
        item.add_argument('--output', required=True)
        item.add_argument('--transport-dir')
        if name == 'weights':
            item.add_argument('--transport-file')
            item.add_argument('--source-receipt', required=True)
            item.add_argument('--source-receipt-sha', required=True)
    args = parser.parse_args()
    if args.mode == 'metadata':
        result = metadata(args.assets, args.transport_dir)
    else:
        result = weights(args.assets, args.source_receipt, args.source_receipt_sha, args.transport_file)
    write_new_json(args.output, result)


if __name__ == '__main__':
    main()
