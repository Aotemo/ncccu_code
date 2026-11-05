"""Validate fit-only F05 fold metadata without exposing held-out rows to training."""
import copy

from .age import make_spec
from .persistence import canonical_digest


def verify_roles(base, train, sealed, *, fold_number, split_seed):
    """Reconstruct the sealed fold and reject any change to original data roles."""
    try:
        if (type(fold_number) is not int or not 0 <= fold_number < 3
                or split_seed != 20261121 or not isinstance(base, dict)
                or not isinstance(train, dict) or not isinstance(sealed, dict)
                or sealed['fold'] != fold_number or sealed['round'] != 'R' + str(265 + fold_number)
                or train['rows'] != base['rows'] or train['source_identity'] != base['source_identity']
                or train['dev_indices'] != base['dev_indices'] or train['audit_indices'] != []
                or train['fit_indices'] != sealed['fit_indices']
                or train['split'] != {'kind':'fit_only_3fold', 'seed':split_seed,
                                      'fold':fold_number, 'folds':3, 'base_split':base.get('split')}
                or train['split_method'] != 'content-group-label-age-size-v1'):
            raise ValueError('fold identity, source rows or dev role differs')
        fit, held, dev = train['fit_indices'], sealed['audit_indices'], train['dev_indices']
        if (not all(isinstance(role, list) and role for role in (fit, held, dev))
                or any(type(i) is not int or not 0 <= i < len(base['rows']) for role in (fit, held, dev) for i in role)
                or any(len(set(role)) != len(role) for role in (fit, held, dev))
                or set(fit) & set(held) or set(fit) & set(dev) or set(held) & set(dev)
                or set(fit) | set(held) != set(base['fit_indices'])
                or set(base['fit_indices']) & set(base['dev_indices'])
                or set(base['fit_indices']) | set(base['dev_indices']) != set(range(len(base['rows'])))):
            raise ValueError('fold roles do not partition original fit and fixed dev')
        groups = [{base['rows'][i]['group'] for i in role} for role in (fit, held, dev)]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('content group crosses fold roles')
        if (sealed['fit_indices_sha256'] != canonical_digest(fit)
                or sealed['audit_indices_sha256'] != canonical_digest(held)
                or sealed['dev_indices_sha256'] != canonical_digest(dev)
                or sealed['age'] != make_spec(train)):
            raise ValueError('sealed indices or fit-only age differs')
        reconstructed = copy.deepcopy(train)
        reconstructed['audit_indices'] = list(held)
        reconstructed['counts']['audit'] = copy.deepcopy(sealed['counts']['audit'])
        if (reconstructed['counts'] != sealed['counts']
                or canonical_digest(reconstructed) != sealed['manifest_canonical_sha256']):
            raise ValueError('fold manifest differs from presealed metadata')
        return {'fit_count':len(fit), 'heldout_count':len(held), 'dev_count':len(dev),
                'fit_indices_sha256':sealed['fit_indices_sha256'],
                'heldout_indices_sha256':sealed['audit_indices_sha256'],
                'dev_indices_sha256':sealed['dev_indices_sha256'],
                'age_contract_sha256':sealed['age']['contract_sha256']}
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('complete sealed fit-only fold metadata required') from error
