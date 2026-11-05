"""Fit-only fold roles and supervised packet checks for the F04 GPU owner."""

from retrain.f05_folds import verify_roles
from retrain.parallel_evaluation import canonical_predictions
from retrain.persistence import canonical_digest


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _round_name(fold):
    if type(fold) is not int or not 0 <= fold < 3:
        raise ValueError('one of the three registered fit-only folds is required')
    return (f'r{265 + fold}-small-f05fold{fold}-seed20261029-'
            'pairsigmoid-24k-p4-9010-20261105')


def build_plan(round_id, fold):
    if round_id != _round_name(fold):
        raise ValueError('fold and registered round differ')
    return [{'device_rank': rank, 'member': round_id, 'rank': rank, 'world': 8}
            for rank in range(8)]


def verified_roles(base, training, sealed, *, fold):
    """Recheck the presealed original-fit partition and fit-only age."""
    _round_name(fold)
    verify_roles(base, training, sealed, fold_number=fold, split_seed=20261121)
    train, heldout = list(sealed['fit_indices']), list(sealed['audit_indices'])
    if (train != sorted(train) or heldout != sorted(heldout)
            or any({base['rows'][index]['label'] for index in role} != {0, 1}
                   for role in (train, heldout))):
        raise ValueError('ordered complete binary fold populations required')
    return {'train': train, 'heldout': heldout}


def make_scores(base, sealed, packets, *, fold, role, round_id, inputs_sha, proof_shas,
                selected, checkpoint_sha, base_manifest_sha, folds_sha):
    """Reduce eight independently proved shards to one private score object."""
    build_plan(round_id, fold)
    if (role not in ('train', 'heldout') or sealed.get('fold') != fold
            or sealed.get('round') != 'R' + str(265 + fold)
            or not all(_digest(value) for value in (inputs_sha, checkpoint_sha,
                                                    base_manifest_sha, folds_sha))
            or not isinstance(selected, dict)
            or set(selected) != {'variant', 'tensor_sha256', 'successful_updates'}
            or selected['variant'] not in ('ema', 'online') or not _digest(selected['tensor_sha256'])
            or type(selected['successful_updates']) is not int or selected['successful_updates'] <= 0
            or not isinstance(proof_shas, dict) or set(proof_shas) != {str(i) for i in range(8)}
            or not all(_digest(value) for value in proof_shas.values())):
        raise ValueError('fixed fold, selected checkpoint and all device proofs required')
    key = 'fit_indices' if role == 'train' else 'audit_indices'
    digest_key = 'fit_indices_sha256' if role == 'train' else 'audit_indices_sha256'
    indices = sealed[key]
    if (not isinstance(indices, list) or not indices or indices != sorted(indices)
            or any(type(index) is not int for index in indices) or len(indices) != len(set(indices))
            or canonical_digest(indices) != sealed[digest_key]
            or not set(indices) <= set(base['fit_indices'])
            or set(indices) & set(base['dev_indices'])):
        raise ValueError('complete ordered sealed fit-only role required')
    load = {'selected_model_sha256': checkpoint_sha,
            'selected_tensor_sha256': selected['tensor_sha256'],
            'variant': selected['variant'], 'successful_updates': selected['successful_updates'],
            'strict_state_load': True, 'storage': 'fp32'}
    if not isinstance(packets, list) or len(packets) != 8:
        raise ValueError('eight supervised rank packets required')
    for packet in packets:
        rank = packet.get('rank')
        if (type(rank) is not int or not 0 <= rank < 8 or packet.get('device_rank') != rank
                or packet.get('member') != round_id or packet.get('fold') != fold
                or packet.get('role') != role or packet.get('error') is not None
                or packet.get('proof_sha256') != proof_shas[str(rank)]
                or packet.get('input_sha256') != inputs_sha or packet.get('model_load') != load
                or packet.get('TTA') is not False or packet.get('official_read') is not False):
            raise ValueError('fold rank, load, role or supervised proof differs')
    records = canonical_predictions(base['rows'], indices, packets, 8, 8)
    rows = [{'index': index, 'label': row['label'], 'logit': float(row['logit'])}
            for index, row in zip(indices, records)]
    return {'format': 'f05-fold-scores-v1', 'fold': fold, 'role': role,
            'base_manifest_sha256': base_manifest_sha, 'folds_sha256': folds_sha,
            'selected_model_sha256': checkpoint_sha,
            'selected_tensor_sha256': selected['tensor_sha256'],
            'selected_variant': selected['variant'], 'role_indices_sha256': sealed[digest_key],
            'rows': rows}
