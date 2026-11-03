"""Release R115 fit-only SSL only after the public eight-rank initial identity is real."""
import json
from pathlib import Path
import time


def verify_teacher_initial(proof, world):
    if (world != 8 or len(proof) != world or {item.get('rank') for item in proof} != set(range(world))
            or len({item.get('teacher_tensor_sha256') for item in proof}) != 1
            or any(not isinstance(item.get('teacher_tensor_sha256'), str)
                   or len(item['teacher_tensor_sha256']) != 64
                   or item['teacher_tensor_sha256'] != item.get('teacher_student_sha256')
                   or item.get('teacher_trainable_tensors') != 0
                   or item.get('teacher_shared_parameter_tensors') != 0 for item in proof)):
        raise ValueError('EMA eight-rank teacher initial equality/no-gradient/disjoint storage differs')
    return True


def maybe_release(root, atomic, sha):
    root = Path(root)
    recipe = json.loads((root / 'SCIENCE.json').read_text()).get('ssl_recipe')
    if recipe is None:
        return False
    barrier = root / 'control/ssl-startup-barrier.json'
    receipt_path = root / 'evidence/SSL-PUBLIC-INITIAL.json'
    if barrier.exists():
        if not receipt_path.is_file():
            raise ValueError('R115 SSL barrier exists without public initial receipt')
        return True
    cfg = json.loads((root / 'TRAIN_CONFIG.json').read_text())
    folder = root / 'outputs' / cfg['run_name'] / cfg['jobs'][0]['name']
    proof_path = folder / 'ssl-stage/public-initial-ranks.json'
    devices_path = folder / 'rank-device-proof.json'
    if not proof_path.is_file() or not devices_path.is_file():
        return False
    from retrain.site_binding import load_site, verify_rank_devices
    from retrain.ssl_stage import validate_recipe, ssl_head_shape
    from retrain.experiment91 import metadata
    site = load_site()
    world = len(site['gpus'])
    validate_recipe(recipe, cfg['jobs'][0], metadata()['manifest.json'], world)
    verify_rank_devices(json.loads(devices_path.read_text()), site)
    proof = json.loads(proof_path.read_text())
    expected = json.loads((root / 'evidence/MODEL_CPU_ADMISSION.json').read_text())['initial_state_control']
    if (world != 8 or len(proof) != world or {item.get('rank') for item in proof} != set(range(world))
            or any(item.get('model_sha256') != expected['model_tensor_sha256']
                   or item.get('encoder_tensor_sha256') != expected['public_encoder_tensor_sha256']
                   or item.get('public_tensor_sha256') != expected['public_tensor_sha256']
                   or item.get('strict_public_exact') is not True
                   or item.get('optimizer_state_entries') != 0
                   or item.get('projector_shape') != ssl_head_shape(recipe)
                   or not isinstance(item.get('non_encoder_sha256'), str)
                   or len(item['non_encoder_sha256']) != 64 for item in proof)):
        raise ValueError('R115 eight-rank public source/empty Adam/projector identity differs')
    if recipe['version'] == 'fit-ema-assignment-1600-v1':
        verify_teacher_initial(proof, world)
    fresh = json.loads((root / 'control/train-fresh-spec.json').read_text())
    if (fresh.get('ssl_recipe') != recipe
            or fresh.get('ssl_startup_barrier') != str(barrier)
            or time.time() >= json.loads((root / 'control/train-phase-start.json').read_text())['soft_deadline_at']):
        raise ValueError('R115 fixed SSL recipe/barrier/start window differs')
    atomic(receipt_path, {'status': 'completed', 'at': time.time(), 'recipe': recipe,
        'rank_count': world, 'public_model_tensor_sha256': expected['model_tensor_sha256'],
        'public_tensor_sha256': expected['public_tensor_sha256'],
        'public_encoder_tensor_sha256': expected['public_encoder_tensor_sha256'],
        'ssl_rank0_non_encoder_sha256': proof[0]['non_encoder_sha256'],
        'rank_proof_sha256': sha(proof_path), 'device_proof_sha256': sha(devices_path),
        'first_ssl_update_executed': False}, exclusive=True)
    atomic(barrier, {'status': 'released', 'at': time.time(), 'recipe': recipe,
        'receipt_sha256': sha(receipt_path)}, exclusive=True)
    return True
