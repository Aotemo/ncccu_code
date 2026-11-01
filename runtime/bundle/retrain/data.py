"""Sealed source identity, content-group splits and stateless sample replay.

Source discovery opens official valid only with include_valid=True and never
opens test. Explicit inference rows may point ImageDataset at test images.
Without a verification cache, originals are hashed during source preparation.
With a persistent cache, sealed metadata is validated at startup and sampled
image bytes are verified on access. DataLoader prefetch is deliberately not state.
"""
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random

from .verification import configured_cache, lexical_path


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name):
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or '/' in name or '\\' in name or ':' in name or '\x00' in name):
        raise ValueError(f'unsafe image name/path: {name!r}')
    return name


def _safe_path(root, relative, split=None):
    if not isinstance(relative, str) or '\\' in relative or '\x00' in relative:
        raise ValueError(f'unsafe relative path: {relative!r}')
    path = PurePosixPath(relative)
    if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] not in ('train', 'valid', 'test')
            or any(p in ('.', '..') for p in path.parts)
            or (split is not None and path.parts[0] != split)):
        raise ValueError(f'unsafe {split or "image"} relative path: {relative!r}')
    _safe_name(path.name)
    root = Path(root).resolve()
    candidate = root.joinpath(*path.parts)
    if not candidate.resolve().is_relative_to(root / path.parts[0]):
        raise ValueError(f'image path escapes its declared split: {relative!r}')
    return candidate


def _read_csv(root, split):
    path = Path(root) / f'{split}.csv'
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('CSV symlink/path may not alias another split')
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if not {'Image_name', 'True_Label'}.issubset(reader.fieldnames or []):
        raise ValueError(f'{split}.csv requires Image_name and True_Label')
    rows, seen = [], set()
    for record in reader:
        name = _safe_name(record['Image_name'])
        key = name.casefold()
        if key in seen:
            raise ValueError(f'duplicate casefold CSV image name: {name}')
        seen.add(key)
        try:
            value = float(record['True_Label'])
        except (ValueError, TypeError) as exc:
            raise ValueError(f'invalid binary label for {name}') from exc
        if value not in (0.0, 1.0):
            raise ValueError(f'invalid binary label for {name}')
        rows.append({'name': name, 'label': int(value), 'age': record.get('Age', '') or ''})
    if not rows:
        raise ValueError(f'empty {split}.csv')
    return rows, hashlib.sha256(raw).hexdigest()


def _image_mapping(root, split):
    directory = Path(root) / split
    if directory.resolve() != Path(root).resolve() / split:
        raise ValueError(f'{split} directory path aliases another split')
    mapping = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        _safe_path(root, f'{split}/{path.name}', split)
        key = path.name.casefold()
        if key in mapping:
            raise ValueError(f'ambiguous casefold image paths: {mapping[key]}, {path.name}')
        mapping[key] = f'{split}/{path.name}'
    return mapping


def load_source(data_root, identity_path=None, verify_images=True, include_valid=False,
                expected_count=51396):
    """Validate official train against this package's pure-data sealed identity.

    verify_images=False skips train image hashes for metadata inspection only.
    TRAIN_VERIFICATION_CACHE opts into on-access verification instead of a full
    startup image walk; it never claims that all originals were checked now.
    Explicit valid inclusion always hashes those previously unsealed images to
    establish a reproducible identity. Exact byte matches share content groups;
    no claim is made that train/valid near-duplicate auditing has been completed.
    """
    root = Path(data_root)
    identity_path = Path(identity_path) if identity_path is not None else (
        Path(__file__).resolve().parents[1] / 'assets' / 'train_identity.json')
    sealed = json.loads(identity_path.read_text(encoding='utf-8'))
    if sealed.get('schema_version') != 1:
        raise ValueError('unsupported source identity schema')
    csv_rows, csv_sha = _read_csv(root, 'train')
    if csv_sha != sealed['source']['csv_sha256']:
        raise ValueError('train CSV SHA256 identity mismatch')
    expected = sealed['rows']
    if (len(csv_rows) != len(expected) or len(expected) != sealed['source']['row_count']
            or (expected_count is not None and len(expected) != expected_count)):
        raise ValueError('train source identity row count mismatch')
    cache = configured_cache(root) if verify_images else None
    mapping = None if cache is not None else _image_mapping(root, 'train')
    expected_names, rows = set(), []
    for position, (original, record) in enumerate(zip(expected, csv_rows)):
        _safe_name(original['name'])
        if cache is not None:
            lexical_path(original['relative_path'], 'train')
        else:
            _safe_path(root, original['relative_path'], 'train')
        if PurePosixPath(original['relative_path']).name.casefold() != original['name'].casefold():
            raise ValueError('identity image name and relative path mismatch')
        key = original['name'].casefold()
        if key in expected_names:
            raise ValueError(f'duplicate casefold identity name: {key}')
        expected_names.add(key)
        if (key != record['name'].casefold() or original['label'] != record['label']
                or original['age'] != record['age']):
            raise ValueError(f'CSV row {position} does not match sealed data identity')
        if mapping is not None and key not in mapping:
            raise ValueError(f'missing train image: {original["name"]}')
        if (not isinstance(original.get('group'), str) or not original['group']
                or not isinstance(original.get('sha256'), str)
                or len(original['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in original['sha256'])):
            raise ValueError(f'invalid source hash/group identity: {original["name"]}')
        relative = original['relative_path'] if mapping is None else mapping[key]
        if (verify_images and cache is None
                and _file_sha(_safe_path(root, relative, 'train')) != original['sha256']):
            raise ValueError(f'image SHA256 identity mismatch: {relative}')
        rows.append(dict(original, relative_path=relative))
    identity = {'schema_version': 1, 'train_csv_sha256': csv_sha,
                'train_rows_sha256': _digest(expected), 'train_count': len(rows)}
    valid_rows = []
    if include_valid:
        records, valid_csv_sha = _read_csv(root, 'valid')
        valid_mapping = _image_mapping(root, 'valid')
        hash_groups = {r['sha256']: r['group'] for r in rows}
        for record in records:
            relative = valid_mapping.get(record['name'].casefold())
            if relative is None:
                raise ValueError(f'missing valid image: {record["name"]}')
            path = _safe_path(root, relative, 'valid')
            sha = _file_sha(path)
            group = hash_groups.setdefault(sha, f'valid-sha256:{sha}')
            valid_rows.append(dict(record, relative_path=relative, sha256=sha, group=group))
        identity.update(valid_csv_sha256=valid_csv_sha, valid_rows_sha256=_digest(valid_rows),
                        valid_count=len(valid_rows))
    if cache is not None:
        cache.record_manifest(identity)
    return {'rows': rows, 'valid_rows': valid_rows, 'identity': identity}


def _age_bin(value):
    try:
        age = float(value)
    except (TypeError, ValueError):
        return 'missing'
    if not math.isfinite(age) or age <= 0:
        return 'missing'
    return 'under40' if age < 40 else '40to59' if age < 60 else '60to79' if age < 80 else '80plus'


def _group_stratum(rows, group):
    labels = {int(rows[i]['label']) for i in group}
    label = 'mixed' if len(labels) > 1 else str(next(iter(labels)))
    ages = sorted(_age_bin(rows[i].get('age')) for i in group)
    sizes = []
    for i in group:
        size = rows[i].get('original_size')
        area = size[0] * size[1] if size and len(size) == 2 else 0
        sizes.append('unknown' if not area else 'small' if area < 3_000_000 else
                     'medium' if area < 8_000_000 else 'large')
    return label, ages[len(ages) // 2], sorted(sizes)[len(sizes) // 2]


def _partition_groups(rows, indices, fractions, seed):
    """Allocate indivisible groups within label/age/size strata.

    Largest groups are allocated first against each stratum's fractional row
    targets. Small/rare strata can be imbalanced; actual counts are recorded.
    """
    groups = defaultdict(list)
    for index in indices:
        groups[rows[index]['group']].append(index)
    strata = defaultdict(list)
    for name in sorted(groups):
        group = groups[name]
        strata[_group_stratum(rows, group)].append(group)
    rng = random.Random(seed)
    parts = [[] for _ in fractions]
    total_targets = [len(indices) * f for f in fractions]
    for key in sorted(strata):
        group_list = strata[key]
        rng.shuffle(group_list)
        group_list.sort(key=len, reverse=True)
        count = sum(map(len, group_list))
        targets = [count * f for f in fractions]
        allocated = [0 for _ in fractions]
        for group in group_list:
            # Maximal remaining quota spreads singleton rows exactly, while
            # never splitting the rare two-image or mixed-label groups.
            destination = max(range(len(fractions)), key=lambda j: (
                targets[j] - allocated[j], total_targets[j] - len(parts[j]), -j))
            parts[destination].extend(group)
            allocated[destination] += len(group)
    return [sorted(part) for part in parts]


def build_manifest(source, split):
    split = dict(split)
    kind = split.setdefault('kind', 'holdout')
    seed = split.setdefault('seed', 20261024)
    rows = list(source['rows'])
    if not rows or any(r.get('label') not in (0, 1) or not r.get('group') for r in rows):
        raise ValueError('manifest requires nonempty binary labeled content groups')
    indices = list(range(len(rows)))
    dev, audit = [], []
    if kind == 'holdout':
        fraction = split.setdefault('valid_fraction', .2)
        if not 0 < fraction < 1:
            raise ValueError('valid_fraction must be between zero and one')
        fit, dev = _partition_groups(rows, indices, [1 - fraction, fraction], seed)
    elif kind == 'cv':
        folds, fold = split.setdefault('folds', 3), split.setdefault('fold', 0)
        fraction = split.setdefault('inner_fraction', .1)
        if not isinstance(folds, int) or folds < 2 or not isinstance(fold, int) or not 0 <= fold < folds:
            raise ValueError('cv requires folds >= 2 and a valid zero-based fold')
        if not 0 < fraction < 1:
            raise ValueError('inner_fraction must be between zero and one')
        outer = _partition_groups(rows, indices, [1 / folds] * folds, seed)
        audit = outer[fold]
        remaining = sorted(i for f, part in enumerate(outer) if f != fold for i in part)
        fit, dev = _partition_groups(rows, remaining, [1 - fraction, fraction], seed + 1000 + fold)
    elif kind == 'refit':
        if split.setdefault('include_valid', False):
            if not source.get('valid_rows'):
                raise ValueError('explicit valid inclusion requires load_source(include_valid=True)')
            rows += list(source['valid_rows'])
        fit = list(range(len(rows)))
    else:
        raise ValueError(f'unknown split kind: {kind}')
    if kind != 'refit':
        for role, values in [('fit', fit), ('dev', dev)] + ([('audit', audit)] if kind == 'cv' else []):
            if not values or {int(rows[i]['label']) for i in values} != {0, 1}:
                raise ValueError(f'{role} split lacks both classes; too few content groups for this protocol')
    counts = {role: {'rows': len(values), 'positive': sum(int(rows[i]['label']) for i in values),
                     'groups': len({rows[i]['group'] for i in values})}
              for role, values in [('fit', fit), ('dev', dev), ('audit', audit)]}
    return {'rows': rows, 'fit_indices': fit, 'dev_indices': dev, 'audit_indices': audit,
            'split': split, 'source_identity': source['identity'], 'counts': counts,
            'split_method': 'content-group-label-age-size-v1'}


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'big')


class BalancedBatchSampler:
    """Lazy permutation streams indexed by consumed attempts, never prefetch.

    Each iterator owns its caches, so advancing one iterator cannot influence a
    resumed or repeated iterator. Only the current class permutations are held.
    """
    def __init__(self, rows, fit_indices, physical_batch, effective_batch,
                 positive_per_batch, seed, start_update, num_updates):
        values = (physical_batch, effective_batch, positive_per_batch, start_update, num_updates)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
            raise ValueError('batch sizes and update counters must be integers')
        if (physical_batch <= 0 or effective_batch < physical_batch or effective_batch % physical_batch
                or not 0 < positive_per_batch < effective_batch or start_update < 0 or num_updates < 0):
            raise ValueError('invalid batch size, positive quota or attempted update range')
        self.microbatches = effective_batch // physical_batch
        if positive_per_batch % self.microbatches:
            raise ValueError('positive quota must divide equally across microbatches')
        if len(set(fit_indices)) != len(fit_indices):
            raise ValueError('duplicate fit indices')
        if any(not isinstance(i, int) or not 0 <= i < len(rows) or rows[i]['label'] not in (0, 1)
               for i in fit_indices):
            raise ValueError('fit indices and labels must be valid')
        self.pools = {c: [i for i in fit_indices if rows[i]['label'] == c] for c in (0, 1)}
        if any(not pool for pool in self.pools.values()):
            raise ValueError('balanced sampling requires both classes')
        self.physical_batch, self.effective_batch = physical_batch, effective_batch
        self.positive_per_batch, self.seed = positive_per_batch, seed
        self.start_update, self.num_updates = start_update, num_updates
        self.steps_per_cycle = math.ceil(len(self.pools[0]) / (effective_batch - positive_per_batch))

    def __len__(self):
        return self.num_updates * self.microbatches

    def __iter__(self):
        caches = {}

        def item(label, offset):
            pool = self.pools[label]
            epoch, within = divmod(offset, len(pool))
            if label not in caches or caches[label][0] != epoch:
                permutation = list(pool)
                random.Random(_seed(self.seed, 'permutation', label, epoch)).shuffle(permutation)
                caches[label] = epoch, permutation
            return caches[label][1][within]

        positives = self.positive_per_batch // self.microbatches
        negatives = self.physical_batch - positives
        for attempt in range(self.start_update, self.start_update + self.num_updates):
            negative_base = attempt * (self.effective_batch - self.positive_per_batch)
            positive_base = attempt * self.positive_per_batch
            for micro in range(self.microbatches):
                batch = ([item(1, positive_base + micro * positives + k) for k in range(positives)]
                         + [item(0, negative_base + micro * negatives + k) for k in range(negatives)])
                # Keep within-class order: each class is a true uninterrupted
                # permutation stream, including across cycle boundaries.
                yield [(index, _seed(self.seed, 'augmentation', attempt, micro, position))
                       for position, index in enumerate(batch)]


def _crop_black_border(image):
    import numpy as np
    preview = image.copy()
    preview.thumbnail((256, 256))
    foreground = np.asarray(preview).max(axis=2) > 12
    ys, xs = np.where(foreground)
    if not len(xs):
        return image
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    # Only trim shallow dark borders. Deep/asymmetric crops fall back to full
    # image rather than trusting a brightness threshold as retinal anatomy.
    if (right - left < .8 * preview.width or bottom - top < .8 * preview.height
            or max(left, preview.width - right) > .1 * preview.width
            or max(top, preview.height - bottom) > .1 * preview.height):
        return image
    sx, sy = image.width / preview.width, image.height / preview.height
    pad = max(2, round(min(image.size) * .01))
    return image.crop((max(0, int(left * sx) - pad), max(0, int(top * sy) - pad),
                       min(image.width, math.ceil(right * sx) + pad),
                       min(image.height, math.ceil(bottom * sy) + pad)))


def flatten_training_views(images, labels, age_features, training_views=None):
    """Flatten sample-major training views without changing main-sample indices."""
    if training_views is None:
        return images, labels, age_features
    if type(training_views) is not int or training_views not in (1, 2):
        raise ValueError('training_views must be None, 1, or 2')
    if training_views == 1:
        return images, labels, age_features
    if images.ndim != 5 or images.shape[1] != 2:
        raise ValueError('two training views require images shaped [B, 2, C, H, W]')
    return (images.flatten(0, 1), labels.repeat_interleave(2, dim=0),
            None if age_features is None else age_features.repeat_interleave(2, dim=0))


class ImageDataset:
    """Map-style Dataset whose augmentation randomness lives in its sample key."""
    def __init__(self, data_root, rows, image_size, training, normalization='imagenet', prepared_size=0, age=None,
                 training_scale=None, training_views=None, include_labels=True, training_photometric=None,
                 training_rotation=None, input_preprocessing=None):
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError('image_size must be a positive integer')
        if normalization not in ('imagenet', 'unit', 'flair_rgb01'):
            raise ValueError('unknown normalization')
        if training_photometric not in (None, 'identity-v1'):
            raise ValueError("training_photometric must be None or 'identity-v1'")
        if training_rotation not in (None, 'identity-v1'):
            raise ValueError("training_rotation must be None or 'identity-v1'")
        if input_preprocessing is not None:
            from .config import LOCAL_CONTRAST_RULE,BRIGHT_CENTER_RULE,NATIVE_BRIGHT_CENTER_RULE,GLOBAL_NATIVE_INSET_RULE
            if input_preprocessing not in (LOCAL_CONTRAST_RULE,BRIGHT_CENTER_RULE,NATIVE_BRIGHT_CENTER_RULE,GLOBAL_NATIVE_INSET_RULE):
                raise ValueError('fixed RGB local contrast policy differs')
        self.input_preprocessing = input_preprocessing
        self.native_preprocessing = (input_preprocessing if input_preprocessing is not None
            and input_preprocessing['version']=='rgb-green-bright-native-half-v1' else None)
        self.global_native_inset = bool(input_preprocessing is not None and input_preprocessing['version']=='rgb-global-native-inset256-v1')
        if self.global_native_inset:self.native_preprocessing=NATIVE_BRIGHT_CENTER_RULE
        self.root, self.rows = Path(data_root), rows
        self.image_size, self.training = image_size, bool(training)
        self.normalization = normalization
        if training_scale is not None and (
                not isinstance(training_scale, (list, tuple)) or len(training_scale) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in training_scale)
                or not 0 < training_scale[0] <= training_scale[1] <= 1):
            raise ValueError('training_scale must contain two finite numbers with 0 < lo <= hi <= 1')
        self.training_scale = (.94, 1.0) if training_scale is None else tuple(training_scale)
        if training_views is not None and (type(training_views) is not int or training_views not in (1, 2)):
            raise ValueError('training_views must be None, 1, or 2')
        self.training_views = 1 if training_views is None else training_views
        self.include_labels = bool(include_labels)
        self.training_photometric = training_photometric
        self.training_rotation = training_rotation
        self.age = age
        if age is not None:
            from .age import validate_spec
            validate_spec(age)
            if age['enabled'] and any('age' not in row for row in rows):
                raise ValueError('age field missing: Dataset schema error')
        self.verification_cache = configured_cache(self.root)
        if (isinstance(prepared_size, bool) or not isinstance(prepared_size, int)
                or prepared_size < 0 or (prepared_size and prepared_size < image_size)):
            raise ValueError('prepared_size must be zero or an integer >= image_size')
        self.prepared_size = prepared_size
        if self.native_preprocessing is not None and not prepared_size:
            raise ValueError('native crop requires explicit prepared_size')
        self.prepared_cache = None
        self.global_cache = None
        if prepared_size and os.environ.get('TRAIN_IMAGE_CACHE'):
            from .preprocessing import PreparedImageCache
            self.prepared_cache = PreparedImageCache(self.root, os.environ['TRAIN_IMAGE_CACHE'], prepared_size,
                                                     self.native_preprocessing)
            if self.global_native_inset:
                if not os.environ.get('TRAIN_GLOBAL_IMAGE_CACHE'):raise ValueError('explicit global read-only cache required for inset input')
                self.global_cache=PreparedImageCache(self.root,os.environ['TRAIN_GLOBAL_IMAGE_CACHE'],prepared_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        import torch

        if isinstance(key, tuple):
            index, augmentation_seed = key
        else:
            index, augmentation_seed = key, None
        if self.training and augmentation_seed is None:
            raise ValueError('training sample key must include an augmentation seed')
        row = self.rows[index]
        # The new cache contains fit/dev train rows only. Official valid is read
        # from sealed original bytes inside the existing once-only final gate.
        native_official = (self.native_preprocessing is not None
            and PurePosixPath(row['relative_path']).parts[0]=='valid')
        if self.prepared_cache is not None and row.get('sha256') and not native_official:
            image = self.prepared_cache.read(row['relative_path'], row['sha256'])
            if self.global_native_inset:image=(self.global_cache.read(row['relative_path'],row['sha256']),image)
        else:
            image = self._read_image(row)
        if self.training and self.training_views == 2:
            # Each view starts from the same unaugmented image. The first view
            # keeps the original sample-key seed and its exact augmentation flow.
            tensor = torch.stack((self._image_tensor(image.copy(), augmentation_seed),
                                  self._image_tensor(image.copy(), _seed(augmentation_seed, 'training-view', 1))))
        else:
            tensor = self._image_tensor(image, augmentation_seed)
        if not self.include_labels:
            if self.age is not None:
                raise ValueError('label-free views cannot include age features')
            return tensor, int(index)
        result = (tensor, float(row.get('label', 0)), int(index))
        if self.age and self.age['enabled']:
            from .age import features
            return (*result, torch.tensor(features(row,self.age),dtype=torch.float32))
        return result

    def _image_tensor(self, image, augmentation_seed):
        from PIL import Image, ImageEnhance, ImageOps
        import numpy as np
        import torch

        if self.global_native_inset:
            from .preprocessing import global_native_inset
            image=global_native_inset(*image,self.input_preprocessing)
        if self.input_preprocessing is not None and self.input_preprocessing['version']=='rgb-green-bright-center-half-v1':
            from .preprocessing import bright_center_crop
            image=bright_center_crop(image,self.input_preprocessing)
        elif self.input_preprocessing is not None and self.native_preprocessing is None:
            from PIL import ImageFilter
            policy = self.input_preprocessing
            pixels = np.asarray(image).astype(np.int16)
            local_mean = np.asarray(image.filter(ImageFilter.GaussianBlur(policy['radius']))).astype(np.int16)
            corrected = policy['image_gain'] * pixels + policy['blur_gain'] * local_mean + policy['offset']
            image = Image.fromarray(np.clip(corrected, *policy['clip']).astype(np.uint8))
        scale, dx, dy = 1.0, .5, .5
        if self.training:
            rng = random.Random(int(augmentation_seed))
            if rng.random() < .5:
                image = ImageOps.mirror(image)
            if self.training_photometric == 'identity-v1':
                # Keep the historical draw count and positions so geometry and the
                # sample-key random stream are unchanged; the factors are unused.
                for _ in range(3):
                    rng.uniform(.9, 1.1)
            else:
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(.9, 1.1))
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(.9, 1.1))
                image = ImageEnhance.Color(image).enhance(rng.uniform(.9, 1.1))
            angle = rng.uniform(-15, 15)
            if self.training_rotation != 'identity-v1':
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR,
                                     expand=True, fillcolor=(0, 0, 0))
            scale, dx, dy = rng.uniform(*self.training_scale), rng.uniform(.25, .75), rng.uniform(.25, .75)
        ratio = self.image_size * scale / max(image.size)
        target = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
        image = image.resize(target, Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.image_size, self.image_size))
        canvas.paste(image, (round((self.image_size - target[0]) * dx),
                             round((self.image_size - target[1]) * dy)))
        tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float().div_(255)
        if self.normalization != 'flair_rgb01':
            mean, std = ((.485, .456, .406), (.229, .224, .225)) if self.normalization == 'imagenet' else (
                (.5, .5, .5), (.5, .5, .5))
            tensor.sub_(tensor.new_tensor(mean)[:, None, None]).div_(tensor.new_tensor(std)[:, None, None])
        return tensor

    def _read_image(self, row):
        from PIL import Image, ImageOps
        verifier = self.verification_cache
        if verifier is None and self.native_preprocessing is not None and self.prepared_cache is not None:
            verifier = self.prepared_cache.verifier
        if (verifier is not None
                and PurePosixPath(row['relative_path']).parts[0] in ('train', 'valid')):
            image_source = io.BytesIO(verifier.read_verified(
                row['relative_path'], row.get('sha256')))
        else:
            image_source = _safe_path(self.root, row['relative_path'])
        with Image.open(image_source) as opened:
            if self.prepared_size:
                from .preprocessing import prepare_image
                if self.global_native_inset:
                    return (prepare_image(opened,self.prepared_size),prepare_image(opened,self.prepared_size,self.native_preprocessing))
                return prepare_image(opened, self.prepared_size,self.native_preprocessing)
            image = ImageOps.exif_transpose(opened).convert('RGB')
        return _crop_black_border(image)
