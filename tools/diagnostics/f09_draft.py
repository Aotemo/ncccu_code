"""F09 shared draft-decode pipeline.

Defines the JPEG draft (decoder DCT half-scale) read path plus an ImageDataset
subclass so dev/official re-forwards use exactly the same decode + crop +
letterbox + normalise code as the frozen pipeline, only the decoder scale
changes. predict.py/cost.py reuse the same primitives.
"""
import io
from pathlib import PurePosixPath

def _divisor():
    return int(os.environ.get('NCCCU_DRAFT_DIVISOR', '2'))


def draft_open(image_source, prepared_size=None, native_preprocessing=None):
    """Open an image file/bytes with a JPEG decoder draft hint, then run the
    frozen prepare path. Returns the PIL image exactly as the frozen
    ImageDataset._read_image would, at reduced decoder scale."""
    import os
    from PIL import Image, ImageOps
    from retrain.preprocessing import prepare_image
    divisor = _divisor()
    with Image.open(image_source) as opened:
        hint = (max(1, opened.width // divisor),
                max(1, opened.height // divisor))
        opened.draft('RGB', hint)
        if prepared_size:
            return prepare_image(opened, prepared_size, native_preprocessing)
        image = ImageOps.exif_transpose(opened).convert('RGB')
    from retrain.data import _crop_black_border
    return _crop_black_border(image)


def make_draft_dataset(dataset_cls, root, rows, image_size, normalization, prepared_size,
                       age=None, input_preprocessing=None):
    """Build dataset_cls-based dataset whose _read_image applies the draft hint."""
    import inspect
    from retrain.data import _crop_black_border
    from PIL import Image, ImageOps

    import os
    divisor = int(os.environ.get('NCCCU_DRAFT_DIVISOR', '2'))

    class _Draft(dataset_cls):
        def _read_image(self, row):
            source = self.root / row['relative_path']
            with Image.open(source) as opened:
                hint = (max(1, opened.width // divisor),
                        max(1, opened.height // divisor))
                opened.draft('RGB', hint)
                if self.prepared_size:
                    from retrain.preprocessing import prepare_image
                    native = getattr(self, 'native_preprocessing', None)
                    try:
                        return prepare_image(opened, self.prepared_size, native)
                    except TypeError:
                        return prepare_image(opened, self.prepared_size)
                image = ImageOps.exif_transpose(opened).convert('RGB')
            return _crop_black_border(image)

    parameters = inspect.signature(dataset_cls.__init__).parameters
    kwargs = {'prepared_size': prepared_size, 'age': age}
    if 'input_preprocessing' in parameters:
        kwargs['input_preprocessing'] = input_preprocessing
    elif input_preprocessing is not None:
        raise ValueError('historical dataset cannot honor configured preprocessing')
    return _Draft(root, rows, image_size, False, normalization, **kwargs)
