"""F09 shared draft-decode pipeline.

Defines the JPEG draft (decoder DCT half-scale) read path plus an ImageDataset
subclass so dev/official re-forwards use exactly the same decode + crop +
letterbox + normalise code as the frozen pipeline, only the decoder scale
changes. predict.py/cost.py reuse the same primitives.
"""
import io
from pathlib import PurePosixPath

DRAFT_HINT_DIVISOR = 2


def draft_open(image_source, prepared_size=None, native_preprocessing=None):
    """Open an image file/bytes with a JPEG decoder draft hint, then run the
    frozen prepare path. Returns the PIL image exactly as the frozen
    ImageDataset._read_image would, at reduced decoder scale."""
    from PIL import Image, ImageOps
    from retrain.preprocessing import prepare_image
    with Image.open(image_source) as opened:
        hint = (max(1, opened.width // DRAFT_HINT_DIVISOR),
                max(1, opened.height // DRAFT_HINT_DIVISOR))
        opened.draft('RGB', hint)
        if prepared_size:
            return prepare_image(opened, prepared_size, native_preprocessing)
        image = ImageOps.exif_transpose(opened).convert('RGB')
    from retrain.data import _crop_black_border
    return _crop_black_border(image)


def make_draft_dataset(dataset_cls, root, rows, image_size, normalization, prepared_size,
                       age=None, input_preprocessing=None):
    """Build dataset_cls-based dataset whose _read_image applies the draft hint."""
    from retrain.data import _crop_black_border
    from PIL import Image, ImageOps

    class _Draft(dataset_cls):
        def _read_image(self, row):
            source = self.root / row['relative_path']
            with Image.open(source) as opened:
                hint = (max(1, opened.width // DRAFT_HINT_DIVISOR),
                        max(1, opened.height // DRAFT_HINT_DIVISOR))
                opened.draft('RGB', hint)
                if self.prepared_size:
                    from retrain.preprocessing import prepare_image
                    return prepare_image(opened, self.prepared_size, self.native_preprocessing)
                image = ImageOps.exif_transpose(opened).convert('RGB')
            return _crop_black_border(image)

    return _Draft(root, rows, image_size, False, normalization,
                  prepared_size=prepared_size, age=age,
                  input_preprocessing=input_preprocessing)
