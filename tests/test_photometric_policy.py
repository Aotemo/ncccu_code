"""Three synthetic checks for the optional training photometric policy.

Parent A runs these on the guarded Pod against the frozen runtime Pillow.
The fixtures are pixel oracles for the legacy sample-key augmentation stream,
not a second augmentation platform.
"""
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/bundle'))
from retrain import data

BaseRandom = random.Random
IMAGE_SIZE = 32
SCALE = (.85, 1.0)
SEED = 20261029


class TracedRandom(BaseRandom):
    def __init__(self, seed, forced_photometric=None):
        super().__init__(seed)
        self.random_calls = 0
        self.bounds = []
        self.forced_photometric = list(forced_photometric or [])
        self.photometric_draws = 0

    def random(self):
        self.random_calls += 1
        return super().random()

    def uniform(self, a, b):
        self.bounds.append((a, b))
        value = super().uniform(a, b)
        # Advance the state exactly as the real path would, then return a forced
        # factor so the pixel-difference assertion cannot depend on the seed.
        if (a, b) == (.9, 1.1) and self.photometric_draws < len(self.forced_photometric):
            self.photometric_draws += 1
            return self.forced_photometric[self.photometric_draws - 1]
        return value


def finish(image, image_size, scale, dx, dy):
    from PIL import Image
    import numpy as np
    import torch

    ratio = image_size * scale / max(image.size)
    target = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    image = image.resize(target, Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (image_size, image_size))
    canvas.paste(image, (round((image_size - target[0]) * dx), round((image_size - target[1]) * dy)))
    tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float().div_(255)
    mean, std = (.485, .456, .406), (.229, .224, .225)
    return tensor.sub_(tensor.new_tensor(mean)[:, None, None]).div_(tensor.new_tensor(std)[:, None, None])


def legacy_render(image, image_size, seed, scale_range):
    """Independent copy of the pre-policy training path, including all 8 draws."""
    from PIL import Image, ImageEnhance, ImageOps

    rng = random.Random(seed)
    scale, dx, dy = 1.0, .5, .5
    if rng.random() < .5:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.9, 1.1))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.9, 1.1))
    image = ImageEnhance.Color(image).enhance(rng.uniform(.9, 1.1))
    image = image.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BILINEAR,
                         expand=True, fillcolor=(0, 0, 0))
    scale, dx, dy = rng.uniform(*scale_range), rng.uniform(.25, .75), rng.uniform(.25, .75)
    return finish(image, image_size, scale, dx, dy)


def geometry_render(image, image_size, seed, scale_range):
    """Same stream and geometry without Brightness/Contrast/Color application."""
    from PIL import Image, ImageOps

    rng = random.Random(seed)
    scale, dx, dy = 1.0, .5, .5
    if rng.random() < .5:
        image = ImageOps.mirror(image)
    for _ in range(3):
        rng.uniform(.9, 1.1)
    image = image.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BILINEAR,
                         expand=True, fillcolor=(0, 0, 0))
    scale, dx, dy = rng.uniform(*scale_range), rng.uniform(.25, .75), rng.uniform(.25, .75)
    return finish(image, image_size, scale, dx, dy)


class PhotometricPolicyTest(unittest.TestCase):
    def test_bright_center_half_crop_pixels_None_rng_and_consumers(self):
        from retrain import config,preprocessing,engine,inference,parallel_evaluation as pe
        self.assertTrue(hasattr(config,'BRIGHT_CENTER_RULE'),'fixed prepared bright-center rule is missing')
        import numpy as np
        import torch
        from PIL import Image,ImageDraw
        policy=config.BRIGHT_CENTER_RULE
        flat=Image.new('RGB',(160,96),(20,30,40))
        self.assertEqual(preprocessing.bright_center_box(flat,policy),(56,24,104,72))
        marker=flat.copy();ImageDraw.Draw(marker).ellipse((110,32,130,52),fill=(5,240,7))
        box=preprocessing.bright_center_box(marker,policy)
        self.assertEqual((box[2]-box[0],box[3]-box[1]),(48,48))
        self.assertLessEqual(abs((box[0]+box[2])/2-120),3)
        self.assertLessEqual(abs((box[1]+box[3])/2-42),3)
        self.assertTrue(np.array_equal(np.asarray(preprocessing.bright_center_crop(marker,policy)),np.asarray(marker.crop(box))))
        for w,h in ((1,1),(2,1),(17,95),(95,17)):
            bounds=preprocessing.bright_center_box(Image.new('RGB',(w,h)),policy)
            self.assertTrue(0<=bounds[0]<bounds[2]<=w and 0<=bounds[1]<bounds[3]<=h)
        cropped=preprocessing.bright_center_crop(self.image,policy)
        actual,traces=self.render(input_preprocessing=policy);legacy,old=self.render()
        explicit_none,_=self.render(input_preprocessing=None)
        self.assertTrue(torch.equal(actual[0],legacy_render(cropped,IMAGE_SIZE,SEED,SCALE)))
        self.assertTrue(torch.equal(explicit_none[0],legacy[0]))
        self.assertEqual(traces[0][1].random_calls,8)
        self.assertEqual(traces[0][1].getstate(),old[0][1].getstate())
        expected=finish(cropped,IMAGE_SIZE,1.,.5,.5)
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.));self.inputs=[]
            def forward(self,x):self.inputs.append(x.detach().clone());return x.mean((1,2,3))*self.scale
        class Context:
            rank=0;world_size=1;primary=True
            def values(self,value):return [value]
            def broadcast(self,value):return value
        rows=[dict(self.row,name=f'synthetic-{i}.png',label=i%2) for i in range(4)]
        job=dict(model='synthetic',image_size=IMAGE_SIZE,prepared_size=0,precision='fp32',normalization='imagenet',
                 physical_batch=2,workers=0,eval_workers=0,input_preprocessing=policy,age=None,seed=SEED)
        model=Model()
        with patch.object(data,'configured_cache',return_value=None),patch.object(data.ImageDataset,'_read_image',side_effect=lambda row:self.image.copy()):
            self.assertEqual(engine.evaluate(model,rows,list(range(4)),self.root,job,torch.device('cpu'))['count'],4)
            self.assertEqual(pe.evaluate(model,rows,list(range(4)),self.root,job,torch.device('cpu'),Context(),variant='online')['count'],4)
            runtime=dict(image_size=IMAGE_SIZE,prepared_size=0,normalization='imagenet',precision='fp32',threshold=.5)
            (self.root/'test.csv').write_text('synthetic fixture only\n')
            with patch.object(inference,'_read_unlabeled_rows',return_value=rows), \
                    patch.object(inference,'_load_model',return_value=(model,job,{},{},torch.device('cpu'))), \
                    patch.object(inference,'_runtime',return_value=runtime):
                self.assertEqual(inference.run_prediction('/synthetic',self.root,self.root/'bright.csv',device='cpu',expected_count=4,batch_size=2)['count'],4)
        self.assertEqual(len(model.inputs),6)
        for inputs in model.inputs:self.assertTrue(torch.equal(inputs,expected.unsqueeze(0).expand(len(inputs),-1,-1,-1)))
        self.assertEqual(before[0],random.getstate());self.assertTrue(np.array_equal(before[1][1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(before[2],torch.get_rng_state()));self.assertFalse(torch.cuda.is_initialized())

    def test_local_contrast_pixels_legacy_rng_and_real_evaluation_consumers(self):
        from retrain import config, engine, inference, parallel_evaluation as pe
        self.assertTrue(hasattr(config, 'LOCAL_CONTRAST_RULE'), 'fixed local contrast policy is missing')
        import numpy as np
        import torch
        from PIL import Image, ImageFilter
        policy = config.LOCAL_CONTRAST_RULE
        image = self.image.copy()
        blur = image.filter(ImageFilter.GaussianBlur(radius=10.0))
        expected_image = Image.new('RGB', image.size)
        expected_image.putdata([tuple(max(0, min(255, 4 * a - 4 * b + 128)) for a, b in zip(pixel, mean))
                               for pixel, mean in zip(np.asarray(image).reshape(-1, 3).tolist(),
                                                      np.asarray(blur).reshape(-1, 3).tolist())])
        actual, traces = self.render(input_preprocessing=policy)
        legacy, original_traces = self.render()
        explicit_none, _ = self.render(input_preprocessing=None)
        self.assertTrue(torch.equal(actual[0], legacy_render(expected_image, IMAGE_SIZE, SEED, SCALE)))
        self.assertFalse(torch.equal(actual[0], legacy[0]))
        self.assertTrue(torch.equal(explicit_none[0], legacy[0]))
        self.assertEqual(traces[0][1].random_calls, 8)
        self.assertEqual(traces[0][1].getstate(), original_traces[0][1].getstate())
        expected = finish(expected_image, IMAGE_SIZE, 1., .5, .5)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with patch.object(data.random, 'Random', side_effect=AssertionError('eval consumed augmentation RNG')):
            evaluation = self.make(training=False, input_preprocessing=policy)
            with patch.object(evaluation, '_read_image', return_value=image.copy()):
                sample = evaluation[0]
        self.assertTrue(torch.equal(sample[0], expected))
        self.assertEqual(sample[1:], (1., 0))
        self.assertEqual(before[0], random.getstate())
        self.assertTrue(np.array_equal(before[1][1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(1.)); self.inputs = []
            def forward(self, inputs):
                self.inputs.append(inputs.detach().clone())
                return inputs.mean((1, 2, 3)) * self.scale
        class Context:
            rank = 0; world_size = 1; primary = True
            def values(self, value): return [value]
            def broadcast(self, value): return value
        rows = [dict(self.row, name=f'synthetic-{i}.png', label=i % 2) for i in range(4)]
        job = dict(model='synthetic', image_size=IMAGE_SIZE, prepared_size=0, precision='fp32',
                   normalization='imagenet', physical_batch=2, workers=0, eval_workers=0,
                   input_preprocessing=policy, age=None, seed=SEED)
        model = Model()
        with patch.object(data, 'configured_cache', return_value=None), \
                patch.object(data.ImageDataset, '_read_image', side_effect=lambda row: image.copy()):
            serial = engine.evaluate(model, rows, list(range(4)), self.root, job, torch.device('cpu'))
            parallel = pe.evaluate(model, rows, list(range(4)), self.root, job, torch.device('cpu'), Context(), variant='online')
            self.assertEqual(serial['count'], 4); self.assertEqual(parallel['count'], 4)
            runtime = dict(image_size=IMAGE_SIZE, prepared_size=0, normalization='imagenet', precision='fp32', threshold=.5)
            (self.root / 'test.csv').write_text('synthetic fixture only\n')
            with patch.object(inference, '_read_unlabeled_rows', return_value=rows), \
                    patch.object(inference, '_load_model', return_value=(model, job, {}, {}, torch.device('cpu'))), \
                    patch.object(inference, '_runtime', return_value=runtime):
                report = inference.run_prediction('/synthetic', self.root, self.root / 'predictions.csv',
                                                device='cpu', expected_count=4, batch_size=2)
            self.assertEqual(report['count'], 4)
        self.assertEqual(len(model.inputs), 6)
        for inputs in model.inputs:
            self.assertTrue(torch.equal(inputs, expected.unsqueeze(0).expand(len(inputs), -1, -1, -1)))
        self.assertFalse(torch.cuda.is_initialized())

    def setUp(self):
        from PIL import Image
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'train').mkdir()
        # Non-constant RGB: any applied photometric factor must move pixels.
        self.image = Image.new('RGB', (40, 28))
        self.image.putdata([(20 + x * 5 % 220, 15 + y * 7 % 230, 25 + (x * y) % 210)
                            for y in range(28) for x in range(40)])
        self.row = {'name': 'synthetic.png', 'relative_path': 'train/synthetic.png',
                    'sha256': 'a' * 64, 'label': 1}

    def make(self, training=True, **kwargs):
        with patch.object(data, 'configured_cache', return_value=None):
            return data.ImageDataset(self.root, [self.row], IMAGE_SIZE, training=training,
                                     training_scale=list(SCALE), **kwargs)

    def render(self, seed=SEED, forced_photometric=None, **kwargs):
        traces = []

        def factory(value):
            rng = TracedRandom(value, forced_photometric)
            traces.append((value, rng))
            return rng

        dataset = self.make(**kwargs)
        with patch.object(dataset, '_read_image', return_value=self.image.copy()), \
                patch.object(data.random, 'Random', side_effect=factory):
            result = dataset[(0, seed)]
        return result, traces

    def test_default_none_matches_legacy_pixels_and_eight_draws(self):
        import torch
        actual, traces = self.render()
        expected = legacy_render(self.image, IMAGE_SIZE, SEED, SCALE)
        self.assertTrue(torch.equal(actual[0], expected))
        self.assertEqual(len(traces), 1)
        rng = traces[0][1]
        # One direct random() for the mirror plus seven uniform() calls, each of
        # which consumes one primitive draw internally: 8 in total.
        self.assertEqual(rng.random_calls, 8)
        self.assertEqual(rng.bounds, [(.9, 1.1)] * 3 + [(-15, 15), SCALE, (.25, .75), (.25, .75)])
        reference = random.Random(SEED)
        reference.random()
        for low, high in rng.bounds:
            reference.uniform(low, high)
        self.assertEqual(rng.getstate(), reference.getstate())
        self.assertEqual(rng.random(), reference.random())

    def test_identity_consumes_photometric_draws_but_applies_only_geometry(self):
        import torch
        forced = (.5, 1.5, .5)
        legacy, legacy_rng = self.render(forced_photometric=forced)
        actual, traces = self.render(training_photometric='identity-v1', forced_photometric=forced)
        expected = geometry_render(self.image, IMAGE_SIZE, SEED, SCALE)
        self.assertEqual(self.make(training_photometric='identity-v1').training_photometric, 'identity-v1')
        self.assertTrue(torch.equal(actual[0], expected))
        # Identical geometry and stream: the legacy branch applies the forced
        # factors, the identity branch only consumes them.
        self.assertFalse(torch.equal(actual[0], legacy[0]))
        rng = traces[0][1]
        # One direct random() plus seven uniform() calls; Random.uniform consumes
        # one primitive draw itself, so the identity branch still makes 8 draws.
        self.assertEqual(rng.random_calls, 8)
        self.assertEqual(rng.bounds, legacy_rng[0][1].bounds)
        self.assertEqual(rng.getstate(), legacy_rng[0][1].getstate())
        self.assertEqual(rng.random(), legacy_rng[0][1].random())

    def test_eval_ignores_policy_without_random_or_enhancer_and_keeps_schema(self):
        import torch
        from PIL import ImageEnhance
        legacy = self.make(training=False)
        policy = self.make(training=False, training_photometric='identity-v1')
        failure = AssertionError('eval must not create augmentation randomness or enhancers')
        with patch.object(data.random, 'Random', side_effect=failure), \
                patch.object(ImageEnhance, 'Brightness', side_effect=failure), \
                patch.object(ImageEnhance, 'Contrast', side_effect=failure), \
                patch.object(ImageEnhance, 'Color', side_effect=failure):
            with patch.object(legacy, '_read_image', return_value=self.image.copy()):
                default = legacy[0]
            with patch.object(policy, '_read_image', return_value=self.image.copy()):
                alternative = policy[0]
        self.assertTrue(torch.equal(default[0], alternative[0]))
        self.assertEqual(default[1:], (1., 0))
        self.assertEqual(alternative[1:], (1., 0))

        unlabeled = self.make(training=False, include_labels=False)
        with patch.object(unlabeled, '_read_image', return_value=self.image.copy()):
            sample = unlabeled[0]
        self.assertEqual(len(sample), 2)
        self.assertEqual(sample[1], 0)
        self.assertEqual(tuple(sample[0].shape), (3, IMAGE_SIZE, IMAGE_SIZE))


if __name__ == '__main__':
    unittest.main()
