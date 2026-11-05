"""Synthetic checks for the fixed A2 gate; no patient images or models."""
import unittest
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from tools.diagnostics.fit_disc_attention import grad_cam, locate
from tools.diagnostics.fit_disc_consensus import box_coverage, choose_fit


class AttentionGateTest(unittest.TestCase):
    def test_fit_sample_excludes_both_prior_holdouts(self):
        old, prior = list(range(96)), list(range(96, 160))
        chosen = choose_fit(list(range(1000)), old + prior, 64, 20261124)
        self.assertEqual(chosen, choose_fit(list(reversed(range(1000))), old + prior, 64, 20261124))
        self.assertEqual(len(set(chosen)), 64)
        self.assertFalse(set(chosen) & set(old + prior))

    def test_localization_and_canvas_edge_clamping(self):
        cam = np.zeros((512, 512))
        cam[10:14, 8:12] = 3
        prediction = locate(cam, [0, 0, 512, 512])
        self.assertEqual(prediction['status'], 'located')
        self.assertEqual(prediction['box'], [0, 0, 256, 256])
        self.assertAlmostEqual(prediction['center_xy'][0], 9.5)
        self.assertAlmostEqual(prediction['center_xy'][1], 11.5)
        self.assertEqual(box_coverage(prediction['box'], [8, 10, 12, 14]), (True, True))

    def test_minmax_uses_only_rect_and_rejects_nonfinite_or_flat(self):
        cam = np.zeros((512, 512))
        cam[120:124, 150:154] = 4
        cam[:50, :] = np.nan
        rect = [100, 100, 300, 300]
        self.assertEqual(locate(cam, rect)['status'], 'located')
        cam[110, 110] = np.nan
        self.assertEqual(locate(cam, rect)['reason'], 'nonfinite_map')
        self.assertEqual(locate(np.ones((512, 512)), rect)['reason'], 'flat_map')
        with self.assertRaises(ValueError):
            locate(cam, [0, 0, 513, 512])

    def test_component_threshold_connectivity_and_distance(self):
        cam = np.zeros((512, 512))
        cam[100:103, 100:105] = 1  # 15 pixels are ignored
        self.assertEqual(locate(cam, [0, 0, 512, 512])['reason'], 'no_qualifying_component')
        cam = np.zeros((512, 512))
        cam[100:104, 100:104] = 1
        cam[104:108, 104:108] = 1  # diagonal adjacency forms one 8-connected component
        self.assertEqual(locate(cam, [0, 0, 512, 512])['status'], 'located')
        cam[350:354, 350:354] = .9
        self.assertEqual(locate(cam, [0, 0, 512, 512])['reason'], 'distant_component')

    def test_peak_priority_ties_and_diffuse_mass(self):
        cam = np.zeros((512, 512))
        cam[80:84, 80:84] = .9
        cam[100:104, 100:104] = 1
        result = locate(cam, [0, 0, 512, 512])
        self.assertEqual(result['status'], 'located')
        self.assertAlmostEqual(result['center_xy'][0], 101.5)
        cam[80:84, 80:84] = 1
        result = locate(cam, [0, 0, 512, 512])
        self.assertAlmostEqual(result['center_xy'][0], 81.5)
        cam = np.full((512, 512), .1)
        cam[0, 0] = 0
        cam[250:254, 250:254] = 1
        self.assertEqual(locate(cam, [0, 0, 512, 512])['reason'], 'diffuse_mass')

    def test_gradcam_uses_last_feature_and_positive_logit_gradient(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(8)])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.network = Network()

            def forward(self, image):
                feature = self.network.features(image)
                return feature[:, 0].mean(dim=(1, 2)) - feature[:, 1].mean(dim=(1, 2))

        image = torch.zeros(1, 3, 512, 512)
        image[:, 0, 10:20, 10:20] = 1
        image.requires_grad_(True)
        model = Model().eval()
        cam = grad_cam(model, image, {})
        self.assertEqual(cam.shape, (512, 512))
        self.assertGreater(float(cam[15, 15]), 0)
        self.assertEqual(float(cam[0, 0]), 0)
        self.assertFalse(model.network.features[7]._forward_hooks)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    paths = ('tools/diagnostics/fit_disc_attention.py', 'tools/diagnostics/fit_disc_consensus.py',
             'tests/test_fit_disc_attention.py')
    print(json.dumps({'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                        for name in paths}}), flush=True)
    unittest.main()
