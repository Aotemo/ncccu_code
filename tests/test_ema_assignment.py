"""Guarded Pod CPU checks for the fit-only EMA assignment signal."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/bundle'))
sys.path.insert(0, str(ROOT / 'runtime'))


class EMAAssignmentTest(unittest.TestCase):
    def test_cross_view_loss_has_independent_probability_oracle_and_no_target_gradients(self):
        import torch
        from retrain.ema_assignment import assignment_loss
        s1 = torch.tensor([[.1 * math.log(4), 0.]], requires_grad=True)
        s2 = torch.tensor([[0., .1 * math.log(4)]], requires_grad=True)
        t1 = torch.tensor([[.04 * math.log(9), 0.]], requires_grad=True)
        t2 = torch.tensor([[0., .04 * math.log(9)]], requires_grad=True)
        center = torch.zeros(1, 2, requires_grad=True)
        loss, stats = assignment_loss(s1, s2, t1, t2, center)
        self.assertAlmostEqual(float(loss.detach()), -.9 * math.log(.2) - .1 * math.log(.8), places=6)
        self.assertAlmostEqual(float(stats['teacher_entropy']), -.9 * math.log(.9) - .1 * math.log(.1), places=6)
        self.assertAlmostEqual(float(stats['teacher_mean_entropy']), math.log(2), places=6)
        loss.backward()
        for value in (s1, s2):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.)
        for value in (t1, t2, center):
            self.assertIsNone(value.grad)

    def test_teacher_and_center_updates_have_known_arithmetic_and_fixed_endpoints(self):
        import torch
        from torch import nn
        from retrain.ema_assignment import teacher_momentum, update_teacher
        student = nn.Linear(2, 2, bias=False)
        with torch.no_grad(): student.weight.fill_(2.)
        teacher = copy.deepcopy(student).requires_grad_(False)
        with torch.no_grad(): teacher.weight.zero_()
        center = torch.zeros(1, 2)
        scores = torch.tensor([[1., 3.], [3., 1.]], requires_grad=True)
        self.assertEqual(teacher_momentum(1), .9995)
        self.assertEqual(teacher_momentum(1600), 1.)
        self.assertLess(teacher_momentum(1), teacher_momentum(800))
        update_teacher(teacher, student, center, scores, update=1)
        self.assertTrue(torch.allclose(teacher.weight, torch.full((2, 2), .001), atol=1e-7))
        self.assertTrue(torch.allclose(center, torch.full((1, 2), .2)))
        self.assertIsNone(teacher.weight.grad)
        self.assertIsNone(scores.grad)
        self.assertFalse(center.requires_grad)

    def test_head_scale_invariance_freeze_unfreeze_and_encoder_only_ownership(self):
        import torch
        from torch import nn
        from retrain.ema import tensor_digest
        from retrain.ema_assignment import AssignmentSSLModel, assignment_loss, freeze_prototypes, teacher_initial_proof
        from retrain.ssl_stage import _derived_state, _non_encoder_state, _optimizer
        class Binary(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Module()
                self.network.features = nn.Conv2d(3, 768, 1)
                self.network.avgpool = nn.AdaptiveAvgPool2d(1)
                self.network.classifier = nn.Sequential(nn.LayerNorm([768, 1, 1]), nn.Flatten(), nn.Linear(768, 1))
                self.age_weight = nn.Parameter(torch.zeros(2))
        torch.manual_seed(31)
        binary = Binary(); head = tensor_digest(_non_encoder_state(binary)); before = tensor_digest(_derived_state(binary))
        rng = torch.get_rng_state().clone()
        student = AssignmentSSLModel(binary, projector_seed=123)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        teacher = copy.deepcopy(student).requires_grad_(False).eval()
        proof = teacher_initial_proof(student, teacher)
        self.assertEqual(proof['teacher_student_sha256'], proof['teacher_tensor_sha256'])
        self.assertEqual(proof['teacher_trainable_tensors'], 0)
        self.assertEqual(proof['teacher_shared_parameter_tensors'], 0)
        optimizer = _optimizer(student)
        owned = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertNotIn(id(binary.network.classifier[-1].weight), owned)
        self.assertNotIn(id(binary.age_weight), owned)
        self.assertFalse(owned & {id(p) for p in teacher.parameters()})
        images = torch.randn(2, 3, 4, 4)
        with torch.no_grad():
            logits = student(images)
            self.assertEqual(logits.shape, (2, 65536))
            self.assertLessEqual(float(logits.abs().max()), 1.000001)
            student.projector.last_layer.weight.mul_(2.)
            self.assertTrue(torch.allclose(logits, student(images), atol=1e-6, rtol=1e-6))
        prototype = student.projector.last_layer.weight
        original = prototype.detach().clone(); center = torch.zeros(1, 65536)
        for update in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            a, b = student(images), student(images * .8)
            with torch.no_grad(): t1, t2 = teacher(images), teacher(images * .8)
            loss, _ = assignment_loss(a, b, t1, t2, center)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))
            self.assertEqual(freeze_prototypes(student, update=update, fit_epoch_updates=1), update == 1)
            optimizer.step()
            if update == 1:
                self.assertTrue(torch.equal(prototype, original))
                self.assertNotIn(prototype, optimizer.state)
            else:
                self.assertFalse(torch.equal(prototype, original))
                self.assertIn(prototype, optimizer.state)
        self.assertNotEqual(tensor_digest(_derived_state(binary)), before)
        self.assertEqual(tensor_digest(_non_encoder_state(binary)), head)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))

    def test_candidates_preserve_complete_R107_supervision(self):
        from retrain import experiment91 as ex
        self.assertTrue(hasattr(ex, 'FIT_EMA_SMALL_BINDING'), 'EMA candidates not registered')
        from retrain.ema_assignment import EMA_RECIPE
        from retrain.ssl_stage import validate_recipe
        original = json.loads((ROOT / 'configs/R107-SCIENCE.json').read_text())
        for label, seed, order, binding in (
                ('R230', 20261029, 145, ex.FIT_EMA_SMALL_BINDING),
                ('R231', 20261027, 146, ex.FIT_EMA_SMALL_SEED27_BINDING)):
            value = json.loads((ROOT / 'configs' / (label + '-SCIENCE.json')).read_text())
            job = copy.deepcopy(original['job']); job['seed'] = seed
            self.assertEqual(value['job'], job)
            self.assertEqual(value['candidate_profile'], binding)
            self.assertEqual(value['candidate']['experiment_order'], order)
            self.assertEqual(value['ssl_recipe'], EMA_RECIPE)
            self.assertEqual(ex.candidate_from_configuration(value), value['candidate'])
            for key in ('public_source', 'paths', 'data_split', 'schedule', 'early_stopping', 'structure'):
                self.assertEqual(value[key], original[key])
            manifest = {'fit_indices': list(range(64)), 'dev_indices': [64], 'audit_indices': []}
            self.assertEqual(validate_recipe(EMA_RECIPE, job, manifest, 8), EMA_RECIPE)
            wrong = copy.deepcopy(EMA_RECIPE); wrong['teacher_temperature'] = .07
            with self.assertRaises(ValueError): validate_recipe(wrong, job, manifest, 8)

    def test_actual_teacher_initial_gate_rejects_shared_trainable_or_changed_targets(self):
        from r115_ssl_gate import verify_teacher_initial
        proof = [dict(rank=rank, teacher_student_sha256='a' * 64, teacher_tensor_sha256='a' * 64,
                      teacher_trainable_tensors=0, teacher_shared_parameter_tensors=0) for rank in range(8)]
        self.assertTrue(verify_teacher_initial(proof, 8))
        for key, change in (('teacher_tensor_sha256', 'b' * 64), ('teacher_trainable_tensors', 1),
                            ('teacher_shared_parameter_tensors', 1)):
            wrong = copy.deepcopy(proof); wrong[3][key] = change
            with self.subTest(key=key), self.assertRaises(ValueError): verify_teacher_initial(wrong, 8)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(max(1, int(os.environ['NCCCU_CPU_THREADS']) // 2))
    torch.set_num_interop_threads(1)
    unittest.main()
