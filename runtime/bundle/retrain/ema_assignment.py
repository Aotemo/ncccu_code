"""Fixed fit-only cross-view assignments from a fresh student's EMA copy."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ema import tensor_digest


EMA_RECIPE = {'version': 'fit-ema-assignment-1600-v1', 'updates': 1600,
    'global_batch': 64, 'per_rank_batch': 8, 'training_views': 2,
    'image_size': 512, 'training_scale': [0.85, 1.0],
    'projector': [768, 2048, 2048, 256, 65536], 'backbone_lr': 1e-4,
    'projector_lr': 1e-3, 'weight_decay': 1e-4,
    'warmup_updates': 160, 'final_lr_factor': 0.01,
    'gradient_clip': 1.0, 'precision': 'fp32',
    'student_temperature': 0.1, 'teacher_temperature': 0.04,
    'center_momentum': 0.9, 'teacher_momentum': [0.9995, 1.0],
    'prototype_scale': 'fixed_unit', 'prototype_freeze': 'first_complete_fit_epoch',
    'target': 'centered_sharpened_stop_gradient_ema_cross_view',
    'learning_rng_domain': 'r115-fit-vicreg-1600-v1',
    'sampling_rng_domain': 'r115-fit-vicreg-1600-v1'}

EMA_HORIZON_RECIPE = dict(EMA_RECIPE, version='fit-ema-assignment-9600-v1', updates=9600)


class AssignmentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(768, 2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, 256))
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=.02)
                nn.init.zeros_(layer.bias)
        # Row normalization is weight normalization with a fixed unit scale.
        self.last_layer = nn.Linear(256, 65536, bias=False)

    def forward(self, value):
        return F.linear(F.normalize(self.mlp(value), dim=-1),
                        F.normalize(self.last_layer.weight, dim=1))


class AssignmentSSLModel(nn.Module):
    def __init__(self, binary, *, projector_seed):
        super().__init__()
        self.features = binary.network.features
        self.avgpool = binary.network.avgpool
        self.final_norm = binary.network.classifier[0]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(projector_seed)
            self.projector = AssignmentHead()

    def forward(self, images):
        return self.projector(self.final_norm(self.avgpool(self.features(images))).flatten(1))


def teacher_initial_proof(student, teacher):
    pointers = {parameter.data_ptr() for parameter in student.parameters()}
    return {'teacher_student_sha256': tensor_digest(student.state_dict()),
        'teacher_tensor_sha256': tensor_digest(teacher.state_dict()),
        'teacher_trainable_tensors': sum(parameter.requires_grad for parameter in teacher.parameters()),
        'teacher_shared_parameter_tensors': sum(parameter.data_ptr() in pointers for parameter in teacher.parameters())}


def assignment_loss(s1, s2, t1, t2, center):
    if (s1.ndim != 2 or any(value.shape != s1.shape for value in (s2, t1, t2))
            or center.shape != (1, s1.shape[1])):
        raise ValueError('EMA assignments require equal global view logits and one prototype center')
    with torch.no_grad():
        logq1 = F.log_softmax((t1.detach() - center.detach()) / .04, dim=-1)
        logq2 = F.log_softmax((t2.detach() - center.detach()) / .04, dim=-1)
        q1, q2 = logq1.exp(), logq2.exp()
        mean = (q1.mean(0) + q2.mean(0)) / 2
        stats = {'teacher_entropy': -((q1 * logq1).sum(-1).mean() + (q2 * logq2).sum(-1).mean()) / 2,
            'teacher_mean_entropy': -(mean * mean.clamp_min(torch.finfo(mean.dtype).tiny).log()).sum()}
    loss = -((q1 * F.log_softmax(s2 / .1, dim=-1)).sum(-1).mean()
             + (q2 * F.log_softmax(s1 / .1, dim=-1)).sum(-1).mean()) / 2
    return loss, stats


def teacher_momentum(update, *, total_updates=1600):
    if total_updates not in (1600, 9600) or not 1 <= update <= total_updates:
        raise ValueError('EMA teacher update outside fixed budget')
    return .9995 + .0005 * (1 - math.cos(math.pi * (update - 1) / (total_updates - 1))) / 2


@torch.no_grad()
def update_teacher(teacher, student, center, global_teacher_scores, *, update, total_updates=1600):
    source, target = dict(student.named_parameters()), dict(teacher.named_parameters())
    if source.keys() != target.keys() or dict(student.named_buffers()).keys() != dict(teacher.named_buffers()).keys():
        raise ValueError('EMA teacher/student parameter or buffer topology differs')
    momentum = teacher_momentum(update, total_updates=total_updates)
    for name, parameter in target.items():
        parameter.mul_(momentum).add_(source[name].detach(), alpha=1 - momentum)
    for name, buffer in teacher.named_buffers():
        buffer.copy_(dict(student.named_buffers())[name])
    center.mul_(.9).add_(global_teacher_scores.detach().mean(0, keepdim=True), alpha=.1)
    return momentum


def freeze_prototypes(student, *, update, fit_epoch_updates):
    frozen = update <= fit_epoch_updates
    if frozen:
        student.projector.last_layer.weight.grad = None
    return frozen
