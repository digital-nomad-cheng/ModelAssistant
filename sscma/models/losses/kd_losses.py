# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from sscma.registry import MODELS


@MODELS.register_module()
class KDLoss(nn.Module):
    """Knowledge Distillation KL Divergence on logits.

    Args:
        temperature (float): Softmax temperature. Default 4.0
        reduction (str): Reduction. Default 'batchmean' to match torch.kl_div KL.
        loss_weight (float): Weight multiplier applied outside (kept for config uniformity).
        use_sigmoid (bool): If True, apply sigmoid+MSE instead of KL (for 1-channel logits like objectness).
    """

    def __init__(self, temperature=4.0, reduction='batchmean', use_sigmoid=False, loss_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        if student_logits.numel() == 0:
            return student_logits.sum() * 0
        if self.use_sigmoid:
            # BCE style distill for single logit
            s = torch.sigmoid(student_logits)
            t = torch.sigmoid(teacher_logits).detach()
            loss = F.mse_loss(s, t, reduction='mean')
            return loss * self.loss_weight
        T = self.temperature
        # reshape to (N, C)
        s = student_logits / T
        t = teacher_logits.detach() / T
        log_p_s = F.log_softmax(s, dim=1)
        p_t = F.softmax(t, dim=1)
        loss = F.kl_div(log_p_s, p_t, reduction=self.reduction) * (T * T)
        return loss * self.loss_weight


@MODELS.register_module()
class BBoxMSELoss(nn.Module):
    """BBox regression distillation (simple MSE on decoded xywh).

    Args:
        loss_weight (float): weight.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, student_boxes: torch.Tensor, teacher_boxes: torch.Tensor):
        if student_boxes.numel() == 0:
            return student_boxes.sum() * 0
        return F.mse_loss(student_boxes, teacher_boxes.detach(), reduction='mean') * self.loss_weight


@MODELS.register_module()
class FeatureMSELoss(nn.Module):
    """Feature map distillation with optional 1x1 projection if channels mismatch.

    Creates adapters lazily on first forward for each pair index.

    Args:
        project_if_mismatch (bool): create 1x1 conv to align channels.
        loss_weight (float): global weight.
        eps (float): for numerical stability in scaling.
    """

    def __init__(self, project_if_mismatch=True, loss_weight=1.0, eps=1e-6):
        super().__init__()
        self.project_if_mismatch = project_if_mismatch
        self.loss_weight = loss_weight
        self.eps = eps
        self.adapters = nn.ModuleList()  # will align to number of feature pairs
        self._built = False

    def _build_if_needed(self, student_feats, teacher_feats, device, dtype):
        if self._built:
            return
        if len(student_feats) != len(teacher_feats):
            raise AssertionError(
                f'Feature list length mismatch: student={len(student_feats)} teacher={len(teacher_feats)}'
            )
        for s, t in zip(student_feats, teacher_feats):
            if s.shape[1] != t.shape[1] and self.project_if_mismatch:
                adapt = nn.Conv2d(s.shape[1], t.shape[1], 1, bias=False)
            else:
                adapt = nn.Identity()
            self.adapters.append(adapt.to(device=device, dtype=dtype))
        self._built = True

    def forward(self, student_feats, teacher_feats):
        if not student_feats or not teacher_feats:
            # graceful zero
            dev = (student_feats[0].device if student_feats
                   else teacher_feats[0].device if teacher_feats
                   else 'cpu')
            return torch.tensor(0.0, device=dev)
        device = student_feats[0].device
        dtype = student_feats[0].dtype
        self._build_if_needed(student_feats, teacher_feats, device, dtype)

        losses = []
        for s, t, adapt in zip(student_feats, teacher_feats, self.adapters):
            if s.shape[2:] != t.shape[2:]:
                s = F.interpolate(s, size=t.shape[2:], mode='bilinear', align_corners=False)
            # Ensure adapter on correct device/dtype even if model moved after build
            if adapt is not None and (next(adapt.parameters(), torch.empty(0)).device != device):
                adapt.to(device=device, dtype=dtype)
            s_adapt = adapt(s)
            losses.append(F.mse_loss(s_adapt, t.detach(), reduction='mean'))
        if not losses:
            return torch.tensor(0.0, device=device)
        return sum(losses) / len(losses) * self.loss_weight
