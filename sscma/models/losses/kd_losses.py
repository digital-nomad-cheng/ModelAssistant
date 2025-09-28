# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from sscma.registry import MODELS
import math


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


@MODELS.register_module()
class ChannelWiseDistillLoss(nn.Module):
    """Channel-wise Knowledge Distillation (CWD) for dense prediction.

    Reference: Channel-wise Knowledge Distillation for Dense Prediction, CVPR 2021.

        Two modes are provided:
                - method='spatial' (default, matches the paper): For each channel we
                    treat its spatial map as a distribution. For a feature F in R^{B,C,H,W}
                    we flatten HxW as S and compute softmax over S independently per channel.
                    KL divergence is then computed between teacher & student per channel
                    distributions and averaged.
                - method='gap_channel' (legacy / previous implementation here): Global
                    average pool (H,W) -> (B,C) then softmax over channels (channel-direction
                    distribution) and KL across channels.

    Args:
        tau (float): Temperature scaling.
        reduction (str): 'mean' or 'sum' over feature levels.
        loss_weight (float): Global multiplier.
        eps (float): Numerical stability constant.
        detach_teacher (bool): Detach teacher features.
        apply_log_softmax_to_student (bool): Use log_softmax for student.
        method (str): 'spatial' or 'gap_channel'.
    """

    def __init__(self,
                 tau: float = 2.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 eps: float = 1e-6,
                 detach_teacher: bool = True,
                 apply_log_softmax_to_student: bool = True,
                 method: str = 'spatial'):
        super().__init__()
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.detach_teacher = detach_teacher
        self.apply_log_softmax_to_student = apply_log_softmax_to_student
        assert method in ('spatial','gap_channel'), "method must be 'spatial' or 'gap_channel'"
        self.method = method

    def _gap_channel_logits(self, feat: torch.Tensor):
        # Global average pooling over spatial -> channel vector
        if feat.dim() != 4:
            raise ValueError('Expected 4D feature map (B,C,H,W)')
        return feat.mean(dim=(2,3))

    def _spatial_channel_kl(self, s: torch.Tensor, t: torch.Tensor):
        # s,t: (B,C,H,W). Flatten spatial -> (B,C,S) then per-channel softmax over S.
        B, C, H, W = s.shape
        S = H * W
        s_flat = s.view(B, C, S) / self.tau
        t_flat = (t.detach() if self.detach_teacher else t).view(B, C, S) / self.tau
        # reshape to treat each (B,C) channel as separate sample: (B*C, S)
        s_flat = s_flat.view(B * C, S)
        t_flat = t_flat.view(B * C, S)
        if self.apply_log_softmax_to_student:
            log_p_s = F.log_softmax(s_flat, dim=1)
        else:
            log_p_s = torch.log(F.softmax(s_flat, dim=1) + self.eps)
        p_t = F.softmax(t_flat, dim=1)
        kl = F.kl_div(log_p_s, p_t, reduction='batchmean') * (self.tau ** 2)
        return kl

    def forward(self, student_feats, teacher_feats):
        if not student_feats or not teacher_feats:
            dev = (student_feats[0].device if student_feats
                   else teacher_feats[0].device if teacher_feats else 'cpu')
            return torch.tensor(0.0, device=dev)

        losses = []
        for s, t in zip(student_feats, teacher_feats):
            if s.shape[2:] != t.shape[2:]:
                # resize student to teacher size for fairness before pooling
                s = F.interpolate(s, size=t.shape[2:], mode='bilinear', align_corners=False)
            if s.shape[1] != t.shape[1]:
                # If channels mismatch, project student with 1x1 conv built on-the-fly (no buffer kept)
                proj = nn.Conv2d(s.shape[1], t.shape[1], kernel_size=1, bias=False).to(device=s.device, dtype=s.dtype)
                # lightweight init (kaiming)
                nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))
                s = proj(s)
            if self.method == 'gap_channel':
                s_c = self._gap_channel_logits(s) / self.tau
                t_c = self._gap_channel_logits(t.detach() if self.detach_teacher else t) / self.tau
                if self.apply_log_softmax_to_student:
                    log_p_s = F.log_softmax(s_c, dim=1)
                else:
                    log_p_s = torch.log(F.softmax(s_c, dim=1) + self.eps)
                p_t = F.softmax(t_c, dim=1)
                kl = F.kl_div(log_p_s, p_t, reduction='batchmean') * (self.tau ** 2)
            else:  # spatial
                kl = self._spatial_channel_kl(s, t)
            losses.append(kl)
        if not losses:
            return torch.tensor(0.0, device=student_feats[0].device)
        if self.reduction == 'mean':
            loss = sum(losses) / len(losses)
        elif self.reduction == 'sum':
            loss = sum(losses)
        else:
            raise ValueError(f'Unsupported reduction {self.reduction}')
        return loss * self.loss_weight
