# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
softmax-based NCE loss, used by this project.
"""

import torch
import torch.nn.functional as F

from torch import nn

from .loss import Loss


class NCE(Loss):
    def __init__(self):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, align_scores, **kargs):
        # note: we reuse the same shape as cls head in BERT (batch_size, 2)
        # but NCE only needs one logits.
        # (so we drop all weights in the second neg logits.)
        align_scores = align_scores[:, :1]
        # duplicate negative examples
        batch_size = align_scores.size(0) // 2
        pos_scores = align_scores[:batch_size]
        neg_scores = align_scores[batch_size:].view(1, batch_size).repeat(
            batch_size, 1)
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        return self.loss(
            scores,
            torch.zeros(
                (batch_size,),
                dtype=torch.long,
                device=align_scores.device),
        )


class T2VContraLoss(Loss):
    """NCE for MM joint space, on softmax text2video matrix.
    """
    def __init__(self):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pooled_video, pooled_text, **kargs):
        batch_size = pooled_video.size(0)
        logits = torch.mm(pooled_text, pooled_video.transpose(1, 0))
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=pooled_video.device)
        return self.loss(logits, targets)


class V2TContraLoss(Loss):
    """NCE for MM joint space, with softmax on video2text matrix."""

    def __init__(self):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pooled_video, pooled_text, **kargs):
        batch_size = pooled_video.size(0)
        logits = torch.mm(pooled_video, pooled_text.transpose(1, 0))
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=pooled_video.device)
        return self.loss(logits, targets)


class MMContraLoss(Loss):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pooled_video, pooled_text, **kwargs):
        logits_per_video = pooled_video @ pooled_text.t()
        logits_per_text = pooled_text @ pooled_video.t()

        targets = torch.arange(
            pooled_video.size(0),
            dtype=torch.long,
            device=pooled_video.device)
        loss_video = self.loss(logits_per_video, targets)
        loss_text = self.loss(logits_per_text, targets)
        return loss_video + loss_text


class DistillContraLoss(MMContraLoss):
    """Symmetric contrastive loss + cosine distillation from a frozen teacher.

    Total loss = InfoNCE(student_video, text) + lambda * cosine_distill(student_video, teacher)

    teacher_embed (shape: [batch, D]) should be pre-extracted embeddings from a
    trained teacher model, produced by extract_teacher_embeddings.py and passed
    through the sample dict by DSDistillAligner.
    """

    def __init__(self, config=None):
        super().__init__()
        self.distill_lambda = getattr(config, "distill_lambda", 0.5) if config is not None else 0.5

    def __call__(self, pooled_video, pooled_text, teacher_embed=None, **kwargs):
        contra_loss = super().__call__(pooled_video, pooled_text)
        if teacher_embed is not None and self.distill_lambda > 0.0:
            student = F.normalize(pooled_video, dim=-1)
            teacher = F.normalize(teacher_embed.to(dtype=pooled_video.dtype, device=pooled_video.device), dim=-1)
            distill_loss = (1.0 - (student * teacher).sum(dim=-1)).mean()
            return contra_loss + self.distill_lambda * distill_loss
        return contra_loss


class SupConLoss(Loss):
    """Supervised Contrastive loss (Khosla et al. 2020).

    Pulls all K views of the same sign together in the shared embedding space.
    Normalises features internally; expects pooled_video of shape (N, D) where
    N = B * K (B signs, K views each).

    Args (from **kwargs / sample dict):
        pooled_video:   (N, D) float tensor — video embeddings (e.g. from
                        MMFusionSeparate.forward_video).
        supcon_labels:  (N,) long tensor — integer sign index, repeated K times
                        per sign (e.g. [0,0,1,1,...,B-1,B-1] for K=2).

    Returns:
        Scalar loss tensor.
    """

    def __init__(self, config=None):
        self.temperature = (
            getattr(config, "supcon_temperature", 0.07)
            if config is not None else 0.07
        )

    def __call__(self, pooled_video, supcon_labels=None, **kwargs):
        if supcon_labels is None:
            return torch.zeros(1, device=pooled_video.device).squeeze()

        device = pooled_video.device
        features = F.normalize(pooled_video, dim=-1)  # (N, D)
        N = features.shape[0]

        # Pairwise cosine similarities scaled by temperature: (N, N)
        sim = torch.mm(features, features.t()) / self.temperature

        # Positive mask: same label AND not the same sample
        labels = supcon_labels.view(-1, 1)                              # (N, 1)
        pos_mask = (labels == labels.t()) & \
                   ~torch.eye(N, dtype=torch.bool, device=device)       # (N, N)

        # Exclude self-similarity from the log-sum-exp denominator
        sim_no_self = sim.masked_fill(
            torch.eye(N, dtype=torch.bool, device=device), float('-inf')
        )
        log_denom = torch.logsumexp(sim_no_self, dim=1, keepdim=True)  # (N, 1)
        log_probs = sim - log_denom                                     # (N, N)

        # Average log-probability over positive pairs, then average over anchors
        n_pos = pos_mask.float().sum(dim=1)                             # (N,)
        valid = n_pos > 0
        if not valid.any():
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        pos_log_probs = (log_probs * pos_mask.float()).sum(dim=1)       # (N,)
        loss = -(pos_log_probs[valid] / n_pos[valid]).mean()
        return loss


class VidNCELoss(Loss):
    """Video-only InfoNCE (instance discrimination). Treats each item in the
    batch as its own class and pushes sign embeddings apart; no text labels or
    positive pairs required. Used as the K=1 real-only baseline.

    Embeddings are L2-normalised and scaled by a temperature before the dot
    product (as in SupConLoss). Without this, forward_video returns raw,
    un-normalised vectors, so the diagonal ``logits[i][i] = ||v_i||^2`` trivially
    dominates every off-diagonal → the cross-entropy is ~0 and the loss produces
    no gradient (the arm never trains).
    """

    def __init__(self, config=None):
        self.loss = nn.CrossEntropyLoss()
        # VidNCE yamls do not set supcon_temperature; OmegaConf returns None for
        # a missing key (not the getattr default), so guard against None.
        temp = getattr(config, "supcon_temperature", None) if config is not None else None
        self.temperature = 0.07 if temp is None else temp

    def __call__(self, pooled_video, **kwargs):
        feats = F.normalize(pooled_video, dim=-1)          # unit-norm rows
        logits = feats @ feats.t() / self.temperature      # (B, B) cosine / tau
        targets = torch.arange(
            logits.size(0), dtype=torch.long, device=logits.device
        )
        return self.loss(logits, targets)


class MTM(Loss):
    """Combination of MFM and MLM."""

    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        video_logits,
        text_logits,
        video_label,
        text_label,
        **kwargs
    ):
        text_logits = torch.cat([
            text_logits,
            torch.zeros(
                (text_logits.size(0), 1), device=text_logits.device)
        ], dim=1)
        vt_logits = torch.cat([video_logits, text_logits], dim=0)
        # loss for video.
        video_label = torch.zeros(
            (video_logits.size(0),),
            dtype=torch.long,
            device=video_logits.device
        )

        # loss for text.
        text_label = text_label.reshape(-1)
        labels_mask = text_label != -100
        selected_text_label = text_label[labels_mask]

        vt_label = torch.cat([video_label, selected_text_label], dim=0)
        return self.loss(vt_logits, vt_label)


class MFMMLM(Loss):
    """Combination of MFM and MLM."""

    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        video_logits,
        text_logits,
        video_label,
        text_label,
        **kwargs
    ):
        # loss for video.
        video_label = torch.zeros(
            (video_logits.size(0),),
            dtype=torch.long,
            device=video_logits.device
        )
        masked_frame_loss = self.loss(video_logits, video_label)

        # loss for text.
        text_label = text_label.reshape(-1)
        labels_mask = text_label != -100
        selected_text_label = text_label[labels_mask]
        masked_lm_loss = self.loss(text_logits, selected_text_label)
        return masked_frame_loss + masked_lm_loss
