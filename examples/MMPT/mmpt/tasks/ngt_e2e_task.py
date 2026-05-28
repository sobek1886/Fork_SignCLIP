"""
NGTEndToEndTask: extends NGTPairTask for end-to-end training with raw clips.

NGTRawPairAligner produces vfeats of shape (K, T, 3, 32, 224, 224) per sample,
which the DataLoader collates to (B, K, T, 3, 32, 224, 224) — 7-D.

This task detects the 7-D shape and routes to _raw_paired_forward, which
flattens the view dimension into the batch before calling model.forward_video
(implemented by MMFusionE2E: runs MViT → clip features → MMBert → pool).
"""

import torch

from .ngt_pair_task import NGTPairTask


class NGTEndToEndTask(NGTPairTask):
    """NGTPairTask variant for raw-clip end-to-end training with MMFusionE2E."""

    def __call__(self, model, sample):
        sample = self.reshape_subsample(sample)

        if sample['vfeats'].dim() == 7:
            # Raw clips: (B, K, T, C, T_frames, H, W)
            return self._raw_paired_forward(sample)

        # Pre-extracted features (4-D): fall through to parent
        return super().__call__(model, sample)

    def _raw_paired_forward(self, sample):
        """Forward pass for a batch of K-view raw clip tensors.

        vfeats: (B, K, T, 3, 32, 224, 224)
        vmasks: (B, K, T)
        caps:   (B, max_len)
        cmasks: (B, max_len)
        """
        B, K, T, C, Tf, H, W = sample['vfeats'].shape

        vfeats_flat = sample['vfeats'].view(B * K, T, C, Tf, H, W)
        vmasks_flat = sample['vmasks'].view(B * K, T)

        caps_flat = (
            sample['caps']
            .unsqueeze(1)
            .expand(B, K, -1)
            .reshape(B * K, -1)
        )
        cmasks_flat = (
            sample['cmasks']
            .unsqueeze(1)
            .expand(B, K, -1)
            .reshape(B * K, -1)
        )

        # MMFusionE2E.forward_video: MViT → clip features → MMBert → pool
        pooled_video = self.model.forward_video(
            vfeats_flat, vmasks_flat, caps_flat, cmasks_flat
        )  # (B*K, 768)

        supcon_labels = torch.arange(
            B, device=pooled_video.device
        ).repeat_interleave(K)

        loss = self.inv_loss_fn(
            pooled_video=pooled_video, supcon_labels=supcon_labels
        )

        return {
            "loss": loss,
            "loss_scalar": loss.item(),
            "max_len": self.config.dataset.max_len,
            "batch_size": B,
            "sample_size": 1,
        }
