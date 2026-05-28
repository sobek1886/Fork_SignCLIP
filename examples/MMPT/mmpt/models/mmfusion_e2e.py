"""
MMFusionE2E: end-to-end model that prepends the Logos MViTv2-S backbone
in front of MMFusionSeparate's 12-layer MMBert video encoder.

forward_video receives raw clip tensors (B, T, 3, 32, 224, 224) instead of
pre-extracted .npy features (B, T, 768).  The backbone converts each clip to
a 768-dim vector; the rest of the forward pass is identical to
MMFusionSeparate (videomlp projection → MMBert → attention-weighted pool).

Config keys (under model:):
  backbone_ckpt:    path to logos_autsl_wlasl_model.pth
  freeze_backbone:  true | false  (default: true — frozen MViT, only MMBert trains)
  ... all other MMFusionSeparate keys (num_hidden_video_layers, vfeat_dim, …) ...
"""

import contextlib

import torch

from .mmfusion import MMFusionSeparate


class MMFusionE2E(MMFusionSeparate):
    """MMFusionSeparate with a Logos MViTv2-S backbone prepended.

    Frozen variant (freeze_backbone=true):
        Only MMBert weights are updated.  MViT runs under torch.no_grad().
        Useful for learning a better temporal aggregator on top of fixed
        Logos features without the GPU memory cost of MViT gradients.

    Fine-tune variant (freeze_backbone=false):
        Gradients flow through MViT as well.  Requires more GPU memory.
        Use a smaller batch size and/or gradient checkpointing.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.freeze_backbone = (
            config.model.freeze_backbone
            if "freeze_backbone" in config.model else True
        )
        self.backbone = self._load_backbone(config)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    # ------------------------------------------------------------------
    # Backbone construction
    # ------------------------------------------------------------------

    def _load_backbone(self, config):
        try:
            from mmaction.registry import MODELS
        except ImportError as exc:
            raise ImportError(
                "mmaction2 is required for MMFusionE2E. "
                "Install it in the MMPT venv:\n"
                "  pip install mmaction2 mmengine mmcv"
            ) from exc

        backbone = MODELS.build(dict(
            type='MViT',
            arch='small',
            drop_path_rate=0.1,
            dim_mul_in_attention=False,
        ))

        ckpt_path = config.model.backbone_ckpt
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt)

        # Strip the 'backbone.' prefix added by the mmaction2 training wrapper
        backbone_state = {
            k[len('backbone.'):]: v
            for k, v in state.items()
            if k.startswith('backbone.')
        }
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        non_head_missing = [k for k in missing if 'head' not in k]
        if non_head_missing:
            print(f'[MMFusionE2E] WARNING: {len(non_head_missing)} backbone keys '
                  f'missing: {non_head_missing[:3]}')
        if unexpected:
            print(f'[MMFusionE2E] WARNING: {len(unexpected)} unexpected keys ignored')

        print(f'[MMFusionE2E] Backbone loaded from {ckpt_path} '
              f'(freeze_backbone={self.freeze_backbone})')
        return backbone.eval()

    # ------------------------------------------------------------------
    # Backbone inference
    # ------------------------------------------------------------------

    def _run_backbone(self, clips):
        """Run MViTv2-S on a batch of clips.

        Args:
            clips: (N, 3, 32, 224, 224) float tensor, already on device.

        Returns:
            feats: (N, 768)
        """
        ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with ctx:
            feat = self.backbone(clips)
            # mmaction2 may return nested list/tuple — unwrap until tensor
            while isinstance(feat, (list, tuple)):
                feat = feat[-1]
            if feat.ndim == 5:      # (N, C, T, H, W) → global avg pool
                feat = feat.mean(dim=[2, 3, 4])
            elif feat.ndim == 3:    # (N, seq, C) → mean over sequence
                feat = feat.mean(dim=1)
        return feat  # (N, 768)

    # ------------------------------------------------------------------
    # Override forward_video
    # ------------------------------------------------------------------

    def forward_video(self, vfeats, vmasks, caps, cmasks, **kwargs):
        """
        Args:
            vfeats:  (B, T, 3, 32, 224, 224) — raw MViT input clips
            vmasks:  (B, T)                  — True for valid clip positions
            caps:    (B, max_len)
            cmasks:  (B, max_len)

        Returns:
            pooled: (B, 768)
        """
        B, T, C, Tf, H, W = vfeats.shape
        clips = vfeats.view(B * T, C, Tf, H, W)       # (B*T, 3, 32, 224, 224)
        clip_feats = self._run_backbone(clips)          # (B*T, 768)
        clip_feats = clip_feats.view(B, T, -1)         # (B, T, 768)
        # Delegate to MMFusionSeparate, which runs videomlp + MMBert + pool
        return super().forward_video(clip_feats, vmasks, caps, cmasks, **kwargs)
