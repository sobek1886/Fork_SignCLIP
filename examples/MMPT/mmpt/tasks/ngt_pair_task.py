"""
NGTPairTask: trains SignCLIP's video encoder with a Supervised Contrastive
loss on paired NGT appearances (real Bushuis + Unreal Engine palmer render).

Each batch item contains K=2 views of the same sign in vfeats of shape
(K, max_video_len, 768).  The task flattens the views into the batch
dimension, runs the video encoder on all B*K clips, then applies SupConLoss
with sign-identity labels [0,0,1,1,...,B-1,B-1].

No text/gloss labels are required; the text side of the model is not called.

Typical config:
    task: NGTPairTask
    dataset:
      meta_processor:  NGTPairMetaProcessor
      video_processor: NGTPairVideoProcessor
      aligner:         NGTPairAligner
      pair_manifest:   /home/psobecki/ngt_pair_manifest.json
    loss:
      loss_cls:            SupConLoss   # primary (and only) loss for NGT-only stage
      supcon_temperature:  0.07
      lambda_inv:          1.0          # weight on SupCon (only loss here, so 1.0)
"""

import torch

from . import tasks
from .. import losses
from .. import processors
from ..datasets import MMDataset
from .task import Task


class NGTPairTask(Task):
    """Task for appearance-invariant contrastive pre-training on paired NGT data.

    Extends Task with:
    - An optional inv_loss_fn (SupConLoss) built from config.loss.loss_cls or
      config.loss.inv_loss_cls.
    - A __call__ that detects paired vfeats (4-D: B×K×T×D), flattens the view
      dimension into the batch, runs forward_video, and applies SupConLoss.
    - Falls back to the standard Task.__call__ for single-view batches so the
      same task class can be reused in a later CLIP fine-tuning stage.
    """

    def build_dataset(self):
        """Build train_data unconditionally from the pair_manifest.

        The base Task.build_dataset() gates on train_path, which NGT configs
        don't set (they use pair_manifest instead).  There is no val/test split
        for NGT pretraining — all signs are used for training.
        """
        meta_processor_cls = getattr(processors, self.config.dataset.meta_processor)
        video_processor_cls = getattr(processors, self.config.dataset.video_processor)
        text_processor_cls = getattr(processors, self.config.dataset.text_processor)
        aligner_cls = getattr(processors, self.config.dataset.aligner)

        self.config.dataset.split = "train"
        meta_processor = meta_processor_cls(self.config.dataset)
        video_processor = video_processor_cls(self.config.dataset)
        text_processor = text_processor_cls(self.config.dataset)
        aligner = aligner_cls(self.config.dataset)
        self.train_data = MMDataset(
            meta_processor, video_processor, text_processor, aligner
        )
        print("train_len", len(self.train_data))
        output = self.train_data[0]
        self.train_data.print_example(output)

    def build_loss(self):
        """Build the primary loss (SupConLoss for NGT-only; MMContraLoss for joint).

        The inv_loss_fn is the SupConLoss.  It can be specified either as:
          loss_cls: SupConLoss      (NGT-only training, no CLIP objective)
        or
          loss_cls: MMContraLoss    (joint training)
          inv_loss_cls: SupConLoss  (added on top)
        """
        super().build_loss()  # builds self.loss_fn from config.loss.loss_cls

        self.inv_loss_fn = None
        self.lambda_inv = 1.0

        if self.config.loss is None:
            return None

        self.lambda_inv = getattr(self.config.loss, 'lambda_inv', 1.0)

        if hasattr(self.config.loss, 'inv_loss_cls') and self.config.loss.inv_loss_cls is not None:
            # Joint mode: a separate inv_loss_cls is specified
            inv_loss_cls = getattr(losses, self.config.loss.inv_loss_cls)
            try:
                self.inv_loss_fn = inv_loss_cls(self.config.loss)
            except TypeError:
                self.inv_loss_fn = inv_loss_cls()
        elif self.config.loss.loss_cls in ('SupConLoss', 'VidNCELoss'):
            # NGT-only mode: the primary loss_fn IS the video-only loss.
            # VidNCELoss is the K=1 arm (SupCon has no positive pairs there);
            # it ignores the supcon_labels kwarg via **kwargs.
            self.inv_loss_fn = self.loss_fn
            self.loss_fn = None  # no CLIP loss

        return self.inv_loss_fn

    def __call__(self, model, sample):
        sample = self.reshape_subsample(sample)

        # Detect paired batch: vfeats has shape (B, K, T, D) with K > 1
        if sample['vfeats'].dim() == 4:
            return self._paired_forward(sample)
        else:
            # Single-view: standard CLIP forward (used in joint training)
            return super().__call__(model, sample)

    def _paired_forward(self, sample):
        """Forward pass for a batch of K-view paired features.

        vfeats:  (B, K, T, D)
        vmasks:  (B, K, T)
        caps:    (B, max_len)   — dummy [CLS][SEP] from NGTPairAligner
        cmasks:  (B, max_len)
        """
        B, K, T, D = sample['vfeats'].shape

        # Flatten views into the batch dimension
        vfeats_flat = sample['vfeats'].view(B * K, T, D)
        vmasks_flat = sample['vmasks'].view(B * K, T)

        # Expand caps/cmasks so each view gets the same dummy text tokens
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

        # Run only the video encoder (no text encoder needed)
        pooled_video = self.model.forward_video(
            vfeats_flat, vmasks_flat, caps_flat, cmasks_flat
        )  # (B*K, D)

        # SupCon labels: sign index repeated K times — [0,0,1,1,...,B-1,B-1]
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
