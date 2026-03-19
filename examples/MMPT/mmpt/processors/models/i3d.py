"""Inception I3D model for RGB video feature extraction.

Architecture from: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
Carreira & Zisserman, CVPR 2017.

This implementation matches the layer naming convention of the BSL-1K / BSL-5K
checkpoints (gulvarol/bsl1k, ECCV'20 and CVPR'21), which follow the
piergiaj/pytorch-i3d naming scheme:
  Conv3d_1a_7x7, Conv3d_2b_1x1, Conv3d_2c_3x3,
  Mixed_3b ... Mixed_5c, logits.

Use `extract_features()` to get 1024-dim Mixed_5c features before the
classification head.  This is what the BSL-5K model was released for
("useful for pretraining or extracting video features", CVPR'21 paper).

Input convention (matching BSL-5K preprocessing):
  - Tensor shape: (B, 3, T, H, W)
  - Values normalised to ~[-1, 1]:  pixel/255, then (x - 0.5) / 0.5

Output of extract_features():
  - Shape: (B, 1024, T', H', W')  where T'≈T/8, H'=W'=H/32
  - Apply adaptive_avg_pool3d(output, 1) then squeeze to get (B, 1024) per clip.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        use_bias=False,
        use_batch_norm=True,
        activation_fn=F.relu,
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Branch 0: 1×1×1
        self.b0 = Unit3D(in_channels, out_channels[0])
        # Branch 1: 1×1×1 → 3×3×3
        self.b1a = Unit3D(in_channels, out_channels[1])
        self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_shape=(3, 3, 3), padding=1)
        # Branch 2: 1×1×1 → 3×3×3
        self.b2a = Unit3D(in_channels, out_channels[3])
        self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_shape=(3, 3, 3), padding=1)
        # Branch 3: MaxPool → 1×1×1
        self.b3a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1)
        self.b3b = Unit3D(in_channels, out_channels[5])

    def forward(self, x):
        return torch.cat([
            self.b0(x),
            self.b1b(self.b1a(x)),
            self.b2b(self.b2a(x)),
            self.b3b(self.b3a(x)),
        ], dim=1)


class InceptionI3d(nn.Module):
    """Inception I3D RGB model.

    Parameters
    ----------
    num_classes : int
        Number of output classes (5383 for BSL-5K CVPR'21 M+D+A model).
        Not used when calling extract_features().
    in_channels : int
        Number of input channels (3 for RGB).
    """

    def __init__(self, num_classes=5383, in_channels=3):
        super().__init__()

        self.Conv3d_1a_7x7 = Unit3D(in_channels, 64, kernel_shape=(7, 7, 7),
                                     stride=(2, 2, 2), padding=(3, 3, 3))
        self.MaxPool3d_2a_3x3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Conv3d_2b_1x1 = Unit3D(64, 64)
        self.Conv3d_2c_3x3 = Unit3D(64, 192, kernel_shape=(3, 3, 3), padding=1)
        self.MaxPool3d_3a_3x3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.Mixed_3b = InceptionModule(192,  [64,  96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionModule(256,  [128, 128, 192, 32, 96, 64])
        self.MaxPool3d_4a_3x3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.Mixed_4b = InceptionModule(480,  [192,  96, 208, 16, 48,  64])
        self.Mixed_4c = InceptionModule(512,  [160, 112, 224, 24, 64,  64])
        self.Mixed_4d = InceptionModule(512,  [128, 128, 256, 24, 64,  64])
        self.Mixed_4e = InceptionModule(512,  [112, 144, 288, 32, 64,  64])
        self.Mixed_4f = InceptionModule(528,  [256, 160, 320, 32, 128, 128])
        self.MaxPool3d_5a_2x2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)

        self.Mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128])

        # Classification head (not used in extract_features, but needed for weight loading)
        self.logits = Unit3D(1024, num_classes, kernel_shape=(1, 1, 1),
                             use_batch_norm=False, use_bias=True, activation_fn=None)

    def extract_features(self, x):
        """Run forward pass up to and including Mixed_5c.

        Parameters
        ----------
        x : Tensor  (B, 3, T, H, W)

        Returns
        -------
        Tensor  (B, 1024, T', H', W')
        """
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        return x  # (B, 1024, T', H', W')

    def forward(self, x):
        features = self.extract_features(x)
        out = F.avg_pool3d(features, features.shape[2:])  # global avg pool
        out = self.logits(out)
        return out.squeeze(-1).squeeze(-1).squeeze(-1)
