"""Lightweight multi-scale CNN for spatial feature extraction.

This module provides a small CNN that processes 2D spatial grids from BioEnv
and outputs per-cell learned features. Designed for gradient-free optimization
(Evolution Strategies) with ~1,880 parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialCNN(nn.Module):
    """Multi-scale CNN with local and regional branches.

    Two parallel convolutional branches capture different spatial scales:
    - Local branch: 3x3 convolution (immediate neighbors)
    - Regional branch: 3x3 dilated convolution (wider landscape context)

    Outputs are concatenated and mixed via a 1x1 convolution.

    Architecture:
        Input: (B, n_input_channels, H, W)
          |
          +-- local:    Conv2d(in->n_branch, 3x3, pad=1)       + ReLU
          |
          +-- regional: Conv2d(in->n_branch, 3x3, pad=3, dil=3) + ReLU
          |
          concat -> (B, 2*n_branch, H, W)
          |
          mix: Conv2d(2*n_branch -> n_output, 1x1)
          |
        Output: (B, n_output_features, H, W)

    Args:
        n_input_channels: Number of input channels.
        n_branch_filters: Filters per branch (local and regional).
        n_output_features: Number of output feature channels.

    Example:
        >>> cnn = SpatialCNN(12, 8, 8)
        >>> x = torch.randn(1, 12, 100, 80)
        >>> out = cnn(x)  # shape: (1, 8, 100, 80)
    """

    def __init__(
        self,
        n_input_channels: int = 12,
        n_branch_filters: int = 8,
        n_output_features: int = 8,
    ):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.n_branch_filters = n_branch_filters
        self.n_output_features = n_output_features

        # Local branch: 3x3 conv with padding=1 preserves spatial dims
        self.local_conv = nn.Conv2d(
            n_input_channels,
            n_branch_filters,
            kernel_size=3,
            padding=1,
        )

        # Regional branch: dilated 3x3 conv with padding=dilation preserves spatial dims
        self.regional_conv = nn.Conv2d(
            n_input_channels,
            n_branch_filters,
            kernel_size=3,
            padding=3,
            dilation=3,
        )

        self.relu = nn.ReLU()

        # Mix: 1x1 conv to combine both branches
        self.mix = nn.Conv2d(
            2 * n_branch_filters,
            n_output_features,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, n_input_channels, H, W).

        Returns:
            Output tensor of shape (B, n_output_features, H, W).
        """
        local_out = self.relu(self.local_conv(x))
        regional_out = self.relu(self.regional_conv(x))
        combined = torch.cat([local_out, regional_out], dim=1)
        return self.mix(combined)
