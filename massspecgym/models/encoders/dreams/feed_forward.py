"""
Generic configurable FeedForward module for DreaMS.

Ported from external/DreaMS/dreams/models/layers/feed_forward.py.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Configurable feed-forward network with variable depth.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        hidden_dim: Hidden dimension (defaults to out_dim).
        depth: Number of layers (1 = single linear).
        dropout: Dropout rate.
        act_last: Whether to apply activation after last layer.
        bias: Whether to use bias in linear layers.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None, depth=2,
                 dropout=0.0, act_last=True, bias=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim

        layers = []
        for i in range(depth):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == depth - 1 else hidden_dim
            layers.append(nn.Linear(d_in, d_out, bias=bias))
            if i < depth - 1 or act_last:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
