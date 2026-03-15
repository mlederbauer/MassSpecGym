"""
Fourier feature encoding for m/z values in DreaMS.

Ported from external/DreaMS/dreams/models/layers/fourier_features.py.
"""

from math import ceil

import torch
import torch.nn as nn
from torch.nn import Parameter


class FourierFeatures(nn.Module):
    """Fourier feature encoding for m/z values.

    Maps scalar m/z values to high-dimensional sinusoidal features,
    enabling the transformer to distinguish fine-grained mass differences.

    Strategies:
    - 'voronov_et_al': log-spaced frequencies (default for DreaMS).
    - 'random': random Gaussian frequencies.
    - 'lin_float_int': linear combination of float and integer frequencies.

    Args:
        strategy: Frequency spacing strategy.
        x_min: Minimum expected input value (for frequency calculation).
        x_max: Maximum expected input value.
        trainable: Whether frequencies are learnable.
        funcs: Which trig functions ('both', 'sin', 'cos').
        sigma: Std for random frequencies.
        num_freqs: Number of frequency components.
    """

    def __init__(self, strategy, x_min, x_max, trainable=True, funcs='both',
                 sigma=10, num_freqs=512):
        assert strategy in {'random', 'voronov_et_al', 'lin_float_int'}
        assert funcs in {'both', 'sin', 'cos'}
        assert x_min < 1

        super().__init__()
        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs

        if strategy == 'random':
            b = torch.randn(num_freqs) * sigma
        elif strategy == 'voronov_et_al':
            b = torch.tensor([
                1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2)))
                for i in range(1, num_freqs)
            ])
        elif strategy == 'lin_float_int':
            b = torch.tensor(
                [1 / (x_min * i) for i in range(2, ceil(1 / x_min), 2)] +
                [1 / (1 * i) for i in range(2, ceil(x_max), 1)]
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        b = b.unsqueeze(0)
        self.b = nn.Parameter(b, requires_grad=self.trainable)

    def num_features(self):
        n = self.b.shape[1]
        if self.funcs == 'both':
            return n * 2
        return n

    def forward(self, x):
        x = 2 * torch.pi * x @ self.b
        if self.funcs == 'both':
            x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        elif self.funcs == 'cos':
            x = torch.cos(x)
        elif self.funcs == 'sin':
            x = torch.sin(x)
        return x
