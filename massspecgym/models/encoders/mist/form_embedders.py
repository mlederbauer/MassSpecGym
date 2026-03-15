"""
Formula embedders for encoding chemical formula counts in the MIST encoder.

Each embedder maps integer atom counts to fixed-dimensional vector representations.
These are used by FormulaTransformer to encode peak formula annotations.
"""

import numpy as np
import torch
import torch.nn as nn

from .chem_constants import NORM_VEC


class IntFeaturizer(nn.Module):
    """Base class for mapping integers to vector representations.

    Creates embeddings for integer values (typically atom counts in chemical formulas).
    Subclasses define ``int_to_feat_matrix`` where each row is the vector for that integer.
    """

    MAX_COUNT_INT = 255
    NUM_EXTRA_EMBEDDINGS = 1

    def __init__(self, embedding_dim):
        super().__init__()
        weights = torch.zeros(self.NUM_EXTRA_EMBEDDINGS, embedding_dim)
        self._extra_embeddings = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self._extra_embeddings, 0.0, 1.0)
        self.embedding_dim = embedding_dim

    def forward(self, tensor):
        orig_shape = tensor.shape
        out_tensor = torch.empty(
            (*orig_shape, self.embedding_dim), device=tensor.device
        )
        extra_embed = tensor >= self.MAX_COUNT_INT

        tensor = tensor.long()
        norm_embeds = self.int_to_feat_matrix[tensor[~extra_embed]]
        extra_embeds = self._extra_embeddings[tensor[extra_embed] - self.MAX_COUNT_INT]

        out_tensor[~extra_embed] = norm_embeds
        out_tensor[extra_embed] = extra_embeds
        return out_tensor.reshape(*orig_shape[:-1], -1)

    @property
    def num_dim(self):
        return self.int_to_feat_matrix.shape[1]

    @property
    def full_dim(self):
        return self.num_dim * NORM_VEC.shape[0]


class FourierFeaturizer(IntFeaturizer):
    """Fourier feature embeddings using sin and cos functions."""

    def __init__(self):
        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2
        freqs = 0.5 ** torch.arange(num_freqs, dtype=torch.float32)
        freqs_time_2pi = 2 * np.pi * freqs
        super().__init__(embedding_dim=2 * freqs_time_2pi.shape[0])

        combo = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        all_features = torch.cat([torch.cos(combo), torch.sin(combo)], dim=1)
        self.int_to_feat_matrix = nn.Parameter(all_features.float(), requires_grad=False)


class FourierFeaturizerSines(IntFeaturizer):
    """Fourier features using only sine functions."""

    def __init__(self):
        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2
        freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs
        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        combo = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(torch.sin(combo).float(), requires_grad=False)


class FourierFeaturizerAbsoluteSines(IntFeaturizer):
    """Fourier features using absolute value of sine functions."""

    def __init__(self):
        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2
        freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs
        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        combo = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(
            torch.abs(torch.sin(combo)).float(), requires_grad=False
        )


class FourierFeaturizerPosCos(IntFeaturizer):
    """Fourier features using positive cosine: (-cos(x) + 1), zero at x=0."""

    def __init__(self, num_funcs=9):
        self.num_funcs = num_funcs
        max_freq = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 1
        freqs = 0.5 ** np.linspace(1, max_freq, num_funcs)
        freqs_time_2pi = torch.from_numpy(2 * np.pi * freqs).float()
        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        combo = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(
            (-torch.cos(combo) + 1).float(), requires_grad=False
        )


class RBFFeaturizer(IntFeaturizer):
    """Radial basis function embeddings."""

    def __init__(self, num_funcs=32):
        super().__init__(embedding_dim=num_funcs)
        width = (self.MAX_COUNT_INT - 1) / num_funcs
        centers = torch.linspace(0, self.MAX_COUNT_INT - 1, num_funcs)
        pre_exp = (
            -0.5
            * ((torch.arange(self.MAX_COUNT_INT)[:, None] - centers[None, :]) / width) ** 2
        )
        self.int_to_feat_matrix = nn.Parameter(torch.exp(pre_exp).float(), requires_grad=False)


class OneHotFeaturizer(IntFeaturizer):
    """One-hot encoding for integers."""

    def __init__(self):
        super().__init__(embedding_dim=self.MAX_COUNT_INT)
        self.int_to_feat_matrix = nn.Parameter(
            torch.eye(self.MAX_COUNT_INT).float(), requires_grad=False
        )


class LearnedFeaturizer(IntFeaturizer):
    """Learned embeddings for integers."""

    def __init__(self, feature_dim=32):
        super().__init__(embedding_dim=feature_dim)
        self.nn_embedder = nn.Embedding(
            self.MAX_COUNT_INT + self.NUM_EXTRA_EMBEDDINGS, feature_dim
        )
        self.int_to_feat_matrix = list(self.nn_embedder.parameters())[0]

    def forward(self, tensor):
        orig_shape = tensor.shape
        out_tensor = self.nn_embedder(tensor.long())
        return out_tensor.reshape(*orig_shape[:-1], -1)


class FloatFeaturizer(IntFeaturizer):
    """Simple normalized float features (divide by per-element max count)."""

    def __init__(self):
        super().__init__(embedding_dim=1)
        self.norm_vec = nn.Parameter(
            torch.from_numpy(NORM_VEC).float(), requires_grad=False
        )

    def forward(self, tensor):
        tens_shape = tensor.shape
        out_shape = [1] * (len(tens_shape) - 1) + [-1]
        return tensor / self.norm_vec.reshape(*out_shape)

    @property
    def num_dim(self):
        return 1


def get_embedder(embedder: str) -> IntFeaturizer:
    """Get embedder instance by name."""
    registry = {
        "fourier": FourierFeaturizer,
        "rbf": RBFFeaturizer,
        "one-hot": OneHotFeaturizer,
        "learnt": LearnedFeaturizer,
        "float": FloatFeaturizer,
        "fourier-sines": FourierFeaturizerSines,
        "abs-sines": FourierFeaturizerAbsoluteSines,
        "pos-cos": FourierFeaturizerPosCos,
    }
    if embedder not in registry:
        raise NotImplementedError(f"Unknown embedder type: {embedder}")
    return registry[embedder]()
