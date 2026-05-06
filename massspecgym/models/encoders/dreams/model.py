"""
DreaMS model: BERT-style MS/MS spectrum encoder.

Ported from external/DreaMS/dreams/models/dreams/dreams.py.
Preserves the exact args-based config for checkpoint compatibility.

The model encodes MS/MS spectra as sequences of (m/z, intensity) tokens,
using Fourier features for m/z encoding and a transformer encoder.
Output: per-token embeddings of dimension d_model (typically 1024).
"""

from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .fourier_features import FourierFeatures
from .feed_forward import FeedForward
from .layers import TransformerEncoder


class DreaMS(pl.LightningModule):
    """DreaMS spectrum encoder model.

    BERT-style transformer that encodes MS/MS spectra into dense embeddings.
    The precursor m/z is prepended as the first token; its embedding serves
    as the spectrum-level representation.

    Args:
        args: Namespace with model configuration. Key fields:
            d_fourier (int): Fourier feature dimension (default 512).
            d_peak (int): Peak embedding dimension (default 512).
            d_mz_token (int): Discrete m/z token dimension (default 0).
            n_layers (int): Number of transformer layers.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            fourier_strategy (str): Fourier frequency strategy.
            fourier_num_freqs (int): Number of Fourier frequencies.
            charge_feature (bool): Whether to include charge as feature.
            graphormer_mz_diffs (bool): Use Graphormer m/z distance bias.
            vanilla_transformer (bool): Use standard PyTorch transformer.
        spec_preproc: SpectrumPreprocessor instance.
    """

    def __init__(self, args: Namespace, spec_preproc=None):
        super().__init__()
        self.save_hyperparameters()

        self.spec_preproc = spec_preproc
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.lr = getattr(args, 'lr', 1e-4)
        self.weight_decay = getattr(args, 'weight_decay', 0.0)
        self.charge_feature = getattr(args, 'charge_feature', False)
        self.d_fourier = getattr(args, 'd_fourier', 512)
        self.d_peak = getattr(args, 'd_peak', 512)
        self.d_mz_token = getattr(args, 'd_mz_token', 0)
        self.d_model = sum(d for d in [self.d_fourier, self.d_peak, self.d_mz_token] if d)
        args.d_model = self.d_model
        self.dformat = getattr(args, 'dformat', None)
        self.hot_mz_bin_size = getattr(args, 'hot_mz_bin_size', 1.0)
        self.graphormer_mz_diffs = getattr(args, 'graphormer_mz_diffs', False)
        self.graphormer_parametrized = getattr(args, 'graphormer_parametrized', False)
        self.mask_val = getattr(args, 'mask_val', -1.0)
        self.vanilla_transformer = getattr(args, 'vanilla_transformer', False)

        if self.graphormer_mz_diffs and self.graphormer_parametrized:
            args.d_graphormer_params = self.d_fourier if self.d_fourier else 1
        else:
            args.d_graphormer_params = 0

        token_dim = 2
        if self.charge_feature:
            token_dim += 1

        if self.d_fourier:
            max_mz = self.dformat.max_mz if self.dformat else 1000.0
            x_min = getattr(self.dformat, 'max_tbxic_stdev', 0.003) if self.dformat else 0.003
            fourier_min_freq = getattr(args, 'fourier_min_freq', None)
            if fourier_min_freq:
                x_min = fourier_min_freq

            self.fourier_enc = FourierFeatures(
                strategy=getattr(args, 'fourier_strategy', 'voronov_et_al'),
                num_freqs=getattr(args, 'fourier_num_freqs', 512),
                x_min=x_min,
                x_max=max_mz,
                trainable=getattr(args, 'fourier_trainable', True),
            )
            self.ff_fourier = FeedForward(
                in_dim=self.fourier_enc.num_features(),
                out_dim=self.d_fourier,
                dropout=getattr(args, 'dropout', 0.0),
                depth=getattr(args, 'ff_fourier_depth', 2),
                hidden_dim=getattr(args, 'ff_fourier_d', self.d_fourier),
                bias=not getattr(args, 'no_ffs_bias', False),
            )

        elif self.d_mz_token:
            max_mz = self.dformat.max_mz if self.dformat else 1000.0
            num_bins = int(max_mz / self.hot_mz_bin_size) + 2
            self.mz_tokenizer = nn.Embedding(num_bins, self.d_mz_token, padding_idx=0)
            self.ff_mz_token = FeedForward(
                in_dim=self.d_mz_token, hidden_dim=self.d_mz_token,
                out_dim=self.d_mz_token, depth=2,
                dropout=getattr(args, 'dropout', 0.0),
            )

        self.ff_peak = FeedForward(
            in_dim=token_dim,
            hidden_dim=self.d_peak,
            out_dim=self.d_peak,
            depth=getattr(args, 'ff_peak_depth', 2),
            dropout=getattr(args, 'dropout', 0.0),
            bias=not getattr(args, 'no_ffs_bias', False),
        )

        if self.vanilla_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, dim_feedforward=self.d_model * 4,
                nhead=self.n_heads, activation='gelu',
                dropout=getattr(args, 'dropout', 0.0),
                batch_first=True, norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.n_layers
            )
        else:
            self.transformer_encoder = TransformerEncoder(args)

    def forward(self, spec, charge=None):
        """Encode a batch of MS/MS spectra.

        Args:
            spec: Tensor of shape (batch, n_peaks, 2) with [m/z, intensity].
                  First peak is the precursor.
            charge: Optional tensor of shape (batch,) with charge values.

        Returns:
            Tensor of shape (batch, n_peaks, d_model) with per-token embeddings.
        """
        padding_mask = spec[:, :, 0] == 0

        if self.charge_feature:
            if charge is None:
                raise ValueError("charge required when charge_feature=True")
            charge_features = ~padding_mask * charge.unsqueeze(-1)
            spec = torch.cat([spec, charge_features.unsqueeze(-1)], dim=-1)

        normalized_spec = spec.clone()
        max_mz = self.dformat.max_mz if self.dformat else 1000.0
        normalized_spec[..., 0] = normalized_spec[..., 0] / max_mz

        peak_embs = self.ff_peak(normalized_spec)

        if self.d_fourier:
            fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
            spec_embs = torch.cat([peak_embs, fourier_features], dim=-1)
        elif self.d_mz_token:
            mz_bins = (spec[..., 0] / self.hot_mz_bin_size).long().clamp(min=0)
            tokenized_mzs = self.ff_mz_token(self.mz_tokenizer(mz_bins))
            spec_embs = torch.cat([peak_embs, tokenized_mzs], dim=-1)
        else:
            spec_embs = peak_embs

        graphormer_dists = None
        if self.graphormer_mz_diffs:
            if self.d_fourier:
                graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
            else:
                mz_vals = spec[..., 0]
                graphormer_dists = (mz_vals.unsqueeze(2) - mz_vals.unsqueeze(1)).unsqueeze(-1)

        if self.vanilla_transformer:
            output = self.transformer_encoder(spec_embs, src_key_padding_mask=padding_mask)
        else:
            output = self.transformer_encoder(spec_embs, padding_mask, graphormer_dists)

        return output

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
