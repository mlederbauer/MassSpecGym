"""
MIST Spectra Encoder for mass spectrometry to fingerprint prediction.

Ported from Goldman et al., "Annotating metabolite mass spectra with domain-inspired
chemical formula transformers", Nature Machine Intelligence, 2023.

Two variants:
- SpectraEncoder: Standard encoder with direct MLP prediction head.
- SpectraEncoderGrowing: Encoder with progressive fingerprint refinement (FPGrowingModule).
"""

from typing import Tuple

import torch
from torch import nn

from . import modules


class SpectraEncoder(nn.Module):
    """Standard spectra encoder for fingerprint prediction.

    Encodes mass spectrometry data into molecular fingerprints using
    a FormulaTransformer backbone with MLP prediction heads.

    Args:
        form_embedder: Type of formula embedding ("float", "pos-cos", etc.).
        output_size: Output fingerprint dimension.
        hidden_size: Hidden dimension size.
        spectra_dropout: Dropout probability.
        top_layers: Number of MLP layers in prediction heads.
        refine_layers: Number of refinement layers (unused in this variant).
        magma_modulo: Dimension for fragment prediction.
        **kwargs: Additional arguments passed to FormulaTransformer.
    """

    def __init__(
        self,
        form_embedder: str = "float",
        output_size: int = 4096,
        hidden_size: int = 50,
        spectra_dropout: float = 0.0,
        top_layers: int = 1,
        refine_layers: int = 0,
        magma_modulo: int = 2048,
        **kwargs,
    ):
        super().__init__()

        spectra_encoder_main = modules.FormulaTransformer(
            hidden_size=hidden_size,
            spectra_dropout=spectra_dropout,
            form_embedder=form_embedder,
            **kwargs,
        )

        fragment_pred_parts = []
        for _ in range(top_layers - 1):
            fragment_pred_parts.append(nn.Linear(hidden_size, hidden_size))
            fragment_pred_parts.append(nn.ReLU())
            fragment_pred_parts.append(nn.Dropout(spectra_dropout))
        fragment_pred_parts.append(nn.Linear(hidden_size, magma_modulo))
        fragment_predictor = nn.Sequential(*fragment_pred_parts)

        top_layer_parts = []
        for _ in range(top_layers - 1):
            top_layer_parts.append(nn.Linear(hidden_size, hidden_size))
            top_layer_parts.append(nn.ReLU())
            top_layer_parts.append(nn.Dropout(spectra_dropout))
        top_layer_parts.append(nn.Linear(hidden_size, output_size))
        top_layer_parts.append(nn.Sigmoid())
        spectra_predictor = nn.Sequential(*top_layer_parts)

        self.spectra_encoder = nn.ModuleList(
            [spectra_encoder_main, fragment_predictor, spectra_predictor]
        )

    def forward(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """Forward pass through the encoder.

        Args:
            batch: Dictionary containing spectra data with keys:
                num_peaks, types, instruments, ion_vec, form_vec, intens.

        Returns:
            Tuple of (fingerprint [batch, output_size], aux_outputs dict).
        """
        encoder_output, aux_out = self.spectra_encoder[0](batch, return_aux=True)
        pred_frag_fps = self.spectra_encoder[1](aux_out["peak_tensor"])
        aux_outputs = {"pred_frag_fps": pred_frag_fps}
        output = self.spectra_encoder[2](encoder_output)
        aux_outputs["h0"] = encoder_output
        return output, aux_outputs


class SpectraEncoderGrowing(nn.Module):
    """Spectra encoder with progressive fingerprint refinement.

    Uses FPGrowingModule to progressively refine fingerprint predictions
    from coarse to fine resolution, enabling multi-scale supervision.

    Args:
        form_embedder: Type of formula embedding.
        output_size: Final fingerprint dimension.
        hidden_size: Hidden dimension size.
        spectra_dropout: Dropout probability.
        top_layers: Number of MLP layers in prediction heads.
        refine_layers: Number of progressive refinement stages.
        magma_modulo: Dimension for fragment prediction.
        **kwargs: Additional arguments passed to FormulaTransformer.
    """

    def __init__(
        self,
        form_embedder: str = "float",
        output_size: int = 4096,
        hidden_size: int = 50,
        spectra_dropout: float = 0.0,
        top_layers: int = 1,
        refine_layers: int = 0,
        magma_modulo: int = 2048,
        **kwargs,
    ):
        super().__init__()

        spectra_encoder_main = modules.FormulaTransformer(
            hidden_size=hidden_size,
            spectra_dropout=spectra_dropout,
            form_embedder=form_embedder,
            **kwargs,
        )

        fragment_pred_parts = []
        for _ in range(top_layers - 1):
            fragment_pred_parts.append(nn.Linear(hidden_size, hidden_size))
            fragment_pred_parts.append(nn.ReLU())
            fragment_pred_parts.append(nn.Dropout(spectra_dropout))
        fragment_pred_parts.append(nn.Linear(hidden_size, magma_modulo))
        fragment_predictor = nn.Sequential(*fragment_pred_parts)

        spectra_predictor = modules.FPGrowingModule(
            hidden_input_dim=hidden_size,
            final_target_dim=output_size,
            num_splits=refine_layers,
            reduce_factor=2,
        )

        self.spectra_encoder = nn.ModuleList(
            [spectra_encoder_main, fragment_predictor, spectra_predictor]
        )

    def forward(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """Forward pass through the growing encoder.

        Args:
            batch: Dictionary containing spectra data.

        Returns:
            Tuple of (final fingerprint [batch, output_size], aux_outputs dict).
                aux_outputs includes: pred_frag_fps, int_preds, h0.
        """
        encoder_output, aux_out = self.spectra_encoder[0](batch, return_aux=True)
        pred_frag_fps = self.spectra_encoder[1](aux_out["peak_tensor"])
        aux_outputs = {"pred_frag_fps": pred_frag_fps}

        output = self.spectra_encoder[2](encoder_output)
        intermediates = output[:-1]
        final_output = output[-1]
        aux_outputs["int_preds"] = intermediates
        aux_outputs["h0"] = encoder_output
        return final_output, aux_outputs
