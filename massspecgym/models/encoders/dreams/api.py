"""
DreaMS pretrained model loading and inference API.

Ported from external/DreaMS/dreams/api.py.
Provides PreTrainedDreaMS for loading checkpoints and computing embeddings.
"""

from argparse import Namespace
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .model import DreaMS
from .preprocessing import SpectrumPreprocessor, DataFormatA


class PreTrainedDreaMS:
    """Pretrained DreaMS model for spectrum embedding inference.

    Loads a DreaMS checkpoint and provides a simple API for computing
    1024-D spectrum embeddings.

    Usage:
        model = PreTrainedDreaMS.from_checkpoint("checkpoints/dreams/embedding_model.ckpt")
        emb = model.embed_spectrum(mzs, intensities, precursor_mz)  # (1024,)
    """

    def __init__(self, model: DreaMS, n_highest_peaks: int = 100):
        self.model = model.eval()
        self.n_highest_peaks = n_highest_peaks
        self.preprocessor = SpectrumPreprocessor(
            dformat=DataFormatA(),
            n_highest_peaks=n_highest_peaks,
        )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        n_highest_peaks: int = 100,
        device: str = None,
        remove_ssl_heads: bool = True,
    ) -> "PreTrainedDreaMS":
        """Load a pretrained DreaMS model from checkpoint.

        Args:
            ckpt_path: Path to DreaMS checkpoint (.ckpt).
            n_highest_peaks: Number of peaks to retain per spectrum.
            device: Device to load model on. Auto-detects if None.
            remove_ssl_heads: Remove unused SSL heads (ff_out, mz_masking_loss, ro_out).

        Returns:
            PreTrainedDreaMS instance ready for inference.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = DreaMS.load_from_checkpoint(ckpt_path, map_location=device)

        if remove_ssl_heads:
            for attr in ('ff_out', 'mz_masking_loss', 'ro_out', 'ff_out_intens'):
                if hasattr(model, attr):
                    delattr(model, attr)

        model = model.to(device)
        return cls(model, n_highest_peaks)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def d_model(self):
        return self.model.d_model

    @torch.no_grad()
    def embed_spectrum(
        self,
        mzs: np.ndarray,
        intensities: np.ndarray,
        precursor_mz: float,
    ) -> np.ndarray:
        """Compute 1024-D embedding for a single spectrum.

        Args:
            mzs: Array of m/z values.
            intensities: Array of intensity values.
            precursor_mz: Precursor m/z.

        Returns:
            1D numpy array of shape (d_model,) - the spectrum embedding.
        """
        spec = self.preprocessor(mzs, intensities, precursor_mz)
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0).to(self.device)
        output = self.model(spec_tensor)
        return output[0, 0, :].cpu().numpy()

    @torch.no_grad()
    def embed_batch(
        self,
        specs: list,
    ) -> np.ndarray:
        """Compute embeddings for a batch of spectra.

        Args:
            specs: List of (mzs, intensities, precursor_mz) tuples.

        Returns:
            Array of shape (batch_size, d_model).
        """
        preprocessed = [
            self.preprocessor(mzs, ints, pmz) for mzs, ints, pmz in specs
        ]
        batch = torch.FloatTensor(np.stack(preprocessed)).to(self.device)
        output = self.model(batch)
        return output[:, 0, :].cpu().numpy()
