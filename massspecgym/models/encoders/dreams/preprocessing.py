"""
Spectrum preprocessing for DreaMS input.

Ported from external/DreaMS/dreams/utils/data.py (SpectrumPreprocessor)
and external/DreaMS/dreams/utils/dformats.py (DataFormatA).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DataFormatA:
    """Default data format constraints for DreaMS."""
    min_peaks_n: int = 3
    max_peaks_n: int = 128
    max_mz: float = 1000.0
    max_prec_mz: float = 1000.0
    charge: int = 1
    max_tbxic_stdev: float = 0.003


class SpectrumPreprocessor:
    """Preprocess raw spectra for DreaMS model input.

    Performs: top-k peak selection, intensity normalization, precursor prepend,
    and padding to fixed length.

    Args:
        dformat: Data format constraints (DataFormatA).
        n_highest_peaks: Number of highest-intensity peaks to retain.
        prec_intens: Intensity value for the prepended precursor peak.
    """

    def __init__(self, dformat=None, n_highest_peaks=100, prec_intens=1.1):
        self.dformat = dformat or DataFormatA()
        self.n_highest_peaks = n_highest_peaks
        self.prec_intens = prec_intens

    def __call__(
        self,
        mzs: np.ndarray,
        intensities: np.ndarray,
        precursor_mz: float,
    ) -> np.ndarray:
        """Preprocess a single spectrum for DreaMS input.

        Args:
            mzs: Array of m/z values.
            intensities: Array of intensity values.
            precursor_mz: Precursor m/z value.

        Returns:
            Array of shape (n_highest_peaks + 1, 2) with [m/z, intensity].
            First row is the precursor; padded positions have zeros.
        """
        mzs = np.asarray(mzs, dtype=np.float32)
        intensities = np.asarray(intensities, dtype=np.float32)

        if len(mzs) == 0:
            result = np.zeros((self.n_highest_peaks + 1, 2), dtype=np.float32)
            result[0] = [precursor_mz, self.prec_intens]
            return result

        max_int = intensities.max()
        if max_int > 0:
            intensities = intensities / max_int

        valid = (mzs > 0) & (mzs <= self.dformat.max_mz)
        mzs = mzs[valid]
        intensities = intensities[valid]

        if len(mzs) > self.n_highest_peaks:
            top_idx = np.argsort(intensities)[::-1][:self.n_highest_peaks]
            mzs = mzs[top_idx]
            intensities = intensities[top_idx]

        sort_idx = np.argsort(mzs)
        mzs = mzs[sort_idx]
        intensities = intensities[sort_idx]

        n_peaks = len(mzs)
        result = np.zeros((self.n_highest_peaks + 1, 2), dtype=np.float32)
        result[0] = [precursor_mz, self.prec_intens]
        result[1:n_peaks + 1, 0] = mzs
        result[1:n_peaks + 1, 1] = intensities

        return result
