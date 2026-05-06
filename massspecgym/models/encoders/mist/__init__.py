"""
MIST (Mass Spectrometry Transformer) encoder for spectrum-to-fingerprint prediction.

Ported from Goldman et al., "Annotating metabolite mass spectra with domain-inspired
chemical formula transformers", Nature Machine Intelligence, 2023.

The encoder predicts molecular fingerprints from tandem mass spectra using a
FormulaTransformer backbone with progressive fingerprint refinement.
"""

from .encoder import SpectraEncoder, SpectraEncoderGrowing
from .chem_constants import (
    VALID_ELEMENTS,
    NORM_VEC,
    formula_to_dense,
    vec_to_formula,
    get_all_subsets,
    get_all_subsets_dense,
    rdbe_filter,
    cross_sum,
    clipped_ppm,
    ion_to_mass,
    ION_LST,
)
