"""
MIST-CF oracle: Chemical formula prediction from MS/MS spectra.

MIST-CF ranks candidate molecular formulas for a given spectrum using an
energy-based scoring model (FormulaTransformer with "abs-sines" embedder).
It is the default formula annotator for MassSpecGym de novo tasks
(non-bonus setting).

Components:
- MistCFNet: The neural scoring model (FormulaTransformer + Linear head).
- predict_formulas(): High-level API for formula prediction.
- enumerate_candidate_formulas(): Pure-Python formula enumeration fallback.

Ported from external/mist-cf/src/mist_cf/mist_cf_score/.
"""

from .predict import predict_formulas, enumerate_candidate_formulas, FormulaCandidate
from .model import MistCFNet, MistCFFormulaTransformer
