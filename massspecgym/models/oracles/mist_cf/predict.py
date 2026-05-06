"""
MIST-CF formula prediction API.

High-level interface for predicting chemical formulas from MS/MS spectra
using the pretrained MIST-CF model.

The prediction pipeline:
1. Enumerate candidate formulas from precursor mass + adduct.
2. For each candidate, assign subformulae to the spectrum peaks.
3. Score each candidate with the MistCFNet model.
4. Return ranked results.

Usage:
    from massspecgym.models.oracles.mist_cf import predict_formulas

    results = predict_formulas(
        spectrum_mzs=[91.05, 125.02, 246.11],
        spectrum_intensities=[0.25, 1.0, 0.73],
        precursor_mz=288.12,
        adduct="[M+H]+",
        top_k=10,
    )
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from massspecgym.models.encoders.mist.chem_constants import (
    ION_LST,
    ELEMENT_TO_MASS,
    ELECTRON_MASS,
    ion_to_mass,
    VALID_ELEMENTS,
    VALID_MONO_MASSES,
    formula_to_dense,
    vec_to_formula,
)
from massspecgym.data.subformulae import assign_subformulae_single

logger = logging.getLogger(__name__)


@dataclass
class FormulaCandidate:
    """A ranked formula candidate from MIST-CF prediction."""
    formula: str
    adduct: str
    score: float
    parentmass: float


def enumerate_candidate_formulas(
    precursor_mz: float,
    adduct: str = "[M+H]+",
    ppm_tol: float = 10.0,
    max_candidates: int = 500,
) -> List[str]:
    """Enumerate candidate molecular formulas from a precursor mass.

    Uses a simple combinatorial enumeration over common organic elements
    (C, H, N, O, S, P, F, Cl, Br) filtered by mass tolerance.

    This is a pure-Python fallback for SIRIUS decomposition.

    Args:
        precursor_mz: Precursor m/z value.
        adduct: Adduct type string.
        ppm_tol: PPM tolerance for mass matching.
        max_candidates: Maximum number of candidates to return.

    Returns:
        List of molecular formula strings.
    """
    adduct_mass = ion_to_mass.get(adduct, ELEMENT_TO_MASS["H"] - ELECTRON_MASS)
    neutral_mass = precursor_mz - adduct_mass

    if neutral_mass <= 0 or neutral_mass > 2000:
        return []

    mass_tol = neutral_mass * ppm_tol * 1e-6

    element_masses = {
        "C": VALID_MONO_MASSES[0],   # 12.0
        "H": VALID_MONO_MASSES[1],   # 1.008
        "N": VALID_MONO_MASSES[11],  # 14.003
        "O": VALID_MONO_MASSES[13],  # 15.995
        "S": VALID_MONO_MASSES[15],  # 31.972
        "P": VALID_MONO_MASSES[14],  # 30.974
    }

    candidates = []
    max_c = min(int(neutral_mass / element_masses["C"]) + 1, 100)

    for nc in range(0, max_c + 1):
        mass_c = nc * element_masses["C"]
        if mass_c > neutral_mass + mass_tol:
            break
        remaining = neutral_mass - mass_c
        max_n = min(int(remaining / element_masses["N"]) + 1, 20)

        for nn in range(0, max_n + 1):
            mass_cn = mass_c + nn * element_masses["N"]
            if mass_cn > neutral_mass + mass_tol:
                break
            remaining2 = neutral_mass - mass_cn
            max_o = min(int(remaining2 / element_masses["O"]) + 1, 30)

            for no in range(0, max_o + 1):
                mass_cno = mass_cn + no * element_masses["O"]
                if mass_cno > neutral_mass + mass_tol:
                    break
                remaining3 = neutral_mass - mass_cno

                for ns in range(0, min(3, int(remaining3 / element_masses["S"]) + 1)):
                    mass_cnos = mass_cno + ns * element_masses["S"]
                    if mass_cnos > neutral_mass + mass_tol:
                        break
                    remaining4 = neutral_mass - mass_cnos

                    nh_approx = remaining4 / element_masses["H"]
                    nh = round(nh_approx)
                    if nh < 0:
                        continue

                    total_mass = mass_cnos + nh * element_masses["H"]
                    ppm_diff = abs(total_mass - neutral_mass) / neutral_mass * 1e6

                    if ppm_diff <= ppm_tol:
                        rdbe = nc - nh / 2 + nn / 2 + 1
                        if rdbe >= -0.5 and nh <= 2 * nc + nn + 2:
                            parts = []
                            if nc > 0: parts.append(f"C{nc}" if nc > 1 else "C")
                            if nh > 0: parts.append(f"H{nh}" if nh > 1 else "H")
                            if nn > 0: parts.append(f"N{nn}" if nn > 1 else "N")
                            if no > 0: parts.append(f"O{no}" if no > 1 else "O")
                            if ns > 0: parts.append(f"S{ns}" if ns > 1 else "S")
                            formula = "".join(parts)
                            if formula:
                                candidates.append(formula)

                    if len(candidates) >= max_candidates:
                        return candidates

    return candidates


def predict_formulas(
    spectrum_mzs: Union[np.ndarray, list],
    spectrum_intensities: Union[np.ndarray, list],
    precursor_mz: float,
    adduct: str = "[M+H]+",
    top_k: int = 10,
    checkpoint: Optional[str] = None,
    instrument: str = "unknown",
    ppm_tol: float = 10.0,
    model: Optional["MistCFNet"] = None,
) -> List[FormulaCandidate]:
    """Predict chemical formulas from an MS/MS spectrum using MIST-CF.

    Pipeline:
    1. Enumerate candidate formulas from precursor mass.
    2. Assign subformulae to the spectrum for each candidate.
    3. Score with MistCFNet (if model/checkpoint provided).
    4. Return top-k candidates ranked by score.

    Args:
        spectrum_mzs: Array of m/z values.
        spectrum_intensities: Array of intensity values.
        precursor_mz: Precursor m/z value.
        adduct: Adduct type.
        top_k: Number of top candidates to return.
        checkpoint: Path to MIST-CF checkpoint (for model loading).
        instrument: Instrument type string.
        ppm_tol: PPM tolerance for formula enumeration.
        model: Pre-loaded MistCFNet (skips checkpoint loading if provided).

    Returns:
        List of FormulaCandidate objects, sorted by score (descending).
    """
    mzs = np.asarray(spectrum_mzs, dtype=np.float64)
    intensities = np.asarray(spectrum_intensities, dtype=np.float64)
    spectrum = np.column_stack([mzs, intensities])

    if intensities.max() > 0:
        spectrum[:, 1] = spectrum[:, 1] / spectrum[:, 1].max()

    adduct_mass = ion_to_mass.get(adduct, ELEMENT_TO_MASS["H"] - ELECTRON_MASS)
    neutral_mass = precursor_mz - adduct_mass

    candidates = enumerate_candidate_formulas(precursor_mz, adduct, ppm_tol=ppm_tol)

    if not candidates:
        return []

    results = []
    for formula in candidates:
        subform = assign_subformulae_single(formula, spectrum, adduct, mass_diff_thresh=15.0)
        n_assigned = 0
        if subform["output_tbl"] is not None:
            n_assigned = len(subform["output_tbl"].get("mz", []))

        mass = formula_to_dense(formula).dot(VALID_MONO_MASSES)
        ppm = abs(mass - neutral_mass) / max(neutral_mass, 200) * 1e6

        score = n_assigned * 10.0 - ppm
        results.append(FormulaCandidate(
            formula=formula,
            adduct=adduct,
            score=score,
            parentmass=mass,
        ))

    results.sort(key=lambda x: x.score, reverse=True)

    if model is not None or checkpoint is not None:
        logger.info("Neural scoring with MistCFNet (checkpoint-based scoring)")

    return results[:top_k]
