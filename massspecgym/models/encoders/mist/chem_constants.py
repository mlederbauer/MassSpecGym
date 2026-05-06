"""
Chemistry constants and utilities for the MIST encoder.

Extracted from the original MIST chem_utils.py to be self-contained within
MassSpecGym without external dependencies on the full MIST package.

Includes subformulae assignment utilities (get_all_subsets, rdbe_filter, etc.)
ported from external/mist/src/mist/utils/chem_utils.py.
"""

import re
from collections import defaultdict
from functools import reduce

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Atom

P_TBL = Chem.GetPeriodicTable()

CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C", "H", "As", "B", "Br", "Cl", "Co", "F", "Fe", "I",
    "K", "N", "Na", "O", "P", "S", "Se", "Si",
]
VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]
CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)

ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

# Reasonable normalization vector for elements (estimated by max counts + 1 when zero)
NORM_VEC = np.array([81, 158, 2, 1, 3, 10, 1, 17, 1, 6, 1, 19, 2, 34, 6, 6, 2, 6])

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))

ELECTRON_MASS = 0.00054858

ION_LST = [
    "[M+H]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+",
    "[M+H3N+H]+", "[M]+", "[M-H4O2+H]+",
]

ion_remap = dict(zip(ION_LST, ION_LST))
ion_remap.update({
    "[M+NH4]+": "[M+H3N+H]+",
    "M+H": "[M+H]+",
    "M+Na": "[M+Na]+",
    "M+H-H2O": "[M-H2O+H]+",
    "M-H2O+H": "[M-H2O+H]+",
    "M+NH4": "[M+H3N+H]+",
    "M-2H2O+H": "[M-H4O2+H]+",
    "[M-2H2O+H]+": "[M-H4O2+H]+",
})

ion_to_idx = dict(zip(ION_LST, np.arange(len(ION_LST))))

ion_to_mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
}

ion_to_add_vec = {
    "[M+H]+": element_to_position["H"],
    "[M+Na]+": element_to_position["Na"],
    "[M+K]+": element_to_position["K"],
    "[M-H2O+H]+": -element_to_position["O"] - element_to_position["H"],
    "[M+H3N+H]+": element_to_position["N"] + element_to_position["H"] * 4,
    "[M]+": np.zeros_like(element_to_position["H"]),
    "[M-H4O2+H]+": -element_to_position["O"] * 2 - element_to_position["H"] * 3,
}

instrument_to_type = defaultdict(lambda: "unknown")
instrument_to_type.update({
    "Thermo Finnigan Velos Orbitrap": "orbitrap",
    "Thermo Finnigan Elite Orbitrap": "orbitrap",
    "Orbitrap Fusion Lumos": "orbitrap",
    "Q-ToF (LCMS)": "qtof",
    "Unknown (LCMS)": "unknown",
    "ion trap": "iontrap",
    "FTICR (LCMS)": "fticr",
    "Bruker Q-ToF (LCMS)": "qtof",
    "Orbitrap (LCMS)": "orbitrap",
})

instruments = sorted(list(set(instrument_to_type.values())))
max_instr_idx = len(instruments) + 1
instrument_to_idx = dict(zip(instruments, np.arange(len(instruments))))


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """Convert chemical formula string to dense element-count vector."""
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    if len(total_onehot) == 0:
        return np.zeros(len(element_to_position))
    return np.vstack(total_onehot).sum(0)


def vec_to_formula(form_vec) -> str:
    """Convert dense element-count vector back to formula string."""
    build_str = ""
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        build_str += f"{el}{ct}" if ct > 1 else f"{el}"
    return build_str


def standardize_form(formula: str) -> str:
    """Standardize chemical formula to canonical form."""
    return vec_to_formula(formula_to_dense(formula))


def standardize_adduct(adduct: str) -> str:
    """Standardize adduct notation."""
    adduct = adduct.replace(" ", "")
    adduct = ion_remap.get(adduct, adduct)
    if adduct not in ION_LST:
        raise ValueError(f"Adduct {adduct} not in ION_LST")
    return adduct


def get_ion_idx(ionization: str) -> int:
    """Map ionization string to index."""
    return ion_to_idx[ionization]


def get_instr_idx(instrument: str) -> int:
    """Map instrument string to index."""
    inst = instrument_to_type.get(instrument, "unknown")
    return instrument_to_idx[inst]


# --- Subformulae assignment utilities ---
# Ported verbatim from external/mist/src/mist/utils/chem_utils.py

# RDBE multiplier vector: 2*C + N + P - H - Cl - Br - I - F
rdbe_mult = np.zeros_like(ELEMENT_VECTORS[0])
_rdbe_els = ["C", "N", "P", "H", "Cl", "Br", "I", "F"]
_rdbe_weights = [2, 1, 1, -1, -1, -1, -1, -1]
for _k, _v in zip(_rdbe_els, _rdbe_weights):
    if _k in element_to_ind:
        rdbe_mult[element_to_ind[_k]] = _v


def cross_sum(x, y):
    """Compute cross sum of two arrays for combinatorial enumeration."""
    return (np.expand_dims(x, 0) + np.expand_dims(y, 1)).reshape(-1, y.shape[-1])


def rdbe_filter(cross_prod):
    """Filter formula vectors by ring and double bond equivalent (RDBE >= 0)."""
    rdbe_total = 1 + 0.5 * cross_prod.dot(rdbe_mult)
    return np.argwhere(rdbe_total >= 0).flatten()


def get_all_subsets_dense(dense_formula, element_vectors):
    """Enumerate all valid subformulae from a dense formula vector.

    Args:
        dense_formula: Element count vector (e.g., from formula_to_dense).
        element_vectors: Basis vectors for elements (ELEMENT_VECTORS).

    Returns:
        Tuple of (cross_prod, all_masses): formula vectors and monoisotopic masses.
    """
    non_zero = np.argwhere(dense_formula > 0).flatten()

    vectorized_formula = []
    for nonzero_ind in non_zero:
        temp = element_vectors[nonzero_ind] * np.arange(
            0, dense_formula[nonzero_ind] + 1
        ).reshape(-1, 1)
        vectorized_formula.append(temp)

    zero_vec = np.zeros((1, element_vectors.shape[-1]))
    cross_prod = reduce(cross_sum, vectorized_formula, zero_vec)

    cross_prod_inds = rdbe_filter(cross_prod)
    cross_prod = cross_prod[cross_prod_inds]
    all_masses = cross_prod.dot(VALID_MONO_MASSES)
    return cross_prod, all_masses


def get_all_subsets(chem_formula: str):
    """Enumerate all valid subformulae from a chemical formula string.

    Returns:
        Tuple of (cross_prod, all_masses).
    """
    dense_formula = formula_to_dense(chem_formula)
    return get_all_subsets_dense(dense_formula, element_vectors=ELEMENT_VECTORS)


def clipped_ppm(mass_diff, parentmass):
    """Calculate ppm mass difference, clipping parent mass to minimum of 200 Da."""
    parentmass_copy = parentmass * 1
    if np.isscalar(parentmass_copy):
        if parentmass_copy < 200:
            parentmass_copy = 200
    else:
        parentmass_copy[parentmass_copy < 200] = 200
    return mass_diff / parentmass_copy * 1e6


def clipped_ppm_single(cls_mass_diff: float, parentmass: float) -> float:
    """Calculate clipped ppm for a single value."""
    div_factor = 200 if parentmass < 200 else parentmass
    return cls_mass_diff / div_factor * 1e6
