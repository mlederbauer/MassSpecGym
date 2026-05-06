"""Small chemistry helpers used by the benchmark integration layer."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def normalize_formula(formula: Optional[str]) -> Optional[str]:
    """Normalize a molecular formula into Hill notation."""
    if not formula:
        return None

    matches = ELEMENT_PATTERN.findall(str(formula))
    if not matches:
        return formula

    counts: Dict[str, int] = {}
    for element, count_str in matches:
        if element:
            count = int(count_str) if count_str else 1
            counts[element] = counts.get(element, 0) + count

    ordered_elements: List[str] = []
    if "C" in counts:
        ordered_elements.append("C")
        if "H" in counts:
            ordered_elements.append("H")

    ordered_elements.extend(
        sorted(element for element in counts if element not in ordered_elements)
    )

    return "".join(
        f"{element}{counts[element] if counts[element] != 1 else ''}"
        for element in ordered_elements
    )


def compute_morgan_fingerprint(
    smiles: str,
    n_bits: int = 4096,
    radius: int = 2,
) -> Optional[np.ndarray]:
    """Convert a SMILES string into a dense Morgan bit vector."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        array = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array
    except Exception:
        return None


def compute_tanimoto_similarity(
    fp_a: Optional[np.ndarray],
    fp_b: Optional[np.ndarray],
) -> float:
    """Compute Tanimoto similarity between two dense bit vectors."""
    if fp_a is None or fp_b is None:
        return 0.0
    intersection = np.sum(np.minimum(fp_a, fp_b))
    union = np.sum(np.maximum(fp_a, fp_b))
    return float(intersection / union) if union > 0 else 0.0
