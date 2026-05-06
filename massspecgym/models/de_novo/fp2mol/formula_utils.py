"""
Formula encoding utilities shared by FP2Mol decoders.

Provides a 30-element FormulaEncoder that maps molecular formula strings
(e.g., "C9H10N2O2") to fixed-size count vectors. Used by FRIGID and DiffMS
for formula conditioning during molecule generation.
"""

import re
from typing import Dict, List, Optional

import torch


class FormulaEncoder:
    """Encoder for molecular formulas to fixed-size numerical vectors.

    Converts molecular formula strings into 30-dimensional vectors of atom counts
    suitable for conditioning molecular generation models.

    The atom vocabulary covers common organic elements plus metals that may appear
    in bioactive molecules, matching the vocabulary used by FRIGID.

    Args:
        normalize: Normalization strategy for atom counts.
            - 'none': No normalization (raw integer counts).
            - 'sum': Divide by total number of atoms.
            - 'max': Divide by maximum count in formula.
            - 'log': Apply log(count + 1) transformation.
    """

    ATOM_VOCAB = [
        'C', 'H', 'N', 'O', 'F', 'S', 'P', 'Cl', 'Br', 'I',
        'B', 'Si', 'Se', 'As', 'Al', 'Sn', 'Li', 'Na', 'K', 'Mg',
        'Ca', 'Fe', 'Zn', 'Cu', 'Mn', 'Co', 'Ni', 'Pt', 'Pd', 'Au'
    ]

    _FORMULA_PATTERN = re.compile(r'([A-Z][a-z]?)(\d*)')

    def __init__(self, normalize: str = 'none'):
        self.atom_vocab = self.ATOM_VOCAB
        self.atom_to_idx = {atom: idx for idx, atom in enumerate(self.atom_vocab)}
        self.normalize = normalize

    def formula_to_counts(self, formula_str: str) -> Dict[str, int]:
        """Parse a molecular formula string into a dictionary of atom counts.

        >>> FormulaEncoder().formula_to_counts("C9H10N2O2")
        {'C': 9, 'H': 10, 'N': 2, 'O': 2}
        """
        if not formula_str:
            return {}
        counts: Dict[str, int] = {}
        for element, count in self._FORMULA_PATTERN.findall(formula_str):
            if element:
                count_val = int(count) if count else 1
                counts[element] = counts.get(element, 0) + count_val
        return counts

    def counts_to_vector(
        self, counts: Dict[str, int], normalize: Optional[str] = None
    ) -> torch.Tensor:
        """Convert atom counts dictionary to a fixed-size vector.

        Args:
            counts: Dictionary mapping atom symbols to counts.
            normalize: Override normalization strategy.

        Returns:
            Tensor of shape (30,) with counts for each atom type.
        """
        normalize = normalize if normalize is not None else self.normalize
        vector = torch.zeros(len(self.atom_vocab), dtype=torch.float32)
        for atom, count in counts.items():
            idx = self.atom_to_idx.get(atom)
            if idx is not None:
                vector[idx] = float(count)

        if normalize == 'sum':
            total = vector.sum()
            if total > 0:
                vector = vector / total
        elif normalize == 'max':
            max_val = vector.max()
            if max_val > 0:
                vector = vector / max_val
        elif normalize == 'log':
            vector = torch.log(vector + 1.0)
        elif normalize != 'none':
            raise ValueError(f"Unknown normalization strategy: {normalize}")

        return vector

    def encode(self, formula_str: str, normalize: Optional[str] = None) -> torch.Tensor:
        """Encode a molecular formula string into a fixed-size vector.

        Args:
            formula_str: Molecular formula string (e.g., "C9H10N2O2").
            normalize: Override normalization strategy.

        Returns:
            Tensor of shape (30,).
        """
        if not formula_str:
            return torch.zeros(len(self.atom_vocab), dtype=torch.float32)
        try:
            counts = self.formula_to_counts(formula_str)
            return self.counts_to_vector(counts, normalize)
        except Exception:
            return torch.zeros(len(self.atom_vocab), dtype=torch.float32)

    def encode_batch(
        self, formulas: List[str], normalize: Optional[str] = None
    ) -> torch.Tensor:
        """Encode a batch of molecular formulas.

        Returns:
            Tensor of shape (batch_size, 30).
        """
        return torch.stack([self.encode(f, normalize) for f in formulas])

    @property
    def vocab_size(self) -> int:
        """Dimensionality of output vectors (30)."""
        return len(self.atom_vocab)

    def decode(self, vector: torch.Tensor, threshold: float = 0.5) -> str:
        """Decode a vector back into an approximate molecular formula string."""
        parts = []
        for idx, count in enumerate(vector.tolist()):
            if count > threshold:
                atom = self.atom_vocab[idx]
                count_int = round(count)
                if count_int > 1:
                    parts.append(f"{atom}{count_int}")
                elif count_int == 1:
                    parts.append(atom)
        return ''.join(parts)
