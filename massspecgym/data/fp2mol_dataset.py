"""
Dataset for FP-to-molecule decoder pretraining.

Loads molecule libraries from Parquet files (standard format) or programmatic
SMILES lists, produces (fingerprint, formula, molecule_representation) training
triples. Performs mandatory InChIKey sanity check against the MassSpecGym
exclusion list to prevent data leakage.

Standard Parquet schema:
    smiles       (string, required)
    inchikey_14  (string, required)
    formula      (string, optional - auto-computed)
    selfies      (string, optional)
    safe         (string, optional)

Use scripts/convert_to_parquet.py to convert raw SMILES/CSV/TSV to Parquet.
"""

import logging
from pathlib import Path
from typing import List, Optional, Set, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from torch.utils.data import Dataset

from massspecgym.data.sanity_check import (
    DataLeakageError,
    check_inchikey_overlap_strict,
)

logger = logging.getLogger(__name__)


class FP2MolDataset(Dataset):
    """Dataset for fingerprint-to-molecule decoder pretraining.

    Accepts Parquet files as the standard input format, or programmatic
    SMILES lists. Performs mandatory InChIKey sanity check at init.

    Args:
        smiles_source: Path to a .parquet file, or a list of SMILES strings.
        mol_repr: Target molecular representation ('smiles', 'selfies', 'safe').
        fingerprint_type: Fingerprint type ('morgan').
        fp_bits: Number of fingerprint bits.
        fp_radius: Morgan fingerprint radius.
        exclude_inchikeys: Path to exclusion list CSV. Defaults to
            data/exclude_inchikeys.csv. Set to False to disable (NOT recommended).
        max_molecules: Maximum number of molecules to load.
        cache_fingerprints: If True, pre-compute and cache all fingerprints.
    """

    def __init__(
        self,
        smiles_source: Union[str, Path, List[str]],
        mol_repr: str = "smiles",
        fingerprint_type: str = "morgan",
        fp_bits: int = 4096,
        fp_radius: int = 2,
        exclude_inchikeys: Union[str, Path, bool, None] = None,
        max_molecules: Optional[int] = None,
        cache_fingerprints: bool = True,
    ):
        super().__init__()
        self.mol_repr = mol_repr
        self.fingerprint_type = fingerprint_type
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.cache_fingerprints = cache_fingerprints

        from massspecgym.models.de_novo.fp2mol.formula_utils import FormulaEncoder
        self.formula_encoder = FormulaEncoder(normalize="none")

        df = self._load_source(smiles_source, max_molecules)
        self._ensure_columns(df)

        if exclude_inchikeys is not False:
            exclude_path = str(exclude_inchikeys) if exclude_inchikeys else None
            check_inchikey_overlap_strict(df["inchikey_14"].tolist(), exclude_path)

        self.smiles = df["smiles"].tolist()
        self.formulas = df["formula"].tolist()
        self._mol_repr_col = None
        if mol_repr in df.columns and mol_repr != "smiles":
            self._mol_repr_col = df[mol_repr].tolist()

        logger.info(f"FP2MolDataset: {len(self.smiles)} molecules loaded")

        self._fp_cache: dict = {}
        if self.cache_fingerprints:
            self._precompute_fingerprints()

    @staticmethod
    def _load_source(source: Union[str, Path, List[str]], max_molecules) -> pd.DataFrame:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                raise ValueError(
                    f"FP2MolDataset only accepts .parquet files. Got: {path.suffix}\n"
                    f"Use scripts/convert_to_parquet.py to convert your data first."
                )
        elif isinstance(source, list):
            df = pd.DataFrame({"smiles": source})
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        if max_molecules and len(df) > max_molecules:
            df = df.head(max_molecules)
        return df.reset_index(drop=True)

    @staticmethod
    def _ensure_columns(df: pd.DataFrame):
        """Ensure required columns exist, computing missing ones."""
        if "smiles" not in df.columns:
            raise ValueError("Parquet must have a 'smiles' column")

        if "inchikey_14" not in df.columns:
            logger.info("Computing inchikey_14 from SMILES...")
            df["inchikey_14"] = df["smiles"].apply(FP2MolDataset._smiles_to_inchikey14)

        if "formula" not in df.columns:
            logger.info("Computing formula from SMILES...")
            df["formula"] = df["smiles"].apply(FP2MolDataset._smiles_to_formula)

        valid = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        n_invalid = (~valid).sum()
        if n_invalid > 0:
            logger.warning(f"Dropping {n_invalid} invalid SMILES")
            df.drop(df[~valid].index, inplace=True)
            df.reset_index(drop=True, inplace=True)

    @staticmethod
    def _smiles_to_inchikey14(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        try:
            ik = Chem.MolToInchiKey(mol)
            return ik.split("-")[0] if ik else ""
        except Exception:
            return ""

    @staticmethod
    def _smiles_to_formula(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        try:
            f = CalcMolFormula(mol)
            return f.split("+")[0].split("-")[0]
        except Exception:
            return ""

    def _precompute_fingerprints(self):
        for idx in range(len(self.smiles)):
            self._fp_cache[idx] = self._compute_fingerprint(self.smiles[idx])

    def _compute_fingerprint(self, smiles: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(self.fp_bits, dtype=torch.float32)
        if self.fingerprint_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
            arr = np.zeros(self.fp_bits, dtype=np.float32)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            return torch.from_numpy(arr)
        raise ValueError(f"Unknown fingerprint type: {self.fingerprint_type}")

    def _convert_mol_repr(self, smiles: str, idx: int) -> str:
        if self._mol_repr_col is not None:
            return self._mol_repr_col[idx]
        if self.mol_repr == "smiles":
            return smiles
        elif self.mol_repr == "selfies":
            try:
                import selfies as sf
                return sf.encoder(smiles)
            except Exception:
                return smiles
        elif self.mol_repr == "safe":
            try:
                from safe import encode as safe_encode
                return safe_encode(smiles)
            except Exception:
                return smiles
        return smiles

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        smiles = self.smiles[idx]
        formula = self.formulas[idx]
        fingerprint = self._fp_cache[idx] if self.cache_fingerprints else self._compute_fingerprint(smiles)
        formula_vec = self.formula_encoder.encode(formula)
        target_repr = self._convert_mol_repr(smiles, idx)

        return {
            "fingerprint": fingerprint,
            "formula": formula,
            "formula_vec": formula_vec,
            "mol": smiles,
            "mol_repr": target_repr,
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        return {
            "fingerprint": torch.stack([b["fingerprint"] for b in batch]),
            "formula_vec": torch.stack([b["formula_vec"] for b in batch]),
            "formula": [b["formula"] for b in batch],
            "mol": [b["mol"] for b in batch],
            "mol_repr": [b["mol_repr"] for b in batch],
        }
