"""
Convert molecule data files (SMILES text, CSV, TSV) to the standard MassSpecGym
Parquet format for FP2Mol decoder pretraining.

Standard Parquet schema:
    smiles       (string, required)
    inchikey_14  (string, auto-computed)
    formula      (string, auto-computed)
    selfies      (string, optional)
    safe         (string, optional)

Usage:
    python scripts/convert_to_parquet.py --input molecules.txt --output molecules.parquet
    python scripts/convert_to_parquet.py --input data.csv --smiles-col SMILES --output out.parquet
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compute_inchikey_14(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        ik = Chem.MolToInchiKey(mol)
        return ik.split("-")[0] if ik else ""
    except Exception:
        return ""


def _compute_formula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        f = CalcMolFormula(mol)
        return f.split("+")[0].split("-")[0]
    except Exception:
        return ""


def _canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def convert_to_parquet(
    input_path: str,
    output_path: str,
    smiles_col: str = "smiles",
    add_selfies: bool = False,
    add_safe: bool = False,
    max_molecules: int = None,
):
    """Convert a molecule file to standard Parquet format."""
    input_path = Path(input_path)
    logger.info(f"Loading {input_path}...")

    if input_path.suffix in (".csv", ".tsv"):
        sep = "\t" if input_path.suffix == ".tsv" else ","
        df = pd.read_csv(input_path, sep=sep)
        if smiles_col not in df.columns:
            candidates = [c for c in df.columns if "smi" in c.lower()]
            if candidates:
                smiles_col = candidates[0]
                logger.info(f"Using column '{smiles_col}' for SMILES")
            else:
                raise ValueError(f"Column '{smiles_col}' not found. Available: {list(df.columns)}")
        smiles_list = df[smiles_col].dropna().tolist()
    else:
        smiles_list = []
        with open(input_path, "r") as f:
            for line in f:
                smi = line.strip().split("\t")[0].split(",")[0].split()[0]
                if smi and smi.lower() != "smiles":
                    smiles_list.append(smi)

    if max_molecules:
        smiles_list = smiles_list[:max_molecules]

    logger.info(f"Processing {len(smiles_list)} SMILES...")
    records = []
    for smi in tqdm(smiles_list, desc="Processing"):
        canonical = _canonicalize(smi)
        if not canonical:
            continue
        record = {
            "smiles": canonical,
            "inchikey_14": _compute_inchikey_14(canonical),
            "formula": _compute_formula(canonical),
        }
        if add_selfies:
            try:
                import selfies as sf
                record["selfies"] = sf.encoder(canonical)
            except Exception:
                record["selfies"] = ""
        if add_safe:
            try:
                from safe import encode as safe_encode
                record["safe"] = safe_encode(canonical)
            except Exception:
                record["safe"] = ""
        records.append(record)

    df_out = pd.DataFrame(records)
    df_out = df_out[df_out["inchikey_14"] != ""]

    df_out.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(df_out)} molecules to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert molecule data to standard Parquet")
    parser.add_argument("--input", required=True, help="Input file (SMILES txt, CSV, TSV)")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name for CSV/TSV")
    parser.add_argument("--add-selfies", action="store_true", help="Add SELFIES column")
    parser.add_argument("--add-safe", action="store_true", help="Add SAFE column")
    parser.add_argument("--max-molecules", type=int, default=None, help="Max molecules to process")
    args = parser.parse_args()

    convert_to_parquet(
        input_path=args.input,
        output_path=args.output,
        smiles_col=args.smiles_col,
        add_selfies=args.add_selfies,
        add_safe=args.add_safe,
        max_molecules=args.max_molecules,
    )
