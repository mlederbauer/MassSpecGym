"""
InChIKey-based data safety sanity check for MassSpecGym.

Ensures that unpaired molecule datasets used for decoder pretraining do not
contain any molecules whose 2D InChIKey (first 14 characters) overlaps with
the MassSpecGym test/validation exclusion list.

Usage as standalone CLI:
    python -m massspecgym.data.sanity_check --input molecules.parquet

Usage as library:
    from massspecgym.data.sanity_check import check_inchikey_overlap
    result = check_inchikey_overlap(my_inchikeys)
    assert result.is_clean, f"Found {result.overlap_count} overlapping molecules!"
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_PATH = Path(__file__).parent.parent.parent / "data" / "exclude_inchikeys.csv"


class DataLeakageError(Exception):
    """Raised when training data overlaps with excluded InChIKeys."""
    pass


@dataclass
class SanityCheckResult:
    """Result of an InChIKey overlap sanity check."""
    total_molecules: int
    overlap_count: int
    overlap_inchikeys: Set[str] = field(default_factory=set)

    @property
    def is_clean(self) -> bool:
        return self.overlap_count == 0


def load_exclusion_set(
    exclude_path: Optional[str] = None,
) -> Set[str]:
    """Load the set of excluded 2D InChIKeys from CSV.

    Args:
        exclude_path: Path to CSV file with column 'inchi' containing 14-char InChIKeys.
            If None, uses the default MassSpecGym exclusion list.

    Returns:
        Set of 14-character 2D InChIKeys.
    """
    if exclude_path is None:
        exclude_path = DEFAULT_EXCLUDE_PATH
    exclude_path = Path(exclude_path)

    if not exclude_path.exists():
        logger.warning(f"Exclusion list not found at {exclude_path}")
        return set()

    keys = set()
    with open(exclude_path, "r") as f:
        for i, line in enumerate(f):
            key = line.strip()
            if i == 0 and key.lower() in ("inchi", "inchikey", "inchikey_14"):
                continue
            if key and len(key) >= 14:
                keys.add(key[:14])
            elif key:
                keys.add(key)
    return keys


def check_inchikey_overlap(
    molecule_inchikeys: Iterable[str],
    exclude_path: Optional[str] = None,
) -> SanityCheckResult:
    """Check whether any molecule InChIKeys overlap with the exclusion list.

    Args:
        molecule_inchikeys: Iterable of InChIKey strings (full or 14-char).
        exclude_path: Path to exclusion CSV. Defaults to data/exclude_inchikeys.csv.

    Returns:
        SanityCheckResult with overlap details.
    """
    exclude_set = load_exclusion_set(exclude_path)
    total = 0
    overlap = set()

    for key in molecule_inchikeys:
        total += 1
        if not key:
            continue
        key_14 = key[:14] if len(key) > 14 else key
        if key_14 in exclude_set:
            overlap.add(key_14)

    return SanityCheckResult(
        total_molecules=total,
        overlap_count=len(overlap),
        overlap_inchikeys=overlap,
    )


def check_inchikey_overlap_strict(
    molecule_inchikeys: Iterable[str],
    exclude_path: Optional[str] = None,
) -> SanityCheckResult:
    """Like check_inchikey_overlap, but raises DataLeakageError on overlap."""
    result = check_inchikey_overlap(molecule_inchikeys, exclude_path)
    if not result.is_clean:
        raise DataLeakageError(
            f"Data leakage detected: {result.overlap_count} molecules overlap with "
            f"MassSpecGym exclusion list. First few: {list(result.overlap_inchikeys)[:5]}"
        )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check InChIKey overlap with MassSpecGym exclusion list")
    parser.add_argument("--input", required=True, help="Path to Parquet or CSV file with molecules")
    parser.add_argument("--exclude", default=None, help="Path to exclusion CSV (default: data/exclude_inchikeys.csv)")
    parser.add_argument("--inchikey-col", default="inchikey_14", help="Column name for InChIKey")
    args = parser.parse_args()

    import pandas as pd
    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    col = args.inchikey_col
    if col not in df.columns:
        for alt in ["inchikey", "inchi", "InChIKey", "INCHIKEY"]:
            if alt in df.columns:
                col = alt
                break

    if col not in df.columns:
        print(f"ERROR: No InChIKey column found. Available: {list(df.columns)}")
        sys.exit(1)

    result = check_inchikey_overlap(df[col].dropna().tolist(), args.exclude)
    if result.is_clean:
        print(f"CLEAN: {result.total_molecules} molecules checked, no overlap found.")
    else:
        print(f"WARNING: {result.overlap_count} overlapping InChIKeys found!")
        print(f"First 10: {list(result.overlap_inchikeys)[:10]}")
        sys.exit(1)
