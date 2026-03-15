"""
Convert MassSpecGym data to MIST-compatible format.

Produces the exact directory layout expected by the MIST encoder and all
MIST-based models:

    {output_dir}/
        labels.tsv            # dataset, spec, ionization, formula, smiles, inchikey, instrument
        split.tsv             # name, split
        spec_files/           # {identifier}.ms per spectrum
        subformulae/
            default_subformulae/
                {identifier}.json

The output matches /home/liuhx25/orcd/pool/data/msg/ identically.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import massspecgym.utils as utils

logger = logging.getLogger(__name__)


def _row_to_ms_string(identifier, formula, parent_mass, adduct, inchikey,
                      smiles, instrument, mzs, intensities, inchi=None):
    """Convert a single MassSpecGym row to a SIRIUS-style .ms file string.

    Produces the exact format matching /home/liuhx25/orcd/pool/data/msg/spec_files/*.ms
    """
    lines = []
    lines.append(f">compound {identifier}")
    lines.append(f">formula {formula}")
    lines.append(f">parentmass {parent_mass}")

    ion_str = adduct
    if adduct == "[M+H]+":
        ion_str = "[M+H]+"
    elif adduct == "[M+Na]+":
        ion_str = "[M+Na]+"
    lines.append(f">ionization {ion_str}")

    lines.append(f">InChi {inchi if inchi else 'None'}")
    lines.append(f">InChiKey {inchikey}")
    lines.append(f"#smiles {smiles}")
    lines.append(f"#instrumentation {instrument}")
    lines.append(f"#_FILE {identifier}")

    if inchi:
        lines.append(f"#InChi {inchi}")

    lines.append("")
    lines.append(">ms2peaks")

    if isinstance(mzs, str):
        mz_arr = [float(m) for m in mzs.split(",")]
        int_arr = [float(i) for i in intensities.split(",")]
    else:
        mz_arr = list(mzs)
        int_arr = list(intensities)

    for mz, inten in zip(mz_arr, int_arr):
        lines.append(f"{mz} {inten}")

    return "\n".join(lines)


def _make_labels_tsv(df: pd.DataFrame) -> pd.DataFrame:
    """Create MIST-format labels.tsv from MassSpecGym DataFrame."""
    labels = pd.DataFrame()
    labels["dataset"] = "MassSpecGym"
    labels["spec"] = df["identifier"]
    labels["ionization"] = df["adduct"]
    labels["formula"] = df["formula"]
    labels["smiles"] = df["smiles"]
    labels["inchikey"] = df["inchikey"] if "inchikey" in df.columns else ""
    labels["instrument"] = df["instrument_type"] if "instrument_type" in df.columns else "unknown"
    return labels


def _make_split_tsv(df: pd.DataFrame) -> pd.DataFrame:
    """Create MIST-format split.tsv from MassSpecGym DataFrame."""
    split = pd.DataFrame()
    split["name"] = df["identifier"]
    split["split"] = df["fold"]
    return split


def convert_massspecgym_to_mist(
    tsv_path: Optional[str] = None,
    output_dir: str = "data/mist_format",
    run_subformulae: bool = True,
    mass_diff_thresh: float = 20.0,
    max_peaks: int = 50,
    num_workers: int = 16,
) -> Path:
    """Convert MassSpecGym dataset to MIST-compatible directory layout.

    Args:
        tsv_path: Path to MassSpecGym.tsv. If None, downloads from HuggingFace.
        output_dir: Target directory for the MIST-format output.
        run_subformulae: Whether to also run subformulae assignment.
        mass_diff_thresh: PPM threshold for subformulae assignment.
        max_peaks: Maximum number of peaks per spectrum.
        num_workers: Workers for parallel subformulae assignment.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    spec_dir = output_dir / "spec_files"
    spec_dir.mkdir(parents=True, exist_ok=True)

    if tsv_path is None:
        logger.info("Downloading MassSpecGym.tsv from HuggingFace...")
        tsv_path = utils.hugging_face_download("MassSpecGym.tsv")

    logger.info(f"Loading {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t")

    logger.info(f"Writing {len(df)} .ms files to {spec_dir}...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing .ms files"):
        identifier = row["identifier"]

        inchikey = row.get("inchikey", "")
        if pd.isna(inchikey):
            inchikey = ""

        inchi = None
        if "inchi" in row and not pd.isna(row.get("inchi", None)):
            inchi = row["inchi"]

        instrument = row.get("instrument_type", "unknown")
        if pd.isna(instrument):
            instrument = "unknown"

        parent_mass = row.get("parent_mass", row.get("precursor_mz", 0))

        ms_str = _row_to_ms_string(
            identifier=identifier,
            formula=row["formula"],
            parent_mass=parent_mass,
            adduct=row["adduct"],
            inchikey=inchikey,
            smiles=row["smiles"],
            instrument=instrument,
            mzs=row["mzs"],
            intensities=row["intensities"],
            inchi=inchi,
        )

        with open(spec_dir / f"{identifier}.ms", "w") as f:
            f.write(ms_str)

    labels_df = _make_labels_tsv(df)
    labels_df.to_csv(output_dir / "labels.tsv", sep="\t", index=False)

    split_df = _make_split_tsv(df)
    split_df.to_csv(output_dir / "split.tsv", sep="\t", index=False)

    logger.info(f"Wrote labels.tsv ({len(labels_df)} rows) and split.tsv to {output_dir}")

    if run_subformulae:
        from massspecgym.data.subformulae import assign_subformulae_dataset

        subform_dir = output_dir / "subformulae" / "default_subformulae"
        logger.info(f"Running subformulae assignment to {subform_dir}...")
        assign_subformulae_dataset(
            spec_source=spec_dir,
            labels_df=labels_df,
            output_dir=subform_dir,
            mass_diff_thresh=mass_diff_thresh,
            max_peaks=max_peaks,
            num_workers=num_workers,
        )

    logger.info(f"MIST format conversion complete: {output_dir}")
    return output_dir
