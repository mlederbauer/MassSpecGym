"""
Subformulae assignment for mass spectra.

Assigns chemical subformulae to MS/MS peaks by matching observed m/z values
to theoretical monoisotopic masses of all possible subfragments of the
precursor formula. This is required by the MIST encoder and all MIST-based
models.

Ported from external/mist/src/mist/utils/spectra_utils.py and
external/mist/src/mist/subformulae/assign_subformulae.py to be
self-contained within MassSpecGym.
"""

import json
import logging
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from massspecgym.models.encoders.mist.chem_constants import (
    ION_LST,
    clipped_ppm,
    get_all_subsets,
    ion_to_mass,
    vec_to_formula,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectrum parsing (from external/mist/src/mist/utils/parse_utils.py)
# ---------------------------------------------------------------------------

def parse_spectra_ms(spectra_file: str) -> Tuple[dict, List[Tuple[str, np.ndarray]]]:
    """Parse a SIRIUS-style .ms file into metadata and peak arrays."""
    lines = [i.strip() for i in open(spectra_file, "r").readlines()]

    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for index, (start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                spectras.append((spectra_header, peak_data))
        else:
            entries = {}
            for i in group_lines:
                if " " not in i:
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end
            metadata.update(entries)
        group_num += 1

    metadata["_FILE_PATH"] = spectra_file
    metadata["_FILE"] = Path(spectra_file).stem
    return metadata, spectras


def parse_spectra_mgf(
    mgf_file: str, max_num: Optional[int] = None
) -> List[Tuple[dict, List[Tuple[str, np.ndarray]]]]:
    """Parse an MGF file into list of (metadata, peak_arrays) tuples."""
    key = lambda x: x.strip() == "BEGIN IONS"
    parsed_spectra = []
    with open(mgf_file, "r") as fp:
        for (is_header, group) in groupby(fp, key):
            if is_header:
                continue
            meta = {}
            cur_spectra = []
            for line in group:
                line = line.strip()
                if not line or line == "END IONS" or line == "BEGIN IONS":
                    continue
                elif "=" in line:
                    k, v = [i.strip() for i in line.split("=", 1)]
                    meta[k] = v
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        cur_spectra.append((float(parts[0]), float(parts[1])))
            if cur_spectra:
                cur_spectra = np.vstack(cur_spectra)
                parsed_spectra.append((meta, [("spec", cur_spectra)]))
            if max_num is not None and len(parsed_spectra) > max_num:
                break
    return parsed_spectra


# ---------------------------------------------------------------------------
# Spectrum processing (from external/mist/src/mist/utils/spectra_utils.py)
# ---------------------------------------------------------------------------

def process_spec_file(meta, tuples, precision=4, max_inten=0.001, max_peaks=60):
    """Process raw spectrum tuples: merge, normalize, sqrt, filter.

    Args:
        meta: Spectrum metadata dict (needs 'parentmass' or 'PEPMASS').
        tuples: List of (header, peak_array) tuples.
        precision: Decimal precision for merging.
        max_inten: Minimum intensity threshold.
        max_peaks: Maximum number of peaks to retain.

    Returns:
        np.ndarray of shape (N, 2) with [mz, intensity] or None.
    """
    if "parentmass" in meta:
        parentmass = meta.get("parentmass", None)
    elif "PARENTMASS" in meta:
        parentmass = meta.get("PARENTMASS", None)
    elif "PEPMASS" in meta:
        parentmass = meta.get("PEPMASS", None)
    else:
        parentmass = 1000000

    parentmass = float(parentmass)

    fused_tuples = [x for _, x in tuples if x.size > 0]
    if len(fused_tuples) == 0:
        return None

    mz_to_inten_pair = {}
    new_tuples = []
    for i in fused_tuples:
        for tup in i:
            mz, inten = tup
            mz_ind = np.round(mz, precision)
            cur_pair = mz_to_inten_pair.get(mz_ind)
            if cur_pair is None:
                mz_to_inten_pair[mz_ind] = tup
                new_tuples.append(tup)
            elif inten > cur_pair[1]:
                cur_pair[1] = inten

    merged_spec = np.vstack(new_tuples)
    merged_spec = merged_spec[merged_spec[:, 0] <= (parentmass + 1)]
    if merged_spec.shape[0] == 0:
        return None
    merged_spec[:, 1] = merged_spec[:, 1] / merged_spec[:, 1].max()
    merged_spec[:, 1] = np.sqrt(merged_spec[:, 1])

    merged_spec = _max_inten_spec(
        merged_spec, max_num_inten=max_peaks, inten_thresh=max_inten
    )
    return merged_spec


def _max_inten_spec(spec, max_num_inten=60, inten_thresh=0):
    """Keep top-k peaks by intensity above threshold."""
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]
    new_sort_order = np.argsort(spec_intens)[::-1]
    if max_num_inten is not None:
        new_sort_order = new_sort_order[:max_num_inten]
    spec_masses = spec_masses[new_sort_order]
    spec_intens = spec_intens[new_sort_order]
    spec_mask = spec_intens > inten_thresh
    spec_masses = spec_masses[spec_mask]
    spec_intens = spec_intens[spec_mask]
    return np.vstack([spec_masses, spec_intens]).transpose(1, 0)


# ---------------------------------------------------------------------------
# Core subformulae assignment (from spectra_utils.assign_subforms)
# ---------------------------------------------------------------------------

def assign_subformulae_single(
    formula: str,
    spectrum: np.ndarray,
    ion_type: str,
    mass_diff_thresh: float = 20.0,
) -> dict:
    """Assign subformulae to a single spectrum.

    Matches the reference MIST implementation exactly:
    1. Enumerate all subformulae of the precursor formula.
    2. Compute monoisotopic masses (with adduct correction).
    3. Match each peak to the nearest subformula by ppm.
    4. Deduplicate by formula, merging intensities.

    Args:
        formula: Precursor molecular formula string (e.g., "C16H17NO4").
        spectrum: Array of shape (N, 2) with [mz, intensity].
        ion_type: Adduct string (e.g., "[M+H]+").
        mass_diff_thresh: Maximum ppm tolerance for matching.

    Returns:
        Dict with keys: cand_form, cand_ion, output_tbl (or None).
    """
    output_dict = {"cand_form": formula, "cand_ion": ion_type, "output_tbl": None}

    if spectrum is None or ion_type not in ION_LST:
        return output_dict

    cross_prod, masses = get_all_subsets(formula)
    spec_masses, spec_intens = spectrum[:, 0], spectrum[:, 1]

    ion_masses = ion_to_mass[ion_type]
    masses_with_ion = masses + ion_masses
    ion_types = np.array([ion_type] * len(masses_with_ion))

    mass_diffs = np.abs(spec_masses[:, None] - masses_with_ion[None, :])
    formula_inds = mass_diffs.argmin(-1)
    min_mass_diff = mass_diffs[np.arange(len(mass_diffs)), formula_inds]
    rel_mass_diff = clipped_ppm(min_mass_diff, spec_masses)

    valid_mask = rel_mass_diff < mass_diff_thresh
    spec_masses = spec_masses[valid_mask]
    spec_intens = spec_intens[valid_mask]
    min_mass_diff = min_mass_diff[valid_mask]
    rel_mass_diff = rel_mass_diff[valid_mask]
    formula_inds = formula_inds[valid_mask]

    if spec_masses.size == 0:
        output_dict["output_tbl"] = None
        return output_dict

    formulas = np.array([vec_to_formula(j) for j in cross_prod[formula_inds]])
    formula_masses = masses_with_ion[formula_inds]
    ion_types = ion_types[formula_inds]

    formula_idx_dict = {}
    uniq_mask = []
    for idx, f in enumerate(formulas):
        uniq_mask.append(f not in formula_idx_dict)
        gather_ind = formula_idx_dict.get(f, None)
        if gather_ind is None:
            formula_idx_dict[f] = idx
            continue
        spec_intens[gather_ind] += spec_intens[idx]

    uniq_mask = np.array(uniq_mask, dtype=bool)
    spec_masses = spec_masses[uniq_mask]
    spec_intens = spec_intens[uniq_mask]
    min_mass_diff = min_mass_diff[uniq_mask]
    rel_mass_diff = rel_mass_diff[uniq_mask]
    formula_masses = formula_masses[uniq_mask]
    formulas = formulas[uniq_mask]
    ion_types = ion_types[uniq_mask]

    if spec_intens.size == 0:
        output_tbl = None
    else:
        output_tbl = {
            "mz": list(spec_masses),
            "ms2_inten": list(spec_intens),
            "mono_mass": list(formula_masses),
            "abs_mass_diff": list(min_mass_diff),
            "mass_diff": list(rel_mass_diff),
            "formula": list(formulas),
            "ions": list(ion_types),
        }
    output_dict["output_tbl"] = output_tbl
    return output_dict


def get_output_dict(
    spec_name: str,
    spec: Optional[np.ndarray],
    form: str,
    mass_diff_type: str,
    mass_diff_thresh: float,
    ion_type: str,
) -> dict:
    """Wrapper matching the reference MIST get_output_dict exactly."""
    assert mass_diff_type == "ppm"
    output_dict = {"cand_form": form, "cand_ion": ion_type, "output_tbl": None}
    if spec is not None and ion_type in ION_LST:
        output_dict = assign_subformulae_single(
            form, spec, ion_type, mass_diff_thresh=mass_diff_thresh
        )
    return output_dict


# ---------------------------------------------------------------------------
# Batch assignment for a full dataset
# ---------------------------------------------------------------------------

def _process_single_ms(spec_name: str, spec_files_dir: str,
                       max_inten: float = 0.001, max_peaks: int = 50):
    """Parse and process a single .ms file."""
    spec_file = Path(spec_files_dir) / f"{spec_name}.ms"
    meta, tuples = parse_spectra_ms(str(spec_file))
    spec = process_spec_file(meta, tuples, max_inten=max_inten, max_peaks=max_peaks)
    return spec_name, spec


def assign_subformulae_dataset(
    spec_source,
    labels_df: pd.DataFrame,
    output_dir,
    mass_diff_thresh: float = 20.0,
    inten_thresh: float = 0.001,
    max_peaks: int = 50,
    feature_id: str = "ID",
    num_workers: int = 16,
) -> Path:
    """Assign subformulae for an entire dataset, writing one JSON per spectrum.

    Args:
        spec_source: Path to .ms directory or .mgf file.
        labels_df: DataFrame with columns: spec, formula, ionization.
        output_dir: Directory to write JSON files.
        mass_diff_thresh: PPM threshold for peak matching.
        inten_thresh: Minimum intensity threshold.
        max_peaks: Maximum number of peaks per spectrum.
        feature_id: MGF field name for spectrum ID.
        num_workers: Number of parallel workers.

    Returns:
        Path to the output directory containing JSON files.
    """
    spec_source = Path(spec_source)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    labels_df = labels_df.astype(str)

    if spec_source.suffix == ".mgf":
        parsed = parse_spectra_mgf(str(spec_source))
        input_specs = {}
        for meta, tuples in parsed:
            name = meta.get(feature_id, meta.get("FEATURE_ID", "unknown"))
            spec = process_spec_file(meta, tuples, max_inten=inten_thresh, max_peaks=max_peaks)
            input_specs[name] = spec
    elif spec_source.is_dir():
        input_specs = {}
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Parsing spectra"):
            spec_name = str(row["spec"])
            try:
                _, spec = _process_single_ms(
                    spec_name, str(spec_source),
                    max_inten=inten_thresh, max_peaks=max_peaks
                )
                input_specs[spec_name] = spec
            except Exception as e:
                logger.warning(f"Failed to parse {spec_name}: {e}")
                input_specs[spec_name] = None
    else:
        raise ValueError(f"spec_source must be a directory or .mgf file, got: {spec_source}")

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Assigning subformulae"):
        spec_name = str(row["spec"])
        formula = str(row["formula"])
        ion_type = str(row["ionization"])
        spec = input_specs.get(spec_name)

        output_dict = get_output_dict(
            spec_name=spec_name,
            spec=spec,
            form=formula,
            mass_diff_type="ppm",
            mass_diff_thresh=mass_diff_thresh,
            ion_type=ion_type,
        )

        with open(output_dir / f"{spec_name}.json", "w") as f:
            json.dump(output_dict, f, indent=4)

    logger.info(f"Wrote {len(labels_df)} subformulae JSONs to {output_dir}")
    return output_dir
