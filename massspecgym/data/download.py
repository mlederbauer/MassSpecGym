"""
Download MassSpecGym datasets from HuggingFace to local data/ directory.

Downloads the core dataset files needed for training, evaluation, and
retrieval tasks.
"""

import logging
from pathlib import Path
from typing import Optional

from massspecgym.utils import hugging_face_download

logger = logging.getLogger(__name__)

CORE_FILES = [
    "MassSpecGym.tsv",
]

RETRIEVAL_FILES = [
    "molecules/MassSpecGym_retrieval_candidates_mass.json",
    "molecules/MassSpecGym_retrieval_candidates_formula.json",
]

MOLECULE_FILES = [
    "molecules/MassSpecGym_retrieval_molecules_1M.tsv",
]


def download_massspecgym_data(
    include_retrieval: bool = True,
    include_molecules: bool = False,
    verbose: bool = True,
) -> dict:
    """Download MassSpecGym files from HuggingFace.

    Files are cached by huggingface_hub to ~/.cache/huggingface/hub/ and
    the function returns a dict mapping file names to local paths.

    Args:
        include_retrieval: Download retrieval candidate files.
        include_molecules: Download molecule library files.
        verbose: Print progress.

    Returns:
        Dict mapping HF file names to local file paths.
    """
    files_to_download = list(CORE_FILES)
    if include_retrieval:
        files_to_download.extend(RETRIEVAL_FILES)
    if include_molecules:
        files_to_download.extend(MOLECULE_FILES)

    paths = {}
    for f in files_to_download:
        if verbose:
            logger.info(f"Downloading {f}...")
        try:
            local_path = hugging_face_download(f)
            paths[f] = local_path
            if verbose:
                logger.info(f"  -> {local_path}")
        except Exception as e:
            logger.warning(f"  Failed to download {f}: {e}")

    return paths


def get_massspecgym_tsv_path() -> str:
    """Get the local path to MassSpecGym.tsv, downloading if needed."""
    return hugging_face_download("MassSpecGym.tsv")


def get_retrieval_candidates_path(bonus: bool = False) -> str:
    """Get path to retrieval candidates JSON, downloading if needed.

    Args:
        bonus: If True, use formula-based candidates (bonus task).
    """
    if bonus:
        return hugging_face_download("molecules/MassSpecGym_retrieval_candidates_formula.json")
    return hugging_face_download("molecules/MassSpecGym_retrieval_candidates_mass.json")
