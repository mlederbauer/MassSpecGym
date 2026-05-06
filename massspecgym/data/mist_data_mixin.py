"""
Mixin for MIST-based models that need subformulae-annotated data.

Provides auto-detection of MIST-format data and lazy conversion from
the standard MassSpecGym TSV format. Used by the MIST encoder, all
FP2Mol decoders, and the MIST-CF oracle.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MIST_DATA_DIR = Path("data/mist_format")


class MISTDataMixin:
    """Mixin providing auto-conversion of MassSpecGym data to MIST format.

    Any model that requires subformulae-annotated spectra can inherit from
    this mixin and call ``ensure_mist_data()`` before using the data.
    """

    def ensure_mist_data(
        self,
        data_dir: Optional[str] = None,
        tsv_path: Optional[str] = None,
        run_subformulae: bool = True,
        num_workers: int = 16,
    ) -> Path:
        """Ensure MIST-format data exists, converting from TSV if needed.

        Checks for:
        1. spec_files/ directory with .ms files
        2. labels.tsv
        3. subformulae/default_subformulae/ with JSON files

        If any are missing, runs the full conversion pipeline.

        Args:
            data_dir: Path to MIST-format data directory.
                Defaults to data/mist_format/.
            tsv_path: Path to MassSpecGym TSV (auto-downloads if None).
            run_subformulae: Whether to generate subformulae JSONs.
            num_workers: Workers for parallel processing.

        Returns:
            Path to the MIST-format data directory.
        """
        if data_dir is None:
            data_dir = DEFAULT_MIST_DATA_DIR
        data_dir = Path(data_dir)

        spec_dir = data_dir / "spec_files"
        labels_file = data_dir / "labels.tsv"
        subform_dir = data_dir / "subformulae" / "default_subformulae"

        needs_conversion = (
            not spec_dir.exists()
            or not labels_file.exists()
            or not any(spec_dir.glob("*.ms"))
        )
        needs_subformulae = run_subformulae and (
            not subform_dir.exists()
            or not any(subform_dir.glob("*.json"))
        )

        if needs_conversion:
            logger.info(
                f"MIST-format data not found at {data_dir}. "
                f"Converting from MassSpecGym TSV..."
            )
            from massspecgym.data.mist_format import convert_massspecgym_to_mist
            convert_massspecgym_to_mist(
                tsv_path=tsv_path,
                output_dir=str(data_dir),
                run_subformulae=run_subformulae,
                num_workers=num_workers,
            )
        elif needs_subformulae:
            logger.info(
                f"Subformulae not found at {subform_dir}. Running assignment..."
            )
            import pandas as pd
            from massspecgym.data.subformulae import assign_subformulae_dataset

            labels_df = pd.read_csv(labels_file, sep="\t")
            assign_subformulae_dataset(
                spec_source=spec_dir,
                labels_df=labels_df,
                output_dir=subform_dir,
                num_workers=num_workers,
            )
        else:
            logger.info(f"MIST-format data found at {data_dir}")

        return data_dir

    def get_subformulae_dir(self, data_dir: Optional[str] = None) -> Path:
        """Get path to subformulae directory, ensuring it exists."""
        data_dir = self.ensure_mist_data(data_dir)
        return data_dir / "subformulae" / "default_subformulae"
