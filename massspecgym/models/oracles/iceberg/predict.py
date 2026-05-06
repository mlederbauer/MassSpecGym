"""
ICEBERG spectrum prediction oracle API.

Thin wrapper around the simulation model at
massspecgym/models/simulation/iceberg/, providing a simple API.

Usage:
    from massspecgym.models.oracles.iceberg import predict_spectrum

    result = predict_spectrum(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        adduct="[M+H]+",
        collision_energy=40.0,
    )
"""

from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass
class PredictedSpectrum:
    """Predicted MS/MS spectrum from ICEBERG."""
    mzs: np.ndarray
    intensities: np.ndarray
    smiles: str
    adduct: str
    collision_energy: float


def predict_spectrum(
    smiles: str,
    adduct: str = "[M+H]+",
    collision_energy: float = 40.0,
    checkpoint_dir: Optional[str] = None,
    device: str = "cpu",
    sparse_k: int = 128,
    threshold: float = 0.001,
) -> PredictedSpectrum:
    """Predict an MS/MS spectrum from a molecular structure using ICEBERG.

    This oracle wraps the ICEBERG JointModel (FragGNN + IntenGNN)
    from massspecgym.models.simulation.iceberg.

    Args:
        smiles: SMILES string of the molecule.
        adduct: Adduct type.
        collision_energy: Collision energy in eV.
        checkpoint_dir: Directory containing gen and inten checkpoints.
        device: Device for inference.
        sparse_k: Number of top peaks to retain.
        threshold: Minimum intensity threshold.

    Returns:
        PredictedSpectrum with m/z and intensity arrays.
    """
    from massspecgym.models.simulation.iceberg.joint_model import JointModel
    from massspecgym.models.simulation.iceberg.gen_model import FragGNN
    from massspecgym.models.simulation.iceberg.inten_model import IntenGNN

    if checkpoint_dir:
        import os
        gen_ckpt = os.path.join(checkpoint_dir, "gen_model.ckpt")
        inten_ckpt = os.path.join(checkpoint_dir, "inten_model.ckpt")
        model = JointModel.from_checkpoints(gen_ckpt, inten_ckpt)
    else:
        model = JointModel(
            FragGNN(hidden_size=256),
            IntenGNN(hidden_size=256),
        )

    result = model.predict_mol(
        smi=smiles,
        collision_eng=collision_energy,
        adduct=adduct,
        threshold=threshold,
        device=device,
        max_nodes=sparse_k,
    )

    spec = result.get("spec", [])
    if spec:
        mzs = np.array([s["mz"] for s in spec])
        intens = np.array([s["intensity"] for s in spec])
    else:
        mzs = np.array([])
        intens = np.array([])

    return PredictedSpectrum(
        mzs=mzs,
        intensities=intens,
        smiles=smiles,
        adduct=adduct,
        collision_energy=collision_energy,
    )
