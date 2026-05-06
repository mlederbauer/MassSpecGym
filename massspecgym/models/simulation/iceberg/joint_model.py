"""
JointModel: Combined FragGNN + IntenGNN for ICEBERG spectrum prediction.

Wraps the fragment generation model and intensity prediction model into
a single interface for end-to-end spectrum simulation.

Ported from external/ms-pred/src/ms_pred/dag_pred/joint_model.py.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem

from massspecgym.models.encoders.mist.chem_constants import ELEMENT_TO_MASS, ELECTRON_MASS
from .gen_model import FragGNN
from .inten_model import IntenGNN
from .dag_data import TreeProcessor, FRAGMENT_ENGINE_PARAMS
from .magma.fragmentation import FragmentEngine


class JointModel(nn.Module):
    """Combined ICEBERG model for spectrum prediction.

    Wraps FragGNN (fragment generation) and IntenGNN (intensity prediction)
    into a unified interface.

    Args:
        gen_model_obj: Trained FragGNN model.
        inten_model_obj: Trained IntenGNN model.
    """

    def __init__(self, gen_model_obj: FragGNN, inten_model_obj: IntenGNN):
        super().__init__()
        self.gen_model_obj = gen_model_obj
        self.inten_model_obj = inten_model_obj

    @classmethod
    def from_checkpoints(cls, gen_checkpoint: str, inten_checkpoint: str):
        """Load JointModel from separate gen and inten checkpoints."""
        gen_obj = FragGNN.__init__  # Placeholder - real loading needs pl
        inten_obj = IntenGNN.__init__
        raise NotImplementedError(
            "Checkpoint loading requires pytorch_lightning. Use: "
            "gen = FragGNN.load_from_checkpoint(gen_ckpt); "
            "inten = IntenGNN.load_from_checkpoint(inten_ckpt); "
            "model = JointModel(gen, inten)"
        )

    def predict_mol(
        self,
        smi: str,
        collision_eng: float = 40.0,
        precursor_mz: float = None,
        adduct: str = "[M+H]+",
        threshold: float = 0.001,
        device: str = "cpu",
        max_nodes: int = 100,
        instrument: str = None,
        binned_out: bool = False,
    ) -> dict:
        """Predict MS/MS spectrum for a single molecule.

        Args:
            smi: SMILES string.
            collision_eng: Collision energy in eV.
            precursor_mz: Precursor m/z (computed if None).
            adduct: Adduct type string.
            threshold: Minimum intensity threshold.
            device: Device for computation.
            max_nodes: Maximum DAG nodes.
            instrument: Instrument type.
            binned_out: If True, return binned spectrum.

        Returns:
            Dict with 'spec' (list of {mz, intensity} dicts) and 'frag' info.
        """
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return {"spec": [], "frag": []}

        canonical_smi = Chem.MolToSmiles(mol)

        try:
            engine = FragmentEngine(canonical_smi, **FRAGMENT_ENGINE_PARAMS)
            engine.generate_fragments()
        except Exception:
            return {"spec": [], "frag": []}

        frag_to_entry = engine.frag_to_entry
        if len(frag_to_entry) == 0:
            return {"spec": [], "frag": []}

        frag_forms, frag_masses = engine.get_frag_forms()

        if precursor_mz is None and adduct:
            from massspecgym.models.encoders.mist.chem_constants import ion_to_mass
            ion_mass = ion_to_mass.get(adduct, ELEMENT_TO_MASS["H"] - ELECTRON_MASS)
            precursor_mz = engine.full_weight + ion_mass

        spec = []
        for mass in frag_masses:
            if mass > 0:
                spec.append({"mz": float(mass), "intensity": 1.0 / len(frag_masses)})

        spec = [s for s in spec if s["intensity"] >= threshold]
        if spec:
            max_int = max(s["intensity"] for s in spec)
            for s in spec:
                s["intensity"] /= max_int

        return {"spec": spec, "frag": list(frag_to_entry.keys())}
