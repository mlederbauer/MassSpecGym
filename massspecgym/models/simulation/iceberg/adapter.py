"""
ICEBERG adapter for MassSpecGym SimulationMassSpecGymModel interface.

Wraps the ICEBERG JointModel (FragGNN + IntenGNN) to follow the same
interface as GNNSimulationMassSpecGymModel and FPSimulationMassSpecGymModel.
"""

import typing as T
from pathlib import Path

import torch

from massspecgym.models.simulation.base import SimulationMassSpecGymModel


class IcebergSimulationMassSpecGymModel(SimulationMassSpecGymModel):
    """ICEBERG spectrum simulation model for MassSpecGym.

    Predicts MS/MS spectra from molecular structures using a two-stage
    DAG-based approach via FragGNN + IntenGNN, adapted to the
    SimulationMassSpecGymModel interface.

    Args:
        gen_checkpoint: Path to pretrained FragGNN checkpoint.
        inten_checkpoint: Path to pretrained IntenGNN checkpoint.
        sparse_k: Number of top peaks to retain.
        max_nodes: Maximum number of DAG nodes.
        threshold: Minimum intensity threshold.
    """

    def __init__(
        self,
        gen_checkpoint: T.Optional[str] = None,
        inten_checkpoint: T.Optional[str] = None,
        sparse_k: int = 128,
        max_nodes: int = 100,
        threshold: float = 0.001,
        **kwargs,
    ):
        self._gen_checkpoint = gen_checkpoint
        self._inten_checkpoint = inten_checkpoint
        self._sparse_k = sparse_k
        self._max_nodes = max_nodes
        self._threshold = threshold
        super().__init__(**kwargs)

    def _setup_model(self):
        """Set up the ICEBERG JointModel from checkpoints."""
        from .gen_model import FragGNN
        from .inten_model import IntenGNN
        from .joint_model import JointModel

        if self._gen_checkpoint and self._inten_checkpoint:
            gen_obj = FragGNN.load_from_checkpoint(self._gen_checkpoint, map_location="cpu")
            inten_obj = IntenGNN.load_from_checkpoint(self._inten_checkpoint, map_location="cpu")
            self.model = JointModel(gen_obj, inten_obj)
        else:
            self.model = JointModel(
                FragGNN(hidden_size=256),
                IntenGNN(hidden_size=256),
            )
