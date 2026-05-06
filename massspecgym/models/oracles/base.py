"""
Base class for all MassSpecGym oracles.

Oracles are pretrained models that are guaranteed data-safe: they never
see test/validation structures during training, making them safe to use
as preprocessing tools (formula annotation, spectrum simulation) without
risk of data leakage.
"""

from abc import ABC, abstractmethod
from typing import Optional


class OracleBase(ABC):
    """Base class for all MassSpecGym oracles.

    Oracles provide pretrained model inference with a simple API.
    All oracles guarantee data safety (no leakage from test/val sets).
    """

    @classmethod
    @abstractmethod
    def load(cls, checkpoint: Optional[str] = None, device: str = "cpu") -> "OracleBase":
        """Load a pretrained oracle.

        Args:
            checkpoint: Path to checkpoint. If None, auto-downloads.
            device: Device to load model on.

        Returns:
            Loaded oracle instance.
        """
        raise NotImplementedError

    def is_data_safe(self) -> bool:
        """Returns True - all oracles are guaranteed data-safe."""
        return True
