from .datasets import MassSpecDataset, RetrievalDataset
from .data_module import MassSpecDataModule

__all__ = [
    "MassSpecDataset",
    "RetrievalDataset",
    "MassSpecDataModule",
]


def __getattr__(name):
    if name == "FP2MolDataset":
        from .fp2mol_dataset import FP2MolDataset
        return FP2MolDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
