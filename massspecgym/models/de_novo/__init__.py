from .base import DeNovoMassSpecGymModel
from .random import RandomDeNovo
from .dummy import DummyDeNovo
from .smiles_tranformer import SmilesTransformer

__all__ = [
    "DeNovoMassSpecGymModel",
    "RandomDeNovo",
    "DummyDeNovo",
    "SmilesTransformer",
    "FP2MolDeNovoModel",
    "FormulaEncoder",
    "FRIGIDDecoder",
    "MolForgeDecoder",
    "DiffMSDecoder",
]


def __getattr__(name):
    """Lazy imports for FP2Mol decoders to avoid pulling heavy optional dependencies."""
    if name == "FP2MolDeNovoModel":
        from .fp2mol.base import FP2MolDeNovoModel
        return FP2MolDeNovoModel
    if name == "FormulaEncoder":
        from .fp2mol.formula_utils import FormulaEncoder
        return FormulaEncoder
    if name == "FRIGIDDecoder":
        from .fp2mol.frigid import FRIGIDDecoder
        return FRIGIDDecoder
    if name == "MolForgeDecoder":
        from .fp2mol.molforge import MolForgeDecoder
        return MolForgeDecoder
    if name == "DiffMSDecoder":
        from .fp2mol.diffms import DiffMSDecoder
        return DiffMSDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
