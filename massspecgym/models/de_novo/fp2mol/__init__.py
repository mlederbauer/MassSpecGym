"""
FP2Mol decoder family: fingerprint-to-molecule de novo generation models.

This package implements a family of de novo molecular generation models that
share a common two-stage pipeline:

    Spectrum --[MIST Encoder]--> Fingerprint + Formula --[Decoder]--> Molecule

Three decoder architectures are provided:
- FRIGID: Masked Diffusion Language Model (MDLM) over SAFE sequences.
- DiffMS: Discrete graph diffusion over molecular graphs.
- MolForge: Autoregressive seq2seq transformer over SMILES/SELFIES.

All decoders extend ``FP2MolDeNovoModel``, which provides:
- Optional MIST encoder for end-to-end spectrum-to-molecule inference.
- Two training modes: ``fp2mol_pretrain`` (decoder only) and ``spec2mol`` (end-to-end).
- Shared formula encoding via ``FormulaEncoder``.
"""

from .base import FP2MolDeNovoModel
from .formula_utils import FormulaEncoder


def __getattr__(name):
    if name == "FRIGIDDecoder":
        from .frigid import FRIGIDDecoder
        return FRIGIDDecoder
    if name == "MolForgeDecoder":
        from .molforge import MolForgeDecoder
        return MolForgeDecoder
    if name == "DiffMSDecoder":
        from .diffms import DiffMSDecoder
        return DiffMSDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
