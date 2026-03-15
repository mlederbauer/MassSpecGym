"""
FRIGID decoder: Fragment Refinement via ICEBERG-Guided Inference Diffusion.

A Masked Diffusion Language Model (MDLM) that generates SAFE molecular sequences
conditioned on fingerprints and chemical formulas via cross-attention.

Requires: ``pip install transformers`` (for BERT backbone).
"""


def __getattr__(name):
    if name == "FRIGIDDecoder":
        from .model import FRIGIDDecoder
        return FRIGIDDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
