"""
DreaMS (Deep Representations Enabling All Mass Spectra) encoder.

A BERT-style transformer encoder for MS/MS spectra that produces 1024-D
spectrum embeddings. Useful for spectral similarity search, retrieval,
and as a general-purpose spectrum representation.

Ported from external/DreaMS/dreams/ (Bushuiev et al., Nature Biotechnology 2025).
Preserves the exact args-based config system for checkpoint compatibility.
"""


def __getattr__(name):
    if name == "DreaMS":
        from .model import DreaMS
        return DreaMS
    if name == "PreTrainedDreaMS":
        from .api import PreTrainedDreaMS
        return PreTrainedDreaMS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
