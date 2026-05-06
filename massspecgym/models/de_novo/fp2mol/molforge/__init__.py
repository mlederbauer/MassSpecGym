"""
MolForge decoder: Seq2Seq Transformer for fingerprint-to-molecule generation.

An encoder-decoder transformer that maps fingerprint bit indices (as a sequence)
to SMILES or SELFIES molecular strings via autoregressive generation.
"""

from .model import MolForgeDecoder
