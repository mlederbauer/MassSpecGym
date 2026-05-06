"""
DiffMS decoder: Discrete graph diffusion for fingerprint-to-molecule generation.

A discrete denoising diffusion model that generates molecular graphs (atom types
and bond types) conditioned on Morgan fingerprints, from Bohde et al. 2025.
"""

from .model import DiffMSDecoder
