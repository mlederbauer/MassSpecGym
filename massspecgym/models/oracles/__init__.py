"""
Official data-safe oracles for MassSpecGym.

Oracles are pretrained tools that MassSpecGym users can freely call without
data leakage concerns. They provide:

- **MIST-CF**: Chemical formula prediction from MS/MS spectra (default formula
  annotator for de novo tasks without bonus).
- **ICEBERG**: MS/MS spectrum simulation from molecular structures (thin wrapper
  around the simulation model in models/simulation/iceberg/).

All oracles follow the OracleBase interface with guaranteed data safety.
"""

from .base import OracleBase
