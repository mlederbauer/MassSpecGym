"""
ICEBERG oracle: MS/MS spectrum simulation from molecular structures.

Thin wrapper around the ICEBERG simulation model at
models/simulation/iceberg/, providing a simple predict_spectrum() API
for oracle usage. No code duplication - all model logic lives in
the simulation package.
"""

from .predict import predict_spectrum
