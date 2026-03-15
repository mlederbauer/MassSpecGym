"""
MAGMa-style combinatorial fragmentation engine for ICEBERG.

Implements the FragmentEngine that enumerates molecular fragments by
combinatorial bond-breaking, producing the DAG structure needed by
ICEBERG's FragGNN and IntenGNN models.

Ported from external/ms-pred/src/ms_pred/magma/fragmentation.py.
"""

from .fragmentation import FragmentEngine, extend
