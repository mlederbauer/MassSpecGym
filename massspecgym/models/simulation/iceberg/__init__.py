"""
ICEBERG spectrum simulation model.

Predicts tandem mass spectra from molecular structures using a two-stage
DAG-based fragmentation approach:
1. FragGNN: Autoregressive fragment generation via bond-breaking DAG.
2. IntenGNN: Intensity prediction for generated fragments.
"""


def __getattr__(name):
    if name == "FragGNN":
        from .gen_model import FragGNN
        return FragGNN
    if name == "IntenGNN":
        from .inten_model import IntenGNN
        return IntenGNN
    if name == "JointModel":
        from .joint_model import JointModel
        return JointModel
    if name == "IcebergSimulationMassSpecGymModel":
        from .adapter import IcebergSimulationMassSpecGymModel
        return IcebergSimulationMassSpecGymModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
