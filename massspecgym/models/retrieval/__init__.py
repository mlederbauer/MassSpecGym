from .base import RetrievalMassSpecGymModel
from .random import RandomRetrieval
from .deepsets import DeepSetsRetrieval
from .fingerprint_ffn import FingerprintFFNRetrieval
from .from_dict import FromDictRetrieval

__all__ = [
    "RetrievalMassSpecGymModel",
    "RandomRetrieval",
    "DeepSetsRetrieval",
    "FingerprintFFNRetrieval",
    "FromDictRetrieval",
    "MISTFingerprintRetrieval",
    "GenerativeRetrieval",
    "IcebergRetrieval",
]


def __getattr__(name):
    if name == "MISTFingerprintRetrieval":
        from .mist_retrieval import MISTFingerprintRetrieval
        return MISTFingerprintRetrieval
    if name == "GenerativeRetrieval":
        from .generative_retrieval import GenerativeRetrieval
        return GenerativeRetrieval
    if name == "IcebergRetrieval":
        from .iceberg_retrieval import IcebergRetrieval
        return IcebergRetrieval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
