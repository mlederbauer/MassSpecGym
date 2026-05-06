from .mist import SpectraEncoder, SpectraEncoderGrowing


def __getattr__(name):
    if name == "DreaMS":
        from .dreams.model import DreaMS
        return DreaMS
    if name == "PreTrainedDreaMS":
        from .dreams.api import PreTrainedDreaMS
        return PreTrainedDreaMS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
