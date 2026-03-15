# MassSpecGym Checkpoints

Place pretrained model checkpoints here for inference.

## Directory Structure

```
checkpoints/
  mist/               # MIST spectrum encoder
    mist_encoder.ckpt
  mist_cf/            # MIST-CF formula prediction oracle
    mist_cf.ckpt
  iceberg/            # ICEBERG spectrum simulation
    gen_model.ckpt
    inten_model.ckpt
  dreams/             # DreaMS spectrum encoder
    embedding_model.ckpt
  frigid/             # FRIGID MDLM decoder
    frigid.ckpt
  diffms/             # DiffMS graph diffusion decoder
    diffms.ckpt
  molforge/           # MolForge seq2seq decoder
    molforge.ckpt
```

## Usage

All models can be loaded from checkpoints using their respective classes:

```python
# MIST encoder
from massspecgym.models.encoders.mist import SpectraEncoderGrowing
encoder = SpectraEncoderGrowing(...)
encoder.load_state_dict(torch.load("checkpoints/mist/mist_encoder.ckpt"))

# DreaMS encoder
from massspecgym.models.encoders.dreams import PreTrainedDreaMS
model = PreTrainedDreaMS.from_checkpoint("checkpoints/dreams/embedding_model.ckpt")

# FRIGID decoder
from massspecgym.models.de_novo import FRIGIDDecoder
model = FRIGIDDecoder.load_from_checkpoint("checkpoints/frigid/frigid.ckpt")
```
