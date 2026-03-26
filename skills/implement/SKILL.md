---
name: implement
description: Step-by-step guide to implementing a new model for one of the three MassSpecGym benchmark challenges (de novo generation, retrieval, or spectrum simulation).
---

# MassSpecGym Model Implementation

## Goal

To implement a custom ML model for one of the three MassSpecGym benchmark challenges: **de novo molecule generation** (MS/MS spectrum --> molecular structure), **molecule retrieval** (MS/MS spectrum --> ranked candidate list), or **spectrum simulation** (molecular structure --> MS/MS spectrum). The implementation must inherit from the appropriate abstract base class (ABC) so that standardised evaluation, data loading, and training infrastructure are applied automatically.

## Instructions

### Step 1: Understand the Benchmark Structure

Read the project overview and understand which of the three tasks you are targeting:

| Task | Input | Output | ABC to inherit |
|------|-------|--------|----------------|
| De novo generation | MS/MS spectrum | Top-$k$ molecular structures | `DeNovoMassSpecGymModel` |
| Molecule retrieval | MS/MS spectrum + candidate set | Ranked candidate scores | `RetrievalMassSpecGymModel` |
| Spectrum simulation | Molecular structure | MS/MS peaks (m/z, intensity) | `SimulationMassSpecGymModel` |

Each task has a **bonus chemical formulae variant** that provides the ground-truth chemical formula as additional input. Decide upfront whether you are targeting the standard or bonus variant, as this determines the dataset class and candidate pool.

For a broad orientation, read the [demo notebook](../../notebooks/demo.ipynb) and existing model implementations in [massspecgym/models/](../../massspecgym/models/).

### Step 2: Set Up the Environment

```bash
# Env: massspecgym
conda create -n massspecgym python==3.11
conda activate massspecgym
pip install -e ".[notebooks,dev]"
```

Verify the installation:

```bash
# Env: massspecgym
python -c "import massspecgym; print(massspecgym.__version__)"
```

### Step 3: Place Your Model File

Add your model under the appropriate subdirectory of `massspecgym/models/`:

```
massspecgym/models/
├── de_novo/         <-- de novo generation models
├── retrieval/       <-- retrieval models
└── simulation/      <-- spectrum simulation models
```

Register your class in the corresponding `__init__.py` so it can be imported as `from massspecgym.models.<task> import MyModel`.

### Step 4: Inherit the Right ABC and Implement `step()`

The only method you are **required** to implement is `step(batch, stage)`. It must return a `dict` containing at minimum a `"loss"` key plus the task-specific prediction key shown below.

**De novo generation**, inherit `DeNovoMassSpecGymModel`:

```python
# Env: massspecgym
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel
from massspecgym.models.base import Stage

class MyDeNovoModel(DeNovoMassSpecGymModel):
    def __init__(self, n_samples: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.mol_pred_kind = "smiles"  # or "rdkit"
        # ... define layers ...

    def step(self, batch: dict, stage: Stage) -> dict:
        spectra = batch["spec"]          # (batch_size, n_peaks, 2) — m/z and intensity
        # ... generate molecules ...
        # mols_pred: list of length batch_size, each entry is a list of n_samples SMILES strings
        mols_pred = [[...] for _ in range(len(spectra))]
        loss = ...
        return dict(loss=loss, mols_pred=mols_pred)
```

**Molecule retrieval**, inherit `RetrievalMassSpecGymModel`:

```python
# Env: massspecgym
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.models.base import Stage
import torch.nn as nn, torch

class MyRetrievalModel(RetrievalMassSpecGymModel):
    def __init__(self, hidden_dim: int = 256, fp_size: int = 4096, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, fp_size), nn.Sigmoid()
        )

    def step(self, batch: dict, stage: Stage) -> dict:
        spec = batch["spec"]                        # (batch_size, n_peaks, 2)
        fp_true = batch["mol"]                      # (batch_size, fp_size)
        candidates = batch["candidates_mol"]        # (total_candidates, fp_size)
        batch_ptr = batch["batch_ptr"]              # (batch_size,) — candidates per sample

        fp_pred = self.encoder(spec).sum(dim=-2)    # pool over peaks
        loss = nn.functional.mse_loss(fp_pred, fp_true)

        fp_pred_rep = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = nn.functional.cosine_similarity(fp_pred_rep, candidates)
        return dict(loss=loss, scores=scores)
```

**Spectrum simulation**, inherit `SimulationMassSpecGymModel`:

```python
# Env: massspecgym
from massspecgym.models.simulation.base import SimulationMassSpecGymModel
from massspecgym.models.base import Stage

class MySimulationModel(SimulationMassSpecGymModel):
    def _setup_model(self):
        # Define architecture here; called in __init__
        self.net = ...

    def forward(self, **kwargs) -> dict:
        # Return sparse peak predictions
        # Must return: pred_mzs, pred_logprobs, pred_batch_idxs
        ...

    def step(self, batch: dict, stage: Stage) -> dict:
        out = self.forward(**batch)
        loss = ...
        return dict(loss=loss, **out)
```

### Step 5: Set Up Data Transforms and Dataset

Choose transforms appropriate for your architecture:

```python
# Env: massspecgym
from massspecgym.data.transforms import (
    SpecTokenizer,       # --> (n_peaks, 2) peak list; for attention-based models
    SpecBinner,          # --> fixed-length binned vector; for FFNs/CNNs
    MolFingerprinter,    # --> Morgan/MACCS/RDKit fingerprint vector
    MolToGraph,          # --> PyG Data object for GNNs
    MolToInChIKey,       # --> InChIKey string (used internally for evaluation)
)
```

Pair with the task-specific dataset:

```python
# Env: massspecgym
from massspecgym.data import MassSpecDataset, RetrievalDataset, SimulationDataset

# De novo
dataset = MassSpecDataset(
    spec_transform=SpecTokenizer(n_peaks=60),
    mol_transform=MolFingerprinter(fp_size=4096),
)

# Retrieval (standard, mass-based candidates)
dataset = RetrievalDataset(
    spec_transform=SpecTokenizer(n_peaks=60),
    mol_transform=MolFingerprinter(fp_size=4096),
    # candidates_pth=None uses default mass-based candidates
    # candidates_pth='bonus' uses formula-based candidates (bonus variant)
)

# Simulation
dataset = SimulationDataset(
    mol_transform=MolToGraph(),
)
```

If the dataset file is not present locally, it is **downloaded automatically** from HuggingFace on first access.

### Step 6: Train and Validate

```python
# Env: massspecgym
from massspecgym.data import MassSpecDataModule
from pytorch_lightning import Trainer

data_module = MassSpecDataModule(dataset=dataset, batch_size=32, num_workers=4)
model = MyRetrievalModel(hidden_dim=256, fp_size=4096)
trainer = Trainer(accelerator="gpu", devices=1, max_epochs=50)

trainer.fit(model, datamodule=data_module)
```

Evaluation metrics are logged automatically at each validation epoch by the parent class. No custom `on_validation_epoch_end` is required.

### Step 7: Test on the Official Split and Save Results

```python
# Env: massspecgym
trainer.test(model, datamodule=data_module)
```

Per-sample predictions are written to a `df_test` CSV when `df_test_path` is set on the model. This file is required for leaderboard submission.

For a command-line training run, use the provided script:

```bash
# Env: massspecgym
python scripts/run.py \
    --job_key my_experiment \
    --run_name my_retrieval_model \
    --model_type deepsets \
    --batch_size 64 \
    --max_epochs 50 \
    --accelerator gpu
```

### Step 8: (Optional) Register Your Model in the Model Zoo

If your model achieves competitive results and you wish to submit it to the leaderboard:

1. Add your results to the appropriate table in `results/`.
2. Open a pull request against [`main`](https://github.com/pluskal-lab/MassSpecGym/tree/main).
3. The review process is described in [skills/review/SKILL.md](../review/SKILL.md).

## Examples

### Minimal retrieval model (DeepSets)

```python
# Env: massspecgym
from massspecgym.data import RetrievalDataset, MassSpecDataModule
from massspecgym.data.transforms import SpecTokenizer, MolFingerprinter
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.models.base import Stage
from pytorch_lightning import Trainer
import torch, torch.nn as nn

class DeepSetsRetrieval(RetrievalMassSpecGymModel):
    def __init__(self, hidden=128, fp_size=4096, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.rho = nn.Sequential(nn.Linear(hidden, fp_size), nn.Sigmoid())

    def step(self, batch, stage):
        fp_pred = self.rho(self.phi(batch["spec"]).sum(-2))
        loss = nn.functional.mse_loss(fp_pred, batch["mol"])
        rep = fp_pred.repeat_interleave(batch["batch_ptr"], dim=0)
        return dict(loss=loss, scores=nn.functional.cosine_similarity(rep, batch["candidates_mol"]))

dataset = RetrievalDataset(
    spec_transform=SpecTokenizer(n_peaks=60),
    mol_transform=MolFingerprinter(fp_size=4096),
)
data_module = MassSpecDataModule(dataset=dataset, batch_size=32, num_workers=4)
model = DeepSetsRetrieval(fp_size=4096)
trainer = Trainer(accelerator="cpu", max_epochs=5)
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)
```

## Constraints

- **ABC Inheritance**: All models must inherit from `DeNovoMassSpecGymModel`, `RetrievalMassSpecGymModel`, or `SimulationMassSpecGymModel`. Direct subclassing of `MassSpecGymModel` or `pl.LightningModule` is not accepted for benchmark submissions.
- **`step()` Return Contract**:
  - Retrieval: dict must contain `"loss"` and `"scores"` (one float per candidate, concatenated across the batch).
  - De novo: dict must contain `"loss"` and `"mols_pred"` (list of lists; outer dim = batch size, inner dim = top-$k$ candidates).
  - Simulation: dict must contain `"loss"`, `"pred_mzs"`, `"pred_logprobs"`, and `"pred_batch_idxs"`.
- **No Manual Metric Computation**: Do not compute or log evaluation metrics inside `step()`. The parent class handles all metric computation in `on_batch_end()`; invoking it manually will cause double-counting.
- **Data Split**: Always use the official `MassSpecDataModule` with the default split. Do not filter or re-order the test set.
- **Data Leakage**: If your model uses pretraining data outside of MassSpecGym (e.g., external molecule libraries for decoder pretraining), you must run the InChIKey sanity check before training. See the [review skill](../review/SKILL.md) for details.
- **Environment**: All code must run in the `massspecgym` conda environment (Python 3.11, PyTorch, PyTorch Lightning).
- **Reproducibility**: Set a fixed random seed before training (e.g., `pl.seed_everything(42)`).

## References

- Bushuiev et al., "MassSpecGym: A benchmark for the discovery and identification of molecules", *NeurIPS 2024 (Spotlight)*. [arXiv:2410.23326](https://doi.org/10.48550/arXiv.2410.23326)
- Zaheer et al., "Deep Sets", *NeurIPS 2017*. [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)
- Goldman et al., "Annotating metabolite mass spectra with domain-inspired chemical formula transformers", *Nature Machine Intelligence*, 2023. [DOI:10.1038/s42256-023-00708-3](https://doi.org/10.1038/s42256-023-00708-3)
