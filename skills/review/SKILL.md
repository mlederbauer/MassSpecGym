---
name: review
description: Validation checklist and procedure for reviewing community-contributed model implementations in MassSpecGym, covering code correctness, data safety, canonicalization artifacts, metric implementation, and fair benchmarking before a leaderboard submission is accepted.
---

# MassSpecGym Model Review

## Goal

To systematically validate a community-contributed model implementation before accepting its performance claims onto the MassSpecGym leaderboard. The review verifies that (i) the implementation correctly inherits and uses the MassSpecGym ABCs, (ii) training data does not leak test or validation molecules — including transitively through shared infrastructure — (iii) the candidate set is free of canonicalization artifacts that enable shortcut learning, (iv) formula information is not used in the mass-based challenge, (v) all shared encoder components are free of inference-time cross-sample leakage, and (vi) the reported metrics use the pinned, standardised implementations. Each failure mode listed here has been documented in the literature and found to materially alter benchmark conclusions; no item can be waived.

## Instructions

### Step 1 — Read the Submission

Locate the contributed model file(s) inside `massspecgym/models/` and the results entry in `results/`. Read the model code in full before proceeding.

Key questions to answer at this stage:

- Which of the three tasks does the model target (de novo, retrieval, simulation)?
- Standard or bonus (formula-conditioned) variant?
- Does it inherit from the correct ABC (`DeNovoMassSpecGymModel`, `RetrievalMassSpecGymModel`, or `SimulationMassSpecGymModel`)?
- Does the model use any external pretraining data, pretrained encoders, or oracle components?

### Step 2 — Verify ABC Inheritance and `step()` Return Contract

Check that `step()` returns the mandatory keys for the relevant task:

| Task | Required keys in `step()` return dict |
|------|---------------------------------------|
| Retrieval | `"loss"`, `"scores"` (one float per candidate, concatenated) |
| De novo | `"loss"`, `"mols_pred"` (list of lists: `[batch_size × top_k]`) |
| Simulation | `"loss"`, `"pred_mzs"`, `"pred_logprobs"`, `"pred_batch_idxs"` |

**Common mistake — wrong `scores` shape for retrieval:** `scores` must be a 1-D tensor of length equal to the total number of candidates across the batch (sum of `batch["batch_ptr"]`), not shape `(batch_size,)` or `(batch_size, n_candidates)`. Returning per-sample scores instead of per-candidate scores silently inflates hit-rate metrics.

**Common mistake — `mols_pred` not a list of lists:** The outer list must have one entry per spectrum in the batch; the inner list must have exactly `top_k` entries (padding with `None` when the model produces fewer valid structures). Missing this structure causes MCES and Tanimoto evaluation to mis-align predictions with ground truth.

Verify that the parent class evaluation methods are **not** called manually inside `step()`:

```python
# Env: massspecgym
# Flag this pattern — it causes double metric computation:
self.evaluate_retrieval_step(...)  # should only be called from on_batch_end()
```

### Step 3 — Check Data Leakage (Direct and Transitive)

#### 3a. Direct Leakage: External Pretraining Data

If the model uses **any external data** (e.g., an unpaired molecule library for fingerprint-to-molecule decoder pretraining, or a pretrained encoder trained on a third-party MS/MS dataset), verify that test and validation InChIKeys are excluded using **connectivity-layer (14-char) matching**. Full 27-character InChIKey matching is insufficient: it treats stereoisomers as distinct molecules and misses stereoisomers of test structures in the pretraining corpus.

Run the built-in sanity check:

```bash
# Env: massspecgym
python -m massspecgym.data.sanity_check \
    --input path/to/pretraining_molecules.parquet \
    --inchikey-col inchikey_14
```

Expected output for a clean dataset:

```
CLEAN: 1000000 molecules checked, no overlap found.
```

If the check reports overlapping InChIKeys, the submission **must not** be accepted until the author removes those molecules and retrains. Deduplication based on the full 27-character InChIKey (instead of the 14-character connectivity layer) under-counts overlaps and is not sufficient.

#### 3b. Transitive Leakage: Shared Oracle Components

A model whose own training data is clean may still inherit test-set information through oracle components trained on overlapping data. The two primary vectors are:

**MIST-CF formula predictor:** Several methods use MIST-CF at inference to predict the molecular formula of a test spectrum and pre-filter candidates. The publicly released MIST-CF checkpoint was trained on data that overlaps with the MassSpecGym test set. Any downstream model conditioned on its predictions inherits this leakage. **Always use the data-safe MIST-CF provided in MassSpecGym v1.5** (`massspecgym/models/oracles/mist_cf/`).

**ICEBERG spectral simulator:** Methods that pretrain on ICEBERG-simulated spectra for structural analogues in the training set should verify that ICEBERG itself was not trained on test-set molecules. Use the data-safe ICEBERG oracle provided in MassSpecGym v1.5 (`massspecgym/models/oracles/iceberg/`).

To verify oracle data safety:

```python
# Env: massspecgym
from massspecgym.models.oracles.base import OracleBase
oracle = oracle_model.load(device="cpu")
assert oracle.is_data_safe(), "Oracle was not trained on a data-safe corpus — use the v1.5 release."
```

### Step 4 — Check for Shortcut Learning and Task Validity Violations

These are the most commonly overlooked failures. Each one can produce leaderboard-topping results with no genuine spectral reasoning.

#### 4a. SMILES Canonicalization Artifact (R1)

The ground-truth SMILES strings in the MassSpecGym spectral library originate from experimental depositions and retain PubChem-style formatting conventions. Decoy candidates are RDKit-canonicalized. This distributional mismatch allows a model to act purely as a syntax checker — the ground-truth molecule is the one SMILES string formatted differently from all others.

**Demonstrated impact:** A rule-based RDKit format check (`smiles != Chem.MolToSmiles(Chem.MolFromSmiles(smiles))`) that never examines the spectrum achieves **>99% Recall@1** on the uncorrected dataset. A spectrum-blind SMILES binary classifier achieves **>90% Recall@1**. Models using SMILES-based encoders (ChemBERTa, ChemFormer) have been found to achieve inflated retrieval scores primarily through this artifact rather than through spectral reasoning.

**Check:** Verify that the submission enforces RDKit canonicalization uniformly across all query and candidate SMILES before encoding. MassSpecGym v1.5 ships with pre-canonicalized candidate sets; verify the submission uses them.

**Sanity test:** Run the spectrum-blind classifier on the candidate set used by the submission. Any result substantially above random (≫ 1/candidate_pool_size) indicates a residual canonicalization artifact.

```python
# Env: massspecgym
# Quick sanity check: RDKit canonical vs. stored SMILES
from rdkit import Chem
non_canonical = sum(
    1 for smi in candidate_smiles
    if smi != Chem.MolToSmiles(Chem.MolFromSmiles(smi))
)
assert non_canonical == 0, f"{non_canonical} non-canonical SMILES in candidate set"
```

#### 4b. Ground-Truth Frequency Bias

Ground-truth molecules in annotated spectral libraries are disproportionately common metabolites, widely available standards, or easily synthesizable compounds. Ranking candidates by PubChem deposition frequency alone — ignoring the spectrum — achieves **>90% Recall@1**. This is an inherent distribution shift in the chemical space of annotated spectra, not a fixable artifact.

**Implication for review:** Results substantially above the PubChem-frequency baseline do not necessarily reflect spectral reasoning; the baseline itself is already inflated by this prior. Flag any submission that does not compare against or acknowledge this bias.

#### 4c. Formula Leakage in the Mass-Based Challenge (R2)

MassSpecGym defines two distinct retrieval tiers:
- **Mass-based (standard):** Candidates are all structures consistent with the observed precursor $m/z$ within tolerance — typically thousands of candidates.
- **Formula-based (bonus):** The ground-truth molecular formula is provided, restricting candidates to structural isomers — an order of magnitude fewer candidates.

**Common mistake:** Using a formula predictor (e.g., MIST-CF) at inference time to pre-filter candidates in the mass-based challenge. This collapses the candidate space by approximately 10× and makes results directly incomparable to other mass-based entries.

**Check:** If the model uses subformula assignment, formula prediction, or any formula-conditioned component at inference, it must be submitted under the **formula-based (bonus) tier**, not the mass-based tier. Verify that results tables do not mix the two tiers.

### Step 5 — Check Shared Encoder Inference Correctness (R3)

If the model uses MIST as a frozen feature extractor, check the attention masking implementation. The original MIST encoder was validated for single-spectrum (batch size 1) inference. In batched settings, incorrect padding handling causes shorter spectra to attend to padding tokens, collapsing their representations toward the longest spectrum in the batch.

**The bug:** `attn += attn_mask` adds a zero-or-negative mask but does not fill padding positions with $-\infty$, leaving them in the softmax distribution.

**Demonstrated impact:** The buggy batched implementation raises average predicted-fingerprint Tanimoto similarity from **0.37 to 0.52** (variance from 0.042 to 0.081) on the MassSpecGym test set. MIST + MolForge inflated from 10.73% to 28.50% Top-1 accuracy — a 17.8 pp artificial gain that reorders the de novo leaderboard.

**Check:**

```python
# Env: massspecgym
# In the MIST encoder forward pass, look for this pattern — it is WRONG:
attn += attn_mask          # BUG: does not mask padding to -inf

# The correct pattern fills padding positions with -inf before softmax:
new_attn_mask = torch.zeros_like(attn_mask)
new_attn_mask = new_attn_mask.masked_fill(attn_mask != 0, float("-inf"))
attn += new_attn_mask      # CORRECT
```

If the submission uses the MIST encoder from `massspecgym/models/encoders/mist/`, verify it is the fixed v1.5 version. If it imports MIST from a third-party codebase, require the author to verify the fix or switch to single-sample inference.

### Step 6 — Verify Metric Implementations (R4)

Metric divergence across codebases is sufficient to reorder leaderboard rankings on identical predictions. The four metrics with known implementation variability are:

| Metric | Correct specification | Common incorrect variant |
|--------|----------------------|--------------------------|
| **InChIKey hit rate** | First 14 characters (connectivity layer only) | Full 27-character match — misses stereoisomers |
| **Tanimoto similarity** | Morgan ECFP4, radius=2, 2048 bits | Radius=3 or 1024 bits — shifts scores systematically |
| **MCES distance** | Standardized solver with pinned timeout | Unconstrained solver — produces incomparable values |
| **Close match rate** | Tanimoto $\geq 0.85$ | $\geq 0.75$ — inflates match rates |
| **Meaningful match rate** | Tanimoto $\geq 0.75$ | $\geq 0.60$ — inflates match rates |

**Check:** Verify that the submission delegates all metric computation to the parent ABCs in MassSpecGym. Any override or re-implementation of metric computation outside the parent class must be flagged and checked against the specifications above. Do not accept metric implementations copied from third-party codebases without independent verification.

```python
# Env: massspecgym
# Correct: InChIKey computed at 14-char connectivity layer
inchikey_14 = Chem.inchi.InchiToInchiKey(Chem.inchi.MolToInchi(mol))[:14]

# Correct: Morgan fingerprint at radius 2, 2048 bits
from rdkit.Chem import AllChem
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
```

### Step 7 — Verify the Evaluation Setup

Confirm that the model is evaluated using the official data split and unmodified metrics:

```python
# Env: massspecgym
# Correct — use the default MassSpecDataModule split
data_module = MassSpecDataModule(dataset=dataset, batch_size=32)

# Flag this — custom split_pth overrides the official benchmark split
data_module = MassSpecDataModule(dataset=dataset, split_pth="my_custom_split.tsv")
```

Check that all required metric values are reported:

- **Retrieval**: `hit_rate@1`, `hit_rate@5`, `hit_rate@20`, `mces@1` — any submission reporting only a subset must be flagged.
- **De novo**: Top-1 and Top-10 accuracy, MCES, and Tanimoto — all three metrics at both $k$ values.
- **Simulation**: Cosine similarity and Jensen–Shannon similarity, plus retrieval hit rates.

Standard and bonus (formula-conditioned) variants must not be compared in the same table row.

### Step 8 — Reproduce the Results

Clone the branch, install the environment, and run the full test suite:

```bash
# Env: massspecgym
git clone <author-fork-url>
cd MassSpecGym
pip install -e ".[dev]"
python scripts/run.py \
    --job_key review_run \
    --run_name <model_name>_review \
    --test_only \
    --no_wandb \
    --seed 42
```

A discrepancy of more than 0.5 percentage points on the primary metric warrants a request for clarification. Larger deviations may indicate the canonicalization artifact, the MIST batching bug, or a non-default metric implementation.

### Step 9 — Final Checklist

Before approving the pull request, confirm all of the following:

- [ ] Model inherits the correct ABC and implements `step()` with the right return contract
- [ ] No manual metric computation inside `step()` or `on_batch_end()`
- [ ] Direct data leakage sanity check passed (14-char InChIKey, or not applicable)
- [ ] Transitive leakage checked: data-safe v1.5 MIST-CF and ICEBERG oracles used
- [ ] Candidate SMILES are RDKit-canonicalized; spectrum-blind classifier scores near-random
- [ ] No formula predictor used to pre-filter candidates in the mass-based challenge
- [ ] MIST encoder (if used) applies correct $-\infty$ padding mask in batched inference
- [ ] All metrics use pinned implementations (14-char InChIKey, ECFP4 r=2 2048-bit, correct match thresholds)
- [ ] Official data split used; standard and bonus variants not conflated
- [ ] Results reproducible within 0.5 pp on primary metric
- [ ] All required metric values reported; no cherry-picking of best $k$
- [ ] `df_test_path` set so per-sample predictions are saved for audit
- [ ] Random seed documented
- [ ] Model registered in `massspecgym/models/<task>/__init__.py`

## Examples

### Checking canonicalization in a candidate set

```python
# Env: massspecgym
from rdkit import Chem

def check_canonicalization(smiles_list):
    non_canonical = [
        smi for smi in smiles_list
        if smi != Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    ]
    print(f"{len(non_canonical)}/{len(smiles_list)} non-canonical SMILES")
    return non_canonical

# Should output: "0/N non-canonical SMILES" for a clean v1.5 candidate set
check_canonicalization(dataset.candidates_smiles)
```

### Running the data leakage sanity check

```bash
# Env: massspecgym
python -m massspecgym.data.sanity_check \
    --input data/my_pretraining_mols.parquet \
    --inchikey-col inchikey_14
```

### Reproducing retrieval results

```python
# Env: massspecgym
from massspecgym.data import RetrievalDataset, MassSpecDataModule
from massspecgym.data.transforms import SpecTokenizer, MolFingerprinter
import pytorch_lightning as pl

pl.seed_everything(42)
dataset = RetrievalDataset(
    spec_transform=SpecTokenizer(n_peaks=60),
    mol_transform=MolFingerprinter(fp_size=4096),
)
data_module = MassSpecDataModule(dataset=dataset, batch_size=32, num_workers=4)

from massspecgym.models.retrieval import DeepSetsRetrieval  # model under review
from pytorch_lightning import Trainer
model = DeepSetsRetrieval.load_from_checkpoint("checkpoints/model_under_review.ckpt")
Trainer(accelerator="gpu", devices=1).test(model, datamodule=data_module)
# Expected output: hit_rate@1, hit_rate@5, hit_rate@20, mces@1
```

## Constraints

- **No Metric Modification**: All metric computation must be delegated to the parent ABC. Any override of `on_batch_end()` that modifies metric logic is grounds for rejection.
- **Data Safety Is Non-Negotiable**: A submission that fails the InChIKey sanity check (14-char matching) cannot be accepted. This applies to both direct leakage and transitive leakage through oracle components.
- **Canonicalization Must Be Verified**: A spectrum-blind SMILES format classifier scoring substantially above random on the submission's candidate set is grounds for rejection until the artifact is removed.
- **Formula Tier Integrity**: Results using formula predictors to pre-filter candidates must be reported under the bonus tier. Mixing tiers in comparison tables is grounds for rejection.
- **MIST Batching Bug**: Any submission using MIST in batched mode without the $-\infty$ padding fix must be corrected before acceptance.
- **Metric Pinning**: All metrics must use the specified implementations. Results computed with deviating implementations (wrong InChIKey layer, wrong fingerprint parameters, non-standard match thresholds) are not comparable to the leaderboard.
- **Reproducibility Threshold**: Results must reproduce within 0.5 percentage points on the primary metric under the same seed.
- **Official Split Only**: The benchmark split defined by `MassSpecDataModule` (default) must be used.
- **Environment**: Review must be conducted in the `massspecgym` conda environment (Python 3.11).

## References

- Bushuiev et al., "MassSpecGym: A benchmark for the discovery and identification of molecules", *NeurIPS 2024 (Spotlight)*. [arXiv:2410.23326](https://doi.org/10.48550/arXiv.2410.23326)
- Goldman et al., "Annotating metabolite mass spectra with domain-inspired chemical formula transformers", *Nature Machine Intelligence*, 2023. [DOI:10.1038/s42256-023-00708-3](https://doi.org/10.1038/s42256-023-00708-3)
- Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in Machine Learning-based Science", *Patterns*, 2023. [DOI:10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804)
