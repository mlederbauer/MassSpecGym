# MassSpecGym Leaderboard Submission Guide

## What your PR must contain

Fork this repository, create a branch, and open a PR to `main` with **exactly this layout**:

```
submissions/<method_name>/
  model_card.yaml       required
  <your-repo>/          optional but strongly recommended
    train.py
    eval.py
    ...
```

Rules:
- `<method_name>` must exactly match the `method_name` field in your model card (spaces → underscores, e.g. `"My Model"` → `submissions/My_Model/`)
- Only one method per PR
- Do **not** edit `results/*.csv` — regenerated automatically
- Do **not** touch any file outside `submissions/<method_name>/`

Your source code is welcome in the PR for review purposes. It will be stripped automatically before merge — only `model_card.yaml` is retained in the repository. Make sure your `code_url` in the model card points to a public repository for long-term access.

---

## model_card.yaml

Copy [`submissions/template/model_card.yaml`](template/model_card.yaml) and fill in every field. Required fields:

| Field | Description |
|-------|-------------|
| `method_name` | Display name on the leaderboard (must match folder name) |
| `paper_url` | arXiv, DOI, or preprint URL — must be accessible |
| `code_url` | Public GitHub/GitLab URL — must be accessible |
| `random_seed` | Integer seed used for reported results |
| `results` | List of task/challenge entries (see below) |
| `uses_mist_encoder` | `true` if MIST spectrum encoder used |
| `uses_mist_cf` | `true` if MIST-CF formula predictor used |
| `uses_iceberg` | `true` if ICEBERG spectral simulator used |
| `pretraining.used` | `true` if any external pretraining data used |
| `uses_official_split` | Must be `true` |

Each entry in `results` requires:

| Field | Description |
|-------|-------------|
| `task` | `de_novo` \| `retrieval` \| `simulation` |
| `challenge` | `standard` \| `bonus` |
| `uses_formula_at_inference` | `true` if formula predictor filters candidates at inference |
| `paper` | Full paper title |
| `doi` | DOI or arXiv URL |
| `publication_date` | `YYYY-MM-DD` |
| `metrics` | All required metric keys with values and 95% bootstrap CIs (see below) |

**Tier integrity:** `uses_formula_at_inference: true` requires `challenge: bonus`. Submitting a formula-assisted model to `challenge: standard` is a hard failure.

### Required metric keys

**De novo (standard and bonus):**
```yaml
metrics:
  Top-1 Accuracy: 0.00
  Top-1 Accuracy CI Low: 0.00
  Top-1 Accuracy CI High: 0.00
  Top-1 MCES: 0.00
  Top-1 MCES CI Low: 0.00
  Top-1 MCES CI High: 0.00
  Top-1 Tanimoto: 0.00
  Top-1 Tanimoto CI Low: 0.00
  Top-1 Tanimoto CI High: 0.00
  Top-10 Accuracy: 0.00
  Top-10 Accuracy CI Low: 0.00
  Top-10 Accuracy CI High: 0.00
  Top-10 MCES: 0.00
  Top-10 MCES CI Low: 0.00
  Top-10 MCES CI High: 0.00
  Top-10 Tanimoto: 0.00
  Top-10 Tanimoto CI Low: 0.00
  Top-10 Tanimoto CI High: 0.00
```

**Retrieval (standard and bonus):**
```yaml
metrics:
  Hit rate @ 1: 0.00
  Hit rate @ 1 CI Low: 0.00
  Hit rate @ 1 CI High: 0.00
  Hit rate @ 5: 0.00
  Hit rate @ 5 CI Low: 0.00
  Hit rate @ 5 CI High: 0.00
  Hit rate @ 20: 0.00
  Hit rate @ 20 CI Low: 0.00
  Hit rate @ 20 CI High: 0.00
  MCES @ 1: 0.00
  MCES @ 1 CI Low: 0.00
  MCES @ 1 CI High: 0.00
```

**Simulation (standard and bonus):**
```yaml
metrics:
  Cosine Similarity: 0.00
  Cosine Similarity CI Low: 0.00
  Cosine Similarity CI High: 0.00
  Jensen-Shannon Similarity: 0.00
  Jensen-Shannon Similarity CI Low: 0.00
  Jensen-Shannon Similarity CI High: 0.00
  Hit rate @ 1: 0.00
  Hit rate @ 1 CI Low: 0.00
  Hit rate @ 1 CI High: 0.00
  Hit rate @ 5: 0.00
  Hit rate @ 5 CI Low: 0.00
  Hit rate @ 5 CI High: 0.00
  Hit rate @ 20: 0.00
  Hit rate @ 20 CI Low: 0.00
  Hit rate @ 20 CI High: 0.00
```

All CIs are mandatory. Use 95% bootstrap CIs from `ReturnScalarBootStrapper` in `massspecgym/utils.py`.

---

## Submission workflow

```
1. You open a PR from your fork to main
2. Maintainer adds 'submission' label → automated review runs, posts report as PR comment
3. Maintainer reads report, performs human review, approves PR
4. Maintainer adds 'ready-to-merge' label → prepare workflow triggers:
     a. checks out your submission (read-only)
     b. strips everything except model_card.yaml
     c. regenerates results/*.csv
     d. pushes a clean branch to the base repo
     e. closes your fork PR with a link to the clean branch
5. Maintainer merges the clean branch into main
```

You can submit from a fork (any branch) — no special access required.

---

## Automated review checks

On every PR with the `submission` label, a report is posted as a comment. Hard failures block merge. It checks:

- Model card present and all required fields filled
- All required metrics present with non-null values and CIs
- Task/tier consistency (standard vs. bonus, formula use)
- SMILES canonicalization
- Pretraining InChIKey overlap (if pretraining data declared)
- Oracle data safety (MIST-CF / ICEBERG v1.5)
- MIST batching bug (if MIST encoder used)
- Metric override detection (custom `on_batch_end` / `evaluate_*`)
- Official data split used
- LLM narrative review of paper and code

---

## Metric specifications

All metrics must use the pinned MassSpecGym implementations. Do not re-implement them.

| Metric | Specification |
|--------|--------------|
| InChIKey hit rate | First 14 characters (connectivity layer) only |
| Tanimoto similarity | Morgan ECFP4, radius=2, 2048 bits |
| MCES distance | `threshold=15`, `always_stronger_bound=True` (`massspecgym/utils.py:MyopicMCES`) |
| Cosine similarity | Standard MS/MS cosine as implemented in `massspecgym` |
| Jensen-Shannon similarity | As implemented in `massspecgym` |

---

## Data safety requirements

### Pretraining data

If your model uses any external pretraining data, declare it under `pretraining` in the model card. The minimum accepted filtering criterion is `exact_match` (27-char InChIKey deduplication). `tanimoto_0.70` is recommended for results claiming genuine generalization.

### Oracle components

If you use MIST-CF or ICEBERG, use the data-safe v1.5 versions in `massspecgym/models/oracles/`. External versions may introduce transitive data leakage.

---

## Checklist before opening PR

- [ ] `submissions/<method_name>/model_card.yaml` filled in completely
- [ ] Folder name matches `method_name` field exactly
- [ ] All required metrics and 95% CIs present
- [ ] `task` and `challenge` correct for each result entry
- [ ] `paper`, `doi`, `publication_date` filled for each entry
- [ ] Paper URL accessible (arXiv, DOI, or preprint)
- [ ] `code_url` is public and accessible
- [ ] Pretraining data declared if used (with filtering criterion)
- [ ] MIST-CF / ICEBERG: v1.5 data-safe versions confirmed if used
- [ ] Official MassSpecGym data split used (`uses_official_split: true`)
- [ ] Random seed documented
- [ ] No changes to `results/*.csv`
- [ ] No changes outside `submissions/<method_name>/`
