# MassSpecGym Leaderboard Submission Guide

To submit a new result to the MassSpecGym leaderboard, open a pull request against `main` that contains exactly two things:

1. A new row in the correct `results/*.csv` file
2. A `submissions/<method_name>/model_card.yaml` describing your method

An automated review runs immediately on your PR and posts a report as a comment. A maintainer performs final human review before merging.

---

## What belongs in this PR

### 1. Results CSV row

Add one row per task/challenge combination to the appropriate file:

| File | Task |
|------|------|
| `results/de_novo.csv` | De novo, mass-based (standard) |
| `results/de_novo_bonus.csv` | De novo, formula-based (bonus) |
| `results/retrieval.csv` | Retrieval, mass-based (standard) |
| `results/retrieval_bonus.csv` | Retrieval, formula-based (bonus) |
| `results/simulation.csv` | Spectrum simulation, mass-based |
| `results/simulation_bonus.csv` | Spectrum simulation, formula-based |

Required columns per task:

**De novo (standard and bonus):**
`Method`, `Top-1 Accuracy`, `Top-1 Accuracy CI Low`, `Top-1 Accuracy CI High`, `Top-1 MCES`, `Top-1 MCES CI Low`, `Top-1 MCES CI High`, `Top-1 Tanimoto`, `Top-1 Tanimoto CI Low`, `Top-1 Tanimoto CI High`, `Top-10 Accuracy`, `Top-10 Accuracy CI Low`, `Top-10 Accuracy CI High`, `Top-10 MCES`, `Top-10 MCES CI Low`, `Top-10 MCES CI High`, `Top-10 Tanimoto`, `Top-10 Tanimoto CI Low`, `Top-10 Tanimoto CI High`, `Paper`, `DOI`, `Comment`, `Publication date`

**Retrieval (standard):**
`Method`, `Hit rate @ 1`, `Hit rate @ 1 CI Low`, `Hit rate @ 1 CI High`, `Hit rate @ 5`, `Hit rate @ 5 CI Low`, `Hit rate @ 5 CI High`, `Hit rate @ 20`, `Hit rate @ 20 CI Low`, `Hit rate @ 20 CI High`, `MCES @ 1`, `MCES @ 1 CI Low`, `MCES @ 1 CI High`, `Paper`, `DOI`, `Comment`, `Publication date`

**Retrieval (bonus):**
`Method`, `Hit rate @ 1`, `Hit rate @ 1 CI Low`, `Hit rate @ 1 CI High`, `Hit rate @ 5`, `Hit rate @ 5 CI Low`, `Hit rate @ 5 CI High`, `Hit rate @ 20`, `Hit rate @ 20 CI Low`, `Hit rate @ 20 CI High`, `MCES @ 1`, `MCES @ 1 CI Low`, `MCES @ 1 CI High`, `Paper`, `DOI`, `Comment`, `Publication date`

**Simulation (standard and bonus):**
`Method`, `Cosine Similarity`, `Cosine Similarity CI Low`, `Cosine Similarity CI High`, `Jensen-Shannon Similarity`, `Jensen-Shannon Similarity CI Low`, `Jensen-Shannon Similarity CI High`, `Hit rate @ 1`, `Hit rate @ 1 CI Low`, `Hit rate @ 1 CI High`, `Hit rate @ 5`, `Hit rate @ 5 CI Low`, `Hit rate @ 5 CI High`, `Hit rate @ 20`, `Hit rate @ 20 CI Low`, `Hit rate @ 20 CI High`, `Paper`, `DOI`, `Comment`, `Publication date`

**All CIs are mandatory.** Use 95% bootstrap CIs computed with the standard MassSpecGym bootstrapper (`ReturnScalarBootStrapper` in `massspecgym/utils.py`). Results without CIs will be rejected.

**Tier integrity:** If your model uses any formula predictor (e.g., MIST-CF) at inference time to pre-filter candidates, it must be submitted to the *bonus* (formula-based) CSV, not the standard one. Mixing tiers is grounds for rejection.

### 2. Model card

Create `submissions/<method_name>/model_card.yaml`. Copy and fill in [`submissions/template/model_card.yaml`](template/model_card.yaml).

`<method_name>` must exactly match the `Method` value in your CSV row (spaces replaced with underscores).

### 3. Code repository (optional but recommended)

You may include your model's source code directly in the PR by placing it under:

```
submissions/<method_name>/<repo_name>/
```

For example:

```
submissions/My_Model/
  model_card.yaml
  my-model-repo/        ← your code here
    train.py
    eval.py
    ...
```

When a local repository directory is present, the automated review will:
- Run all static code checks (MIST bug, metric overrides, formula leakage, split detection) against the local files
- Include the full source in the LLM narrative review instead of only the README

If no local directory is provided, the review falls back to cloning `code_url` from the model card. Including local code is strongly encouraged — it enables a more thorough automated review and reduces the burden on human maintainers.

---

## Metric specifications

All metrics must be computed using the pinned MassSpecGym implementations:

| Metric | Specification |
|--------|--------------|
| InChIKey hit rate | First 14 characters (connectivity layer) only |
| Tanimoto similarity | Morgan ECFP4, radius=2, 2048 bits |
| MCES distance | `threshold=15`, `always_stronger_bound=True` (see `massspecgym/utils.py:MyopicMCES`) |
| Cosine similarity | Standard MS/MS cosine as implemented in `massspecgym` |
| Jensen-Shannon similarity | As implemented in `massspecgym` |

Do not re-implement these metrics. Use the parent ABC evaluation methods. Any override of `on_batch_end()` or custom metric computation outside the MassSpecGym parent classes must be explicitly justified.

---

## What the automated review checks

On every PR, a GitHub Action runs [`scripts/review_submission.py`](../scripts/review_submission.py) and posts a structured report. It checks:

- Model card present and all required fields filled
- Results CSV row present, correct columns, all CIs non-null
- Task/tier consistency (standard vs. bonus)
- SMILES canonicalization: 0 non-RDKit-canonical entries in the candidate set
- Pretraining InChIKey overlap (if pretraining data declared): runs `massspecgym.data.sanity_check`
- Oracle data safety (if MIST-CF or ICEBERG used): checks v1.5 data-safe versions referenced
- MIST batching bug: if MIST encoder used, scans linked code for the `-inf` mask fix
- Metric override: scans for custom `on_batch_end` / `evaluate_*` reimplementations
- Official data split used (no custom `split_pth`)
- LLM narrative review of paper and code (fetches arXiv/PDF if accessible)

**Hard failures block merge.** Warnings require explicit maintainer sign-off.

---

## Data safety requirements

### Pretraining data

If your model uses any external pretraining data (decoder pretraining, encoder pretraining, or fine-tuning on data outside MassSpecGym):

- Declare it in `model_card.yaml` under `pretraining`
- Specify the filtering criterion (`exact_match` or `tanimoto_0.70` recommended)
- Provide a publicly accessible parquet/CSV if possible

The MassSpecGym leaderboard accepts results trained with at least exact-match exclusion. We recommend Tanimoto ≥ 0.70 filtering for results claimed to reflect genuine generalization (see the MassSpecGym v1.5 paper for motivation).

### Oracle components

If you use MIST-CF or ICEBERG as auxiliary components, you must use the data-safe v1.5 versions provided in `massspecgym/models/oracles/`. Using externally trained versions of these models may introduce transitive data leakage.

---

## Checklist before opening PR

- [ ] `submissions/<method_name>/model_card.yaml` filled in completely
- [ ] Results CSV row added to the correct file with all required metrics and CIs
- [ ] Method name in CSV matches `submissions/<method_name>/` folder name (underscores for spaces)
- [ ] Paper URL is accessible (arXiv, DOI, or preprint link)
- [ ] Code repository URL is public and accessible
- [ ] If pretraining data used: declared in model card with filtering criterion
- [ ] If MIST-CF or ICEBERG used: v1.5 data-safe version confirmed
- [ ] Evaluation run with official MassSpecGym data split (no custom `split_pth`)
- [ ] Random seed documented
