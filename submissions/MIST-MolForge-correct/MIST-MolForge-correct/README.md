# MIST-MolForge

This repository provides a minimal, self-contained implementation for reproducing the results of **"One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra"** ([Neo et al., 2025](https://arxiv.org/abs/2508.04180)), which combines the [MIST](https://github.com/samgoldman97/mist) spectrum-to-fingerprint encoder with the [MolForge](https://github.com/knu-lcbc/MolForge) fingerprint-to-molecule decoder for de novo molecular structure elucidation from tandem mass spectra.

The pipeline is:
1. MIST encoder: mass spectrum → 4096-bit molecular fingerprint probabilities
2. Thresholding: probabilities → discrete on-bits
3. MolForge decoder: fingerprint tokens → SMILES molecules
4. Evaluation on [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) and CANOPUS (NPLIB1)

## Data and Model Artifacts

### MIST Encoder Checkpoints

The pre-trained MIST encoder checkpoints for MassSpecGym and NPLIB1 (CANOPUS) are consistent with those used by [DiffMS](https://arxiv.org/abs/2502.09571) (Bohde et al., 2025) and are hosted on Zenodo:

> **Download:** [https://zenodo.org/records/15122968](https://zenodo.org/records/15122968)

From the archive, extract `mist_msg.pt` and `mist_canopus.pt` into the `checkpoints/` directory.

### Datasets

The MassSpecGym and CANOPUS datasets need to be in a specific format that includes subformulae assignments for the MIST encoder. These pre-processed datasets can be obtained by running the data preparation scripts from the [DiffMS repository](https://github.com/coleygroup/DiffMS):

```bash
# Clone the DiffMS repo
git clone https://github.com/coleygroup/DiffMS.git
cd DiffMS

# Download and prepare CANOPUS (NPLIB1) data
bash data_processing/01_download_canopus_data.sh

# Download and prepare MassSpecGym data
bash data_processing/02_download_msg_data.sh
```

After preparation, the resulting `data/canopus/` and `data/msg/` directories should each contain:
- `labels.tsv`
- Split TSV files
- `spec_files/*.ms`
- Precomputed subformula JSON files

Update the paths in the config files under `configs/` to point to your local data directories.

### MolForge Decoder Checkpoint and SentencePiece Models

The MolForge decoder checkpoint (`decoder_molforge.pth`) and SentencePiece vocabulary models are also hosted on OSF (Open Science Framework). For anonymous release/review, we create the anonimized link below:

> **Download (anonymized):** [MolForge Decoder](https://osf.io/dqcwm/overview?view_only=17065170fb0e4a98a64f6d1ae6a5bd8f)

After downloading, place the files under `checkpoints/` as follows:

```
checkpoints/
  mist_msg.pt
  mist_canopus.pt
  decoder_molforge.pth
  molforge_sp/
    combined_morgan4096_vocab_sp.model
    combined_smiles_vocab_sp_morgan4096.model
```

## Setup

Install the local package:

```bash
pip install -e .
```

Optional MCES metrics dependency:

```bash
pip install -e .[mces]
```

Initialize the upstream MolForge submodule:

```bash
git submodule update --init --recursive
```

## Repository Layout

- `src/mist/`: retained MIST encoder and benchmark data-loading code
- `src/mist_molforge/`: integration layer for benchmark orchestration, chemistry helpers, metrics, and the MolForge adapter
- `MolForge/`: upstream MolForge repository, tracked as a git submodule ([knu-lcbc/MolForge](https://github.com/knu-lcbc/MolForge))
- `configs/`: MassSpecGym and CANOPUS benchmark configs
- `checkpoints/`: MIST checkpoints, MolForge checkpoint, and SentencePiece models

## Benchmark Commands

After `pip install -e .`, the single canonical benchmark entrypoint is:

```bash
# MassSpecGym
mist-molforge-benchmark --config configs/spec2mol_benchmark_msg.yaml

# CANOPUS (NPLIB1)
mist-molforge-benchmark --config configs/spec2mol_benchmark_canopus.yaml
```

Typical GPU usage:

```bash
mist-molforge-benchmark \
  --config configs/spec2mol_benchmark_msg.yaml \
  --device cuda \
  --output-dir results/msg
```

```bash
mist-molforge-benchmark \
  --config configs/spec2mol_benchmark_canopus.yaml \
  --device cuda \
  --output-dir results/canopus
```

Useful overrides:

```bash
mist-molforge-benchmark \
  --config configs/spec2mol_benchmark_msg.yaml \
  --thresholds 0.5 0.172 \
  --max-spectra 128 \
  --batch-size 24 \
  --device cuda
```

## Config Structure

The benchmark configs contain four sections:

- `data.*`: dataset files and directories
- `mist_encoder.*`: MIST encoder architecture and checkpoint
- `molforge.*`: MolForge submodule root, checkpoint, SentencePiece models, and decode settings
- `evaluation.*`: split and optional sample cap

## References

- **MIST + MolForge:** Neo, N. K. N., Jing, L., Preston, N. Y. Z., Serene, K. X. T., & Shen, B. (2025). *One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra.* AI4Mat-NeurIPS-2025 Workshop. [arXiv:2508.04180](https://arxiv.org/abs/2508.04180)
- **DiffMS:** Bohde, M., Manjrekar, M., Wang, R., Ji, S., & Coley, C. W. (2025). *DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra.* [arXiv:2502.09571](https://arxiv.org/abs/2502.09571)
- **MIST:** Goldman, S., Bradshaw, J., Xin, J., & Coley, C. W. (2023). *A Machine Learning Model for Predicting Molecular Structures from Mass Spectra.* Nature Machine Intelligence, 5(11), 1245--1254.
- **MolForge:** Ucak, U. S., Kang, T., Ko, J., & Lee, J. (2023). *MolForge.* [GitHub](https://github.com/knu-lcbc/MolForge)
- **MassSpecGym:** Bushuiev, R. et al. (2024). *MassSpecGym: A Benchmark for the Discovery and Identification of Molecules.* NeurIPS 2024.

## Notes

- The package entrypoint is `mist_molforge.benchmark:main`.
- This repo does not bundle MassSpecGym or CANOPUS data. See [Datasets](#datasets) above for download instructions.
- Checkpoints are local runtime artifacts; whether to commit them, ignore them, or move them to Git LFS is a separate repository policy decision.
