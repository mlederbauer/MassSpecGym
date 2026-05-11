---
title: MVP
emoji: 🏆
colorFrom: blue
colorTo: pink
sdk: streamlit
app_file: app.py
pinned: false
short_description: msms annotation tool
python_version: 3.11.7
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# MultiView Projection (MVP) for Spectra Annotation

###  Yan Zhou Chen, Soha Hassoun
#### Department of Computer Science, Tufts University
This repository provides the implementation of MultiView Projection (MVP). MVP can be used to rank a set of molecular candidates given a spectrum.

## Table of Contents
1. [Install & setup]
2. [Data prep]
3. [MassSpecGym data download]
4. [Use our pretrained model]
5. [Training from scratch]
6. [References]

## Install & setup
1. Clone the repository: git clone <REPO_link>
2. Install evironment or only key packages:
```
conda env create -f environment.yml
``` 
#### Key packages
- python
- dgl
- pytorch
- rdkit
- pytorch-geometric
- numpy
- scikit-learn
- scipy
- massspecgym
- lightning

## Data prep
We provide sample spectra data and candidates in `data/sample`. 
For preprocessing:
1. If using formSpec, compute subformula labels
2. Run our preprocess code to obatain fingerprints and consensus spectra files

```
# If using formSpec
python subformula_assign/assign_subformulae.py --spec-files ../data/sample/data.tsv --output-dir ../data/sample/subformulae_default --max-formulae 60 --labels-file ../data/sample/data.tsv
python data_preprocess.py --spec_type formSpec --dataset_pth ../data/sample/data.tsv --candidates_pth  ../data/sample/candidates_mass.json --subformula_dir_pth ../data/sample/subformulae_default/ --output_dir ../data/sample/

# If using binnedSpec
python data_preprocess.py --spec_type binnedSpec --dataset_pth ../data/sample/data.tsv --candidates_pth  ../data/sample/candidates_mass.json --output_dir ../data/sample/

```
We include sample subformula, fingerprint, and consensus spectra data in `../data/sample/`.

## Use our pretrained model
You can use our pretrained model (on MassSpecGym) to rank molecular candidates by providing the spectra data and a list of candidates.

After prepping your data, modify the params_binnedSpec.yaml or params_formSpec.yaml with your dataset paths:

```
# If using formSpec
python test.py --param_pth params_formSpec.yaml

# If using binnedSpec
python test.py --param_pth params_binnedSpec.yaml
```

We provide a notebook showing sample result files in `notebooks/demo.ipynb`

## MassSpecGym data download
Our model is trained on [MassSpecGym dataset](https://github.com/pluskal-lab/MassSpecGym). Follow their instruction to download the spectra and candidate dataset.

You can preprocess the MassSpecGym dataset as descirbed in the above section or download the preprocessed files as follows:
```
mkdir data/msgym/
cd data/msgym
wget 
wget 
```
## Training from scratch
To train a model from scratch:
1. Prepare data as described in the data prep section
2. Modify the configuration in params file as necessary
3. Train using the following
```
# If using formSpec
python train.py --param_pth params_formSpec.yaml

# If using binnedSpec
python train.py --param_pth params_binnedSpec.yaml
```

## References


#### Contact
Soha.Hassoun@tufts.edu
=======