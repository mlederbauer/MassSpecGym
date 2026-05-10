# 1. preprocess data (subformula labels should be obtained through MIST)
python subformula_assign/assign_subformulae.py --spec-files ../data/sample/data.tsv --output-dir ../data/sample/subformulae_default --max-formulae 60 --labels-file ../data/sample/data.tsv
python data_preprocess.py --spec_type formSpec --dataset_pth ../data/sample/data.tsv --candidates_pth  ../data/sample/candidates_mass.json --subformula_dir_pth ../data/sample/subformulae_default/ --output_dir ../data/sample/

# 2. train model on msgym
python train.py --param_pth params_formSpec.yaml

# 3. test model on msgym
python train.py --param_pth params_binnedSpec.yaml
