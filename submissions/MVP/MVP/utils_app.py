from matchms.importing import load_from_mgf
import yaml
import numpy as np
from mvp.subformula_assign.utils.spectra_utils import assign_subforms
import tempfile
import json
import os
from functools import partial
from pytorch_lightning import Trainer
from massspecgym.models.base import Stage
from mvp.data.data_module import TestDataModule
from mvp.data.datasets import ContrastiveDataset
from mvp.utils.data import get_spec_featurizer, get_mol_featurizer, get_test_ms_dataset
from mvp.utils.models import get_model
import pandas as pd

# check formspec requirements
def check_formspec_requirements(spectra):
    for spec in spectra:
        if 'formula' not in spec.metadata or 'adduct' not in spec.metadata:
            return False
    return True

# preprocess spectra
def preprocess_spectra(mgf_path, model_choice, mass_diff_thresh=20, dataset_pth=None, subformula_dir=None):

    if dataset_pth is None:
        dataset_pth = os.path.join(tempfile.gettempdir(), f"mvp_data.tsv")
    if subformula_dir is None:
        subformula_dir = os.path.join(tempfile.gettempdir(), f"mvp_subformulae")
    os.makedirs(subformula_dir, exist_ok=True)

    # load mgf file
    spectra = list(load_from_mgf(mgf_path))

    columns = ['identifier', 'formula', 'adduct', 'precursor_mz', 'precursor_formula', 'mzs', 'intensities', 'fold']
    data = []
    try:
        for spec in spectra:
            identifier = spec.metadata['title']
            formula = spec.metadata.get('formula', None)
            adduct = spec.metadata.get('adduct', None)
            precursor_mz = spec.metadata['precursor_mz']
            precursor_formula = spec.metadata['formula'] # technically incorrect, but we don't use it
            mzs = spec.peaks.mz
            intensities = spec.peaks.intensities

            if model_choice == "formSpec":
                if formula is None or adduct is None:
                    return None, None
                ms = [(m, i) for m, i in zip(mzs, intensities)]

                # annotate peaks
                x = assign_subforms(formula, np.array(ms), adduct, mass_diff_thresh=mass_diff_thresh)
                if x['output_tbl'] is None:
                    continue
                
                # save json file
                json_file = os.path.join(subformula_dir, f"{identifier}.json")
                with open(json_file, 'w') as f:
                    json.dump(x['output_tbl'], f)

            mzs = ','.join([str(m) for m in mzs])
            intensities = ','.join([str(i) for i in intensities])
            data.append([identifier, formula, adduct, precursor_mz, precursor_formula, mzs, intensities, 'test'])
        
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(dataset_pth, sep='\t', index=False)

        return dataset_pth, subformula_dir
    except Exception as e:
        return None, None

def setup_config(model_choice, dataset_pth, candidates_pth, subformula_dir):

    if model_choice == "binnedSpec":
        param_file = f"mvp/params_binnedSpec.yaml"
        checkpoint_path = f"pretrained_models/msgym_binnedSpec.ckpt"
    elif model_choice == "formSpec":
        param_file = f"mvp/params_formSpec.yaml"
        checkpoint_path = f"pretrained_models/msgym_formSpec.ckpt"

    # load yaml
    with open(param_file, 'r') as f:
        params = yaml.safe_load(f)

    params['dataset_pth'] = dataset_pth
    params['candidates_pth'] = candidates_pth
    params['subformula_dir_pth'] = subformula_dir
    params['experiment_dir'] = tempfile.mkdtemp()
    params['checkpoint_pth'] = checkpoint_path
    params['df_test_path'] = os.path.join(params['experiment_dir'], f"results_{model_choice}.pkl")

    return params


def run_inference(params):
    
    # Load dataset
    spec_featurizer = get_spec_featurizer(params['spectra_view'], params)
    mol_featurizer = get_mol_featurizer(params['molecule_view'], params)
    dataset = get_test_ms_dataset(params['spectra_view'], params['molecule_view'], spec_featurizer, mol_featurizer, params, external_test=True)

    # Init data module
    collate_fn = partial(ContrastiveDataset.collate_fn, spec_enc=params['spec_enc'], spectra_view=params['spectra_view'], stage=Stage.TEST)
    data_module = TestDataModule(
        dataset=dataset,
        collate_fn=collate_fn,
        split_pth=params['split_pth'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers']
    )

    model = get_model(params['model'], params)
    print(model.hparams)
    model.df_test_path = params['df_test_path']
    model.external_test = True
    model.hparams['use_fp'] = False
    model.hparams["contr_views"] = [['spec_enc', 'mol_enc']]
    model.hparams['use_cons_spec'] = False
    
    # Init trainer
    trainer = Trainer(
        accelerator=params['accelerator'],
        devices=params['devices'],
        default_root_dir=params['experiment_dir']
    )

    # Prepare data module to test
    data_module.prepare_data()
    data_module.setup(stage="test")
        
    # Test
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":

    # test run
    mgf_path = "data/app/data.mgf"
    model_choice = "formSpec"
    candidates_pth = "data/app/identifier_to_candidates.json"
    mass_diff_thresh = 20
    dataset_pth, subformula_dir = preprocess_spectra(mgf_path, model_choice, mass_diff_thresh=mass_diff_thresh)
    params = setup_config(model_choice, dataset_pth, candidates_pth, subformula_dir)
    print(params)
    run_inference(params)