import os
import json
import numpy as np

from mvp.data.transforms import SpecBinner, SpecBinnerLog, SpecFormulaFeaturizer
from massspecgym.data.transforms import SpecTransform, MolTransform
from mvp.data.transforms import MolToGraph
import mvp.data.datasets as jestr_datasets
import typing as T
from mvp.definitions import MSGYM_FORMULA_VECTOR_NORM
import matchms

class Subformula_Loader:
    def __init__(self, spectra_view, dir_path) -> None:

        self.dir_path = dir_path
        if spectra_view == 'SpecFormula':
            self.load = self.load_subformula_data
        elif spectra_view == "SpecFormulaMz":
            self.load = self.load_subformula_dict
        else:
            raise Exception("Spectra view is not supported.")

    def __call__(self, ids):
        id_to_form_spec = {}
        for id in ids:
            data = self.load(id)
            if data:
                id_to_form_spec[id] = data
        
        return id_to_form_spec
    
    def load_subformula_data(self, spec_id: str):
        '''MIST subformula format:https://github.com/samgoldman97/mist/blob/main_v2/src/mist/utils/spectra_utils.py 
        '''
        try:
            file = os.path.join(self.dir_path, spec_id+".json")
            with open(file) as f:
                data = json.load(f)
            mzs = np.array(data['output_tbl']['mz'])
            formulas = np.array(data['output_tbl']['formula'])
            intensities = np.array(data['output_tbl']['ms2_inten'])

            # sort by mzs
            ind = mzs.argsort()
            mzs = mzs[ind]
            formulas = formulas[ind]
            intensities = intensities[ind]
            return {'formulas': formulas, 'formula_mzs': mzs, 'formula_intensities': intensities}
        except:
            return None

    def load_subformula_dict(self, spec_id: str):
        '''MIST subformula format:https://github.com/samgoldman97/mist/blob/main_v2/src/mist/utils/spectra_utils.py 
        '''
        try:
            file = os.path.join(self.dir_path, spec_id+".json")
            with open(file) as f:
                data = json.load(f)
            mzs = np.array(data['output_tbl']['mz'])
            formulas = np.array(data['output_tbl']['formula'])
            intensities = np.array(data['output_tbl']['ms2_inten'])

            mz_to_formulas = {mz:f for mz, f in zip(mzs, formulas)}
            for mz, f in zip(mzs, formulas):
                mz_to_formulas[mz] = f
            
            ind = mzs.argsort()
            mzs = mzs[ind]
            formulas = formulas[ind]
            intensities = intensities[ind]
            return {'formulas': mz_to_formulas, 'formula_mzs': mzs, 'formula_intensities': intensities}
        except:
            return None

def make_tmp_subformula_spectra(row):
        return {'formulas':[row['formula']], 'formula_mzs':[float(row['precursor_mz'])], 'formula_intensities':[1.0]}

def get_spec_featurizer(spectra_view: T.Union[str, list[str]],
                         params) -> T.Union[SpecTransform, T.Dict[str, SpecTransform]]:
    
    featurizers = {"BinnedSpectra": SpecBinner,
        "SpecBinnerLog": SpecBinnerLog,
        "SpecFormula": SpecFormulaFeaturizer}

    spectra_featurizer = {}

    if isinstance(spectra_view, str):
        spectra_view = [spectra_view]

    for view in spectra_view:
        featurizer_params = {'max_mz': params['max_mz']}
        if view in ["BinnedSpectra", "SpecBinnerLog"]:
            featurizer_params.update({'bin_width': params['bin_width']})
        elif view in ["SpecFormula"]:
            featurizer_params.update({'element_list': params['element_list'], 'add_intensities': params['add_intensities'], 'formula_normalize_vector': MSGYM_FORMULA_VECTOR_NORM})
        
        spectra_featurizer[view] = featurizers[view](**featurizer_params)

    return spectra_featurizer

def get_mol_featurizer(molecule_view: T.Union[str, T.List[str]], params) -> MolTransform:
    featurizes = {'MolGraph':MolToGraph}
    mol_featurizer = {}

    if isinstance(molecule_view, str):
        molecule_view = [molecule_view]
    for view in molecule_view:
        featurizer_params = {}
        if view in ('MolGraph'):
            featurizer_params.update({'atom_feature': params['atom_feature'], 'bond_feature': params['bond_feature'], 'element_list': params['element_list']})
        
        if len(molecule_view) == 1:
            return featurizes[view](**featurizer_params)

        mol_featurizer[view] = featurizes[view](**featurizer_params)
    
    return mol_featurizer

def get_test_ms_dataset(spectra_view: T.Union[str, T.List[str]],
                 mol_view: T.Union[str, T.List[str]],
                 spectra_featurizer: SpecTransform,
                 mol_featurizer: MolTransform,
                 params,
                external_test: bool = False,):
    
    use_formulas = False
    
    views = []
    for v in [spectra_view, mol_view]:
        if isinstance(v, str):
            views.append(v)
        else: views.extend(v)
    views = frozenset(views)

    dataset_params = {'spectra_view': spectra_view, 'pth': params['dataset_pth'], 'spec_transform': spectra_featurizer, 'mol_transform': mol_featurizer, "candidates_pth": params['candidates_pth']}
    if "SpecFormula" in views or "SpecFormulaMz" in views:
        dataset_params.update({'subformula_dir_pth': params['subformula_dir_pth']})
        use_formulas = True 
    
    if params['use_cons_spec']:
        dataset_params.update({'cons_spec_dir_pth': params['cons_spec_dir_pth']})

    if params['pred_fp'] or params['use_fp']:
        dataset_params.update({'fp_dir_pth': '', 'fp_size': params['fp_size'], 'fp_radius': params['fp_radius']})

    return jestr_datasets.ExpandedRetrievalDataset(use_formulas=use_formulas, external_test=external_test, **dataset_params)
    
def get_ms_dataset(spectra_view: str,
                 mol_view: str,
                 spectra_featurizer: SpecTransform,
                 mol_featurizer: MolTransform,
                 params):
    

    # set up dataset_parameters
    dataset_params = {'pth': params['dataset_pth'], 'spec_transform': spectra_featurizer, 'mol_transform': mol_featurizer, 'spectra_view': spectra_view}
    use_formulas = False
    if "SpecFormula" in spectra_view:
        dataset_params.update({'subformula_dir_pth': params['subformula_dir_pth']})
        use_formulas = True

    if params['pred_fp'] or params['use_fp']:
        dataset_params.update({'fp_dir_pth': params['fp_dir_pth']})
        
    if params['use_cons_spec']:
        dataset_params.update({'cons_spec_dir_pth': params['cons_spec_dir_pth']})

    # select dataset
    if params['aug_cands']:
        return jestr_datasets.MassSpecDataset_Candidates(**dataset_params)
    elif use_formulas:
        return jestr_datasets.MassSpecDataset_PeakFormulas(**dataset_params)
    
    return jestr_datasets.JESTR1_MassSpecDataset(**dataset_params)

class PrepMatchMS:
    def __init__(self, spectra_view) -> None:

        if spectra_view == 'SpecFormula':
            self.prepare = self.specFormula
        elif spectra_view == "SpecFormulaMz":
            self.prepare = self.specFormulaMz
        elif spectra_view in ('SpecBinnerLog', 'BinnedSpectra', 'SpecMzIntTokenizer'):
            self.prepare = self.specMzInt
        else:
            raise Exception("Spectra view is not supported.")
        
    def specFormulaMz(self, row):
        
        return matchms.Spectrum(
            mz = np.array([float(m) for m in row["mzs"].split(",")]),
            intensities = np.array(
                        [float(i) for i in row["intensities"].split(",")]
                    ),
            metadata = {'precursor_mz': row['precursor_mz'], 'formulas': row['formulas']}
        )
    
    def specFormula(self, row):

        return matchms.Spectrum(
            mz = np.array(row['formula_mzs']),
            intensities = np.array(row['formula_intensities']),
            metadata = {'precursor_mz': row['precursor_mz'], 'formulas': np.array(row['formulas']), 'precursor_formula': row['precursor_formula']}
        )
    
    def specMzInt(self, row):
        return matchms.Spectrum(
            mz = row['mzs'],
            intensities = row['intensities'],
            metadata = {'precursor_mz': row['precursor_mz']}
        )