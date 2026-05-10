import pandas as pd
import json
import typing as T
import numpy as np
import torch
import massspecgym.utils as utils
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import dgl
from collections import defaultdict
from massspecgym.data.transforms import SpecTransform, MolTransform, MolToInChIKey
from massspecgym.data.datasets import MassSpecDataset
import mvp.utils.data as data_utils
from torch.nn.utils.rnn import pad_sequence
from massspecgym.models.base import Stage
import pickle
import math
import itertools
from rdkit.Chem import AllChem
from rdkit import Chem
class JESTR1_MassSpecDataset(MassSpecDataset):
    def __init__(
        self,
        spectra_view: str,
        fp_dir_pth: str = None,
        cons_spec_dir_pth: str = None,
        NL_spec_dir_pth: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.use_fp = False
        self.use_cons_spec = False
        self.use_NL_spec = False
        self.spectra_view = spectra_view

        # load fingerprints
        self._load_fp(fp_dir_pth)

        # load consensus
        self._load_cons_spec(cons_spec_dir_pth)

        # load NL specs
        self._load_NL_spec(NL_spec_dir_pth)

    def _load_fp(self, fp_dir_pth):
        if fp_dir_pth is not None:
            self.use_fp = True
            if fp_dir_pth:
                with open(fp_dir_pth, 'rb') as f:
                    self.smiles_to_fp = pickle.load(f)
            else:
                self.smiles_to_fp = {}
    
    def _load_cons_spec(self, cons_spec_dir_pth):
        if cons_spec_dir_pth is not None:
            self.use_cons_spec = True
            with open(cons_spec_dir_pth, 'rb') as f:
                cons_specs = pickle.load(f)

            # Convert spectra to matchms spectra
            matchMS_preparer = data_utils.PrepMatchMS(self.spectra_view)
            spectra = cons_specs.apply(matchMS_preparer.prepare,axis=1)

            self.cons_specs = dict(zip(cons_specs['smiles'].tolist(), spectra))

    def _load_NL_spec(self, NL_spec_dir_pth):
        if NL_spec_dir_pth is not None:
            self.use_NL_spec = True
            with open(NL_spec_dir_pth, 'rb') as f:
                NL_specs = pickle.load(f)

            # Convert spectra to matchms spectra
            matchMS_preparer = data_utils.PrepMatchMS(self.spectra_view)
            self.NL_specs = NL_specs.apply(matchMS_preparer.prepare,axis=1)


    def __getitem__(self, i, transform_spec: bool = True, transform_mol: bool = True):

        spec = self.spectra[i]
        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]

        # Apply all transformations to the spectrum
        item = {}
        if transform_spec and self.spec_transform:
            if isinstance(self.spec_transform, dict):
                for key, transform in self.spec_transform.items():
                    item[key] = transform(spec) if transform is not None else spec
            else:
                item["spec"] = self.spec_transform(spec)
        else:
            item["spec"] = spec

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]

        if self.use_fp and self.smiles_to_fp:
            item['fp'] = torch.Tensor(self.smiles_to_fp[mol].ToList())
        
        if self.use_cons_spec:
            item['cons_spec'] = self.spec_transform[self.spectra_view](self.cons_specs[mol])

        if self.use_NL_spec:
            item['NL_spec'] = self.spec_transform[self.spectra_view](self.NL_specs[i])

        # Apply all transformations to the molecule
        if transform_mol and self.mol_transform:
            if isinstance(self.mol_transform, dict):
                for key, transform in self.mol_transform.items():
                    item[key] = transform(mol) if transform is not None else mol
            else:
                item["mol"] = self.mol_transform(mol)
        else:
            item["mol"] = mol
        return item

class MassSpecDataset_PeakFormulas(JESTR1_MassSpecDataset):
    def __init__(
        self,
        spectra_view: str,
        spec_transform: T.Optional[T.Union[SpecTransform, T.Dict[str, SpecTransform]]],
        mol_transform: T.Optional[T.Union[MolTransform, T.Dict[str, MolTransform]]],
        pth: T.Optional[Path],
        subformula_dir_pth: str,
        fp_dir_pth: str = None,
        NL_spec_dir_pth: str = None,
        cons_spec_dir_pth: str = None,
        return_mol_freq: bool = False,
        return_identifier: bool = True,
        dtype: T.Type = torch.float32
    ):
        """
        Args:
        """
        self.pth = pth
        self.spec_transform = spec_transform
        self.mol_transform = mol_transform
        self.return_mol_freq = return_mol_freq
        self.pred_fp = False
        self.use_fp = False
        self.use_cons_spec = False
        self.use_NL_spec = False
        self.spectra_view = spectra_view

        if isinstance(self.pth, str):
            self.pth = Path(self.pth)

        self.spectra_view = spectra_view
        print("Data path: ", self.pth)
        self.metadata = pd.read_csv(self.pth, sep="\t")

        # Used for training on consensus spectra
        # with open(self.pth, 'rb') as f:
        #     self.metadata = pickle.load(f)
        # self.metadata['identifier'] = self.metadata['smiles'].tolist()

        # load subformulas
        all_spec_ids = self.metadata['identifier'].tolist()
        subformulaLoader = data_utils.Subformula_Loader(spectra_view=spectra_view, dir_path=subformula_dir_pth)
        id_to_spec = subformulaLoader(all_spec_ids)

        # create subformula spectra if no subformula is available
        tmp_ids = [spec_id for spec_id in all_spec_ids if spec_id not in id_to_spec]
        tmp_df = self.metadata[self.metadata['identifier'].isin(tmp_ids)]
        tmp_df['spec'] = tmp_df.apply(lambda row: data_utils.make_tmp_subformula_spectra(row), axis=1)
        id_to_spec.update(dict(zip(tmp_df['identifier'].tolist(), tmp_df['spec'].tolist())))
        
        
        # load fingerprints
        self._load_fp(fp_dir_pth)

        # load consensus spectra
        self._load_cons_spec(cons_spec_dir_pth)

        # load NL specs
        self._load_NL_spec(NL_spec_dir_pth)

        self.metadata = self.metadata[self.metadata['identifier'].isin(id_to_spec)]
        formula_df = pd.DataFrame.from_dict(id_to_spec, orient='index').reset_index().rename(columns={'index': 'identifier'})
        self.metadata = self.metadata.merge(formula_df, on='identifier')

        # create matchms spectra
        matchMS_preparer = data_utils.PrepMatchMS(spectra_view=spectra_view)
        self.spectra = self.metadata.apply(matchMS_preparer.prepare,axis=1)
                
        if self.return_mol_freq:
            if "inchikey" not in self.metadata.columns:
                self.metadata["inchikey"] = self.metadata["smiles"].apply(utils.smiles_to_inchi_key)
            self.metadata["mol_freq"] = self.metadata.groupby("inchikey")["inchikey"].transform("count")

        self.return_identifier = return_identifier
        self.dtype = dtype
    
    def __getitem__(self, i, transform_spec: bool = True, transform_mol: bool = True):
        item = super().__getitem__(i, transform_spec, transform_mol = False)
        mol = item['mol'] #smiles

        # transform mol
        if transform_mol:
            if isinstance(self.mol_transform, dict):
                for key, transform in self.mol_transform.items():
                    item[key] = transform(mol) if transform is not None else mol
            else:
                item["mol"] = self.mol_transform(mol)

        return item

class ContrastiveDataset(Dataset):
    def __init__(
        self,
        spec_mol_data,
    ):
        super().__init__()
    
        indices = spec_mol_data.indices
        self.spec_mol_data = spec_mol_data
        self.smiles_to_specmol_ids = spec_mol_data.dataset.metadata.loc[indices].groupby('smiles').indices
        self.smiles_to_spec_couter = defaultdict(int)
        self.smiles_list = list(self.smiles_to_specmol_ids.keys())

    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, i:int) -> dict:
        mol = self.smiles_list[i]

        # select spectrum (iterate through list of spectra)
        specmol_ids = self.smiles_to_specmol_ids[mol]
        counter = self.smiles_to_spec_couter[mol]
        specmol_id = specmol_ids[counter % len(specmol_ids)]

        item = self.spec_mol_data.__getitem__(specmol_id)
        self.smiles_to_spec_couter[mol] = counter+1
        # item['smiles'] = mol
        # item['spec_id'] = specmol_id
        return item

    @staticmethod
    def collate_fn(batch: T.Iterable[dict], spec_enc: str, spectra_view: str, stage=None, mask_peak_ratio: float = 0.0, aug_cands: bool = False) -> dict:
        mol_key = 'cand' if stage == Stage.TEST else 'mol'
        non_standard_collate = ['mol', 'cand', 'aug_cands', 'cons_spec', 'aug_cands_fp', 'NL_spec']
        require_pad = False
        if 'Formula' in spectra_view or 'Tokens' in spectra_view:
            require_pad = True
            padding_value=-5 if spec_enc in ('Transformer_Formula', 'Formula_BinnedSpec', 'Transformer_MzInt') else 0
            non_standard_collate.append(spectra_view)
        else:
            non_standard_collate.remove('cons_spec')
            non_standard_collate.remove('NL_spec')

        collated_batch = {}
        # standard collate
        for k in batch[0].keys():
            if k not in non_standard_collate:
                collated_batch[k] = default_collate([item[k] for item in batch])
                
        # batch graphs
        batch_mol = []
        batch_mol_nodes= []

        for item in batch:
            batch_mol.append(item[mol_key])
            batch_mol_nodes.append(item[mol_key].num_nodes())

        collated_batch[mol_key] = dgl.batch(batch_mol)
        collated_batch['mol_n_nodes'] = batch_mol_nodes
        
        # pad peaks/formulas
        if require_pad:
            peaks = []
            n_peaks = []
            for item in batch:
                peaks.append(item[spectra_view])
                n_peaks.append(len(item[spectra_view]))
            collated_batch[spectra_view] = pad_sequence(peaks, batch_first=True, padding_value=padding_value)
            collated_batch['n_peaks'] = n_peaks
        
            if 'cons_spec' in batch[0]:
                peaks = []
                n_peaks = []
                for item in batch:
                    peaks.append(item['cons_spec'])
                    n_peaks.append(len(item['cons_spec']))
                collated_batch['cons_spec'] = pad_sequence(peaks, batch_first=True, padding_value=padding_value)
                collated_batch['cons_n_peaks'] = n_peaks

            if 'NL_spec' in batch[0]:
                peaks = []
                n_peaks = []
                for item in batch:
                    peaks.append(item['NL_spec'])
                    n_peaks.append(len(item['NL_spec']))
                collated_batch['NL_spec'] = pad_sequence(peaks, batch_first=True, padding_value=padding_value)
                collated_batch['NL_n_peaks'] = n_peaks


        # mask peaks
        if mask_peak_ratio > 0.0 and stage == Stage.TRAIN:
            n_mask_peaks = [math.floor(n_peak* mask_peak_ratio) for n_peak in n_peaks]
            mask_peak_idx = [np.random.choice(n_peak, n_mask, replace=False) for n_peak, n_mask in zip(n_peaks, n_mask_peaks)]
            for i, peaks in enumerate(collated_batch[spectra_view]):
                peaks[mask_peak_idx[i]] = -5.0
        
        # batch candidates
        if aug_cands:
            candidates = \
                sum([item["aug_cands"] for item in batch], start=[])
            collated_batch['aug_cands'] = dgl.batch(candidates)

            if 'aug_cands_fp' in batch[0]:
                cand_fp = [item['aug_cands_fp'] for item in batch]
                collated_batch['aug_cands_fp'] = torch.flatten(torch.Tensor(cand_fp), end_dim=1)

        return collated_batch
    
 

class ExpandedRetrievalDataset:
    '''Used for testing only 
    Assumes 'fold' column defines the split'''
    def __init__(self,
                 use_formulas: bool = True,
                 mol_label_transform: MolTransform = MolToInChIKey(),
                 candidates_pth: T.Optional[T.Union[Path, str]] = None,
                 fp_size: int = None,
                 fp_radius: int = None,
                 external_test: bool = False,
                **kwargs):
        
        self.external_test = external_test
        
        self.instance = MassSpecDataset_PeakFormulas(**kwargs, return_mol_freq=False) if use_formulas else JESTR1_MassSpecDataset(**kwargs, return_mol_freq=False)
        # super().__init__(**kwargs)

        if self.use_fp:
            self.fpgen = AllChem.GetMorganGenerator(radius=fp_radius,fpSize=fp_size)

        self.candidates_pth = candidates_pth
        self.mol_label_transform = mol_label_transform
        
        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(self.candidates_pth, "r") as file:
            candidates = json.load(file)

        self.candidates = {}
        for s, cand in candidates.items():
            self.candidates[s] = [c for c in cand if '.' not in c]
        
        self.spec_cand = [] #(spec index, cand_smiles, true_label)

        # use for external dataset where target smiles is not known
        # self.candidates should be a dict of identifier to candidates
        if self.external_test or 'smiles' not in self.metadata.columns:
            if not isinstance(self.metadata.iloc[0]['identifier'], str):
                self.metadata['smiles'] = self.metadata['identifier'].apply(str)
            else:
                self.metadata['smiles'] = self.metadata['identifier']
        test_smiles = self.metadata[self.metadata['fold'] == "test"]['smiles'].tolist()
        test_ms_id = self.metadata[self.metadata['fold'] == "test"]['identifier'].tolist()
        
        spec_id_to_index = dict(zip(self.metadata['identifier'], self.metadata.index))
        for spec_id, s in zip(test_ms_id, test_smiles):
            candidates = self.candidates[s]
            # mol_label = self.mol_label_transform(s)
            # labels = [self.mol_label_transform(c) == mol_label for c in candidates]
            if not self.external_test:
                labels = [c == s for c in candidates]

                if len(candidates) == 0:
                    print(f"Skipping {spec_id}; empty candidate set")
                    continue
                if not any(labels):
                    print(f"Target smiles not in candidate set")
            else:
                labels = [False] * len(candidates)

            self.spec_cand.extend([(spec_id_to_index[spec_id], candidates[j], k) for j, k in enumerate(labels)])
    
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)
    
    def __len__(self):
        return len(self.spec_cand)

    def __getitem__(self, i):
        spec_i = self.spec_cand[i][0]
        cand_smiles = self.spec_cand[i][1]
        label = self.spec_cand[i][2]

        item = self.instance.__getitem__(spec_i, transform_mol=False)
        item['cand'] = self.mol_transform(cand_smiles)
        item['cand_smiles'] = cand_smiles
        item['label'] = label

        if self.use_fp:
            item['fp'] = torch.Tensor(self.fpgen.GetFingerprint(Chem.MolFromSmiles(cand_smiles)).ToList())

        return item