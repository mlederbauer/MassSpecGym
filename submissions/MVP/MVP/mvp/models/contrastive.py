import typing as T
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
import numpy as np
import os
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.models.base import Stage
from massspecgym import utils

from mvp.utils.loss import contrastive_loss, cand_spec_sim_loss, fp_loss, cons_spec_loss
import mvp.utils.models as model_utils

import torch.nn.functional as F


class ContrastiveModel(RetrievalMassSpecGymModel):
    def __init__(
        self,
        external_test = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.external_test = external_test

        if 'use_fp' not in self.hparams:
            self.hparams.use_fp = False

        if 'loss_strategy' not in self.hparams:
            self.hparams.loss_strategy = 'static'
            self.hparams.contr_wt = 1.0
            self.hparams.use_contr = True

        self.spec_enc_model = model_utils.get_spec_encoder(self.hparams.spec_enc, self.hparams)
        self.mol_enc_model = model_utils.get_mol_encoder(self.hparams.mol_enc, self.hparams)
        
        if self.hparams.pred_fp:
            self.fp_loss = fp_loss(self.hparams.fp_loss_type)
            self.fp_pred_model = model_utils.get_fp_pred_model(self.hparams)
        if self.hparams.use_cons_spec:
            self.cons_spec_enc_model = model_utils.get_spec_encoder(self.hparams.spec_enc, self.hparams)
            self.cons_loss = cons_spec_loss(self.hparams.cons_loss_type)
            
        self.spec_view = self.hparams.spectra_view
        
        # result storage for testing results
        self.result_dct = defaultdict(lambda: defaultdict(list))
                    
    def forward(self, batch, stage):
        g = batch['cand'] if stage == Stage.TEST else batch['mol']
        
        if self.hparams.use_cons_spec and stage != Stage.TEST:
            spec = batch['cons_spec']
            n_peaks = batch['cons_n_peaks'] if 'cons_n_peaks' in batch else None
            spec_enc = self.cons_spec_enc_model(spec, n_peaks)
        else:
            spec = batch[self.spec_view]
            n_peaks = batch['n_peaks'] if 'n_peaks' in batch else None
            spec_enc = self.spec_enc_model(spec, n_peaks)

        fp = batch['fp'] if self.hparams.use_fp else None
        mol_enc = self.mol_enc_model(g, fp=fp)

        return spec_enc, mol_enc

    def compute_loss(self, batch: dict, spec_enc, mol_enc, output):
        loss = 0
        losses = {}
        contr_loss, cong_loss, noncong_loss = contrastive_loss(spec_enc, mol_enc, self.hparams.contr_temp)
        contr_loss = self.loss_wts['contr_wt'] *contr_loss
        losses['contr_loss'] = contr_loss.detach().item()
        losses['cong_loss'] = cong_loss.detach().item()
        losses['noncong_loss'] = noncong_loss.detach().item()
        
        loss+=contr_loss
        if self.hparams.pred_fp:
            fp_loss_val = self.loss_wts['fp_wt'] *self.fp_loss(output['fp'], batch['fp'])
            loss+= fp_loss_val
            losses['fp_loss'] = fp_loss_val.detach().item()

        if 'aug_cand_enc' in output:  
            aug_cand_loss = self.loss_wts['aug_cand_wt'] * cand_spec_sim_loss(spec_enc, output['aug_cand_enc'])
            loss+= aug_cand_loss
            losses['aug_cand_loss'] = aug_cand_loss.detach().item()
        
        if 'ind_spec' in output:
            spec_loss = self.loss_wts['cons_spec_wt'] * self.cons_loss(spec_enc, output['ind_spec'])
            loss+=spec_loss
            losses['cons_spec_loss'] = spec_loss.detach().item()

        losses['loss'] = loss 

        return losses
    
    def step(
        self, batch: dict, stage= Stage.NONE):
        
        # Compute spectra and mol encoding
        spec_enc, mol_enc = self.forward(batch, stage)

        if stage == Stage.TEST:
            return dict(spec_enc=spec_enc, mol_enc=mol_enc)
        
        # Aux tasks
        output = {}
        if self.hparams.pred_fp:
            output['fp'] = self.fp_pred_model(mol_enc)
        
        if self.hparams.use_cons_spec:
            spec = batch[self.spec_view]
            n_peaks = batch['n_peaks'] if 'n_peaks' in batch else None
            output['ind_spec'] = self.spec_enc_model(spec, n_peaks)

        # Calculate loss
        losses = self.compute_loss(batch, spec_enc, mol_enc, output)

        return losses

    def on_batch_end(self, outputs, batch: dict, batch_idx: int, stage: Stage) -> None:
        # total loss
        self.log(
            f'{stage.to_pref()}loss',
            outputs['loss'],
            batch_size=len(batch['identifier']),
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            # on_step=True
        )

        # contr loss
        if self.hparams.use_contr:
            self.log(
                f'{stage.to_pref()}contr_loss',
                outputs['contr_loss'],
                batch_size=len(batch['identifier']),
                sync_dist=True,
                prog_bar=False,
                on_epoch=True,
                # on_step=True
            )

            # noncongruent pairs
            self.log(
                f'{stage.to_pref()}noncong_loss',
                outputs['noncong_loss'],
                batch_size=len(batch['identifier']),
                sync_dist=True,
                prog_bar=False,
                on_epoch=True,
                # on_step=True
            )
            
            # congruent pairs
            self.log(
                f'{stage.to_pref()}cong_loss',
                outputs['cong_loss'],
                batch_size=len(batch['identifier']),
                sync_dist=True,
                prog_bar=False,
                on_epoch=True,
                # on_step=True
            )


        if self.hparams.pred_fp:

            self.log(
                f'{stage.to_pref()}_fp_loss',
                outputs['fp_loss'],
                batch_size=len(batch['identifier']),
                sync_dist=True,
                prog_bar=False,
                on_epoch=True,
            )
        
        if self.hparams.use_cons_spec:
            self.log(
                f'{stage.to_pref()}cons_loss',
                outputs['cons_spec_loss'],
                batch_size=len(batch['identifier']),
                sync_dist=True,
                prog_bar=False,
                on_epoch=True,
            )
    
    def test_step(self, batch, batch_idx):
        # Unpack inputs
        identifiers = batch['identifier']
        cand_smiles = batch['cand_smiles']
        id_to_ct = defaultdict(int)
        for i in identifiers: id_to_ct[i]+=1
        batch_ptr = torch.tensor(list(id_to_ct.values()))

        outputs = self.step(batch, stage=Stage.TEST)
        spec_enc = outputs['spec_enc']
        mol_enc = outputs['mol_enc']

        # Calculate scores
        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        
        scores = nn.functional.cosine_similarity(spec_enc, mol_enc)
        scores = torch.split(scores, list(id_to_ct.values()))

        cand_smiles = utils.unbatch_list(batch['cand_smiles'], indexes)
        labels = utils.unbatch_list(batch['label'], indexes)
        
        return dict(identifiers=list(id_to_ct.keys()), scores=scores, cand_smiles=cand_smiles, labels=labels)
    
    def on_test_batch_end(self, outputs, batch: dict, batch_idx: int, stage: Stage = Stage.TEST) -> None:
        
        # save scores
        for i, cands, scores, l in zip(outputs['identifiers'], outputs['cand_smiles'], outputs['scores'], outputs['labels']):
            self.result_dct[i]['candidates'].extend(cands)
            self.result_dct[i]['scores'].extend(scores.cpu().tolist())
            self.result_dct[i]['labels'].extend([x.cpu().item() for x in l])
            
    def _compute_rank(self, scores, labels):
        if not any(labels):
            return -1
        scores = np.array(scores)
        target_score = scores[labels][0]
        rank = np.count_nonzero(scores >=target_score)
        return rank
    
    def on_test_epoch_end(self) -> None:

        self.df_test = pd.DataFrame.from_dict(self.result_dct, orient='index').reset_index().rename(columns={'index': 'identifier'})

        # Compute rank
        self.df_test['rank'] = self.df_test.apply(lambda row: self._compute_rank(row['scores'], row['labels']), axis=1)
        if not self.df_test_path:
            self.df_test_path = os.path.join(self.hparams['experiment_dir'], 'result.pkl')
        # self.df_test_path.parent.mkdir(parents=True, exist_ok=True)
        self.df_test.to_pickle(self.df_test_path)

    def get_checkpoint_monitors(self) -> T.List[dict]:
        monitors = [
            {"monitor": f"{Stage.TRAIN.to_pref()}loss", "mode": "min", "early_stopping": False}, # monitor train loss
        ]
        return monitors
    
    def _update_loss_weights(self)-> None:
        if self.hparams.loss_strategy == 'linear':
            for loss in self.loss_wts:
                self.loss_wts[loss] += self.loss_updates[loss]
        elif self.hparams.loss_strategy == 'manual':
            for loss in self.loss_wts:
                if self.current_epoch in self.loss_updates[loss]:
                    self.loss_wts[loss] = self.loss_updates[loss][self.current_epoch]

    def on_train_epoch_end(self) -> None:
        self._update_loss_weights()      
    
class MultiViewContrastive(ContrastiveModel):

    def __init__(self,
                 **kwargs):
        
        super().__init__(**kwargs)
    
        # build fingerprint encoder model
        if self.hparams.use_fp: 
            self.fp_enc_model = model_utils.get_fp_enc_model(self.hparams)
        
        # build NL encoder model
        # if self.hparams.use_NL_spec:
        #     self.NL_enc_model = model_utils.get_spec_encoder(self.hparams.spec_enc, self.hparams)
            
    def forward(self, batch, stage):
        g = batch['cand'] if stage == Stage.TEST else batch['mol']
        
        spec = batch[self.spec_view]
        n_peaks = batch['n_peaks'] if 'n_peaks' in batch else None

        spec_enc = self.spec_enc_model(spec, n_peaks)
        mol_enc = self.mol_enc_model(g)
        views = {'spec_enc': spec_enc, 'mol_enc': mol_enc}
        
        if self.hparams.use_fp:
            fp_enc = self.fp_enc_model(batch['fp'])
            views['fp_enc'] = fp_enc

        if self.hparams.use_cons_spec:
            spec = batch['cons_spec']
            n_peaks = batch['cons_n_peaks'] if 'cons_n_peaks' in batch else None
            spec_enc = self.cons_spec_enc_model(spec, n_peaks)
            views['cons_spec_enc'] = spec_enc

        if self.hparams.use_NL_spec:
            spec = batch['NL_spec']
            n_peaks = batch['NL_n_peaks'] if 'NL_n_peaks' in batch else None
            spec_enc = self.NL_enc_model(spec, n_peaks) 
            views['NL_spec_enc'] = spec_enc
        return views
    
    def step(
        self, batch: dict, stage= Stage.NONE):
        
        # Compute spectra and mol encoding
        views = self.forward(batch, stage)

        if stage == Stage.TEST:
            return views
        
        # Calculate loss
        losses = self.compute_loss(batch, views)

        return losses
    
    def compute_loss(self, batch: dict, views: dict):
        loss = 0
        losses = {}
        for v1, v2 in self.hparams.contr_views:
            contr_loss, cong_loss, noncong_loss = contrastive_loss(views[v1], views[v2], self.hparams.contr_temp)
            loss+=contr_loss

            losses[f'{v1[:-4]}-{v2[:-4]}_contr_loss'] = contr_loss.detach().item()
            losses[f'{v1[:-4]}-{v2[:-4]}_cong_loss'] = cong_loss.detach().item()
            losses[f'{v1[:-4]}-{v2[:-4]}_noncong_loss'] = noncong_loss.detach().item()
        
        losses['loss'] = loss 

        return losses
    
    def on_batch_end(self, outputs, batch: dict, batch_idx: int, stage: Stage) -> None:
        # total loss
        self.log(
            f'{stage.to_pref()}loss',
            outputs['loss'],
            batch_size=len(batch['identifier']),
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            # on_step=True
        )

        for v1, v2 in self.hparams.contr_views:
            self.log(
            f'{stage.to_pref()}{v1[:-4]}-{v2[:-4]}_contr_loss',
            outputs[f'{v1[:-4]}-{v2[:-4]}_contr_loss'],
            batch_size=len(batch['identifier']),
            sync_dist=True,
            on_epoch=True,
        )
            self.log(
            f'{stage.to_pref()}{v1[:-4]}-{v2[:-4]}_cong_loss',
            outputs[f'{v1[:-4]}-{v2[:-4]}_cong_loss'],
            batch_size=len(batch['identifier']),
            sync_dist=True,
            on_epoch=True,
        )
            self.log(
            f'{stage.to_pref()}{v1[:-4]}-{v2[:-4]}_noncong_loss',
            outputs[f'{v1[:-4]}-{v2[:-4]}_noncong_loss'],
            batch_size=len(batch['identifier']),
            sync_dist=True,
            on_epoch=True,
        )
            
    def test_step(self, batch, batch_idx):
        # Unpack inputs
        identifiers = batch['identifier']
        cand_smiles = batch['cand_smiles']
        id_to_ct = defaultdict(int)
        for i in identifiers: id_to_ct[i]+=1
        batch_ptr = torch.tensor(list(id_to_ct.values()))

        outputs = self.step(batch, stage=Stage.TEST)
        scores = {}
        for v1, v2 in self.hparams.contr_views:
            # if 'cons_spec_enc' in (v1, v2):
            #     continue
            v1_enc = outputs[v1]
            v2_enc = outputs[v2]
            
            s = nn.functional.cosine_similarity(v1_enc, v2_enc)
            scores[f'{v1[:-4]}-{v2[:-4]}_scores'] = torch.split(s, list(id_to_ct.values()))

        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)

        cand_smiles = utils.unbatch_list(batch['cand_smiles'], indexes)
        labels = utils.unbatch_list(batch['label'], indexes)
        
        return dict(identifiers=list(id_to_ct.keys()), scores=scores, cand_smiles=cand_smiles, labels=labels)
    
    def on_test_batch_end(self, outputs, batch: dict, batch_idx: int, stage: Stage = Stage.TEST) -> None:

        # save scores
        for i, cands, l in zip(outputs['identifiers'], outputs['cand_smiles'], outputs['labels']):
            self.result_dct[i]['candidates'].extend(cands)
            self.result_dct[i]['labels'].extend([x.cpu().item() for x in l])

        for v1, v2 in self.hparams.contr_views:
            for i, scores in zip(outputs['identifiers'], outputs['scores'][f'{v1[:-4]}-{v2[:-4]}_scores']):
                self.result_dct[i][f'{v1[:-4]}-{v2[:-4]}_scores'].extend(scores.cpu().tolist())

    def _get_top_cand(self, scores, candidates):
        return candidates[np.argmax(np.array(scores))]

    def on_test_epoch_end(self) -> None:

        self.df_test = pd.DataFrame.from_dict(self.result_dct, orient='index').reset_index().rename(columns={'index': 'identifier'})

        # Compute rank
        if not self.external_test:
            for v1, v2 in self.hparams.contr_views:
                self.df_test[f'{v1[:-4]}-{v2[:-4]}_rank'] = self.df_test.apply(lambda row: self._compute_rank(row[f'{v1[:-4]}-{v2[:-4]}_scores'], row['labels']), axis=1)
        
        if self.external_test:
            self.df_test.drop('labels', axis=1, inplace=True)
            for v1, v2 in self.hparams.contr_views:
                self.df_test[f'top_{v1[:-4]}-{v2[:-4]}_cand'] = self.df_test.apply(lambda row: self._get_top_cand(row[f'{v1[:-4]}-{v2[:-4]}_scores'], row['candidates']), axis=1)
        self.df_test.to_pickle(self.df_test_path)