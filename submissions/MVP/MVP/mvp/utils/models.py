from mvp.models.spec_encoder import SpecEncMLP_BIN, SpecFormulaTransformer
from mvp.models.mol_encoder import MolEnc
from mvp.models.encoders import MLP
from mvp.models.contrastive import ContrastiveModel, MultiViewContrastive

def get_spec_encoder(spec_enc:str, args):
    return {"MLP_BIN": SpecEncMLP_BIN,
            "Transformer_Formula": SpecFormulaTransformer}[spec_enc](args)

def get_mol_encoder(mol_enc: str, args):
    return {'GNN': MolEnc}[mol_enc](args, in_dim=78)

def get_fp_pred_model(args):
    return MLP(in_dim=args.final_embedding_dim, hidden_dims=[args.fp_size], final_activation='sigmoid', dropout=args.fp_dropout)

def get_fp_enc_model(args):
    return MLP(in_dim=args.fp_size, hidden_dims=[args.final_embedding_dim,args.final_embedding_dim*2,args.final_embedding_dim,], final_activation=None, dropout=0.0)

def get_model(model:str,
              params):
    
    if model == 'contrastive':
        model= ContrastiveModel(**params)
    elif model == "MultiviewContrastive":
        model = MultiViewContrastive(**params)
    else:
        raise Exception(f"Model {model} not implemented.")
    
    # If checkpoint path is provided, load the model from the checkpoint instead
    if params['checkpoint_pth'] is not None and params['checkpoint_pth'] != "":
        model = type(model).load_from_checkpoint(
            params['checkpoint_pth'],
            log_only_loss_at_stages=params['log_only_loss_at_stages'],
            df_test_path=params['df_test_path']
        )
        print("Loaded Model from checkpoint")

    return model