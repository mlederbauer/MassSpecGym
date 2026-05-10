import torch.nn as nn
import torch
from mvp.models.encoders import MLP
from torch_geometric.nn import global_mean_pool


class SpecEncMLP_BIN(nn.Module):
    def __init__(self, args, out_dim=None):
        super(SpecEncMLP_BIN, self).__init__()

        if not out_dim:
            out_dim = args.final_embedding_dim

        bin_size = int(args.max_mz / args.bin_width)
        self.dropout = nn.Dropout(args.fc_dropout)
        self.mz_fc1 = nn.Linear(bin_size, out_dim * 2)
        self.mz_fc2 = nn.Linear(out_dim* 2, out_dim * 2)
        self.mz_fc3 = nn.Linear(out_dim * 2, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, mzi_b, n_peaks=None):
                
       h1 = self.mz_fc1(mzi_b)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       h1 = self.mz_fc2(h1)
       h1 = self.relu(h1)
       h1 = self.dropout(h1)
       mz_vec = self.mz_fc3(h1)
       mz_vec = self.dropout(mz_vec)
       
       return mz_vec

    
class SpecFormulaTransformer(nn.Module):
    def __init__(self, args, out_dim=None):
        super(SpecFormulaTransformer, self).__init__()
        in_dim = len(args.element_list)
        if args.add_intensities: # intensity
            in_dim+=1
        if args.spectra_view == "SpecFormulaMz": #mz
            in_dim+=1 

        self.returnEmb = False
        
        self.formulaEnc = MLP(in_dim=in_dim, hidden_dims=args.formula_dims, dropout=args.formula_dropout)
        
        self.use_cls = args.use_cls
        if args.use_cls:
            self.cls_embed = torch.nn.Embedding(1,args.formula_dims[-1])
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.formula_dims[-1], nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        if not out_dim:
            out_dim = args.final_embedding_dim
        self.fc = nn.Linear(args.formula_dims[-1], out_dim)
        
    def forward(self, spec, n_peaks):
        h = self.formulaEnc(spec)
        pad = (spec == -5)
        pad = torch.all(pad, -1)

        if self.use_cls:
            cls_embed = self.cls_embed(torch.tensor(0).to(spec.device))
            h = torch.concat((cls_embed.repeat(spec.shape[0], 1).unsqueeze(1), h), dim=1)
            pad = torch.concat((torch.tensor(False).repeat(pad.shape[0],1).to(spec.device), pad), dim=1)
            h = self.transformer(h, src_key_padding_mask=pad)
            h = h[:,0,:]
        else:
            h = self.transformer(h, src_key_padding_mask=pad)

            if self.returnEmb:
                # repad h
                h[pad] = -5
                return h
            
            h = h[~pad].reshape(-1, h.shape[-1])
            indecies = torch.tensor([i for i, count in enumerate(n_peaks) for _ in range(count)]).to(h.device)
            h = global_mean_pool(h, indecies)
            
        h = self.fc(h)

        return h
    

