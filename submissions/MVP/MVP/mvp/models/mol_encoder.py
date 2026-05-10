import torch
import torch.nn as nn
import dgl
from dgllife.model import GCN, GAT

class MolEnc(nn.Module):

    def __init__(self,
                 args,
                 in_dim,):
        super().__init__()

        self.return_emb = False

        if args.model in ('crossAttenContrastive', 'filipContrastive'):
            self.return_emb = True

        dropout = [args.gnn_dropout for _ in range(len(args.gnn_channels))]
        batchnorm = [True for _ in range(len(args.gnn_channels))]
        gnn_map = {
            "gcn": GCN(in_dim, args.gnn_channels, batchnorm = batchnorm, dropout = dropout),
            "gat": GAT(in_dim, args.gnn_channels, args.attn_heads)
        }
        self.GNN = gnn_map[args.gnn_type]
        self.pool = dgl.nn.pytorch.glob.MaxPooling()
        
        if not self.return_emb:
            self.fc1_graph = nn.Linear(args.gnn_channels[len(args.gnn_channels) - 1], args.gnn_hidden_dim * 2)
            self.fc2_graph = nn.Linear(args.gnn_hidden_dim * 2, args.final_embedding_dim)

            self.dropout = nn.Dropout(args.fc_dropout)
            self.relu = nn.ReLU()

    def forward(self, g, fp=None) -> torch.Tensor:
        g1 = g
        f1 = g.ndata['h']

        f = self.GNN(g1, f1)
        if self.return_emb:
            return f
        h = self.pool(g1, f)
        if fp is not None:
            h = torch.concat((h, fp), dim=-1)
        h1 = self.relu(self.fc1_graph(h))
        h1 = self.dropout(h1)
        h1 = self.fc2_graph(h1)
        h1 = self.dropout(h1)

        return h1
    
