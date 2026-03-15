"""
Graph Transformer for DiffMS discrete graph diffusion.

Implements the XEy (node-edge-global) Transformer architecture that simultaneously
updates node features X, edge features E, and global features y (fingerprint).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .diffusion_utils import assert_correctly_masked, PlaceHolder


def masked_softmax(x, mask, **kwargs):
    """Softmax with mask (matching reference DiffMS exactly)."""
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class Xtoy(nn.Module):
    """Aggregate node features to global feature.

    Matches reference DiffMS src/models/layers.py exactly:
    output = Linear(cat([mean, min, max, variance], dim=-1))
    """
    def __init__(self, dx, dy):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        # x_mask: (bs, n, 1) -> expand to (bs, n, dx)
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class Etoy(nn.Module):
    """Aggregate edge features to global feature.

    Matches reference DiffMS src/models/layers.py exactly.
    """
    def __init__(self, d, dy):
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class NodeEdgeBlock(nn.Module):
    """Self-attention block that jointly updates node, edge, and global features."""

    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = dx // n_head
        self.n_head = n_head

        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)

        self.e_add = nn.Linear(de, dx)
        self.e_mul = nn.Linear(de, dx)

        self.y_e_mul = nn.Linear(dy, dx)
        self.y_e_add = nn.Linear(dy, dx)
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)

        self.y_y = nn.Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)

        Q = (self.q(X) * x_mask).reshape(bs, n, self.n_head, self.df).unsqueeze(2)
        K = (self.k(X) * x_mask).reshape(bs, n, self.n_head, self.df).unsqueeze(1)
        Y = Q * K / math.sqrt(self.df)

        E1 = (self.e_mul(E) * e_mask1 * e_mask2).reshape(bs, n, n, self.n_head, self.df)
        E2 = (self.e_add(E) * e_mask1 * e_mask2).reshape(bs, n, n, self.n_head, self.df)
        Y = Y * (E1 + 1) + E2

        newE = Y.flatten(start_dim=3)
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * e_mask1 * e_mask2

        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = (self.v(X) * x_mask).reshape(bs, n, self.n_head, self.df).unsqueeze(1)
        weighted_V = (attn * V).sum(dim=2).flatten(start_dim=2)

        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V
        newX = self.x_out(newX) * x_mask

        new_y = self.y_y(y) + self.x_y(X, x_mask) + self.e_y(E, e_mask1, e_mask2)
        new_y = self.y_out(new_y)

        return newX, newE, new_y


class XEyTransformerLayer(nn.Module):
    """Full transformer layer updating X, E, y with self-attention and FFN."""

    def __init__(self, dx, de, dy, n_head, dim_ffX=2048, dim_ffE=128, dim_ffy=2048,
                 dropout=0.1, layer_norm_eps=1e-5, **kwargs):
        super().__init__()
        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kwargs)

        self.linX1 = nn.Linear(dx, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, dx)
        self.normX1 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.normX2 = nn.LayerNorm(dx, eps=layer_norm_eps)
        self.dropoutX1 = nn.Dropout(dropout)
        self.dropoutX2 = nn.Dropout(dropout)
        self.dropoutX3 = nn.Dropout(dropout)

        self.linE1 = nn.Linear(de, dim_ffE)
        self.linE2 = nn.Linear(dim_ffE, de)
        self.normE1 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.normE2 = nn.LayerNorm(de, eps=layer_norm_eps)
        self.dropoutE1 = nn.Dropout(dropout)
        self.dropoutE2 = nn.Dropout(dropout)
        self.dropoutE3 = nn.Dropout(dropout)

        self.lin_y1 = nn.Linear(dy, dim_ffy)
        self.lin_y2 = nn.Linear(dim_ffy, dy)
        self.norm_y1 = nn.LayerNorm(dy, eps=layer_norm_eps)
        self.norm_y2 = nn.LayerNorm(dy, eps=layer_norm_eps)
        self.dropout_y1 = nn.Dropout(dropout)
        self.dropout_y2 = nn.Dropout(dropout)
        self.dropout_y3 = nn.Dropout(dropout)

    def forward(self, X, E, y, node_mask):
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        X = self.normX1(X + self.dropoutX1(newX))
        E = self.normE1(E + self.dropoutE1(newE))
        y = self.norm_y1(y + self.dropout_y1(new_y))

        X = self.normX2(X + self.dropoutX3(self.linX2(self.dropoutX2(F.relu(self.linX1(X))))))
        E = self.normE2(E + self.dropoutE3(self.linE2(self.dropoutE2(F.relu(self.linE1(E))))))
        y = self.norm_y2(y + self.dropout_y3(self.lin_y2(self.dropout_y2(F.relu(self.lin_y1(y))))))

        return X, E, y


class GraphTransformer(nn.Module):
    """Graph Transformer backbone for DiffMS.

    Processes noisy graph features (node types, bond types, global fingerprint)
    through multiple XEyTransformerLayers and projects to output dimensions.

    Args:
        n_layers: Number of transformer layers.
        input_dims: Dict with X, E, y input dimensions.
        hidden_mlp_dims: Dict with X, E, y hidden MLP dimensions.
        hidden_dims: Dict with X, E, y hidden dimensions.
        output_dims: Dict with X, E, y output dimensions.
    """

    def __init__(self, n_layers, input_dims, hidden_mlp_dims, hidden_dims, output_dims,
                 act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU()):
        super().__init__()
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]), act_fn_in,
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]), act_fn_in,
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]), act_fn_in,
        )

        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(
                dx=hidden_dims["dx"], de=hidden_dims["de"], dy=hidden_dims["dy"],
                n_head=hidden_dims["n_head"],
                dim_ffX=hidden_dims["dim_ffX"], dim_ffE=hidden_dims["dim_ffE"],
            )
            for _ in range(n_layers)
        ])

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], self.out_dim_X),
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], self.out_dim_E),
        )
        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], self.out_dim_y),
        )

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        X = self.mlp_in_X(X)
        E = self.mlp_in_E(E)
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        return PlaceHolder(X=X, E=E, y=y)
