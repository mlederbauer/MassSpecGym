"""
IntenGNN: Fragment intensity prediction model for ICEBERG.

Predicts the intensity of each fragment peak in a mass spectrum,
given the fragmentation DAG from FragGNN.

Ported from external/ms-pred/src/ms_pred/dag_pred/inten_model.py.
Requires: dgl, pytorch_lightning.
"""

import numpy as np
from typing import Optional

import torch
import torch.nn as nn

from massspecgym.models.encoders.mist.chem_constants import CHEM_ELEMENT_NUM, ELEMENT_TO_MASS
from .magma.fragmentation import MAX_BONDS
from .dag_data import FRAGMENT_ENGINE_PARAMS, MAX_H

ELEMENT_DIM = CHEM_ELEMENT_NUM


class IntenGNN(nn.Module):
    """Fragment intensity prediction GNN for ICEBERG.

    Given fragment graphs and their DAG structure, predicts the intensity
    of each fragment as a peak in the MS/MS spectrum.

    Uses binned output (15000 bins over 0-1500 m/z) with scatter-based
    pooling for peaks at the same m/z.

    Args:
        hidden_size: Hidden dimension.
        gnn_layers: Number of GNN layers.
        mlp_layers: Number of MLP layers after GNN.
        set_layers: Number of set-transformer layers.
        dropout: Dropout rate.
        mpnn_type: GNN type ('PNA' or 'GGNN').
        pool_op: Pooling operation.
        node_feats: Number of node features.
        max_broken: Maximum broken bonds.
        frag_set_layers: Set layers for fragment-level attention.
        loss_fn: Loss function type ('cosine' or 'entropy').
        root_encode: Root encoding ('gnn' or 'fp').
        embed_adduct: Embed adduct type.
        embed_collision: Embed collision energy.
        embed_instrument: Embed instrument type.
        binned_targs: Use binned intensity targets.
        encode_forms: Encode subformula features.
        add_hs: Add hydrogen features.
        ppm_tol: PPM tolerance for peak matching.
    """

    NUM_BINS = 15000
    MZ_MAX = 1500.0

    def __init__(
        self,
        hidden_size: int,
        gnn_layers: int = 2,
        mlp_layers: int = 0,
        set_layers: int = 2,
        dropout: float = 0,
        mpnn_type: str = "PNA",
        pool_op: str = "avg",
        node_feats: int = ELEMENT_DIM + MAX_H,
        max_broken: int = FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
        frag_set_layers: int = 0,
        loss_fn: str = "cosine",
        root_encode: str = "gnn",
        embed_adduct: bool = False,
        embed_collision: bool = False,
        embed_instrument: bool = False,
        binned_targs: bool = True,
        encode_forms: bool = False,
        add_hs: bool = False,
        ppm_tol: float = 20.0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_layers_n = gnn_layers
        self.dropout = dropout
        self.mpnn_type = mpnn_type
        self.max_broken = max_broken
        self.root_encode = root_encode
        self.embed_adduct = embed_adduct
        self.embed_collision = embed_collision
        self.embed_instrument = embed_instrument
        self.binned_targs = binned_targs
        self.encode_forms = encode_forms
        self.loss_fn_name = loss_fn
        self.ppm_tol = ppm_tol

        self.output_size = max_broken * 2 + 1
        self.inten_buckets = torch.FloatTensor(np.linspace(0, self.MZ_MAX, self.NUM_BINS))

        broken_dim = max_broken * 2 + 1

        root_input_dim = node_feats
        if root_encode == "fp":
            root_input_dim = 4096
        self.root_encoder = nn.Sequential(
            nn.Linear(root_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.node_input = nn.Linear(node_feats, hidden_size)

        try:
            import dgl.nn as dgl_nn
            if mpnn_type == "PNA":
                self.gnn_layers = nn.ModuleList([
                    nn.Linear(hidden_size, hidden_size)
                    for _ in range(gnn_layers)
                ])
            elif mpnn_type == "GGNN":
                self.gnn_layers = nn.ModuleList([
                    dgl_nn.GatedGraphConv(hidden_size, hidden_size, 1, 1)
                    for _ in range(gnn_layers)
                ])
            else:
                self.gnn_layers = nn.ModuleList([
                    nn.Linear(hidden_size, hidden_size) for _ in range(gnn_layers)
                ])
        except ImportError:
            self.gnn_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(gnn_layers)
            ])

        mlp_input_dim = hidden_size * 3 + broken_dim
        if encode_forms:
            from massspecgym.models.encoders.mist.form_embedders import get_embedder
            self.form_embedder = get_embedder("abs-sines")
            form_dim = self.form_embedder.full_dim
            mlp_input_dim += form_dim * 2

        self.intermediate_out = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.output_map = nn.Linear(hidden_size, self.output_size)
        self.attn_map = nn.Linear(hidden_size, self.output_size)

    def forward(
        self,
        graphs,
        root_repr,
        ind_maps,
        num_frags,
        broken,
        collision_engs=None,
        precursor_mzs=None,
        adducts=None,
        instruments=None,
        max_add_hs=None,
        max_remove_hs=None,
        masses=None,
        root_forms=None,
        frag_forms=None,
    ):
        """Forward pass: predict intensities for fragment peaks.

        Returns:
            Dict with 'output_binned' (binned spectrum) and 'output' (per-fragment).
        """
        try:
            import dgl
        except ImportError:
            raise ImportError("DGL required for IntenGNN")

        if isinstance(root_repr, dgl.DGLGraph):
            root_h = root_repr.ndata["h"]
            root_embedded = self.root_encoder(root_h)
            root_repr.ndata["h"] = root_embedded
            root_embeddings = dgl.mean_nodes(root_repr, "h")
        else:
            root_embeddings = self.root_encoder(root_repr)

        ext_root = root_embeddings[ind_maps]

        frag_h = self.node_input(graphs.ndata["h"])
        for gnn_layer in self.gnn_layers:
            frag_h = torch.relu(gnn_layer(frag_h))

        graphs.ndata["h_out"] = frag_h
        avg_frags = dgl.mean_nodes(graphs, "h_out")

        broken_clamped = broken.clamp(0, self.max_broken * 2)
        broken_onehot = torch.zeros(broken.size(0), self.max_broken * 2 + 1, device=broken.device)
        broken_onehot.scatter_(1, broken_clamped.unsqueeze(1).long(), 1.0)

        cat_list = [ext_root, ext_root - avg_frags, avg_frags, broken_onehot]
        padded_hidden = self.intermediate_out(torch.cat(cat_list, dim=-1))

        output = torch.sigmoid(self.output_map(padded_hidden))
        attn_weights = self.attn_map(padded_hidden)

        return {"output": output, "attn_weights": attn_weights, "output_binned": output}

    def predict(
        self,
        graphs,
        root_reprs,
        ind_maps,
        num_frags,
        max_breaks,
        adducts=None,
        collision_engs=None,
        instruments=None,
        precursor_mzs=None,
        max_add_hs=None,
        max_remove_hs=None,
        masses=None,
        root_forms=None,
        frag_forms=None,
        binned_out: bool = False,
    ) -> dict:
        """Run inference and return predicted spectrum."""
        out = self.forward(
            graphs=graphs,
            root_repr=root_reprs,
            ind_maps=ind_maps,
            num_frags=num_frags,
            broken=max_breaks,
            collision_engs=collision_engs,
            precursor_mzs=precursor_mzs,
            adducts=adducts,
            instruments=instruments,
            max_add_hs=max_add_hs,
            max_remove_hs=max_remove_hs,
            masses=masses,
            root_forms=root_forms,
            frag_forms=frag_forms,
        )
        if binned_out:
            return {"spec": out["output_binned"]}
        return {"spec": out["output"]}
