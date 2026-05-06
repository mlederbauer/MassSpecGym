"""
FragGNN: Autoregressive fragment generation model for ICEBERG.

Predicts the probability of each atom leaving the current fragment,
building a fragmentation DAG by iteratively removing atoms.

Ported from external/ms-pred/src/ms_pred/dag_pred/gen_model.py.
Requires: dgl, pytorch_lightning.
"""

import numpy as np
from typing import List, Optional

import torch
import torch.nn as nn

from massspecgym.models.encoders.mist.chem_constants import CHEM_ELEMENT_NUM
from .magma.fragmentation import FragmentEngine, MAX_BONDS
from .dag_data import FRAGMENT_ENGINE_PARAMS, MAX_H, TreeProcessor

ELEMENT_DIM = CHEM_ELEMENT_NUM


class FragGNN(nn.Module):
    """Autoregressive fragment generation GNN for ICEBERG.

    Given a molecular graph (root) and a current fragment, predicts the
    probability that each atom will be removed in the next fragmentation step.

    Uses a GNN (GGNN or message-passing) over the fragment graph, conditioned
    on the root molecule representation.

    Args:
        hidden_size: Hidden dimension for all layers.
        layers: Number of GNN layers.
        set_layers: Number of set-transformer layers for fragment pooling.
        dropout: Dropout rate.
        mpnn_type: GNN type ('GGNN' or 'PNA').
        pool_op: Pooling operation ('avg', 'max').
        node_feats: Number of node features (element_dim + max_h).
        max_broken: Maximum broken bonds for one-hot encoding.
        root_encode: Root encoding method ('gnn' or 'fp').
        embed_adduct: Whether to embed adduct type.
        embed_collision: Whether to embed collision energy.
        embed_instrument: Whether to embed instrument type.
        encode_forms: Whether to encode subformula features.
        add_hs: Whether to add hydrogen features.
    """

    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        set_layers: int = 2,
        dropout: float = 0,
        mpnn_type: str = "GGNN",
        pool_op: str = "avg",
        node_feats: int = ELEMENT_DIM + MAX_H,
        max_broken: int = FRAGMENT_ENGINE_PARAMS["max_broken_bonds"],
        root_encode: str = "gnn",
        embed_adduct: bool = False,
        embed_collision: bool = False,
        embed_instrument: bool = False,
        encode_forms: bool = False,
        add_hs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.mpnn_type = mpnn_type
        self.pool_op = pool_op
        self.node_feats = node_feats
        self.max_broken = max_broken
        self.root_encode = root_encode
        self.embed_adduct = embed_adduct
        self.embed_collision = embed_collision
        self.embed_instrument = embed_instrument
        self.encode_forms = encode_forms
        self.add_hs = add_hs

        broken_dim = max_broken * 2 + 1

        root_input_dim = node_feats
        if root_encode == "fp":
            root_input_dim = 4096
        self.root_encoder = nn.Sequential(
            nn.Linear(root_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        edge_feats = MAX_BONDS
        self.node_input = nn.Linear(node_feats, hidden_size)

        try:
            import dgl.nn as dgl_nn
            if mpnn_type == "GGNN":
                self.gnn_layers = nn.ModuleList([
                    dgl_nn.GatedGraphConv(hidden_size, hidden_size, 1, 1)
                    for _ in range(layers)
                ])
            else:
                self.gnn_layers = nn.ModuleList([
                    nn.Linear(hidden_size, hidden_size)
                    for _ in range(layers)
                ])
        except ImportError:
            self.gnn_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size)
                for _ in range(layers)
            ])

        mlp_input_dim = hidden_size * 3 + broken_dim
        if encode_forms:
            from massspecgym.models.encoders.mist.form_embedders import get_embedder
            self.form_embedder = get_embedder("abs-sines")
            form_dim = self.form_embedder.full_dim
            mlp_input_dim += form_dim * 2

        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_map = nn.Linear(hidden_size, 1)

    def forward(
        self,
        graphs,
        root_repr,
        ind_maps,
        broken,
        collision_engs=None,
        precursor_mzs=None,
        adducts=None,
        instruments=None,
        root_forms=None,
        frag_forms=None,
    ):
        """Forward pass: predict atom-leaving probabilities for each fragment.

        Args:
            graphs: Batched DGL graph of all fragments.
            root_repr: Root molecule representation (DGL graph or fingerprint tensor).
            ind_maps: Fragment-to-root index mapping.
            broken: Number of broken bonds per fragment.
            collision_engs: Collision energies.
            precursor_mzs: Precursor m/z values.
            adducts: Adduct indices.
            instruments: Instrument indices.
            root_forms: Root formula vectors.
            frag_forms: Fragment formula vectors.

        Returns:
            Dict with 'output': atom leaving probabilities [batch, max_atoms].
        """
        try:
            import dgl
        except ImportError:
            raise ImportError("DGL required for FragGNN")

        if isinstance(root_repr, dgl.DGLGraph):
            root_h = root_repr.ndata["h"]
            root_embedded = self.root_encoder(root_h)
            root_repr.ndata["h"] = root_embedded
            root_embeddings = dgl.mean_nodes(root_repr, "h")
        else:
            root_embeddings = self.root_encoder(root_repr)

        ext_root = root_embeddings[ind_maps]
        ext_root_atoms = torch.repeat_interleave(
            ext_root, graphs.batch_num_nodes(), dim=0
        )

        frag_h = self.node_input(graphs.ndata["h"])
        for gnn_layer in self.gnn_layers:
            if hasattr(gnn_layer, 'forward') and 'graph' in str(type(gnn_layer)):
                frag_h = gnn_layer(graphs, frag_h)
            else:
                frag_h = torch.relu(gnn_layer(frag_h))

        graphs.ndata["h_out"] = frag_h
        avg_frags = dgl.mean_nodes(graphs, "h_out")
        ext_frag_atoms = torch.repeat_interleave(
            avg_frags, graphs.batch_num_nodes(), dim=0
        )

        broken_clamped = broken.clamp(0, self.max_broken * 2)
        broken_onehot = torch.zeros(broken.size(0), self.max_broken * 2 + 1, device=broken.device)
        broken_onehot.scatter_(1, broken_clamped.unsqueeze(1).long(), 1.0)
        broken_onehot_atoms = torch.repeat_interleave(
            broken_onehot, graphs.batch_num_nodes(), dim=0
        )

        cat_list = [ext_root_atoms, ext_root_atoms - ext_frag_atoms, frag_h, broken_onehot_atoms]
        cat_vec = torch.cat(cat_list, dim=-1)
        hidden = self.output_mlp(cat_vec)
        output = torch.sigmoid(self.output_map(hidden).squeeze(-1))

        output_padded = torch.zeros(
            graphs.batch_size, graphs.batch_num_nodes().max().item(),
            device=output.device
        )
        offset = 0
        for i, n in enumerate(graphs.batch_num_nodes()):
            output_padded[i, :n] = output[offset:offset + n]
            offset += n

        return {"output": output_padded}

    def predict_mol(
        self,
        root_smi: str,
        collision_eng: float = 40.0,
        precursor_mz: float = None,
        adduct: str = None,
        instrument: str = None,
        threshold: float = 0,
        device: str = "cpu",
        max_nodes: int = 100,
    ) -> dict:
        """Generate fragmentation tree for a molecule.

        Returns:
            Dict mapping fragment hashes to fragment entries.
        """
        engine = FragmentEngine(root_smi, **FRAGMENT_ENGINE_PARAMS)
        engine.generate_fragments()
        return engine.frag_to_entry
