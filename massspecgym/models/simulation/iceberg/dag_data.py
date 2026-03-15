"""
DAG data structures and featurization for ICEBERG.

Provides TreeProcessor for converting fragmentation trees into featurized
DGL graphs, and dataset classes for training/inference.

Ported from external/ms-pred/src/ms_pred/dag_pred/dag_data.py.
Requires: dgl, torch.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from massspecgym.models.encoders.mist.chem_constants import (
    VALID_ELEMENTS,
    CHEM_ELEMENT_NUM,
    element_to_ind,
    ELEMENT_VECTORS,
    VALID_MONO_MASSES,
    ELEMENT_TO_MASS,
    formula_to_dense,
    vec_to_formula,
)
from .magma.fragmentation import FragmentEngine, MAX_BONDS, create_new_ids

logger = logging.getLogger(__name__)

FRAGMENT_ENGINE_PARAMS = {"max_broken_bonds": 6, "max_tree_depth": 3}

MAX_H = 20
ELEMENT_DIM = CHEM_ELEMENT_NUM
NUM_ATOM_GROUPS = 8


class TreeProcessor:
    """Processes fragmentation trees into featurized representations for ICEBERG.

    Converts a FragmentEngine's frag_to_entry into DGL graphs with node features
    (element type + hydrogen count), edge features (bond type), and tree structure.

    Args:
        pe_embed_k: Positional embedding dimension (0 to disable).
        root_encode: Root encoding method ('gnn' or 'fp').
        binned_targs: Whether to use binned intensity targets.
        add_hs: Whether to add hydrogen atoms explicitly.
        embed_elem_group: Whether to embed element group features.
    """

    def __init__(
        self,
        pe_embed_k: int = 10,
        root_encode: str = "gnn",
        binned_targs: bool = False,
        add_hs: bool = False,
        embed_elem_group: bool = False,
    ):
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.binned_targs = binned_targs
        self.add_hs = add_hs
        self.embed_elem_group = embed_elem_group
        self.bins = np.linspace(0, 1500, 15000)

    def get_node_feats(self) -> int:
        """Return the number of node features."""
        base = ELEMENT_DIM + MAX_H
        if self.embed_elem_group:
            base += NUM_ATOM_GROUPS
        return base

    def get_frag_info(self, frag: int, engine: FragmentEngine) -> dict:
        """Get atom indices and formula for a fragment."""
        keep_atoms, keep_symbols = engine.get_present_atoms(frag)
        old_to_new = {old: new for new, old in enumerate(keep_atoms)}
        new_to_old = {new: old for old, new in old_to_new.items()}
        form = engine.formula_from_kept_inds(np.array(keep_atoms))
        return {"new_to_old": new_to_old, "old_to_new": old_to_new, "form": form}

    def featurize_frag(self, frag: int, engine: FragmentEngine, add_random_walk: bool = False):
        """Featurize a single fragment as a graph.

        Returns dict with node features (atom types + H counts) and edge info.
        """
        try:
            import dgl
        except ImportError:
            raise ImportError("DGL is required for ICEBERG. Install with: pip install dgl")

        keep_atoms, keep_symbols = engine.get_present_atoms(frag)
        bond_types_list, bond_inds_list = engine.get_present_edges(frag)

        if len(keep_atoms) == 0:
            return None

        old_to_new = {old: new for new, old in enumerate(keep_atoms)}
        num_nodes = len(keep_atoms)

        atom_symbols = [engine.atom_symbols[i] for i in keep_atoms]
        h_counts = engine.atom_hs[np.array(keep_atoms)]

        node_feats = self._build_node_feats(atom_symbols, h_counts)

        src, dst, edge_feats = [], [], []
        for btype, (a1, a2) in zip(bond_types_list, bond_inds_list):
            if a1 in old_to_new and a2 in old_to_new:
                src.append(old_to_new[a1])
                dst.append(old_to_new[a2])
                edge_feats.append(btype)
                src.append(old_to_new[a2])
                dst.append(old_to_new[a1])
                edge_feats.append(btype)

        if len(src) == 0:
            g = dgl.graph(([], []), num_nodes=num_nodes)
        else:
            g = dgl.graph((src, dst), num_nodes=num_nodes)

        g.ndata["h"] = torch.FloatTensor(node_feats)
        if len(edge_feats) > 0:
            ef = torch.zeros(len(edge_feats), MAX_BONDS)
            for i, bt in enumerate(edge_feats):
                if bt < MAX_BONDS:
                    ef[i, bt] = 1.0
            g.edata["e"] = ef
        else:
            g.edata["e"] = torch.zeros(0, MAX_BONDS)

        return g

    def _build_node_feats(self, atom_symbols: List[str], h_counts: np.ndarray) -> np.ndarray:
        """Build node feature matrix: one-hot element + one-hot H count."""
        n = len(atom_symbols)
        feats = np.zeros((n, ELEMENT_DIM + MAX_H))
        for i, sym in enumerate(atom_symbols):
            if sym in element_to_ind:
                feats[i, element_to_ind[sym]] = 1.0
            h = min(int(h_counts[i]), MAX_H - 1)
            feats[i, ELEMENT_DIM + h] = 1.0
        return feats

    def process_tree_gen(self, tree: dict, convert_to_dgl: bool = True):
        """Process a fragmentation tree for FragGNN training/inference."""
        return self._process_tree(tree, include_targets=True, convert_to_dgl=convert_to_dgl)

    def process_tree_inten(self, tree: dict, convert_to_dgl: bool = True):
        """Process a fragmentation tree for IntenGNN training."""
        return self._process_tree(tree, include_targets=True, convert_to_dgl=convert_to_dgl)

    def process_tree_inten_pred(self, tree: dict, convert_to_dgl: bool = True):
        """Process a fragmentation tree for IntenGNN prediction (no targets)."""
        return self._process_tree(tree, include_targets=False, convert_to_dgl=convert_to_dgl)

    def _process_tree(self, tree: dict, include_targets: bool = True, convert_to_dgl: bool = True):
        """Core tree processing: build graph features from frag_to_entry."""
        engine = tree.get("engine")
        frag_to_entry = tree.get("frag_to_entry", {})

        if engine is None or len(frag_to_entry) == 0:
            return None

        frag_to_id, id_to_frag = create_new_ids(frag_to_entry)
        frag_hashes = list(frag_to_id.keys())
        n_frags = len(frag_hashes)

        graphs = []
        ind_maps = []
        broken_list = []
        form_list = []
        mass_list = []

        root_hash = None
        for h, entry in frag_to_entry.items():
            if entry["tree_depth"] == 0:
                root_hash = h
                break

        root_frag = frag_to_entry[root_hash]["frag"] if root_hash else engine.get_root_frag()

        if convert_to_dgl:
            root_graph = self.featurize_frag(root_frag, engine)
        else:
            root_graph = None

        for h in frag_hashes:
            entry = frag_to_entry[h]
            frag = entry["frag"]
            if convert_to_dgl:
                g = self.featurize_frag(frag, engine)
                if g is not None:
                    graphs.append(g)
            ind_maps.append(frag_to_id[h])
            broken_list.append(entry.get("max_broken", 0))
            form_list.append(entry.get("form", ""))
            mass_list.append(entry.get("base_mass", 0.0))

        result = {
            "root_repr": root_graph,
            "n_frags": n_frags,
            "ind_maps": np.array(ind_maps),
            "broken": np.array(broken_list),
            "forms": form_list,
            "masses": np.array(mass_list),
            "frag_to_entry": frag_to_entry,
            "frag_to_id": frag_to_id,
            "engine": engine,
        }

        if convert_to_dgl and graphs:
            import dgl
            result["graphs"] = dgl.batch(graphs)
        else:
            result["graphs"] = None

        return result
