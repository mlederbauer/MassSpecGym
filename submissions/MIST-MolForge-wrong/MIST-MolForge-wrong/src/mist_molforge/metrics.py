"""Elucidation metrics for MIST + MolForge benchmarking."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

try:
    import pulp  # type: ignore
    from myopic_mces import MCES  # type: ignore
except Exception:  # pragma: no cover
    pulp = None
    MCES = None


def remove_stereo_from_mol(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    Chem.RemoveStereochemistry(rw)
    return rw.GetMol()


def mol_to_inchi_no_stereo(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    mol_ns = remove_stereo_from_mol(mol)
    if mol_ns is None:
        return None
    try:
        return Chem.MolToInchi(mol_ns)
    except Exception:
        return None


def canonical_smiles_no_stereo_from_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        mol = remove_stereo_from_mol(mol)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _rdkit_topological_fp(mol: Chem.Mol):
    fpgen = AllChem.GetRDKitFPGenerator()
    return fpgen.GetFingerprint(mol)


def _morgan_bitvect(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _cosine_similarity(fp1, fp2) -> float:
    try:
        return float(DataStructs.CosineSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _tanimoto_similarity(fp1, fp2) -> float:
    try:
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _compute_mces_distance(
    true_smiles_no_stereo: str,
    pred_smiles_no_stereo: str,
    time_limit_s: int = 120,
    threshold: int = 30,
) -> float:
    if MCES is None or pulp is None:
        raise ImportError(
            "MCES requires packages `myopic-mces` and `pulp` in the active env."
        )
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s)
    return float(
        MCES(
            true_smiles_no_stereo,
            pred_smiles_no_stereo,
            solver=solver,
            threshold=threshold,
            always_stronger_bound=False,
            solver_options=dict(msg=0, timeLimit=time_limit_s),
        )[1]
    )


def compute_metrics_for_one(
    true_smiles: str,
    pred_smiles: Sequence[Optional[str]],
    ks: Sequence[int] = (1, 10),
    compute_mces: bool = True,
    mces_time_limit_s: int = 120,
    mces_threshold: int = 30,
) -> Dict[str, float]:
    """Compute top-k elucidation metrics for a single example."""
    ks = sorted(set(int(k) for k in ks))
    max_k = max(ks) if ks else 10

    true_smi_ns = canonical_smiles_no_stereo_from_smiles(true_smiles)
    if true_smi_ns is None:
        return defaultdict(float)

    true_mol = Chem.MolFromSmiles(true_smi_ns)
    if true_mol is None:
        return defaultdict(float)

    true_inchi_ns = mol_to_inchi_no_stereo(true_mol)
    true_fp = _morgan_bitvect(true_mol, n_bits=2048, radius=2)
    true_rdkit_fp = _rdkit_topological_fp(true_mol)
    true_num_bonds = int(true_mol.GetNumBonds())

    rdkit_tan: List[float] = []
    morgan_tan: List[float] = []
    cosine: List[float] = []
    mces: List[float] = []
    pred_inchi_ns: List[Optional[str]] = []

    for pred in list(pred_smiles)[:max_k]:
        if not pred:
            rdkit_tan.append(0.0)
            morgan_tan.append(0.0)
            cosine.append(0.0)
            mces.append(float(2 * true_num_bonds))
            pred_inchi_ns.append(None)
            continue

        pred_ns = canonical_smiles_no_stereo_from_smiles(pred)
        if pred_ns is None:
            rdkit_tan.append(0.0)
            morgan_tan.append(0.0)
            cosine.append(0.0)
            mces.append(float(2 * true_num_bonds))
            pred_inchi_ns.append(None)
            continue

        pred_mol = Chem.MolFromSmiles(pred_ns)
        if pred_mol is None:
            rdkit_tan.append(0.0)
            morgan_tan.append(0.0)
            cosine.append(0.0)
            mces.append(float(2 * true_num_bonds))
            pred_inchi_ns.append(None)
            continue

        pred_inchi_ns.append(mol_to_inchi_no_stereo(pred_mol))

        try:
            pred_fp = _morgan_bitvect(pred_mol, n_bits=2048, radius=2)
            pred_rdkit_fp = _rdkit_topological_fp(pred_mol)
            morgan_tan.append(_tanimoto_similarity(true_fp, pred_fp))
            rdkit_tan.append(_tanimoto_similarity(true_rdkit_fp, pred_rdkit_fp))
            cosine.append(_cosine_similarity(true_fp, pred_fp))
        except Exception:
            morgan_tan.append(0.0)
            rdkit_tan.append(0.0)
            cosine.append(0.0)

        if compute_mces:
            try:
                mces.append(
                    _compute_mces_distance(
                        true_smiles_no_stereo=true_smi_ns,
                        pred_smiles_no_stereo=pred_ns,
                        time_limit_s=mces_time_limit_s,
                        threshold=mces_threshold,
                    )
                )
            except Exception:
                mces.append(float(true_num_bonds + int(pred_mol.GetNumBonds())))
        else:
            mces.append(float(true_num_bonds + int(pred_mol.GetNumBonds())))

    prefix_max_rdkit = [0.0]
    prefix_max_morgan = [0.0]
    prefix_max_cos = [0.0]
    prefix_min_mces = [100.0]
    for index in range(len(rdkit_tan)):
        prefix_max_rdkit.append(max(prefix_max_rdkit[-1], rdkit_tan[index]))
        prefix_max_morgan.append(max(prefix_max_morgan[-1], morgan_tan[index]))
        prefix_max_cos.append(max(prefix_max_cos[-1], cosine[index]))
        prefix_min_mces.append(min(prefix_min_mces[-1], mces[index]))

    earliest_idx = -1
    if true_inchi_ns is not None:
        for index, pred_inchi in enumerate(pred_inchi_ns):
            if pred_inchi is not None and pred_inchi == true_inchi_ns:
                earliest_idx = index
                break

    out = defaultdict(float)
    for k in ks:
        idx = min(k, len(rdkit_tan))
        out[f"acc@{k}"] = 1.0 if (earliest_idx != -1 and earliest_idx < k) else 0.0
        out[f"mces@{k}"] = float(prefix_min_mces[idx])
        out[f"tanimoto@{k}"] = float(prefix_max_morgan[idx])
        out[f"rdkit_tanimoto@{k}"] = float(prefix_max_rdkit[idx])
        out[f"cosine@{k}"] = float(prefix_max_cos[idx])
        out[f"close_match@{k}"] = 1.0 if prefix_max_rdkit[idx] >= 0.675 else 0.0
        out[f"meaningful_match@{k}"] = 1.0 if prefix_max_rdkit[idx] >= 0.4 else 0.0
    return out


def aggregate_metrics(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average per-example metrics across a full benchmark run."""
    if not metrics:
        return {}
    aggregate: defaultdict[str, float] = defaultdict(float)
    for metric in metrics:
        for key, value in metric.items():
            aggregate[key] += float(value)
    count = float(len(metrics))
    return {key: value / count for key, value in aggregate.items()}
