"""CLI implementation for MIST + MolForge benchmarking."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

# Allow `python src/mist_molforge/benchmark.py ...` from the repo root.
if __package__ in {None, ""}:
    import sys

    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from mist.data.datasets import SpectraMolDataset, get_paired_loader, get_paired_spectra
from mist.data.featurizers import get_paired_featurizer
from mist.data.splitter import PresetSpectraSplitter
from mist.models.spectra_encoder import SpectraEncoderGrowing

if __package__ in {None, ""}:
    from mist_molforge.chem_utils import (
        compute_morgan_fingerprint,
        compute_tanimoto_similarity,
        normalize_formula,
    )
    from mist_molforge.metrics import aggregate_metrics, compute_metrics_for_one
    from mist_molforge.molforge_adapter import MolForgeDecoder
else:
    from .chem_utils import (
        compute_morgan_fingerprint,
        compute_tanimoto_similarity,
        normalize_formula,
    )
    from .metrics import aggregate_metrics, compute_metrics_for_one
    from .molforge_adapter import MolForgeDecoder

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOLFORGE = {
    "root_dir": "MolForge",
    "checkpoint": "checkpoints/decoder_molforge.pth",
    "src_sp_model": "checkpoints/molforge_sp/combined_morgan4096_vocab_sp.model",
    "trg_sp_model": "checkpoints/molforge_sp/combined_smiles_vocab_sp_morgan4096.model",
    "src_seq_len": 104,
    "trg_seq_len": 130,
    "beam_size": 10,
    "append_src_eos": False,
}


def resolve_project_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark MIST+MolForge on MSG/CANOPUS test sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--molforge-checkpoint",
        type=str,
        default=None,
        help="Override MolForge decoder checkpoint path.",
    )
    parser.add_argument(
        "--molforge-root",
        type=str,
        default=None,
        help="Path to the upstream MolForge checkout or submodule root.",
    )
    parser.add_argument(
        "--molforge-src-sp-model",
        type=str,
        default=None,
        help="Override the MolForge source SentencePiece model path.",
    )
    parser.add_argument(
        "--molforge-trg-sp-model",
        type=str,
        default=None,
        help="Override the MolForge target SentencePiece model path.",
    )
    parser.add_argument(
        "--molforge-append-src-eos",
        action="store_true",
        help="Append EOS to the tokenized MolForge source sequence.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.172],
        help="Fingerprint probability thresholds to evaluate.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mist_molforge_benchmark",
    )
    parser.add_argument("--compute-mces", action="store_true")
    parser.add_argument("--mces-time-limit", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_config_paths(cfg: dict) -> dict:
    cfg = dict(cfg)
    data_cfg = dict(cfg.get("data", {}))
    for key in ("labels_file", "split_file", "spec_folder", "subform_folder"):
        if key in data_cfg:
            data_cfg[key] = resolve_project_path(data_cfg[key])
    cfg["data"] = data_cfg

    mist_cfg = dict(cfg.get("mist_encoder", {}))
    if "checkpoint" in mist_cfg:
        mist_cfg["checkpoint"] = resolve_project_path(mist_cfg["checkpoint"])
    cfg["mist_encoder"] = mist_cfg

    molforge_cfg = {**DEFAULT_MOLFORGE, **dict(cfg.get("molforge", {}))}
    for key in ("root_dir", "checkpoint", "src_sp_model", "trg_sp_model"):
        if key in molforge_cfg:
            molforge_cfg[key] = resolve_project_path(molforge_cfg[key])
    cfg["molforge"] = molforge_cfg
    return cfg


def load_mist_encoder(cfg: dict, device: torch.device) -> torch.nn.Module:
    checkpoint_path = cfg["checkpoint"]
    enc = SpectraEncoderGrowing(
        form_embedder=cfg.get("form_embedder", "pos-cos"),
        output_size=cfg.get("output_size", 4096),
        hidden_size=cfg.get("hidden_size", 512),
        spectra_dropout=cfg.get("spectra_dropout", 0.1),
        peak_attn_layers=cfg.get("peak_attn_layers", 2),
        num_heads=cfg.get("num_heads", 8),
        set_pooling=cfg.get("set_pooling", "cls"),
        refine_layers=cfg.get("refine_layers", 4),
        pairwise_featurization=cfg.get("pairwise_featurization", True),
        embed_instrument=cfg.get("embed_instrument", False),
        inten_transform=cfg.get("inten_transform", "float"),
        magma_modulo=cfg.get("magma_modulo", 2048),
        inten_prob=cfg.get("inten_prob", 0.1),
        remove_prob=cfg.get("remove_prob", 0.5),
        cls_type=cfg.get("cls_type", "ms1"),
        spec_features=cfg.get("spec_features", "peakformula"),
        mol_features=cfg.get("mol_features", "fingerprint"),
        top_layers=cfg.get("top_layers", 1),
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MIST checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if isinstance(state_dict, dict) and any(key.startswith("encoder.") for key in state_dict):
        state_dict = {
            key.replace("encoder.", ""): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
    enc.load_state_dict(state_dict, strict=False)
    enc = enc.to(device)
    enc.eval()
    return enc


def load_spec_data(
    data_cfg: dict,
    encoder_cfg: dict,
    split: str,
    max_spectra: Optional[int],
) -> Tuple[SpectraMolDataset, List]:
    labels_file = data_cfg["labels_file"]
    split_file = data_cfg["split_file"]
    spec_folder = data_cfg["spec_folder"]
    subform_folder = data_cfg["subform_folder"]

    for path, name in [
        (labels_file, "labels_file"),
        (split_file, "split_file"),
        (spec_folder, "spec_folder"),
        (subform_folder, "subform_folder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required {name} not found: {path}")

    spectra_list, mol_list = get_paired_spectra(
        labels_file=labels_file,
        spec_folder=spec_folder,
        prog_bars=True,
        split_file=split_file,
        split_val=split,
    )
    full_dataset = list(zip(spectra_list, mol_list))

    splitter = PresetSpectraSplitter(split_file=split_file)
    _, (train_data, val_data, test_data) = splitter.get_splits(full_dataset)
    split_pairs = {"train": train_data, "val": val_data, "test": test_data}[split]
    if max_spectra is not None:
        split_pairs = split_pairs[: int(max_spectra)]

    featurizer = get_paired_featurizer(
        spec_features=encoder_cfg.get("spec_features", "peakformula"),
        mol_features=encoder_cfg.get("mol_features", "fingerprint"),
        subform_folder=subform_folder,
        fp_names=["morgan4096"],
        magma_modulo=encoder_cfg.get("magma_modulo", 2048),
        cls_type=encoder_cfg.get("cls_type", "ms1"),
        inten_transform=encoder_cfg.get("inten_transform", "float"),
        inten_prob=encoder_cfg.get("inten_prob", 0.1),
        remove_prob=encoder_cfg.get("remove_prob", 0.5),
    )
    return SpectraMolDataset(split_pairs, featurizer), split_pairs


def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def dedup_keep_order(smiles_list: Sequence[Optional[str]]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for smiles in smiles_list:
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        deduped.append(smiles)
    return deduped


def run_one_threshold(
    *,
    threshold: float,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    compute_mces: bool,
    mces_time_limit: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    mist_encoder = load_mist_encoder(cfg["mist_encoder"], device)
    dataset, split_pairs = load_spec_data(
        cfg["data"],
        cfg["mist_encoder"],
        split=cfg.get("evaluation", {}).get("split", "test"),
        max_spectra=cfg.get("evaluation", {}).get("max_spectra"),
    )

    decoder = MolForgeDecoder(
        checkpoint_path=cfg["molforge"]["checkpoint"],
        src_sp_model_path=cfg["molforge"]["src_sp_model"],
        trg_sp_model_path=cfg["molforge"]["trg_sp_model"],
        device=device,
        molforge_root=cfg["molforge"].get("root_dir"),
        src_seq_len=int(cfg["molforge"].get("src_seq_len", 104)),
        trg_seq_len=int(cfg["molforge"].get("trg_seq_len", 130)),
        beam_size=int(cfg["molforge"].get("beam_size", 10)),
        append_src_eos=bool(cfg["molforge"].get("append_src_eos", False)),
    )

    dataloader = get_paired_loader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    started = time.time()
    written = 0
    with pred_path.open("w", encoding="utf-8") as handle:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generate (t={threshold})")):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            with torch.no_grad():
                fp_probs, _ = mist_encoder(batch)
                fp_probs = fp_probs.detach().cpu().numpy()

            for row_index in range(fp_probs.shape[0]):
                global_idx = batch_idx * batch_size + row_index
                if global_idx >= len(split_pairs):
                    break

                spec, mol = split_pairs[global_idx]
                true_smiles = mol.get_smiles()
                true_formula = normalize_formula(
                    rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(true_smiles))
                )

                probs = fp_probs[row_index]
                on_bits = np.where(probs >= float(threshold))[0].astype(int).tolist()
                pred_fp_binary = (probs >= float(threshold)).astype(np.float32)

                pred_smiles_raw, pred_scores = decoder.generate_topk_smiles(on_bits, top_k=10)
                pred_smiles = dedup_keep_order(canonicalize_smiles(s) for s in pred_smiles_raw)[:10]

                gt_fp = compute_morgan_fingerprint(true_smiles, n_bits=4096, radius=2)
                mist_fp_tanimoto = (
                    compute_tanimoto_similarity(gt_fp, pred_fp_binary)
                    if gt_fp is not None
                    else 0.0
                )

                record = {
                    "spec_name": spec.get_spec_name(),
                    "true_smiles": true_smiles,
                    "true_formula": true_formula,
                    "threshold": float(threshold),
                    "num_on_bits": int(len(on_bits)),
                    "on_bits": on_bits,
                    "mist_fp_tanimoto": float(mist_fp_tanimoto),
                    "pred_smiles_top10": pred_smiles,
                    "pred_scores_top10": [float(score) for score in pred_scores[: len(pred_smiles_raw)]],
                }
                handle.write(json.dumps(record) + "\n")
                written += 1

    elapsed = time.time() - started
    records: List[Dict[str, Any]] = []
    with pred_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))

    per_sample_metrics = Parallel(n_jobs=-1)(
        delayed(compute_metrics_for_one)(
            true_smiles=record["true_smiles"],
            pred_smiles=record.get("pred_smiles_top10", []),
            ks=(1, 10),
            compute_mces=bool(compute_mces),
            mces_time_limit_s=int(mces_time_limit),
        )
        for record in tqdm(records, desc=f"Metrics (t={threshold})")
    )
    aggregate = aggregate_metrics(per_sample_metrics)

    mist_fp_tanimoto_vals = [
        record.get("mist_fp_tanimoto")
        for record in records
        if record.get("mist_fp_tanimoto") is not None
    ]
    aggregate["mist_fp_tanimoto_mean"] = (
        float(np.mean(mist_fp_tanimoto_vals)) if mist_fp_tanimoto_vals else 0.0
    )

    out = {
        "meta": {
            "threshold": float(threshold),
            "num_samples": int(written),
            "elapsed_seconds": float(elapsed),
            "device": str(device),
            "molforge_checkpoint": cfg["molforge"]["checkpoint"],
            "mist_checkpoint": cfg["mist_encoder"]["checkpoint"],
            "compute_mces": bool(compute_mces),
            "mces_time_limit": int(mces_time_limit),
            "molforge_root": cfg["molforge"].get("root_dir"),
        },
        "metrics": aggregate,
    }
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = normalize_config_paths(load_yaml(resolve_project_path(args.config)))

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["split"] = args.split
    if args.max_spectra is not None:
        cfg["evaluation"]["max_spectra"] = int(args.max_spectra)

    cfg.setdefault("molforge", {})
    cfg["molforge"]["root_dir"] = resolve_project_path(
        args.molforge_root or cfg["molforge"].get("root_dir")
    )
    cfg["molforge"]["checkpoint"] = resolve_project_path(
        args.molforge_checkpoint or cfg["molforge"].get("checkpoint")
    )
    cfg["molforge"]["src_sp_model"] = resolve_project_path(
        args.molforge_src_sp_model or cfg["molforge"].get("src_sp_model")
    )
    cfg["molforge"]["trg_sp_model"] = resolve_project_path(
        args.molforge_trg_sp_model or cfg["molforge"].get("trg_sp_model")
    )
    cfg["molforge"]["append_src_eos"] = bool(args.molforge_append_src_eos)

    required_molforge_paths = {
        "checkpoint": cfg["molforge"]["checkpoint"],
        "src_sp_model": cfg["molforge"]["src_sp_model"],
        "trg_sp_model": cfg["molforge"]["trg_sp_model"],
    }
    for name, path in required_molforge_paths.items():
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Required MolForge {name} not found: {path}")

    device = torch.device(args.device)
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    for threshold in args.thresholds:
        subdir = out_root / f"thr_{threshold:.3f}".replace(".", "p")
        run_one_threshold(
            threshold=float(threshold),
            cfg=cfg,
            device=device,
            out_dir=subdir,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            compute_mces=bool(args.compute_mces),
            mces_time_limit=int(args.mces_time_limit),
        )


if __name__ == "__main__":
    main()
