"""Thin adapter that delegates decoding to the upstream MolForge implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import sys

import torch
import sentencepiece as spm

PAD_ID = 0
EOS_ID = 2


def _strip_module_prefix(state_dict: dict) -> dict:
    stripped = {}
    for key, value in state_dict.items():
        stripped[key[len("module.") :]] = value if key.startswith("module.") else value
    return stripped


def load_molforge_state_dict(checkpoint_path: str) -> dict:
    """Load a trusted MolForge checkpoint and return the model state dict."""
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "model_state_dict" not in obj:
        raise ValueError(
            f"Unexpected MolForge checkpoint format: {type(obj)} "
            f"keys={getattr(obj, 'keys', lambda: [])()}"
        )
    return _strip_module_prefix(obj["model_state_dict"])


@dataclass
class MolForgeModules:
    transformer_cls: type
    decoder_module: object


def _resolve_molforge_root(molforge_root: Optional[str]) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    candidates = []
    if molforge_root:
        path = Path(molforge_root)
        candidates.append(path if path.is_absolute() else project_root / path)
    candidates.extend(
        [
            project_root / "MolForge",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "MolForge repository not found. Expected it under `MolForge`, or pass "
        "`molforge_root` explicitly."
    )


def _load_upstream_modules(molforge_root: Path) -> MolForgeModules:
    root_str = str(molforge_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from MolForge.transformer import Transformer  # type: ignore
    import MolForge.decoder as decoder_module  # type: ignore

    return MolForgeModules(transformer_cls=Transformer, decoder_module=decoder_module)


class MolForgeDecoder:
    """Benchmark-friendly wrapper around upstream MolForge Transformer + beam search."""

    def __init__(
        self,
        checkpoint_path: str,
        src_sp_model_path: str,
        trg_sp_model_path: str,
        device: torch.device,
        molforge_root: Optional[str] = None,
        src_seq_len: int = 104,
        trg_seq_len: int = 130,
        beam_size: int = 10,
        append_src_eos: bool = False,
    ):
        self.device = device
        self.src_seq_len = int(src_seq_len)
        self.trg_seq_len = int(trg_seq_len)
        self.beam_size = int(beam_size)
        self.append_src_eos = bool(append_src_eos)
        self.molforge_root = _resolve_molforge_root(molforge_root)

        self.modules = _load_upstream_modules(self.molforge_root)
        self.decoder_module = self.modules.decoder_module
        self.decoder_module.beam_size = self.beam_size
        self.decoder_module.trg_seq_len = self.trg_seq_len

        self.src_sp = spm.SentencePieceProcessor()
        self.trg_sp = spm.SentencePieceProcessor()
        if not self.src_sp.Load(src_sp_model_path):
            raise FileNotFoundError(f"Unable to load MolForge source SP model: {src_sp_model_path}")
        if not self.trg_sp.Load(trg_sp_model_path):
            raise FileNotFoundError(f"Unable to load MolForge target SP model: {trg_sp_model_path}")

        args = SimpleNamespace(
            src_seq_len=self.src_seq_len,
            trg_seq_len=self.trg_seq_len,
            rank=device,
        )
        self.model = self.modules.transformer_cls(
            src_vocab_size=self.src_sp.GetPieceSize(),
            trg_vocab_size=self.trg_sp.GetPieceSize(),
            args=args,
        ).to(self.device)

        state_dict = load_molforge_state_dict(checkpoint_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(
                f"Unexpected keys when loading MolForge checkpoint: {unexpected[:5]} ..."
            )
        self.model.eval()
        self.missing_keys = missing

    def _encode_source_bits(self, on_bits: Sequence[int]) -> torch.LongTensor:
        sentence = " ".join(str(int(bit)) for bit in sorted(int(bit) for bit in on_bits))
        token_ids = self.src_sp.EncodeAsIds(sentence)
        if self.append_src_eos:
            token_ids = token_ids + [EOS_ID]
        token_ids = token_ids[: self.src_seq_len]
        if len(token_ids) < self.src_seq_len:
            token_ids = token_ids + [PAD_ID] * (self.src_seq_len - len(token_ids))
        return torch.LongTensor(token_ids)

    @torch.no_grad()
    def generate_topk_smiles(
        self,
        on_bits: Sequence[int],
        top_k: int = 10,
    ) -> Tuple[List[str], List[float]]:
        self.decoder_module.beam_size = max(1, int(top_k))

        src_ids = self._encode_source_bits(on_bits).unsqueeze(0).to(self.device)
        encoder_mask = (src_ids != PAD_ID).unsqueeze(1).to(self.device)

        src_embedded = self.model.src_embedding(src_ids)
        src_positioned = self.model.src_positional_encoder(src_embedded)
        encoder_output = self.model.encoder(src_positioned, encoder_mask)

        candidates, scores = self.decoder_module.beam_search(
            self.model,
            encoder_output,
            encoder_mask,
            self.trg_sp,
            self.device,
            return_candidates=True,
        )
        smiles = [str(candidate).replace(" ", "").strip() for candidate in candidates[:top_k]]
        return smiles, [float(score) for score in scores[:top_k]]
