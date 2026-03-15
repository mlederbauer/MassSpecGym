"""
MolForge decoder: Encoder-Decoder Transformer for fingerprint-to-SMILES generation.

Ported from Neo et al., "One Small Step with Fingerprints, One Giant Leap for
De Novo Molecule Generation from Mass Spectra", 2025.

The model treats a molecular fingerprint as a sequence of on-bit indices,
encodes them through a transformer encoder, then autoregressively decodes
SMILES (or SELFIES) token sequences.

Architecture (matching MolForge paper):
- Encoder: 6 layers, 512d, 8 heads, FFN 2048
- Decoder: 6 layers, 512d, 8 heads, FFN 2048
- Source: fingerprint on-bit indices -> SentencePiece tokens -> encoder
- Target: SMILES/SELFIES -> SentencePiece tokens -> decoder
"""

import math
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from massspecgym.models.base import Stage
from massspecgym.models.de_novo.fp2mol.base import FP2MolDeNovoModel
from .decoder_search import greedy_search, beam_search


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MolForgeDecoder(FP2MolDeNovoModel):
    """MolForge encoder-decoder transformer for fingerprint-to-SMILES generation.

    The model encodes fingerprint on-bit indices as a source sequence and
    autoregressively generates SMILES/SELFIES as the target sequence.

    Fingerprints are represented as space-separated on-bit indices
    (e.g., "1 80 94 114 237") and tokenized with a source vocabulary.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        dim_feedforward: FFN intermediate dimension.
        dropout: Dropout rate.
        src_vocab_size: Source (fingerprint) vocabulary size.
        trg_vocab_size: Target (SMILES/SELFIES) vocabulary size.
        src_seq_len: Maximum source sequence length.
        trg_seq_len: Maximum target sequence length.
        pad_id: Padding token ID.
        sos_id: Start-of-sequence token ID.
        eos_id: End-of-sequence token ID.
        beam_size: Beam size for beam search decoding.
        decode_method: Decoding method ('greedy' or 'beam').
        fingerprint_bits: Number of fingerprint bits.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        src_vocab_size: int = 6000,
        trg_vocab_size: int = 109,
        src_seq_len: int = 104,
        trg_seq_len: int = 130,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        beam_size: int = 10,
        decode_method: str = "greedy",
        fingerprint_bits: int = 2048,
        *args,
        **kwargs,
    ):
        super().__init__(
            fingerprint_bits=fingerprint_bits,
            use_formula=False,
            *args, **kwargs,
        )

        self.d_model = d_model
        self.src_seq_len = src_seq_len
        self.trg_seq_len = trg_seq_len
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.decode_method = decode_method
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.src_positional_encoder = PositionalEncoding(d_model, src_seq_len, dropout)
        self.trg_positional_encoder = PositionalEncoding(d_model, trg_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_linear = nn.Linear(d_model, trg_vocab_size)
        self.criterion = nn.NLLLoss(ignore_index=pad_id)

    def _fp_to_onbit_indices(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """Convert binary fingerprint to a padded sequence of on-bit indices.

        Args:
            fingerprint: Binary FP tensor [batch, fp_bits].

        Returns:
            LongTensor of on-bit indices [batch, src_seq_len], padded with pad_id.
        """
        batch_size = fingerprint.shape[0]
        device = fingerprint.device
        result = torch.full((batch_size, self.src_seq_len), self.pad_id, dtype=torch.long, device=device)

        for i in range(batch_size):
            on_bits = torch.nonzero(fingerprint[i] > 0.5, as_tuple=True)[0]
            n = min(len(on_bits), self.src_seq_len - 1)
            result[i, :n] = on_bits[:n] + 3  # offset by special tokens (pad=0, sos=1, eos=2)
            result[i, n] = self.eos_id

        return result

    def encode_source(self, src_input: torch.Tensor) -> tuple:
        """Encode source (fingerprint indices) through the transformer encoder.

        Returns:
            Tuple of (encoder output [B, src_len, d_model], encoder mask [B, 1, src_len]).
        """
        e_mask = (src_input != self.pad_id).unsqueeze(1).float()
        src_emb = self.src_positional_encoder(self.src_embedding(src_input))
        e_output = self.encoder(src_emb, src_key_padding_mask=(src_input == self.pad_id))
        return e_output, e_mask

    def compute_decoder_loss(self, batch: dict) -> torch.Tensor:
        """Compute NLL loss with teacher forcing."""
        fingerprint = batch["fingerprint"]
        mol_repr = batch.get("mol_repr", batch.get("mol"))

        src_input = self._fp_to_onbit_indices(fingerprint)
        e_output, e_mask = self.encode_source(src_input)

        if isinstance(mol_repr, (list, tuple)):
            trg_ids = self._tokenize_targets(mol_repr)
        else:
            trg_ids = mol_repr

        trg_ids = trg_ids.to(self.device)
        trg_input = trg_ids[:, :-1]
        trg_output = trg_ids[:, 1:]

        d_mask = nn.Transformer.generate_square_subsequent_mask(trg_input.size(1)).to(self.device)
        trg_emb = self.trg_positional_encoder(self.trg_embedding(trg_input))
        d_output = self.decoder(
            trg_emb, e_output, tgt_mask=d_mask,
            memory_key_padding_mask=(src_input == self.pad_id),
        )
        output = F.log_softmax(self.output_linear(d_output), dim=-1)
        loss = self.criterion(output.reshape(-1, self.trg_vocab_size), trg_output.reshape(-1))
        return loss

    def _tokenize_targets(self, smiles_list: list) -> torch.Tensor:
        """Simple character-level tokenization for SMILES (placeholder).

        In production, this should use SentencePiece. For now, uses a simple
        character-to-ID mapping as a fallback.
        """
        batch_size = len(smiles_list)
        max_len = min(max(len(s) for s in smiles_list) + 2, self.trg_seq_len)
        result = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)

        for i, smi in enumerate(smiles_list):
            result[i, 0] = self.sos_id
            for j, ch in enumerate(smi[:max_len - 2]):
                result[i, j + 1] = ord(ch) % (self.trg_vocab_size - 3) + 3
            end_pos = min(len(smi) + 1, max_len - 1)
            result[i, end_pos] = self.eos_id

        return result

    @torch.no_grad()
    def decode_from_fingerprint(
        self,
        fingerprint: torch.Tensor,
        formula=None,
        num_samples: int = 1,
    ) -> list:
        """Generate SMILES from fingerprint via greedy or beam search."""
        self.eval()
        batch_size = fingerprint.shape[0]
        all_preds = []

        for b_idx in range(batch_size):
            fp_single = fingerprint[b_idx:b_idx + 1]
            src_input = self._fp_to_onbit_indices(fp_single)
            e_output, e_mask = self.encode_source(src_input)

            sample_preds = []
            for _ in range(num_samples):
                if self.decode_method == "beam":
                    results = beam_search(
                        self.decoder, self.trg_embedding, self.trg_positional_encoder,
                        self.output_linear, e_output, e_mask,
                        self.sos_id, self.eos_id, self.pad_id,
                        self.trg_seq_len, self.device, self.beam_size,
                    )
                    if results:
                        token_ids = results[0]
                    else:
                        token_ids = []
                else:
                    token_ids = greedy_search(
                        self.decoder, self.trg_embedding, self.trg_positional_encoder,
                        self.output_linear, e_output, e_mask,
                        self.sos_id, self.eos_id, self.pad_id,
                        self.trg_seq_len, self.device,
                    )

                smi = self._detokenize(token_ids)
                sample_preds.append(smi)

            all_preds.append(sample_preds)
        return all_preds

    def _detokenize(self, token_ids: list) -> Optional[str]:
        """Convert token IDs back to SMILES (placeholder for SentencePiece)."""
        try:
            chars = []
            for tid in token_ids:
                if tid == self.eos_id:
                    break
                if tid >= 3:
                    chars.append(chr((tid - 3) % 128 + 32))
            smi = "".join(chars)
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol) if mol is not None else None
        except Exception:
            return None
