"""
Greedy and beam search decoding for MolForge.

Implements autoregressive decoding strategies for the encoder-decoder
transformer, converting encoder memory into target token sequences.
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass(order=True)
class BeamNode:
    """A node in the beam search tree."""
    score: float
    tokens: List[int] = field(compare=False)
    finished: bool = field(default=False, compare=False)


def greedy_search(
    decoder: "torch.nn.Module",
    trg_embedding: "torch.nn.Module",
    positional_encoder: "torch.nn.Module",
    output_linear: "torch.nn.Module",
    e_output: torch.Tensor,
    e_mask: torch.Tensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    device: torch.device,
) -> List[int]:
    """Greedy autoregressive decoding.

    At each step, the most likely next token is selected.

    Args:
        decoder: Transformer decoder module.
        trg_embedding: Target embedding layer.
        positional_encoder: Positional encoding module.
        output_linear: Linear projection to vocabulary.
        e_output: Encoder output [1, src_len, d_model].
        e_mask: Encoder attention mask [1, 1, src_len].
        sos_id: Start-of-sequence token ID.
        eos_id: End-of-sequence token ID.
        pad_id: Padding token ID.
        max_len: Maximum output length.
        device: Target device.

    Returns:
        List of generated token IDs (excluding SOS, including EOS if generated).
    """
    last_words = torch.full((1, max_len), pad_id, dtype=torch.long, device=device)
    last_words[0, 0] = sos_id
    cur_len = 1

    for i in range(max_len - 1):
        d_mask = _subsequent_mask(cur_len, device)
        trg = trg_embedding(last_words[:, :cur_len])
        trg = positional_encoder(trg)
        d_output = decoder(trg, e_output, tgt_mask=d_mask, memory_key_padding_mask=~e_mask.squeeze(1).bool() if e_mask is not None else None)
        output = F.log_softmax(output_linear(d_output[:, -1, :]), dim=-1)
        next_id = output.argmax(dim=-1).item()
        last_words[0, cur_len] = next_id
        cur_len += 1
        if next_id == eos_id:
            break

    return last_words[0, 1:cur_len].tolist()


def beam_search(
    decoder: "torch.nn.Module",
    trg_embedding: "torch.nn.Module",
    positional_encoder: "torch.nn.Module",
    output_linear: "torch.nn.Module",
    e_output: torch.Tensor,
    e_mask: torch.Tensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    device: torch.device,
    beam_size: int = 10,
) -> List[List[int]]:
    """Beam search decoding.

    Maintains top-k hypotheses at each step, returning all completed hypotheses
    ranked by log-probability score.

    Returns:
        List of token ID lists, best-scoring first.
    """
    beams = [BeamNode(score=0.0, tokens=[sos_id])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for beam in beams:
            if beam.finished:
                completed.append(beam)
                continue

            seq = torch.tensor([beam.tokens], dtype=torch.long, device=device)
            d_mask = _subsequent_mask(seq.size(1), device)
            trg = trg_embedding(seq)
            trg = positional_encoder(trg)
            d_output = decoder(trg, e_output, tgt_mask=d_mask, memory_key_padding_mask=~e_mask.squeeze(1).bool() if e_mask is not None else None)
            log_probs = F.log_softmax(output_linear(d_output[:, -1, :]), dim=-1)

            topk_scores, topk_ids = log_probs.topk(beam_size, dim=-1)
            for j in range(beam_size):
                token_id = topk_ids[0, j].item()
                new_score = beam.score + topk_scores[0, j].item()
                new_tokens = beam.tokens + [token_id]
                finished = token_id == eos_id
                candidates.append(BeamNode(score=new_score, tokens=new_tokens, finished=finished))

        candidates.sort(key=lambda n: n.score, reverse=True)
        beams = candidates[:beam_size]

        if all(b.finished for b in beams):
            break

    all_results = completed + beams
    all_results.sort(key=lambda n: n.score, reverse=True)

    return [node.tokens[1:] for node in all_results]


def _subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """Generate causal (upper-triangular) mask for autoregressive decoding."""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask
