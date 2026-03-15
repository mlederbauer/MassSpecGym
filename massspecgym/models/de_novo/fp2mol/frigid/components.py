"""
Conditioning components for the FRIGID decoder.

Contains formula and fingerprint conditioners that produce embeddings for
cross-attention injection into the BERT backbone, matching the FRIGID paper.
"""

import math

import torch
import torch.nn as nn


class FormulaSequenceEncoder(nn.Module):
    """Encodes molecular formula as a fixed-length sequence of atom embeddings.

    Each position represents an atom type (C, H, N, O, ...) with its count,
    creating a 30-length sequence for cross-attention with SAFE tokens.

    Args:
        num_atom_types: Number of atom types in the vocabulary (30).
        embedding_dim: Dimension of output embeddings (matches BERT hidden_size).
        max_count: Maximum count value for count embeddings.
        use_count_embedding: If True, use separate learnable count embeddings.
    """

    def __init__(
        self,
        num_atom_types: int = 30,
        embedding_dim: int = 768,
        max_count: int = 200,
        use_count_embedding: bool = True,
    ):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.embedding_dim = embedding_dim
        self.use_count_embedding = use_count_embedding

        self.atom_embeddings = nn.Embedding(num_atom_types, embedding_dim)
        if use_count_embedding:
            self.count_embeddings = nn.Embedding(max_count + 1, embedding_dim)
        self.position_embeddings = nn.Embedding(num_atom_types, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.atom_embeddings.weight, mean=0.0, std=0.02)
        if self.use_count_embedding:
            nn.init.normal_(self.count_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, formula_vectors: torch.Tensor):
        """Encode formula vectors as a sequence of embeddings.

        Args:
            formula_vectors: Tensor [batch_size, num_atom_types] with counts.

        Returns:
            Tuple of (embeddings [B, 30, D], mask [B, 30]).
        """
        batch_size = formula_vectors.shape[0]
        device = formula_vectors.device

        atom_indices = torch.arange(self.num_atom_types, device=device)
        atom_indices = atom_indices.unsqueeze(0).expand(batch_size, -1)

        atom_emb = self.atom_embeddings(atom_indices)

        if self.use_count_embedding:
            counts_clipped = formula_vectors.clamp(0, 200).long()
            count_emb = self.count_embeddings(counts_clipped)
        else:
            count_emb = formula_vectors.unsqueeze(-1) * atom_emb

        pos_emb = self.position_embeddings(atom_indices)
        embeddings = self.layer_norm(atom_emb + count_emb + pos_emb)
        mask = (formula_vectors > 0).float()

        return embeddings, mask


class SetSelfAttention(nn.Module):
    """Permutation-equivariant self-attention for unordered sets.

    No positional encoding - treats input as an unordered set of elements.
    Used by FingerprintSequenceEncoder to learn combinatorial relationships
    among active fingerprint bits.
    """

    def __init__(self, hidden_size: int = 768, num_attention_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None):
        q = self._transpose_for_scores(self.query(hidden_states))
        k = self._transpose_for_scores(self.key(hidden_states))
        v = self._transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if mask is not None:
            ext_mask = (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            scores = scores + ext_mask

        probs = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))

        output = self.output_dropout(self.output_dense(context))
        return self.output_layer_norm(hidden_states + output)


class FingerprintSequenceEncoder(nn.Module):
    """Set-aware encoder for Morgan fingerprints.

    Treats fingerprint as an unordered set of active bits:
    1. Each bit index gets a learnable embedding (substructure semantics).
    2. No positional encoding (set has no order).
    3. Self-attention learns combinatorial relationships.

    Args:
        num_bits: Number of fingerprint bits (4096).
        embedding_dim: Embedding dimension.
        max_seq_len: Maximum number of active bits to retain (256).
        activation_threshold: Threshold for considering a bit active.
        dropout: Dropout probability.
        num_self_attention_layers: Number of self-attention layers (3).
        num_attention_heads: Number of attention heads.
    """

    def __init__(
        self,
        num_bits: int = 4096,
        embedding_dim: int = 768,
        max_seq_len: int = 256,
        activation_threshold: float = 0.0,
        dropout: float = 0.1,
        num_self_attention_layers: int = 2,
        num_attention_heads: int = 12,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.activation_threshold = activation_threshold

        self.bit_embeddings = nn.Embedding(num_bits, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.self_attention_layers = nn.ModuleList([
            SetSelfAttention(hidden_size=embedding_dim, num_attention_heads=num_attention_heads, dropout=dropout)
            for _ in range(num_self_attention_layers)
        ])
        nn.init.normal_(self.bit_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, fingerprint: torch.Tensor):
        """Encode fingerprint as a set of active bit embeddings.

        Args:
            fingerprint: Tensor [batch_size, num_bits] - binary or soft Morgan FP.

        Returns:
            Tuple of (embeddings [B, L, D], mask [B, L]).
        """
        if fingerprint.dim() == 1:
            fingerprint = fingerprint.unsqueeze(0)
        batch_size = fingerprint.shape[0]
        device = fingerprint.device

        active_mask = fingerprint > self.activation_threshold
        num_active = active_mask.sum(dim=1)
        max_active = min(self.max_seq_len, max(int(num_active.max().item()), 1))

        scores = active_mask.float()
        if self.training:
            scores = scores + torch.rand_like(scores) * 0.5

        _, topk_indices = torch.topk(scores, k=max_active, dim=1, sorted=False)

        pos_indices = torch.arange(max_active, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (pos_indices < num_active.unsqueeze(1)).float()

        no_active = num_active == 0
        if no_active.any():
            topk_indices[no_active, 0] = 0
            mask[no_active] = 0.0

        embeddings = self.dropout(self.layer_norm(self.bit_embeddings(topk_indices)))
        for self_attn in self.self_attention_layers:
            embeddings = self_attn(embeddings, mask)
        embeddings = embeddings * mask.unsqueeze(-1)

        return embeddings, mask


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for conditioning SAFE tokens on formula/fingerprint.

    SAFE token representations (queries) attend to conditioning sequence
    embeddings (keys/values) with residual connection and layer norm.
    """

    def __init__(self, hidden_size: int = 768, num_attention_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, condition_embeddings, condition_mask=None):
        """Apply cross-attention.

        Args:
            hidden_states: SAFE token hidden states [B, safe_len, H].
            condition_embeddings: Conditioning embeddings [B, cond_len, H].
            condition_mask: Mask for conditioning tokens [B, cond_len].

        Returns:
            Updated hidden states [B, safe_len, H].
        """
        q = self._transpose_for_scores(self.query(hidden_states))
        k = self._transpose_for_scores(self.key(condition_embeddings))
        v = self._transpose_for_scores(self.value(condition_embeddings))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if condition_mask is not None:
            ext_mask = (1.0 - condition_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            scores = scores + ext_mask

        probs = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))

        output = self.output_dropout(self.output_dense(context))
        return self.output_layer_norm(hidden_states + output)


class CrossAttentionFormulaConditioner(nn.Module):
    """Complete cross-attention formula conditioner.

    1. Encodes formula as a 30-length sequence (one per atom type).
    2. Provides per-layer cross-attention layers for injection into BERT.
    """

    def __init__(
        self,
        num_atom_types: int = 30,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        use_count_embedding: bool = True,
    ):
        super().__init__()
        self.formula_encoder = FormulaSequenceEncoder(
            num_atom_types=num_atom_types,
            embedding_dim=hidden_size,
            use_count_embedding=use_count_embedding,
        )
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_size, num_attention_heads, dropout)
            for _ in range(num_layers)
        ])

    def encode_formula(self, formula_vectors):
        return self.formula_encoder(formula_vectors)


class CrossAttentionFingerprintConditioner(nn.Module):
    """Cross-attention conditioner for fingerprints using set-aware encoding.

    Uses FingerprintSequenceEncoder to produce set embeddings, optionally
    with its own cross-attention layers (or sharing formula's layers).
    """

    def __init__(
        self,
        num_bits: int = 4096,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 12,
        max_seq_len: int = 256,
        activation_threshold: float = 0.0,
        dropout: float = 0.1,
        num_self_attention_layers: int = 2,
        create_cross_attention_layers: bool = True,
    ):
        super().__init__()
        self.fingerprint_encoder = FingerprintSequenceEncoder(
            num_bits=num_bits,
            embedding_dim=hidden_size,
            max_seq_len=max_seq_len,
            activation_threshold=activation_threshold,
            dropout=dropout,
            num_self_attention_layers=num_self_attention_layers,
            num_attention_heads=num_attention_heads,
        )
        if create_cross_attention_layers:
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionLayer(hidden_size, num_attention_heads, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.cross_attention_layers = None

    def encode_fingerprint(self, fingerprint_vectors):
        return self.fingerprint_encoder(fingerprint_vectors)
