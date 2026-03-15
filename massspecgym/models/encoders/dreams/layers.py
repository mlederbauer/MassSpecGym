"""
DreaMS transformer encoder layers.

Ported from external/DreaMS/dreams/models/dreams/layers.py.
Includes custom MultiheadAttention with optional Graphormer m/z bias,
ScaleNorm, and TransformerEncoder with gradient checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ScaleNorm(nn.Module):
    """Scale normalization (Nguyen & Salazar, 2019)."""

    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = Parameter(torch.tensor(float(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class MultiheadAttention(nn.Module):
    """Multi-head attention with optional Graphormer m/z distance bias.

    Args:
        args: Namespace with d_model, n_heads, att_dropout, no_transformer_bias,
              attn_mech, d_graphormer_params.
    """

    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.dropout = getattr(args, 'att_dropout', getattr(args, 'dropout', 0.0))
        self.use_transformer_bias = not getattr(args, 'no_transformer_bias', False)
        self.attn_mech = getattr(args, 'attn_mech', 'dot-product')
        self.d_graphormer_params = getattr(args, 'd_graphormer_params', 0)

        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.weights = Parameter(torch.Tensor(4 * self.d_model, self.d_model))
        if self.use_transformer_bias:
            self.biases = Parameter(torch.Tensor(4 * self.d_model))

        if self.d_graphormer_params:
            self.lin_graphormer = nn.Linear(self.d_graphormer_params, self.n_heads, bias=False)

        mean, std = 0, (2 / (5 * self.d_model)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.biases, 0.)

        if self.attn_mech == 'additive_v':
            self.additive_v = Parameter(torch.Tensor(self.n_heads, self.head_dim))
            nn.init.normal_(self.additive_v, mean=mean, std=std)

    def proj_qkv(self, q, k, v):
        w_q, w_k, w_v, w_o = self.weights.chunk(4, dim=0)
        if self.use_transformer_bias:
            b_q, b_k, b_v, b_o = self.biases.chunk(4, dim=0)
            q = F.linear(q, w_q, b_q)
            k = F.linear(k, w_k, b_k)
            v = F.linear(v, w_v, b_v)
        else:
            q = F.linear(q, w_q)
            k = F.linear(k, w_k)
            v = F.linear(v, w_v)
        return q, k, v

    def proj_o(self, x):
        w_q, w_k, w_v, w_o = self.weights.chunk(4, dim=0)
        if self.use_transformer_bias:
            b_q, b_k, b_v, b_o = self.biases.chunk(4, dim=0)
            return F.linear(x, w_o, b_o)
        return F.linear(x, w_o)

    def forward(self, q, k, v, mask, graphormer_dists=None, do_proj_qkv=True):
        bs, n, d = q.size()

        def _split_heads(tensor):
            return tensor.reshape(bs, n, self.n_heads, self.head_dim).transpose(1, 2)

        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)
        q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)

        if self.attn_mech == 'dot-product':
            att_weights = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1))
        elif self.attn_mech in ('additive_v', 'additive_fixed'):
            att_weights = (q.unsqueeze(-2) - k.unsqueeze(-3))
            if self.attn_mech == 'additive_v':
                att_weights = att_weights * self.additive_v[None, :, None, None, :]
            att_weights = att_weights.sum(dim=-1)
        else:
            raise NotImplementedError(f'"{self.attn_mech}" not implemented')

        att_weights = att_weights * self.scale

        if graphormer_dists is not None:
            if self.d_graphormer_params:
                att_bias = self.lin_graphormer(graphormer_dists).permute(0, 3, 1, 2)
            else:
                att_bias = graphormer_dists.sum(dim=-1).unsqueeze(1)
            att_weights = att_weights + att_bias

        if mask is not None:
            att_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)

        _att = att_weights.reshape(-1, n, n)
        output = torch.bmm(_att, v.reshape(bs * self.n_heads, -1, self.head_dim))
        output = output.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        output = self.proj_o(output)

        return output, att_weights


class TransformerFeedForward(nn.Module):
    """Feed-forward block inside transformer layer."""

    def __init__(self, args):
        super().__init__()
        self.dropout = getattr(args, 'ff_dropout', getattr(args, 'dropout', 0.0))
        self.d_model = args.d_model
        self.ff_dim = 4 * args.d_model
        use_bias = not getattr(args, 'no_transformer_bias', False)
        self.in_proj = nn.Linear(self.d_model, self.ff_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.d_model, bias=use_bias)

        std = (2 / (self.ff_dim + self.d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=0, std=std)
        nn.init.normal_(self.out_proj.weight, mean=0, std=std)
        if use_bias:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)


class TransformerEncoder(nn.Module):
    """DreaMS transformer encoder with ScaleNorm and optional Graphormer bias.

    Args:
        args: Namespace with n_layers, d_model, pre_norm, scnorm,
              residual_dropout, and attention/FF config.
    """

    def __init__(self, args):
        super().__init__()
        self.residual_dropout = getattr(args, 'residual_dropout', getattr(args, 'dropout', 0.0))
        self.n_layers = args.n_layers
        self.pre_norm = getattr(args, 'pre_norm', True)
        self._gradient_checkpointing = False

        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.n_layers)])
        self.ffs = nn.ModuleList([TransformerFeedForward(args) for _ in range(self.n_layers)])

        num_scales = self.n_layers * 2 + 1 if self.pre_norm else self.n_layers * 2
        if getattr(args, 'scnorm', True):
            self.scales = nn.ModuleList([ScaleNorm(args.d_model ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(num_scales)])

    def _layer_forward(self, i, x, src_mask, graphormer_dists):
        if self.pre_norm:
            residual = x
            x = self.scales[2 * i](x)
            x, _ = self.atts[i](x, x, x, src_mask, graphormer_dists)
            x = F.dropout(x, p=self.residual_dropout, training=self.training)
            x = residual + x

            residual = x
            x = self.scales[2 * i + 1](x)
            x = self.ffs[i](x)
            x = F.dropout(x, p=self.residual_dropout, training=self.training)
            x = residual + x
        else:
            residual = x
            x, _ = self.atts[i](x, x, x, src_mask, graphormer_dists)
            x = F.dropout(x, p=self.residual_dropout, training=self.training)
            x = self.scales[2 * i](residual + x)

            residual = x
            x = self.ffs[i](x)
            x = F.dropout(x, p=self.residual_dropout, training=self.training)
            x = self.scales[2 * i + 1](residual + x)

        return x

    def forward(self, src_inputs, src_mask, graphormer_dists=None):
        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)
        for i in range(self.n_layers):
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, i, x, src_mask, graphormer_dists,
                    use_reentrant=False
                )
            else:
                x = self._layer_forward(i, x, src_mask, graphormer_dists)

        if self.pre_norm:
            x = self.scales[-1](x)
        return x
