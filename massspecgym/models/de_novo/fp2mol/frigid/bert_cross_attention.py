"""
BERT encoder with cross-attention layers for FRIGID conditioning.

Extends the HuggingFace BertForMaskedLM with cross-attention sublayers in each
transformer block, allowing formula and fingerprint conditioning embeddings
to be injected via cross-attention at every layer.

Matches the reference implementation from external/genms/src/genmol/bert_with_cross_attention.py
exactly in structure: BertLayerWithCrossAttention wraps BertLayer and adds
cross-attention after self-attention + FFN.

Supports two modes:
- Shared cross-attention: formula + fingerprint concatenated into single conditioning.
- Independent cross-attention: separate cross-attention paths for each modality.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertPreTrainedModel,
    BertForMaskedLM,
    BertEmbeddings,
    BertPooler,
)
from transformers.models.bert.configuration_bert import BertConfig


class BertLayerWithCrossAttention(nn.Module):
    """BERT layer extended with optional cross-attention over conditioning tokens.

    Performs:
    1. Standard self-attention + FFN (via wrapped BertLayer)
    2. Cross-attention to formula conditioning sequence (if provided)
    3. Cross-attention to fingerprint conditioning sequence (if provided)
    """

    def __init__(self, config, cross_attention_layer=None,
                 fingerprint_cross_attention_layer=None):
        super().__init__()
        self.bert_layer = BertLayer(config)
        self.cross_attention = cross_attention_layer
        self.has_cross_attention = cross_attention_layer is not None
        self.fingerprint_cross_attention = fingerprint_cross_attention_layer
        self.has_fingerprint_cross_attention = fingerprint_cross_attention_layer is not None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        condition_embeddings=None,
        condition_mask=None,
        fingerprint_embeddings=None,
        fingerprint_mask=None,
    ):
        layer_outputs = self.bert_layer(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0]

        if self.has_cross_attention and condition_embeddings is not None:
            hidden_states = self.cross_attention(
                hidden_states=hidden_states,
                condition_embeddings=condition_embeddings,
                condition_mask=condition_mask,
            )

        if self.has_fingerprint_cross_attention and fingerprint_embeddings is not None:
            hidden_states = self.fingerprint_cross_attention(
                hidden_states=hidden_states,
                condition_embeddings=fingerprint_embeddings,
                condition_mask=fingerprint_mask,
            )

        return (hidden_states,) + layer_outputs[1:]


class BertEncoderWithCrossAttention(nn.Module):
    """BERT encoder with per-layer cross-attention for conditioning sequences."""

    def __init__(self, config, cross_attention_layers=None,
                 fingerprint_cross_attention_layers=None):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            cross_attn = cross_attention_layers[i] if cross_attention_layers else None
            fp_cross_attn = fingerprint_cross_attention_layers[i] if fingerprint_cross_attention_layers else None
            self.layer.append(
                BertLayerWithCrossAttention(
                    config,
                    cross_attention_layer=cross_attn,
                    fingerprint_cross_attention_layer=fp_cross_attn,
                )
            )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        condition_embeddings=None,
        condition_mask=None,
        fingerprint_embeddings=None,
        fingerprint_mask=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                condition_embeddings=condition_embeddings,
                condition_mask=condition_mask,
                fingerprint_embeddings=fingerprint_embeddings,
                fingerprint_mask=fingerprint_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, None, all_hidden_states,
                                  all_self_attentions] if v is not None)


class BertModelWithCrossAttention(BertPreTrainedModel):
    """BERT model with cross-attention for conditioning, drop-in replacement for BertModel."""

    def __init__(self, config, add_pooling_layer=True, cross_attention_layers=None,
                 fingerprint_cross_attention_layers=None, use_shared_cross_attention=False):
        super().__init__(config)
        self.config = config
        self.use_shared_cross_attention = use_shared_cross_attention

        self.embeddings = BertEmbeddings(config)

        if use_shared_cross_attention:
            fingerprint_cross_attention_layers = None

        self.encoder = BertEncoderWithCrossAttention(
            config, cross_attention_layers, fingerprint_cross_attention_layers
        )
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None):
        return super().get_extended_attention_mask(attention_mask, input_shape, device)


class BertForMaskedLMWithCrossAttention(nn.Module):
    """BertForMaskedLM extended with cross-attention conditioning.

    Wraps BertModelWithCrossAttention + BertOnlyMLMHead (cls) for
    masked language modeling with formula/fingerprint conditioning.
    """

    def __init__(
        self,
        config: BertConfig,
        cross_attention_layers: Optional[nn.ModuleList] = None,
        fingerprint_cross_attention_layers: Optional[nn.ModuleList] = None,
        use_shared_cross_attention: bool = True,
    ):
        super().__init__()

        self.bert = BertModelWithCrossAttention(
            config,
            add_pooling_layer=False,
            cross_attention_layers=cross_attention_layers,
            fingerprint_cross_attention_layers=fingerprint_cross_attention_layers,
            use_shared_cross_attention=use_shared_cross_attention,
        )

        # Reuse the MLM head from standard BertForMaskedLM
        _tmp = BertForMaskedLM(config)
        self.cls = _tmp.cls
        del _tmp

    @property
    def embeddings(self):
        return self.bert.embeddings

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None):
        return self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

    def parameters(self, recurse=True):
        yield from self.bert.parameters(recurse)
        yield from self.cls.parameters(recurse)
