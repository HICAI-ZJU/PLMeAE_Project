import torch
from torch import nn

from fairseq.models import FairseqEncoder

from .modules import TransformerLayer, LearnedPositionalEmbedding, ESM1bLayerNorm

class ProteinBertEncoder(FairseqEncoder):
    def __init__(self, args, max_position_num, layer_num, attention_head_num, embed_dim, ffn_embed_dim, alphabet) -> None:
        super().__init__(alphabet)
        self.args = args
        self.max_position_num = max_position_num
        self.layer_num = layer_num
        self.attention_head_num = attention_head_num
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = True

        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    self.ffn_embed_dim,
                    self.attention_head_num,
                    add_bias_kv=False,
                )
                for _ in range(self.layer_num)
            ]
        )
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(
            self.max_position_num, self.embed_dim, self.padding_idx
        )
        self.emb_layer_norm_before = (
            ESM1bLayerNorm(self.embed_dim) if self.emb_layer_norm_before else None
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

    def forward(self, tokens, attn_mask=None, repr_layers=[], need_head_weights=False, return_contacts=False):
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)
        x = x + self.embed_positions(tokens)

        if self.emb_layer_norm_before:
            x = self.emb_layer_norm_before(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        # repr_layers 很关键

        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x
        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)
        if not padding_mask.any():
            padding_mask = None
        for layer_idx, layer in enumerate(self.layers):

            x, attn = layer(
                x, self_attn_mask=attn_mask, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
        # last hidden representation should have layer norm applied
        if (layer_idx + 2) in repr_layers:
            hidden_representations[layer_idx + 2] = x

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
        return result
