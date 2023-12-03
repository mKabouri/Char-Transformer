import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedFoward

class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, attention_dim, nb_heads):
        super().__init__()
        # mha for multi-head attention
        self.masked_mha = MultiHeadAttention(embedding_dim, attention_dim, nb_heads)
        self.layer_norm1 = nn.LayerNorm(attention_dim)

        self.mha = MultiHeadAttention(embedding_dim, attention_dim, nb_heads)
        self.layer_norm2 = nn.LayerNorm(attention_dim)

        self.feed_forward = FeedFoward(attention_dim)
        self.layer_norm3 = nn.LayerNorm(attention_dim)

    def forward(self, input): # input shape: (seq_len, embedding_dim)
        masked_mha_output = input + self.masked_mha(input, mask=True)
        norm1_output = self.layer_norm1(masked_mha_output)

        mha_output = norm1_output + self.mha(norm1_output, mask=False)
        norm2_output = self.layer_norm2(mha_output)

        ff_output = norm2_output + self.feed_forward(norm2_output)
        norm3_output = self.layer_norm3(ff_output) 

        return norm3_output  # (seq_len, attention_dim)

