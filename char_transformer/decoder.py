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
        output = input + self.masked_mha(input, mask=True)
        output = self.layer_norm1(output)

        output += self.mha(output, mask=False)
        output = self.layer_norm2(output)

        output += self.feed_forward(output)
        output = self.layer_norm3(output) 

        return output # (seq_len, attention_dim)
