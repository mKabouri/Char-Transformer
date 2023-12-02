import torch
import torch.nn as nn
import torch.nn.functional as F

from single_attention_head import SingleAttentionHead

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim, nb_heads): # nb_heads = 6 in the original paper
        super().__init__()
        self.nb_heads = nb_heads
        self.multi_head_linear = nn.Linear(self.nb_heads*attention_dim, attention_dim)
        self.heads = [
            SingleAttentionHead(embedding_dim, attention_dim) for _ in range(nb_heads)
        ]

    def forward(self, input, mask): # input: (seq_len, attention_dim)
        output_heads = torch.cat([self.heads[i](input, mask) for i in range(self.nb_heads)], dim=-1) # (seq_len, nb_heads*attention_dim)
        output_multi_head_attention = self.multi_head_linear(output_heads)
        return output_multi_head_attention # (seq_len, attention_dim)
