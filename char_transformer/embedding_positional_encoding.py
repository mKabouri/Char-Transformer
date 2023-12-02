import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class EmbeddingPositionEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, len_seq):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim) # Embedding table
        self.pos_enc = torch.empty((len_seq, embedding_dim)) # Positional encoding table

        positions = torch.arange(0., len_seq).unsqueeze(1)
        components_even_idx = torch.arange(0., embedding_dim, 2)
        # ith component even
        self.pos_enc[:, 1::2] = torch.sin(positions/(10000**(components_even_idx/embedding_dim)))
        # ith component odd
        self.pos_enc[:, 0::2] = torch.cos(positions/(10000**(components_even_idx/embedding_dim)))

    def get_pos_enc(self):
        return self.pos_enc

    def get_embedding(self):
        return self.embed
    
    def plot_pos_encoding(self):
        pass

    def forward(self, input):
        output_emb = self.embed(input) # shape: (sequence_length, embed_dim)
        return (output_emb + self.pos_enc).detach()
