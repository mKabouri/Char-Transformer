import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding_positional_encoding import EmbeddingPositionEncoding
from decoder import TransformerDecoder

class CharTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 len_seq,
                 attention_dim,
                 nb_heads,
                 output_dim):
        super().__init__()
        self.input_encoding = EmbeddingPositionEncoding(vocab_size, embedding_dim, len_seq)
        
        self.decoder = TransformerDecoder(embedding_dim, attention_dim, nb_heads) # (seq_len, attention_dim)

        self.fc = nn.Linear(attention_dim, vocab_size)

    def forward(self, input_tensor_seq):
        output = self.input_encoding(input_tensor_seq)
        output = self.decoder(output)
        output = self.fc(output)
        return output
