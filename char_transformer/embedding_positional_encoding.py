import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class EmbeddingPositionEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, len_seq):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_enc = torch.empty((len_seq, embedding_dim))

        positions = torch.arange(0., len_seq).unsqueeze(1)
        components_even_idx = torch.arange(0., emb_dim, 2)
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
        output_emb = self.embed(input) # shape: (embed)
        return (output_emb + self.pos_enc).detach()


if __name__ == '__main__':
    text_data = "Hello World!"

    vocab_size = utils.get_vocab_size(text_data)
    emb_dim = 2
    len_seq = len(text_data)

    encoding = EmbeddingPositionEncoding(vocab_size, emb_dim, len_seq)

    ctoi = utils.get_ctoi(text_data)

    tensor_data = utils.text_to_tensor(text_data, ctoi)

    assert encoding(tensor_data).size() == torch.Size([12, 2])

    print(encoding(tensor_data).size())
