import torch
import torch.nn as nn

import utils
from embedding_positional_encoding import EmbeddingPositionEncoding    
from single_attention_head import SingleAttentionHead
from multi_head_attention import MultiHeadAttention
from decoder import TransformerDecoder

text_data = "Hello World!"

vocab_size = utils.get_vocab_size(text_data)
emb_dim = 2
len_seq = len(text_data)

encoding = EmbeddingPositionEncoding(vocab_size, emb_dim, len_seq)

ctoi = utils.get_ctoi(text_data)
tensor_data = utils.text_to_tensor(text_data, ctoi)

encoding_output = encoding(tensor_data)
assert encoding_output.size() == torch.Size([len_seq, emb_dim])

attention_dim = emb_dim
one_attention_head = SingleAttentionHead(emb_dim, attention_dim)
assert one_attention_head(encoding_output, False).size() == torch.Size([len_seq, attention_dim])
    
h = 6
multi_head_attention = MultiHeadAttention(emb_dim, attention_dim, h)
assert multi_head_attention(encoding_output, False).size() == encoding_output.size()

decoder = TransformerDecoder(emb_dim, attention_dim, h)
output_decoder = decoder(encoding_output)
assert output_decoder.size() == encoding_output.size()
