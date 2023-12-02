import torch
import torch.nn as nn

import utils
from embedding_positional_encoding import EmbeddingPositionEncoding    
from single_attention_head import SingleAttentionHead
from multi_head_attention import MultiHeadAttention

text_data = "Hello World!"

vocab_size = utils.get_vocab_size(text_data)
emb_dim = 2
len_seq = len(text_data)

encoding = EmbeddingPositionEncoding(vocab_size, emb_dim, len_seq)

ctoi = utils.get_ctoi(text_data)
tensor_data = utils.text_to_tensor(text_data, ctoi)

encoding_output = encoding(tensor_data)
assert encoding_output.size() == torch.Size([len_seq, emb_dim])

attention_dim = 5
one_attention_head = SingleAttentionHead(emb_dim, attention_dim)
assert one_attention_head(encoding_output).size() == torch.Size([len_seq, attention_dim])
    
multi_head_attention = MultiHeadAttention(emb_dim, attention_dim, 6)

assert multi_head_attention(encoding_output).size() == tensor_data.size()
