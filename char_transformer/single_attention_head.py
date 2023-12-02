import torch
import torch.nn as nn
import torch.nn.functional as F

# Receive input of shape (seq_len, embed_dim)
class SingleAttentionHead(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super().__init__()
        self.query_linear = nn.Linear(embedding_dim, attention_dim)
        self.key_linear = nn.Linear(embedding_dim, attention_dim)
        self.value_linear = nn.Linear(embedding_dim, attention_dim)

        self.dk = torch.tensor(attention_dim) # To scale in Scaled dot product
    
    def scaled_dot_product(self, query, key, value):
        output_query_value = torch.matmul(query,\
                                          key.permute(1, 0))/torch.sqrt(self.dk) # (seq_len, seq_len)
        
        output_softmax = F.softmax(output_query_value, dim=1)
        output_attention = torch.matmul(output_softmax, value)
        return output_attention # (seq_len, attention_dim)

    def forward(self, input):
        output_query = self.query_linear(input) # (seq_len, attention_dim)
        output_key = self.key_linear(input) # (seq_len, attention_dim)
        output_value = self.value_linear(input) # (seq_len, attention_dim)

        # Scaled Dot-Product Attention
        output_attention = self.scaled_dot_product(output_query,\
                                                   output_key,\
                                                   output_value)
        return output_attention # (seq_len, attention_dim)
