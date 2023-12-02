import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedFoward(nn.Module):
    def __init__(self, attention_dim):
        super().__init__()
        self.fc1 = nn.Linear(attention_dim, 1024)
        self.fc2 = nn.Linear(1024, attention_dim)

    def forward(self, input): # input shape: (h*seq_len, attention_dim)
        output = F.relu(self.fc1(input))
        output = self.fc2(output)
        return output # (h*seq_len, attention_dim)
