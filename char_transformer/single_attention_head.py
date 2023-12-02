import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        query_linear = nn.Linear()
        key_linear = nn.Linear()
        value_linear = nn.Linear()

    def forward(self, input):
        pass