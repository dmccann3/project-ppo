import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Network(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out
    

