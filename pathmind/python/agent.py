import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(30,64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.Softmax(dim=-1)
        )
    
    def forward(self,x):
        x=self.network(x)
        return x