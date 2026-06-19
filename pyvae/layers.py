import torch
import torch.nn as nn
import torch.nn.functional as F  

class InformedLinear(nn.Module):

    def __init__(self, adj: torch.Tensor, activation: str = "tanh"):
        super().__init__()
        in_f, out_f = adj.shape  
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.register_buffer("mask", adj.T.float())
        self.register_buffer("bias_mask", (adj.sum(dim=0) > 0).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        w = self.weight * self.mask
        b = self.bias * self.bias_mask
        out = x @ w.T + b

        if self.activation == "tanh":
            out = torch.tanh(out)
        
        elif self.activation == "relu":
            out = torch.relu(out)
        
        else: # "linear" — no activation
            pass 
        
        return out
    