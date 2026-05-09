import torch
import torch.nn as nn
import torch.nn.functional as F 
class LlamaFFNExpert(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias =  False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias = False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias = False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
    
    