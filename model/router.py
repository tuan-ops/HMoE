import torch
import torch.nn as nn
import torch.functional as F 
class Router(nn.Module):
    def __init__(self, d_model: int, num_expert: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_expert, bias = False)
    def forward(self, x: torch.tensor):
        logits = self.linear(x)
        probs = torch.softmax(logits, dim = -1)
        return probs
def top_k(probs: torch.Tensor, k: int):
    selected_probs, selected_indices = torch.topk(probs, k = k, dim = -1)
    selected_probs = selected_probs/selected_probs.sum(
        dim = -1,
        keepdim = True
    ).clamp_min(1e-9)
    return selected_probs, selected_indices
def top_p(probs: torch.Tensor, p: float):
    batch_size, seq_len, num_experts = probs.shape
    probs_flat = probs.reshape(-1, num_experts)
    sorted_probs, sorted_indices = torch.sort(
        probs_flat,
        dim = -1,
        descending=True
    )
    cumulative_probs = torch.cumsum(sorted_probs, dim = -1)
    firts_reach = (cumulative_probs >= p).float().argmax(dim = -1)
    slot_ids = torch.arange(num_experts, device=probs.device).unsqueeze(0)
    selected_mask = slot_ids <= firts_reach.unsqueeze(-1)
    token_indices, slot_indices = selected_mask.nonzero(as_tuple=True)
    selected_indices = sorted_indices[token_indices, slot_indices]
    selected_probs = sorted_probs[token_indices, slot_indices]
    denom = torch.zeros(
        probs_flat.shape[0],
        device=probs.device,
        dtype = probs.dtype,
    )
    denom.scatter_add_(0, token_indices, selected_probs)
    selected_probs = selected_probs/denom[token_indices].clamp_min(1e-9)
    return selected_probs, selected_indices, token_indices, batch_size, seq_len