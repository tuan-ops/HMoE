import torch
import torch.nn as nn
from model.hmoe_layer import HMoELayer
class TransformerBlockWithHMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        expert_ffn_dims: list[int],
        routing_type: str = "top_p",
        top_k: int = 2,
        top_p: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.hmoe = HMoELayer(
            d_model=d_model,
            expert_ffn_dims=expert_ffn_dims,
            routing_type=routing_type,
            top_k=top_k,
            top_p=top_p,
        )
    def forward(self, x, attn_mask = None):
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, attn_mask = attn_mask)
        x = x + attn_out
        h = self.ffn_norm(x)
        hmoe_out, aux_loss = self.hmoe(h)
        x = x + hmoe_out
        return x, aux_loss
        