import torch
import torch.nn as nn
from model.transformer_block import TransformerBlockWithHMoE

class TinyHMoELanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        expert_ffn_dims: list[int] | None = None,
        routing_type: str = "top_p",
        top_k: int = 2,
        top_p: float = 0.1,
        max_seq: int = 256
    ):
        super().__init__()
        if expert_ffn_dims is None:
            expert_ffn_dims = [512, 768, 1024, 1280]
        if routing_type not in {"top_k", "top_p"}:
            raise ValueError(
                f"routing_type must be 'top_k' or 'top_p', got {routing_type}"
            )
        self.vocab_size = vocab_size,
        self.d_model = d_model,
        self.num_layers = num_layers,
        self.num_heads = num_heads,
        self.expert_ffn_dims = expert_ffn_dims,
        self.routing_type = routing_type,
        self.top_k = top_k,
        self.top_p = top_p,
        self.max_seq = max_seq
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlockWithHMoE(
                d_model=d_model,
                num_heads=num_heads,
                expert_ffn_dims=expert_ffn_dims,
                routing_type=routing_type,
                top_k = top_k,
                top_p=top_p
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias = False)
    def forward(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        if seq_len > self.max_seq:
            raise ValueError(
                {f"seq_len={seq_len} exceeds max_seq={self.max_seq}"}
            )
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        total_aux_loss = torch.tensor(0.0, device=device)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        ).bool()

        for block in self.blocks:
            x, aux_loss = block(x, attn_mask=causal_mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, total_aux_loss
        