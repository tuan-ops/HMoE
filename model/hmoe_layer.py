import torch
import torch.nn as nn
from model.ffn import LlamaFFNExpert
from model.router import Router, top_k, top_p
class HMoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        expert_ffn_dims: list[int],
        routing_type: str = "top_k",
        top_k: int = 2,
        top_p: float = 0.6,
        p_penalty_coef: float = 0.1,
        entropy_coef: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.expert_ffn_dims = expert_ffn_dims
        self.num_experts = len(expert_ffn_dims)
        self.routing_type = routing_type
        self.top_k = top_k
        self.top_p = top_p
        self.p_penalty_coef = p_penalty_coef
        self.entropy_coef = entropy_coef
        self.router = Router(d_model, self.num_experts)
        self.experts = nn.ModuleList([
            LlamaFFNExpert(d_model, d_ffn)
            for d_ffn in expert_ffn_dims
            ])
    def forward(self, x: torch.Tensor):
        batch, seq_len, d_model = x.shape
        probs = self.router(x)
        if self.routing_type == "top_k":
            output, aux_loss = self.forward_top_k(x, probs)
        elif self.routing_type == "top_p":
            output, aux_loss = self.forward_top_p(x, probs)
        else:
            raise ValueError(f"Unknown routing_type: {self.routing_type}")
        return output, aux_loss
    def forward_top_k(self, x, probs):
        batch, seq_len, d_model = x.shape
        selected_probs, selected_indices = top_k(
            probs,
            k = self.top_k
        )
        output = torch.zeros_like(x)
        x_flat = x.reshape(batch * seq_len, d_model)
        output_flat = output.reshape(batch * seq_len, d_model)
        selected_probs_flat = selected_probs.reshape(-1, self.top_k)
        selected_indices_flat = selected_indices.reshape(-1, self.top_k)
        for expert_id, expert in enumerate(self.experts):
            mask = selected_indices_flat == expert_id
            if not mask.any():
                continue
            token_indices, slot_indices = mask.nonzero(as_tuple=True)
            selected_x = x_flat[token_indices]
            weights = selected_probs_flat[token_indices, slot_indices].unsqueeze(-1)

            expert_out = expert(selected_x)
            output_flat[token_indices] += weights * expert_out

        output = output_flat.reshape(batch, seq_len, d_model)
        aux_loss = self.compute_p_penalty_top_k(probs, selected_indices)
        return output, aux_loss
    def forward_top_p(self, x, probs):
        batch, seq_len, d_model = x.shape
        (
            selected_probs,
            selected_indices,
            token_indices,
            batch_size,
            seq_len
        ) = top_p(probs, p = self.top_p)
        output = torch.zeros_like(x)
        x_flat = x.reshape(batch * seq_len, d_model)
        output_flat = output.reshape(batch * seq_len, d_model)
        for expert_id, expert in enumerate(self.experts):
            mask = selected_indices == expert_id
            if not mask.any():
                continue
            selected_token_indices = token_indices[mask]
            selected_x = x_flat[selected_token_indices]
            weights = selected_probs[mask].unsqueeze(-1)
            expert_out = expert(selected_x)
            output_flat[selected_token_indices] += weights * expert_out
        output = output_flat.reshape(batch, seq_len, d_model)
        aux_loss = self.compute_p_penalty_top_p(
            probs=probs,
            selected_indices=selected_indices,
            token_indices=token_indices,
        )

        entropy_loss = self.compute_router_entropy(probs)

        return output, aux_loss + entropy_loss
    def compute_p_penalty_top_k(self, probs, selected_indices):
        batch, seq_len, num_experts = probs.shape
        total_tokens = batch * seq_len
        p_hat = probs.mean(dim=(0,1))
        activation_count = torch.zeros(
            num_experts,
            device=probs.device,
            dtype=probs.dtype
        )
        flat_indices = selected_indices.reshape(-1)
        activation_count.scatter_add_(
            0,
            flat_indices,
            torch.ones_like(flat_indices, dtype = probs.dtype)
        )
        activation_ratio = activation_count / total_tokens
        expert_sizes = torch.tensor(
            self.expert_ffn_dims,
            device = probs.device,
            dtype = probs.dtype
        ) 
        expert_sizes = expert_sizes / expert_sizes.mean()
        m_i = activation_ratio * expert_sizes
        loss = len(self.experts) * torch.sum(p_hat * m_i)
        return self.p_penalty_coef * loss
    def compute_p_penalty_top_p(self, probs, selected_indices, token_indices):
        batch, seq_len, num_experts = probs.shape
        total_tokens = batch * seq_len
        p_hat = probs.mean(dim = (0,1))
        activation_count = torch.zeros(
            num_experts,
            device=probs.device,
            dtype=probs.dtype
        )
        activation_count.scatter_add_(
            0,
            selected_indices,
            torch.ones_like(selected_indices, dtype = probs.dtype)
        )
        activation_ratio = activation_count / total_tokens
        expert_sizes = torch.tensor(
            self.expert_ffn_dims,
            device = probs.device,
            dtype = probs.dtype
        )
        m_i = expert_sizes / expert_sizes.mean()
        loss = len(self.experts) * torch.sum(p_hat * m_i)
        return self.p_penalty_coef * loss
    def compute_router_entropy(self, probs):
        loss = len(self.experts) * torch.sum(probs * torch.log(probs.clamp_min(1e-9)),
                                             dim = -1)
        return self.entropy_coef * loss

                
        
            