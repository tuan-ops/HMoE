import torch
import torch.nn as nn
# Xây dựng sử dụng hàm gate thông qua router để đánh giá các expert khi có đầu vào là input x
# Đầu ra mong muốn trả về expert được chọn và trọng số đi kèm
class Router(nn.Module):
    def __init__(self, d_model: int, num_expert: int):
        super().__init__()
# Layer tuyến tính biến đổi vector d_model chiều thành vector num_expert chiều
        self.linear = nn.Linear(d_model, num_expert, bias = False)
# Sử dụng lưu thông tin 
        self.last_router_probs = None
        self.last_topk_idx = None
        self.last_topk_weight = None
        self.last_expert_mask = None
        
    def forward(self, x: torch.Tensor):
# Tính điểm số thô cho mỗi expert
        logits = self.linear(x)
# Chuyển số thô thành xác suất thuộc [0, 1]
        probs = torch.softmax(logits, dim = -1)
# Lưu lại xác suất của mỗi expert
        self.last_router_probs = probs.detach()
        return probs
# Xây dựng chiến lược top-k: chọn k-expert có xác suất cao nhất cho mỗi token 
def top_k(probs: torch.Tensor, k: int):
    num_experts = probs.shape[-1]
    if not (1 <= k <= num_experts):
        raise ValueError(f"top_k must be in [1, {num_experts}], got {k}")
    selected_probs, selected_indices = torch.topk(probs, k = k, dim = -1)
    # Chuẩn hóa lại xác suất k expert được chọn
    selected_probs = selected_probs/selected_probs.sum(
        dim = -1,
        keepdim = True
    ).clamp_min(1e-9)
    return selected_probs, selected_indices
# Xây dựng chiến lược top-p: chọn số expert linh hoạt sao cho tổng xác suất chạm hoặc vượt ngưỡng p
def top_p(probs: torch.Tensor, p: float):
    if not (0.0 < p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    batch_size, seq_len, num_experts = probs.shape
# Gộp batch_size và seq_len thành một chiều token -> tổng số token 
# Mỗi token sẽ có số expert khác nhau 
    probs_flat = probs.reshape(-1, num_experts)
    sorted_probs, sorted_indices = torch.sort(
        probs_flat,
        dim = -1,
        descending=True
    )
# Tính xác suất cộng dồn [0.5, 0.3, 0.1, 0.1] -> [0.5, 0.8, 0.9, 1.0]
    cumulative_probs = torch.cumsum(sorted_probs, dim = -1)
# Tìm vị trí đầu tiên đạt ngưỡng p
# first_reach.shape = [B*T] -> token i chọn tới first_reach[i]
    first_reach = (cumulative_probs >= p).float().argmax(dim = -1)
# Lấy vị trí đầu tiên vượt mức p
# slot_ids.shape = [1, T] -> chuyển cột thành hàng
    slot_ids = torch.arange(num_experts, device=probs.device).unsqueeze(0)
# first_reach.shape = [B*T, 1] chuyển hàng thành cột
    selected_mask = slot_ids <= first_reach.unsqueeze(-1)
# hàng -> token, cột -> expert
    token_indices, slot_indices = selected_mask.nonzero(as_tuple=True)
    selected_indices = sorted_indices[token_indices, slot_indices]
    selected_probs = sorted_probs[token_indices, slot_indices]
# Chuẩn hóa xác suất các expert do từng token chọn
    denom = torch.zeros(
        probs_flat.shape[0],
        device=probs.device,
        dtype = probs.dtype,
    )
    denom.scatter_add_(0, token_indices, selected_probs)
    selected_probs = selected_probs/denom[token_indices].clamp_min(1e-9)
    return selected_probs, selected_indices, token_indices, batch_size, seq_len