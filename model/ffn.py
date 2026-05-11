import torch
import torch.nn as nn
import torch.nn.functional as F 
# Tạo module neural network mới và kế thừa từ nn.Module
# có nghĩa là class này được dùng như một layer trong pytorch có tham số train được
# có thể gọi các hàm như: .parameters, .train(), ...
class LlamaFFNExpert(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
# Expert có 3 layer tuyến tính
        self.gate_proj = nn.Linear(d_model, d_ffn, bias =  False)
# Hàm cổng đánh giá input đầu vào 
        self.up_proj = nn.Linear(d_model, d_ffn, bias = False)
# Hàm up đẩy input từ d_model lên d_ffn
        self.down_proj = nn.Linear(d_ffn, d_model, bias = False)
# Hàm down hạ input từ d_ffn về lại d_model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # Hàm xử lý input x
        gate = F.silu(self.gate_proj(x))
# Hàm silu = x * sigmoid(x) nếu x rất âm -> tắt x gần 0 -> lấy một phần x rất dương -> gần như giữ một phần
        up = self.up_proj(x)
        return self.down_proj(gate * up)
    
    