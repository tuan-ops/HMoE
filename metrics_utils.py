import os
import json
import math 
import time 
from typing import Dict, Any, Optional 
import torch
import torch.nn.functional as F 
# Hàm đếm tổng số tham số 
def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params
    }
@torch.no_grad()
# Hàm đánh giá thông qua loss và perplexity
def evaluate(
    model: torch.nn.Module,
    dataloader,
    vocab_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        output = model(input_ids)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction = "sum"
        )
        total_loss += loss.item()
        total_tokens += labels.numel()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    model.train()
    return {
        "val_loss": avg_loss,
        "val_ppl": perplexity
    }
# Hàm tính toán tốc độ huấn luyện 
class TrainSpeedMeter:

    def __init__(self):
        self.start_time = None
        self.tokens_seen = 0

    def start(self):
        self.start_time = time.time()
        self.tokens_seen = 0

    def update(self, input_ids: torch.Tensor):
        self.tokens_seen += input_ids.numel()

    def get_metrics(self) -> Dict[str, float]:
        if self.start_time is None:
            return {
                "train_time_sec": 0.0,
                "tokens_seen": float(self.tokens_seen),
                "tokens_per_sec": 0.0,
            }

        elapsed = time.time() - self.start_time

        if elapsed <= 0:
            tokens_per_sec = 0.0
        else:
            tokens_per_sec = self.tokens_seen / elapsed

        return {
            "train_time_sec": elapsed,
            "tokens_seen": float(self.tokens_seen),
            "tokens_per_sec": tokens_per_sec,
        }

# Hàm tính toán bộ nhớ trên VRAM
def reset_peak_vram(device: torch.device):

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / 1024 / 1024

    return peak_mb


def save_metrics_json(
    metrics: Dict[str, Any],
    save_dir: str,
    filename: str = "metrics.json",
):
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[metrics] saved: {path}")


def print_metrics_table(metrics: Dict[str, Any]):
    print("\nRUN METRICS ")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:24s}: {value:.4f}")
        else:
            print(f"{key:24s}: {value}")
