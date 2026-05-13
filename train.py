
import argparse
import os
import re
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.model import TinyHMoELanguageModel
from metrics_utils import (
    count_parameters,
    evaluate,
    TrainSpeedMeter,
    reset_peak_vram,
    get_peak_vram_mb,
    save_metrics_json,
    print_metrics_table
)

@dataclass
class TrainConfig:
    data_path: str = "data/tiny_text.txt"

    vocab_size: int = 0
    max_seq: int = 128

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4

    batch_size: int = 8
    num_steps: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    model_type: str = "hmoe"
    routing_type: str = "top_k"
    top_k: int = 2
    top_p: float = 0.6

    save_dir: str = "checkpoints"
    save_every: int = 100
    seed: int = 42
# Hàm suy luận đưa ra dự đoán token tiếp theo
def validate_cloze(model, dataset, device):
    model.eval()

    test_prompts = [
        ("[ASM] mov rax, ", "rbx"),
        ("[AI] The router assigns each token a probability distribution over ", "experts"),
        ("[AI] Top-k routing selects the ", "highest"),
        ("[MATH] The softmax function ", "sigma"),
        ("[BIO] CRISPR-Cas9 introduces ", "double"),
    ]

    results = []

    with torch.no_grad():
        for prompt, target in test_prompts:
            input_indices = [dataset.stoi[c] for c in prompt if c in dataset.stoi]

            if len(input_indices) == 0:
                results.append(
                    f"Prompt: '{prompt}' | Pred: '' | [SKIP: prompt không có ký tự trong vocab]"
                )
                continue

            input_tensor = torch.tensor(
                input_indices,
                dtype=torch.long
            ).unsqueeze(0).to(device)

            pred_str = ""
            current_input = input_tensor
# vòng lặp sẽ sinh 20 từ tiếp theo từ câu promt
            for _ in range(20):
                logits, _ = model(current_input[:, -dataset.seq_len:])

# Sử dụng chiến lược greedy để chọn từ tiếp theo
                next_token = torch.argmax(logits[0,-1, :]).item()
                
                char = dataset.itos[next_token]
                pred_str += char

                current_input = torch.cat(
                    [
                        current_input,
                        torch.tensor([[next_token]], device=device)
                    ],
                    dim=1
                )
                if target.lower() in pred_str.lower():
                    break
            pred_display = pred_str.strip()
            target_lower = target.lower()
            pred_lower = pred_display.lower()
            is_correct = target_lower in pred_lower

            if is_correct:
                end_pos = pred_lower.find(target_lower) + len(target)
                pred_display = pred_display[:end_pos]

            results.append(
                f"Prompt: '{prompt}' | Pred: '{pred_display}' | "
                f"{'[OK]' if is_correct else '[FAIL]'}"
            )

    model.train()
    return results
# Xử lý đầu vào mỗi kí tự sẽ là một token
def char_tokenize(text: str):
    """Character-level tokenizer"""
    return list(text) 

class CharTextDataset(Dataset):
    def __init__(self, file_path: str, seq_len: int):
        super().__init__()

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = char_tokenize(text)

        vocab = sorted(list(set(tokens)))
        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

        self.vocab_size = len(vocab)
        self.seq_len = seq_len

        self.data = torch.tensor([self.stoi[tok] for tok in tokens], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collect_hmoe_router_stats(model):
    stats = {}

    for name, module in model.named_modules():
        # Chỉ lấy module HMoELayer thật sự, bỏ qua Router con
        if not hasattr(module, "experts"):
            continue

        if not hasattr(module, "last_router_probs"):
            continue

        if not hasattr(module, "last_expert_mask"):
            continue

        probs = module.last_router_probs
        expert_mask = module.last_expert_mask

        if probs is None or expert_mask is None:
            continue

        # Đảm bảo lấy đúng shape [batch, seq_len, num_experts]
        if probs.dim() != 3 or expert_mask.dim() != 3:
            continue

        # Nếu mask toàn 0 thì báo debug để biết layer nào lỗi
        mask_sum = expert_mask.sum().item()

        mean_probs = probs.mean(dim=(0, 1))

        # usage[i] = tỉ lệ token thực tế có chọn expert i
        usage = expert_mask.float().mean(dim=(0, 1))

        entropy = -torch.sum(
            probs * torch.log(probs.clamp_min(1e-9)),
            dim=-1
        ).mean()

        usage_mean = usage.mean().clamp_min(1e-9)
        balance_cv = usage.std() / usage_mean

        layer_stats = {
            "entropy": entropy.item(),
            "balance_cv": balance_cv.item(),
            "mean_probs": mean_probs.detach().cpu().tolist(),
            "usage": usage.detach().cpu().tolist(),
            "mask_sum": mask_sum,
        }

        if hasattr(module, "last_topk_idx") and module.last_topk_idx is not None:
            layer_stats["topk_idx_shape"] = tuple(module.last_topk_idx.shape)

        stats[name] = layer_stats

    return stats
def print_hmoe_router_stats(model, step: int):
    stats = collect_hmoe_router_stats(model)

    if not stats:
        print("[router] Không tìm thấy HMoE router stats.")
        return

    print(f"\n[Router stats @ step {step}]")

    for layer_name, s in stats.items():
        print(f"  Layer: {layer_name}")
        print(f"    entropy    : {s['entropy']:.4f}")
        print(f"    balance_cv : {s['balance_cv']:.4f}")

        usage_str = " | ".join(
            f"E{i}: {v:.3f}"
            for i, v in enumerate(s["usage"])
        )

        prob_str = " | ".join(
            f"E{i}: {v:.3f}"
            for i, v in enumerate(s["mean_probs"])
        )

        print(f"    usage      : {usage_str}")
        print(f"    mean_probs : {prob_str}")

        if "topk_idx_shape" in s:
            print(f"    topk shape : {s['topk_idx_shape']}")
def print_token_expert_mix_and_p_penalty(
    model,
    dataset,
    input_ids,
    max_tokens: int = 80,
):
    """
    Chỉ ra từng token đang chọn expert nhỏ/lớn nào,
    weight bao nhiêu, và token đó đóng góp p-penalty thế nào.
    """

    print("\n[Token expert mix + p-penalty view]")

    found = False

    for layer_name, module in model.named_modules():
        if not hasattr(module, "last_topk_idx"):
            continue

        if module.last_topk_idx is None or module.last_topk_weight is None:
            continue

        found = True

        topk_idx = module.last_topk_idx.detach().cpu()          # [B, T, K]
        topk_weight = module.last_topk_weight.detach().cpu()    # [B, T, K]

        expert_sizes = torch.tensor(
            module.expert_ffn_dims,
            dtype=torch.float32,
        )

        num_experts = len(expert_sizes)

        # Chuẩn hóa size để biết expert nào nhỏ/to
        # Ví dụ [2,3,4,5] / mean = [0.57,0.86,1.14,1.43]
        size_norm = expert_sizes / expert_sizes.mean()

        print(f"\n  Layer: {layer_name}")
        print("  Expert size map:")

        for i, size in enumerate(expert_sizes.tolist()):
            if i < num_experts // 2:
                size_tag = "SMALL"
            else:
                size_tag = "LARGE"

            print(
                f"    E{i}: d_ffn={int(size)} "
                f"size_norm={size_norm[i].item():.3f} "
                f"{size_tag}"
            )

        sample_token_ids = input_ids[0].detach().cpu().tolist()
        seq_len = min(max_tokens, len(sample_token_ids), topk_idx.shape[1])

        print("\n  Token routes:")

        total_small_weight = 0.0
        total_large_weight = 0.0
        total_p_token = 0.0

        for t in range(seq_len):
            token_id = sample_token_ids[t]
            ch = dataset.itos[token_id]

            experts = topk_idx[0, t].tolist()
            weights = topk_weight[0, t].tolist()

            small_w = 0.0
            large_w = 0.0
            p_token = 0.0

            route_parts = []

            for e, w in zip(experts, weights):
                e_size_norm = size_norm[e].item()

                # Đóng góp p-penalty token:
                # weight router * normalized expert size
                contrib = float(w) * e_size_norm
                p_token += contrib

                if e < num_experts // 2:
                    small_w += float(w)
                    tag = "S"
                else:
                    large_w += float(w)
                    tag = "L"

                route_parts.append(
                    f"E{e}({tag},size={e_size_norm:.2f}):w={float(w):.2f},p={contrib:.3f}"
                )

            total_small_weight += small_w
            total_large_weight += large_w
            total_p_token += p_token

            route_str = " + ".join(route_parts)

            print(
                f"    t={t:02d} token={repr(ch):>6s} "
                f"small_w={small_w:.2f} "
                f"large_w={large_w:.2f} "
                f"p_token={p_token:.3f} "
                f"| {route_str}"
            )

        if seq_len > 0:
            print("\n  Summary on printed tokens:")
            print(f"    avg_small_weight : {total_small_weight / seq_len:.3f}")
            print(f"    avg_large_weight : {total_large_weight / seq_len:.3f}")
            print(f"    avg_p_token      : {total_p_token / seq_len:.3f}")

        # Chỉ in layer đầu tiên
        break

    if not found:
        print("  Không tìm thấy top-k route cache. Kiểm tra HMoELayer đã lưu last_topk_idx/last_topk_weight chưa.")
def print_active_parameter_efficiency(model):
    print("\n[HMoE active expert size efficiency]")

    for layer_name, module in model.named_modules():
        if not hasattr(module, "last_topk_idx"):
            continue

        if module.last_topk_idx is None or module.last_topk_weight is None:
            continue

        topk_idx = module.last_topk_idx
        topk_weight = module.last_topk_weight

        expert_dims = torch.tensor(
            module.expert_ffn_dims,
            device=topk_weight.device,
            dtype=topk_weight.dtype,
        )

        selected_dims = expert_dims[topk_idx]  # [B, T, K]

        weighted_dim = (topk_weight * selected_dims).sum(dim=-1)  # [B, T]

        avg_active_dim = weighted_dim.mean()
        max_dim = expert_dims.max()
        relative_active_dim = avg_active_dim / max_dim

        print(f"  Layer: {layer_name}")
        print(f"    avg_active_expert_dim : {avg_active_dim.item():.2f}")
        print(f"    max_expert_dim        : {max_dim.item():.2f}")
        print(f"    relative_active_dim   : {relative_active_dim.item():.4f}")

        break
def build_model(cfg: TrainConfig, device: torch.device):

    if cfg.model_type == "moe":
        expert_ffn_dims = [
            cfg.d_model * 4,
            cfg.d_model * 4,
            cfg.d_model * 4,
            cfg.d_model * 4,
        ]

    elif cfg.model_type == "hmoe":
        expert_ffn_dims = [
            cfg.d_model * 2,
            cfg.d_model * 3,
            cfg.d_model * 4,
            cfg.d_model * 5,
        ]

    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    model = TinyHMoELanguageModel(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        expert_ffn_dims=expert_ffn_dims,
        routing_type=cfg.routing_type,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        max_seq=cfg.max_seq,
    )

    return model.to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
):
    os.makedirs(cfg.save_dir, exist_ok=True)

    path = os.path.join(
        cfg.save_dir,
        f"{cfg.model_type}_{cfg.routing_type}_step_{step}.pt",
    )

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )

    print(f"[save] saved checkpoint: {path}")



def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[info] device       = {device}")
    print(f"[info] routing_type = {cfg.routing_type}")
    print(f"[info] model_type   = {cfg.model_type}")

    dataset = CharTextDataset(
        file_path=cfg.data_path,
        seq_len=cfg.max_seq,
    )

    cfg.vocab_size = dataset.vocab_size

    print(f"[info] vocab_size   = {cfg.vocab_size}")
    print(f"[info] data length   = {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_model(cfg, device)
    model.train()

    param_info = count_parameters(model)
    print(f"[info] total_params = {param_info['total_params']:,}")
    print(f"[info] trainable    = {param_info['trainable_params']:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    step = 0
    printed_token_route = False

    # Đo tốc độ train và VRAM peak bằng các hàm đã có trong metrics_utils.py
    speed_meter = TrainSpeedMeter()
    speed_meter.start()
    reset_peak_vram(device)

    # Lưu giá trị cuối để đưa vào metrics JSON
    last_lm_loss = None
    last_aux_loss = None
    last_total_loss = None

    while step < cfg.num_steps:
        for input_ids, labels in dataloader:
            if step >= cfg.num_steps:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            speed_meter.update(input_ids)

            logits, aux_loss = model(input_ids)

            # Loss từng token/ký tự: [batch_size, seq_len]
            token_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)

            # Loss trung bình để backward như bình thường
            lm_loss = token_loss.mean()

            if aux_loss.dim() > 0:
                aux_loss = aux_loss.mean()

            loss = lm_loss + aux_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            last_lm_loss = lm_loss.item()
            last_aux_loss = aux_loss.item()
            last_total_loss = loss.item()

            if step % 50 == 0:
                print(f"\n[Step {step:04d}]")
                print(f"  lm_loss  : {last_lm_loss:.4f}")
                print(f"  aux_loss : {last_aux_loss:.4f}")
                print(f"  total    : {last_total_loss:.4f}")

                print_hmoe_router_stats(model, step)

                # Chỉ in bảng token-route đúng 1 lần đầu tiên, tối đa 50 token.
                if not printed_token_route:
                    print_token_expert_mix_and_p_penalty(
                        model=model,
                        dataset=dataset,
                        input_ids=input_ids,
                        max_tokens=50,
                    )
                    printed_token_route = True

                val_results = validate_cloze(model, dataset, device)
                for res in val_results:
                    print(f"  > {res}")

                print("-" * 50)

            # Lưu checkpoint định kỳ theo save_every
            if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
                save_checkpoint(model, optimizer, cfg, step)

            step += 1

    save_checkpoint(model, optimizer, cfg, step)

    speed_metrics = speed_meter.get_metrics()
    peak_vram_mb = get_peak_vram_mb(device)

    metrics = {
        "model_type": cfg.model_type,
        "routing_type": cfg.routing_type,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "d_model": cfg.d_model,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "max_seq": cfg.max_seq,
        "batch_size": cfg.batch_size,
        "num_steps": cfg.num_steps,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "vocab_size": cfg.vocab_size,
        "total_params": param_info["total_params"],
        "trainable_params": param_info["trainable_params"],
        "final_lm_loss": last_lm_loss,
        "final_aux_loss": last_aux_loss,
        "final_total_loss": last_total_loss,
        "train_time_sec": speed_metrics["train_time_sec"],
        "tokens_seen": speed_metrics["tokens_seen"],
        "tokens_per_sec": speed_metrics["tokens_per_sec"],
        "peak_vram_mb": peak_vram_mb,
    }

    print_metrics_table(metrics)

    save_metrics_json(
        metrics=metrics,
        save_dir=cfg.save_dir,
        filename=f"{cfg.model_type}_{cfg.routing_type}_metrics.json",
    )

    print("[done] training finished")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/tiny_text.txt")

    parser.add_argument(
        "--routing_type",
        type=str,
        default="top_k",
        choices=["top_k", "top_p"],
    )

    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument(
    "--model_type",
    type=str,
    default="hmoe",
    choices=["moe", "hmoe"],
)

    parser.add_argument("--max_seq", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return TrainConfig(
        data_path=args.data_path,
        max_seq=args.max_seq,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_type=args.model_type,
        routing_type=args.routing_type,
        top_k=args.top_k,
        top_p=args.top_p,
        save_dir=args.save_dir,
        save_every=args.save_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)