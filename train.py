
import argparse
import os
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.model import TinyHMoELanguageModel


@dataclass
class TrainConfig:
    data_path: str = "data/tiny_text.txt"

    vocab_size: int = 0
    max_seq: int = 64

    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4

    batch_size: int = 8
    num_steps: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    model_type: str = "hmoe"
    routing_type: str = "top_k"
    top_k: int = 2
    top_p: float = 0.6

    save_dir: str = "checkpoints"
    save_every: int = 100
    seed: int = 42

def validate_cloze(model, dataset, device):
    model.eval()
    # Các câu test đặc trưng cho 2 domain
    test_prompts = [
        ("The RSP register manages the ", "stack"),
        ("The encoder block in a Transformer consists of", "attention")
    ]
    results = []
    
    with torch.no_grad():
        for prompt, target in test_prompts:
            # Tokenize input
            input_indices = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
            
            # Dự đoán 5 ký tự tiếp theo
            pred_str = ""
            current_input = input_tensor
            for _ in range(7): # Dự đoán độ dài vừa đủ cho 'stack' hoặc 'patches'
                logits, _ = model(current_input[:, -64:])
                probs = F.softmax(logits[0, -1,:]/0.6, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                char = dataset.itos[next_token]
                pred_str += char
                current_input = torch.cat([current_input, torch.tensor([[next_token]]).to(device)], dim=1)
            
            is_correct = target.lower() in pred_str.lower()
            results.append(f"Prompt: '{prompt}' | Pred: '{pred_str.strip()}' | {'[OK]' if is_correct else '[FAIL]'}")
    
    model.train()
    return results

class CharTextDataset(Dataset):

    def __init__(self, file_path: str, seq_len: int):
        super().__init__()

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if len(text) < seq_len + 2:
            raise ValueError(
                f"File text quá ngắn. Cần ít nhất {seq_len + 1} ký tự."
            )

        chars = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.vocab_size = len(chars)
        self.seq_len = seq_len

        self.data = torch.tensor(
            [self.stoi[ch] for ch in text],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        input_ids = chunk[:-1]
        labels = chunk[1:]
        return input_ids, labels



def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    step = 0

    while step < cfg.num_steps:
        for input_ids, labels in dataloader:
            if step >= cfg.num_steps:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits, aux_loss = model(input_ids)

            lm_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                labels.reshape(-1),
            )
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

            if step % 50 == 0:
                print(f"\n[Step {step:04d}] Loss: {lm_loss.item():.4f}")
                val_results = validate_cloze(model, dataset, device)
                for res in val_results:
                    print(f"  > {res}")
                print("-" * 50)

            step += 1

    save_checkpoint(model, optimizer, cfg, step)
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
    choices=["dense", "moe", "hmoe"],
)

    parser.add_argument("--max_seq", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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