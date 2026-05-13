# HMoE — Heterogeneous Mixture of Experts Language Model

Mô hình ngôn ngữ nhỏ tích hợp cơ chế **Heterogeneous Mixture of Experts (HMoE)** — mỗi expert có kích thước FFN khác nhau, router động phân phối token tới expert phù hợp theo chiến lược `top_k` hoặc `top_p`.

---

## Cấu trúc thư mục

```
hmoe_from_scratch/
├── model/
│   ├── ffn.py               # LlamaFFNExpert: một expert FFN dạng SwiGLU
│   ├── router.py            # Router softmax + top_k / top_p selection
│   ├── hmoe_layer.py        # HMoELayer: ghép router + experts + aux loss
│   ├── transformer_block.py # TransformerBlockWithHMoE: attention + HMoE
│   └── model.py             # TinyHMoELanguageModel: model hoàn chỉnh
├── data_set.py              # Sinh dataset 4 domain (ASM, AI, MATH, BIO)
├── train.py                 # Script huấn luyện chính
├── metrics_utils.py         # Đo loss, perplexity, tốc độ, VRAM
├── inspect_checkpoint.py    # Kiểm tra nội dung file checkpoint .pt
└── data/                    # (tự sinh) chứa tiny_text.txt + domain files
```

---

## Kiến trúc mô hình

```
Input token IDs  [B, T]
       │
  Token Embedding + Positional Embedding  →  [B, T, d_model]
       │
  ┌────┴──── x N layers ────────────────────┐
  │  LayerNorm → MultiheadAttention         │
  │  LayerNorm → HMoELayer                  │
  │    ├── Router (softmax linear)          │
  │    ├── top_k / top_p selection          │
  │    └── Experts [E0(small)...E3(large)]  │
  └─────────────────────────────────────────┘
       │
  LayerNorm → LM Head (Linear)  →  logits [B, T, vocab]
```

**Điểm khác biệt với MoE thông thường:** các expert có `d_ffn` khác nhau (heterogeneous), khuyến khích router phân công token "khó" cho expert lớn và token "dễ" cho expert nhỏ — tiết kiệm FLOPs theo nội dung.

---

## Chiến lược routing

| Loại | Mô tả | Tham số |
|------|--------|---------|
| `top_k` | Chọn đúng k expert có xác suất cao nhất | `--top_k 2` |
| `top_p` | Chọn linh hoạt cho đến khi tổng xác suất ≥ p | `--top_p 0.6` |

---

## Hàm loss huấn luyện

| Loss | Mục đích |
|------|----------|
| `lm_loss` | Cross-entropy dự đoán token tiếp theo |
| `p_penalty` | Phạt router nếu tập trung quá vào một expert lớn |
| `entropy_loss` | (top_p only) Khuyến khích router không quá tự tin vào một expert |

`total_loss = lm_loss + aux_loss`

> **Lưu ý:** `lm_loss` thấp (dưới 0.5) không đồng nghĩa model generate tốt. Dataset sinh từ template lặp lại khiến model học thống kê ký tự rất nhanh nhưng chưa chắc tổng quát hóa được khi generate autoregressive.

---

## Cài đặt

```bash
pip install torch
```

Không cần thư viện ngoài nào khác.

---

## Sinh dữ liệu

```bash
python data_set.py
```

Tạo ra:

- `data/tiny_text.txt` — mixed dataset 4 domain để train
- `data/domain_0_asm.txt` — Assembly/CPU (easy)
- `data/domain_1_ai.txt` — Transformer/AI (medium)
- `data/domain_2_math.txt` — Toán học (hard)
- `data/domain_3_bio.txt` — Sinh học phân tử (hard+)

> **Quan trọng:** Phải chạy `data_set.py` trước khi train. Nếu dùng file data sai (không sinh từ script này), cloze test sẽ cho kết quả vô nghĩa vì vocab và pattern không khớp domain.

---

## Huấn luyện

```bash
# Chạy nhanh để thử (vài phút)
python train.py \
  --routing_type top_k \
  --top_k 2 \
  --num_steps 400 \
  --d_model 256 \
  --num_layers 4 \
  --batch_size 32

# Chạy đầy đủ để model hội tụ tốt
python train.py \
  --routing_type top_k \
  --top_k 2 \
  --num_steps 3000 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 4 \
  --max_seq 128 \
  --batch_size 32 \
  --learning_rate 3e-4
```

### Tham số đầy đủ

| Tham số | Mặc định (parse_args) | Ý nghĩa |
|---------|----------------------|---------|
| `--data_path` | `data/tiny_text.txt` | Đường dẫn file văn bản |
| `--routing_type` | `top_k` | Chiến lược routing: `top_k` hoặc `top_p` |
| `--top_k` | `2` | Số expert chọn (top_k) |
| `--top_p` | `0.6` | Ngưỡng xác suất tích lũy (top_p) |
| `--model_type` | `hmoe` | `hmoe` (expert khác cỡ) hoặc `moe` (expert đồng cỡ) |
| `--d_model` | `256` | Chiều hidden vector |
| `--num_layers` | `4` | Số transformer block |
| `--num_heads` | `4` | Số attention head |
| `--max_seq` | `128` | Độ dài sequence tối đa |
| `--batch_size` | `32` | Kích thước batch |
| `--num_steps` | `400` | Tổng số bước train |
| `--learning_rate` | `3e-4` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay (AdamW) |
| `--save_dir` | `checkpoints/` | Thư mục lưu checkpoint |
| `--save_every` | `100` | Lưu checkpoint mỗi N bước |
| `--seed` | `42` | Random seed |

> **Lưu ý:** `TrainConfig` dataclass có giá trị mặc định khác với `parse_args` (ví dụ `d_model=128` vs `256`). Luôn chạy qua `python train.py --...` thay vì khởi tạo `TrainConfig()` trực tiếp trong code.

---

## Checkpoint

Checkpoint được lưu tự động theo hai cách:

- **Định kỳ** mỗi `--save_every` bước: `checkpoints/{model_type}_{routing_type}_step_{N}.pt`
- **Cuối training**: lưu thêm một lần nữa khi vòng lặp kết thúc

Để kiểm tra nội dung checkpoint:

```bash
python inspect_checkpoint.py
# Sửa đường dẫn trong __main__ cho đúng file cần xem
```

Output bao gồm: config, danh sách tensor và shape, thông tin optimizer.

> **Lưu ý:** `history` chưa được lưu vào checkpoint — `inspect_checkpoint.py` sẽ báo "Không có history" — đây là hành vi bình thường của phiên bản hiện tại.

---

## Metrics sau training

Sau khi train xong, script tự động lưu file JSON:

```
checkpoints/{model_type}_{routing_type}_metrics.json
```

Bao gồm: config đầy đủ, số tham số, final loss, thời gian train, tokens/sec, peak VRAM (nếu dùng GPU).

---

## Giải thích output khi train

Mỗi 50 bước script in loss, router stats, và cloze test:

```
[Step 0050]
  lm_loss  : 1.8400      ← cross-entropy dự đoán ký tự tiếp theo
  aux_loss : 0.0031      ← penalty cân bằng expert
  total    : 1.8431

[Router stats @ step 50]
  Layer: blocks.0.hmoe
    entropy    : 1.2341  ← cao = phân tán đều; thấp = tập trung 1 expert
    balance_cv : 0.3120  ← thấp = dùng đều; cao = lệch hẳn sang 1 expert
    usage      : E0: 0.412 | E1: 0.388 | E2: 0.124 | E3: 0.076
    mean_probs : E0: 0.310 | E1: 0.290 | E2: 0.220 | E3: 0.180

[Token expert mix + p-penalty view]   ← chỉ in 1 lần ở bước đầu tiên
  t=00 token=' '  small_w=0.60 large_w=0.40 ...

  > Prompt: '[ASM] mov rax, ' | Pred: 'rbx' | [OK]
  > Prompt: '[AI] Top-k routing selects the ' | Pred: 'highest' | [OK]
```

**Đọc router stats:**

| Chỉ số | Tốt | Xấu |
|--------|-----|-----|
| `entropy` | 0.8 – 1.4 | < 0.3 (router bị collapse vào 1 expert) |
| `balance_cv` | < 0.4 | > 1.0 (lệch nặng, expert nhỏ không được dùng) |

**Đọc cloze test:**

- `[OK]` — model generate ra chuỗi chứa target word trong 20 ký tự
- `[FAIL]` — model chưa học được pattern của domain đó
- Dùng **greedy decoding** (argmax) để kết quả nhất quán, không bị nhiễu bởi sampling

---

## Lưu ý kỹ thuật

- Tokenization ở **cấp ký tự** (character-level), không dùng BPE/WordPiece. Vocab size thường 80–120 ký tự tùy dataset.
- Model sử dụng **positional embedding học được** (không phải sinusoidal), giới hạn bởi `max_seq`.
- Expert FFN theo kiến trúc **SwiGLU** (`gate × up → down`), giống LLaMA.
- **Gradient clipping** `max_norm=1.0` áp dụng mỗi bước — cần thiết vì gradient từ nhiều expert cộng lại dễ spike.
- **Causal mask** (triangular upper) tạo trong `model.py`, đảm bảo attention chỉ nhìn về quá khứ.
- `token_loss` tính với `reduction="none"` giữ nguyên shape `[B, T]` rồi mới `.mean()` — cho phép mở rộng sau này như per-token weighting hoặc per-domain loss tracking.
- `print_token_expert_mix_and_p_penalty` chỉ in **1 lần duy nhất** ở bước đầu tiên để tránh output quá dài.
- `print_active_parameter_efficiency` được định nghĩa trong `train.py` nhưng chưa được gọi trong vòng lặp — có thể bật thủ công khi cần debug routing efficiency.
