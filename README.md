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
| `entropy_loss` | (top_p) Khuyến khích router phân phối đều, không quá tự tin |

`total_loss = lm_loss + aux_loss`

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

---

## Huấn luyện

```bash
# Chạy nhanh để thử
python train.py \
  --routing_type top_k \
  --top_k 2 \
  --num_steps 300 \
  --d_model 128 \
  --num_layers 2

# Chạy đầy đủ với top_p
python train.py \
  --routing_type top_p \
  --top_p 0.6 \
  --num_steps 1000 \
  --d_model 256 \
  --num_layers 4 \
  --batch_size 16
```

### Tham số đầy đủ

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--data_path` | `data/tiny_text.txt` | Đường dẫn file văn bản |
| `--routing_type` | `top_k` | Chiến lược routing: `top_k` hoặc `top_p` |
| `--top_k` | `2` | Số expert chọn (top_k) |
| `--top_p` | `0.6` | Ngưỡng xác suất tích lũy (top_p) |
| `--model_type` | `hmoe` | `hmoe` (expert khác cỡ) hoặc `moe` (expert đồng cỡ) |
| `--d_model` | `256` | Chiều hidden vector |
| `--num_layers` | `4` | Số transformer block |
| `--num_heads` | `4` | Số attention head |
| `--max_seq` | `64` | Độ dài sequence tối đa |
| `--batch_size` | `8` | Kích thước batch |
| `--num_steps` | `100` | Tổng số bước train |
| `--learning_rate` | `1e-3` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay (AdamW) |
| `--save_dir` | `checkpoints/` | Thư mục lưu checkpoint |
| `--save_every` | `100` | Lưu mỗi N bước |
| `--seed` | `42` | Random seed |

---

## Checkpoint

Checkpoint được lưu tự động theo định dạng:

```
checkpoints/{model_type}_{routing_type}_step_{N}.pt
```

Để kiểm tra nội dung checkpoint:

```bash
python inspect_checkpoint.py
# Sửa đường dẫn trong __main__ cho đúng file cần xem
```

Output bao gồm: config, tên tensor, shape, thông tin optimizer, lịch sử loss.

---

## Giải thích output khi train

Mỗi 50 bước, script in ra:

```
[Step 0050]
  lm_loss  : 3.1200      ← cross-entropy dự đoán token
  aux_loss : 0.0043      ← penalty cân bằng expert
  total    : 3.1243

[Router stats @ step 50]
  entropy    : 1.2341    ← cao = router phân tán; thấp = tập trung 1 expert
  balance_cv : 0.3120    ← thấp = expert được dùng đều; cao = lệch

[Token expert mix + p-penalty view]
  t=00 token=' '  small_w=0.60 large_w=0.40 ...

  > Prompt: '[ASM] mov rax, ' | Pred: 'rbx...' | [OK]
```

---

## Lưu ý kỹ thuật

- Tokenization ở **cấp ký tự** (character-level), không dùng BPE/WordPiece.
- Model sử dụng **positional embedding học được** (không phải sinusoidal).
- Expert FFN theo kiến trúc **SwiGLU** (gate × up → down), giống LLaMA.
- Gradient clipping với `max_norm=1.0` được áp dụng mỗi bước.
- Causal mask (triangular upper) được tạo trong `model.py`, đảm bảo attention chỉ nhìn về quá khứ.