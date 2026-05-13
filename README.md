# HMoE — Heterogeneous Mixture of Experts Language Model

Dự án xây dựng một mô hình ngôn ngữ nhỏ dùng **Heterogeneous Mixture of Experts (HMoE)**. Thay vì dùng các expert có cùng kích thước như MoE thông thường, HMoE dùng nhiều expert có kích thước FFN khác nhau. Router sẽ quyết định mỗi token đi qua expert nào theo chiến lược `top_k` hoặc `top_p`.

Mục tiêu chính của project:

- huấn luyện language model nhỏ ở mức character-level;
- so sánh **MoE đồng cỡ expert** và **HMoE khác cỡ expert**;
- quan sát router phân phối token vào expert như thế nào;
- đo loss, tốc độ train, VRAM, số tham số và lưu metrics ra JSON để so sánh.

---

## Cấu trúc thư mục

```text
hmoe_from_scratch/
├── model/
│   ├── ffn.py               # LlamaFFNExpert: expert FFN dạng SwiGLU
│   ├── router.py            # Router softmax + top_k / top_p selection
│   ├── hmoe_layer.py        # HMoELayer: router + experts + auxiliary loss
│   ├── transformer_block.py # Transformer block: attention + HMoE
│   └── model.py             # TinyHMoELanguageModel hoàn chỉnh
├── data_set.py              # Sinh dataset 4 domain: ASM, AI, MATH, BIO
├── train.py                 # Script huấn luyện chính
├── metrics_utils.py         # Đếm params, đo speed, VRAM, lưu metrics JSON
├── inspect_checkpoint.py    # Kiểm tra checkpoint .pt
└── data/                    # Tự sinh sau khi chạy data_set.py
```

---

## Kiến trúc mô hình

```text
Input token IDs [B, T]
       │
Token Embedding + Positional Embedding
       │
       ▼
┌────────────────────────────────────────────┐
│ TransformerBlock x num_layers              │
│                                            │
│  LayerNorm → MultiheadAttention → Residual │
│  LayerNorm → HMoELayer/MoELayer → Residual │
│                                            │
│  HMoELayer:                                │
│    ├── Router softmax                      │
│    ├── top_k hoặc top_p routing            │
│    └── Experts E0, E1, E2, E3              │
└────────────────────────────────────────────┘
       │
LayerNorm
       │
LM Head
       │
logits [B, T, vocab_size]
```

Mỗi expert là một FFN kiểu SwiGLU:

```text
output = down_proj(silu(gate_proj(x)) * up_proj(x))
```

---

## MoE và HMoE khác nhau thế nào?

Trong `train.py`, hàm `build_model()` đang tạo expert size như sau:

```python
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
```

Vì vậy:

| Model | Expert size | Ý nghĩa |
|---|---|---|
| `moe` | `[4d, 4d, 4d, 4d]` | Expert đồng cỡ, capacity đều |
| `hmoe` | `[2d, 3d, 4d, 5d]` | Expert nhỏ/lớn khác nhau, tiết kiệm tham số hơn |

HMoE không tự động đảm bảo mỗi expert học đúng một domain. Router học cách phân phối token dựa trên loss của mô hình. Vì model đang dùng character-level tokenizer, router có thể học theo ký tự/pattern thay vì học chuyên biệt domain rõ ràng.

---

## Chiến lược routing

| Routing | Mô tả | Tham số |
|---|---|---|
| `top_k` | Mỗi token chọn đúng `k` expert có xác suất cao nhất | `--top_k 2` |
| `top_p` | Mỗi token chọn số expert linh hoạt đến khi tổng xác suất đạt ngưỡng `p` | `--top_p 0.6` |

Với `top_k=2`, mỗi token luôn đi qua 2 expert. Vì vậy tổng `usage` của các expert thường xấp xỉ `2.0`.

---

## Loss huấn luyện

Trong training loop:

```text
total_loss = lm_loss + aux_loss
```

| Loss | Ý nghĩa |
|---|---|
| `lm_loss` | Cross-entropy dự đoán ký tự tiếp theo |
| `aux_loss` | Loss phụ từ HMoELayer, gồm penalty routing/balance tùy routing |

Khi so sánh chất lượng dự đoán giữa MoE và HMoE, nên ưu tiên nhìn `final_lm_loss`. `final_total_loss` có thêm auxiliary loss nên không phải chỉ số thuần language modeling.

---

## Cài đặt

```bash
pip install torch
```

Nếu muốn đo VRAM và tốc độ GPU, cần cài PyTorch bản hỗ trợ CUDA. Kiểm tra CUDA bằng:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu kết quả là `False`, model sẽ chạy CPU và `peak_vram_mb` sẽ bằng `0.0`.

---

## Sinh dữ liệu

Chạy:

```bash
python data_set.py
```

Script sẽ tạo:

```text
data/tiny_text.txt
data/domain_0_asm.txt
data/domain_1_ai.txt
data/domain_2_math.txt
data/domain_3_bio.txt
```

Trong đó:

| File | Domain |
|---|---|
| `tiny_text.txt` | dữ liệu train trộn 4 domain |
| `domain_0_asm.txt` | Assembly / CPU |
| `domain_1_ai.txt` | Transformer / AI |
| `domain_2_math.txt` | Toán học |
| `domain_3_bio.txt` | Sinh học phân tử |

---

## Huấn luyện

### Train HMoE mặc định

```bash
python train.py \
  --model_type hmoe \
  --routing_type top_k \
  --top_k 2 \
  --num_steps 500 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 4 \
  --max_seq 128 \
  --batch_size 32 \
  --learning_rate 3e-4
```

### Train MoE baseline

```bash
python train.py \
  --model_type moe \
  --routing_type top_k \
  --top_k 2 \
  --num_steps 500 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 4 \
  --max_seq 128 \
  --batch_size 32 \
  --learning_rate 3e-4
```

---

## Tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--data_path` | `data/tiny_text.txt` | Đường dẫn file train |
| `--routing_type` | `top_k` | Chọn `top_k` hoặc `top_p` |
| `--top_k` | `2` | Số expert được chọn khi dùng top-k |
| `--top_p` | `0.6` | Ngưỡng xác suất tích lũy khi dùng top-p |
| `--model_type` | `hmoe` | Chọn `hmoe` hoặc `moe` |
| `--max_seq` | `128` | Độ dài context tối đa |
| `--d_model` | `256` | Kích thước hidden vector |
| `--num_layers` | `4` | Số Transformer block |
| `--num_heads` | `4` | Số attention head |
| `--batch_size` | `32` | Batch size |
| `--num_steps` | `500` | Số bước train |
| `--learning_rate` | `3e-4` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay của AdamW |
| `--save_dir` | `checkpoints` | Thư mục lưu checkpoint và metrics |
| `--save_every` | `100` | Lưu checkpoint mỗi N step |
| `--seed` | `42` | Random seed |

---

## Output trong lúc train

Mỗi 50 step, script in:

```text
[Step 0050]
  lm_loss  : 1.2345
  aux_loss : 0.0123
  total    : 1.2468
```

Sau đó in router stats cho từng layer:

```text
[Router stats @ step 50]
  Layer: blocks.0.hmoe
    entropy    : 1.2841
    balance_cv : 0.2130
    usage      : E0: 0.617 | E1: 0.320 | E2: 0.626 | E3: 0.437
    mean_probs : E0: 0.260 | E1: 0.257 | E2: 0.213 | E3: 0.270
    topk shape : (32, 128, 2)
```

Giải thích:

| Chỉ số | Ý nghĩa |
|---|---|
| `entropy` | Độ phân tán của xác suất router. Cao hơn nghĩa là router ít tự tin hơn, phân phối đều hơn |
| `balance_cv` | Độ lệch sử dụng expert. Thấp hơn nghĩa là usage giữa expert đều hơn |
| `usage` | Tỉ lệ token thực tế được gán cho từng expert |
| `mean_probs` | Xác suất mềm trung bình router gán cho từng expert |
| `topk shape` | Shape của expert index được chọn: `[batch, seq_len, top_k]` |

Lưu ý: với `top_k=2`, tổng các giá trị `usage` thường gần `2.0`, vì mỗi token chọn 2 expert.

---

## Token route debug

Ở step đầu tiên, script in thêm bảng token-route cho tối đa 50 token:

```text
[Token expert mix + p-penalty view]
  t=00 token='A' small_w=0.48 large_w=0.52 p_token=1.152 | E1(S,...):w=0.48 + E3(L,...):w=0.52
```

Ý nghĩa:

| Chỉ số | Ý nghĩa |
|---|---|
| `small_w` | Tổng routing weight rơi vào các expert nhỏ |
| `large_w` | Tổng routing weight rơi vào các expert lớn |
| `p_token` | Ước lượng chi phí/kích thước expert active cho token đó, dựa trên routing weight và expert size chuẩn hóa |
| `w` | Routing weight, tức hệ số trộn output của expert vào output cuối của token |

Với `top_k=2`, các routing weight của token thường có tổng bằng `1.0`.

---

## Cloze test trong lúc train

Script có hàm `validate_cloze()` để sinh thử vài prompt cố định, ví dụ:

```text
Prompt: '[AI] Top-k routing selects the ' | Pred: 'highest' | [OK]
```

Cách đọc:

| Kết quả | Ý nghĩa |
|---|---|
| `[OK]` | Chuỗi sinh ra có chứa target |
| `[FAIL]` | Chưa sinh được target trong tối đa 20 ký tự |

Cloze test chỉ là kiểm tra nhanh trong lúc train, không thay thế đánh giá loss/perplexity.

---

## Checkpoint

Checkpoint được lưu vào:

```text
checkpoints/{model_type}_{routing_type}_step_{N}.pt
```

Ví dụ:

```text
checkpoints/hmoe_top_k_step_100.pt
checkpoints/moe_top_k_step_100.pt
```

Script lưu checkpoint định kỳ theo `--save_every` và lưu thêm một checkpoint cuối training.

Kiểm tra checkpoint:

```bash
python inspect_checkpoint.py
```

---

## Metrics JSON

Sau khi train xong, script tự lưu metrics vào:

```text
checkpoints/{model_type}_{routing_type}_metrics.json
```

Ví dụ:

```text
checkpoints/hmoe_top_k_metrics.json
checkpoints/moe_top_k_metrics.json
```

Metrics gồm:

| Key | Ý nghĩa |
|---|---|
| `model_type` | `hmoe` hoặc `moe` |
| `routing_type` | `top_k` hoặc `top_p` |
| `top_k`, `top_p` | Tham số routing |
| `d_model`, `num_layers`, `num_heads`, `max_seq` | Config mô hình |
| `batch_size`, `num_steps`, `learning_rate`, `weight_decay` | Config train |
| `vocab_size` | Số ký tự trong vocab |
| `total_params` | Tổng số tham số |
| `trainable_params` | Số tham số train được |
| `final_lm_loss` | LM loss cuối training |
| `final_aux_loss` | Auxiliary loss cuối training |
| `final_total_loss` | Tổng loss cuối training |
| `train_time_sec` | Tổng thời gian train |
| `tokens_seen` | Tổng số token đã train |
| `tokens_per_sec` | Tốc độ train |
| `peak_vram_mb` | VRAM peak nếu chạy CUDA; CPU thì bằng 0 |

Xem JSON:

```bash
python -m json.tool checkpoints/hmoe_top_k_metrics.json
```

---

## So sánh HMoE và MoE

Train hai model với cùng cấu hình:

```bash
python train.py --model_type moe  --routing_type top_k --top_k 2 --num_steps 500
python train.py --model_type hmoe --routing_type top_k --top_k 2 --num_steps 500
```

Sau đó dùng script so sánh metrics, ví dụ:

```bash
python compare_metric.py checkpoints/moe_top_k_metrics.json checkpoints/hmoe_top_k_metrics.json
```

Bảng so sánh thường có dạng:

```text

METRICS COMPARISON
metric           | moe        | hmoe       | diff       | better
-----------------+------------+------------+------------+-------
final_lm_loss    | 0.2647     | 0.2555     | 0.0092     | hmoe  
final_total_loss | 0.3456     | 0.3279     | 0.0177     | hmoe  
tokens_per_sec   | 31186.1992 | 39070.2396 | -7884.0404 | hmoe  
peak_vram_mb     | 1011.5376  | 931.3872   | 80.1504    | hmoe  
total_params     | 13719552   | 12146688   | 1572864    | hmoe  
train_time_sec   | 65.6701    | 52.4184    | 13.2517    | hmoe  
```

Cách đọc:

| Metric | Tốt hơn khi |
|---|---|
| `final_lm_loss` | thấp hơn |
| `final_total_loss` | thấp hơn, nhưng có ảnh hưởng aux loss |
| `tokens_per_sec` | cao hơn |
| `peak_vram_mb` | thấp hơn |
| `total_params` | thấp hơn |
| `train_time_sec` | thấp hơn |

Nên so sánh hiệu suất trên CUDA nếu có GPU. Khi chạy CPU, `tokens_per_sec` và `train_time_sec` có thể bị ảnh hưởng mạnh bởi overhead Python, DataLoader, nhiệt máy và tác vụ nền.

---

## Một số lưu ý 

### 1. Loss thấp không đồng nghĩa generate tốt

Dataset được sinh từ template lặp lại, character-level model có thể đạt loss thấp nhanh. Vì vậy nên xem loss như chỉ số train trên dataset này, không phải bằng chứng tổng quát hóa mạnh.

### 2. HMoE không chắc luôn tốt hơn MoE

Nếu MoE có nhiều tham số hơn, MoE có thể đạt loss thấp hơn. Nếu HMoE ít tham số hơn, HMoE thường có lợi về VRAM và tốc độ trên CUDA. Kết quả phụ thuộc vào:

- số tham số;
- `top_k` hoặc `top_p`;
- seed;
- số step;
- CPU hay CUDA;
- cấu hình expert size.

Với `top_k=2`, tổng `usage` xấp xỉ 2.0 là bình thường.

### 4. Chuyên biệt expert không phải lúc nào cũng rõ

Vì tokenizer đang ở cấp ký tự, các domain dùng nhiều ký tự chung. Router có thể học theo pattern/ký tự thay vì tách rõ `ASM`, `AI`, `MATH`, `BIO`. Muốn kiểm tra chuyên biệt mạnh hơn, nên bổ sung đánh giá domain usage hoặc ablation từng expert.

---

## Lưu ý kỹ thuật

- Tokenizer là **character-level**, không dùng BPE/WordPiece.
- Position embedding là **learned positional embedding**.
- Attention dùng causal mask để chỉ nhìn về quá khứ.
- Expert FFN dùng **SwiGLU** giống phong cách LLaMA.
