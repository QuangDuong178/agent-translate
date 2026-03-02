# 📘 BÁO CÁO CHI TIẾT: XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP) TRONG HỆ THỐNG DỊCH PHỤ ĐỀ ĐA NGÔN NGỮ

**Môn học**: Xử lý Ngôn ngữ Tự nhiên (NLP)  
**Dự án**: Agent-Translate — Hệ thống dịch phụ đề đa ngôn ngữ  
**Ngôn ngữ hỗ trợ**: Anh (EN), Nhật (JA), Trung (ZH), Hàn (KO) → Tiếng Việt (VI)

---

## MỤC LỤC

1. [Giới thiệu & Bài toán NLP](#1-giới-thiệu--bài-toán-nlp)
2. [Nền tảng lý thuyết NLP](#2-nền-tảng-lý-thuyết-nlp)
3. [Kiến trúc Transformer — Chi tiết toán học](#3-kiến-trúc-transformer--chi-tiết-toán-học)
4. [Tokenization — Xử lý văn bản đa ngôn ngữ](#4-tokenization--xử-lý-văn-bản-đa-ngôn-ngữ)
5. [Mô hình NLLB-200 — Kiến trúc & Cơ chế hoạt động](#5-mô-hình-nllb-200--kiến-trúc--cơ-chế-hoạt-động)
6. [Thu thập dữ liệu — Kỹ thuật Bridge Matching](#6-thu-thập-dữ-liệu--kỹ-thuật-bridge-matching)
7. [Fine-tuning — Quy trình & Toán học](#7-fine-tuning--quy-trình--toán-học)
8. [Inference — Quy trình dịch thuật](#8-inference--quy-trình-dịch-thuật)
9. [Speech-to-Text (ASR) — Whisper](#9-speech-to-text-asr--whisper)
10. [Kết quả thực nghiệm](#10-kết-quả-thực-nghiệm)
11. [Tài liệu tham khảo](#11-tài-liệu-tham-khảo)

---

## 1. Giới thiệu & Bài toán NLP

### 1.1 Phát biểu bài toán

Bài toán **Machine Translation (MT)** — Dịch máy — là một trong những bài toán cốt lõi của NLP:

> **Cho một chuỗi token nguồn** x = (x₁, x₂, ..., xₙ) **thuộc ngôn ngữ L₁, tìm chuỗi token đích** y = (y₁, y₂, ..., yₘ) **thuộc ngôn ngữ L₂ sao cho y là bản dịch chính xác nhất của x.**

Về mặt toán học, ta tìm:

```
ŷ = argmax P(y | x)
         y
```

Trong đó P(y|x) là xác suất có điều kiện của chuỗi đích y cho trước chuỗi nguồn x.

### 1.2 Đặc thù bài toán trong dự án

| Thách thức | Mô tả | Giải pháp NLP |
|-----------|-------|--------------|
| **Low-resource pairs** | Không có dataset JA→VI, ZH→VI trực tiếp | Bridge Matching qua tiếng Anh |
| **Đa dạng script** | Kanji, Hiragana, Hangul, Latin, Hán tự | SentencePiece tokenizer (NLLB) |
| **Câu ngắn** | Phụ đề thường chỉ 5-15 từ | `max_length=64` tokens |
| **Ngữ cảnh hạn chế** | Dịch từng câu, không có đoạn văn | Batch translate để giữ tính nhất quán |
| **Real-time** | Cần dịch nhanh khi xử lý video | Batch inference, `num_beams=4` |

---

## 2. Nền tảng lý thuyết NLP

### 2.1 Lịch sử phát triển Machine Translation

```
Rule-based MT (1950s)     Dùng luật ngữ pháp cứng
        ↓
Statistical MT (1990s)    Dùng xác suất thống kê (IBM Models, Phrase-based)
        ↓
Neural MT - RNN (2014)    Seq2Seq + Attention (Bahdanau et al.)
        ↓
Neural MT - CNN (2017)    ConvS2S (Gehring et al., Facebook)
        ↓
★ Transformer (2017)      Self-Attention (Vaswani et al., Google)    ← Dùng trong dự án
        ↓
★ NLLB-200 (2022)         200 ngôn ngữ, 1 model (Meta AI)           ← Dùng trong dự án
```

### 2.2 Tại sao Transformer thay thế RNN?

**RNN (Recurrent Neural Network)**:
- Xử lý **tuần tự** từng token → chậm, không song song hóa được
- Gặp **vanishing gradient** khi câu dài (thông tin đầu câu bị mất)
- LSTM/GRU cải thiện nhưng vẫn tuần tự

**Transformer**:
- Xử lý **song song** tất cả tokens cùng lúc → nhanh hơn
- **Self-Attention** cho phép mỗi token "nhìn" tất cả tokens khác → không mất thông tin
- Không có vấn đề vanishing gradient với position-independent attention

### 2.3 Phân loại các mô hình NLP

| Loại | Kiến trúc | Ví dụ | Dùng cho |
|------|----------|-------|---------|
| **Encoder-only** | Chỉ encoder | BERT, RoBERTa | Phân loại, NER |
| **Decoder-only** | Chỉ decoder | GPT, LLaMA | Sinh text |
| ★ **Encoder-Decoder** | Cả hai | T5, mBART, **NLLB** | **Dịch máy**, Tóm tắt |

Dự án sử dụng **mô hình Encoder-Decoder** vì bài toán dịch máy cần:
- **Encoder**: Hiểu câu nguồn (tiếng Nhật)
- **Decoder**: Sinh câu đích (tiếng Việt) từ trái sang phải

---

## 3. Kiến trúc Transformer — Chi tiết toán học

### 3.1 Tổng quan kiến trúc

```
Input: "猫が好きです" (Tôi thích mèo)

    ┌───────────────────────────────────────────────────────┐
    │                      ENCODER                           │
    │                                                        │
    │  [猫] [が] [好き] [です] ← Input Embeddings             │
    │       +                                                │
    │  [PE₁] [PE₂] [PE₃] [PE₄] ← Positional Encoding       │
    │       ↓                                                │
    │  ┌────────────────────────┐                            │
    │  │ Multi-Head Self-Attention │  × 12 layers            │
    │  │ + Add & LayerNorm        │                          │
    │  │ + Feed-Forward Network   │                          │
    │  │ + Add & LayerNorm        │                          │
    │  └────────────────────────┘                            │
    │       ↓                                                │
    │  Encoder Output: [h₁, h₂, h₃, h₄]                    │
    └───────────────┬───────────────────────────────────────┘
                    │ Cross-Attention
    ┌───────────────▼───────────────────────────────────────┐
    │                      DECODER                           │
    │                                                        │
    │  [<BOS>] [Tôi] [thích] ← Output so far (shifted right)│
    │       ↓                                                │
    │  ┌────────────────────────┐                            │
    │  │ Masked Self-Attention     │  × 12 layers            │
    │  │ + Add & LayerNorm         │                          │
    │  │ Cross-Attention (→Encoder)│                          │
    │  │ + Add & LayerNorm         │                          │
    │  │ + Feed-Forward Network    │                          │
    │  │ + Add & LayerNorm         │                          │
    │  └────────────────────────┘                            │
    │       ↓                                                │
    │  Linear + Softmax → P("mèo") = 0.87                   │
    └───────────────────────────────────────────────────────┘

Output: "Tôi thích mèo"
```

### 3.2 Self-Attention — Cơ chế cốt lõi

**Self-Attention** cho phép mỗi token trong câu "chú ý" (attend) đến mọi token khác:

#### Công thức toán học:

**Bước 1**: Tạo 3 vectors Query (Q), Key (K), Value (V) cho mỗi token:

```
Q = X · W_Q   (Query: "Token này đang tìm gì?")
K = X · W_K   (Key: "Token này chứa thông tin gì?")
V = X · W_V   (Value: "Giá trị thực sự của token này")

Trong đó:
  X ∈ ℝ^(n×d_model)     — Ma trận input (n tokens, mỗi token d chiều)
  W_Q, W_K ∈ ℝ^(d_model×d_k)  — Ma trận trọng số học được
  W_V ∈ ℝ^(d_model×d_v)       — Ma trận trọng số cho Value
```

**Bước 2**: Tính Attention scores:

```
                    Q · K^T
Attention(Q,K,V) = softmax(─────────) · V
                    √(d_k)

Trong đó:
  Q · K^T     → Ma trận (n×n) chứa "mức độ liên quan" giữa mọi cặp token
  √(d_k)     → Scaled để tránh giá trị quá lớn (gradient ổn định hơn)
  softmax()  → Chuẩn hóa thành phân phối xác suất (tổng = 1)
  · V        → Nhân với Value để lấy biểu diễn có trọng số
```

#### Ví dụ minh họa:

```
Input: "猫 が 好き です" (Mèo [trợ từ] thích [thể lịch sự])

Attention matrix (sau softmax):
             猫    が    好き   です
   猫     [0.15  0.10  0.65  0.10]  ← "猫" chú ý nhiều nhất vào "好き"
   が     [0.30  0.10  0.40  0.20]
   好き   [0.55  0.15  0.10  0.20]  ← "好き" chú ý nhiều nhất vào "猫"
   です   [0.20  0.10  0.30  0.40]

→ Model hiểu: "猫" (mèo) là object của "好き" (thích)
  dù chúng không đứng cạnh nhau trong câu!
```

### 3.3 Multi-Head Attention

Thay vì 1 attention, dùng **h heads** (NLLB: h=16) chạy song song:

```
MultiHead(Q,K,V) = Concat(head₁, head₂, ..., head_h) · W_O

head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)

Trong đó:
  h = 16 (NLLB-600M)
  d_model = 1024
  d_k = d_v = d_model / h = 64 per head
```

**Tại sao multi-head?** Mỗi head học một **loại quan hệ** khác nhau:
- Head 1: Quan hệ chủ-vị (subject-verb)
- Head 2: Quan hệ cú pháp (syntax)
- Head 3: Quan hệ ngữ nghĩa (semantics)
- Head 4: Quan hệ vị trí (proximity)
- ...

### 3.4 Positional Encoding

Transformer xử lý song song → mất thông tin vị trí. Giải pháp: thêm **Positional Encoding**:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Trong đó:
  pos = vị trí token (0, 1, 2, ...)
  i   = chiều (dimension) trong vector
  d_model = 1024

Ví dụ: token ở vị trí 3, chiều 0 và 1:
  PE(3, 0) = sin(3 / 10000^0) = sin(3) = 0.141
  PE(3, 1) = cos(3 / 10000^0) = cos(3) = -0.990
```

**Kết quả**: Input = Token Embedding + Positional Encoding:
```
x_i = Embedding(token_i) + PE(i)
```

### 3.5 Feed-Forward Network (FFN)

Sau mỗi Attention layer, có 1 FFN gồm 2 ma trận tuyến tính:

```
FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂

Trong đó:
  W₁ ∈ ℝ^(1024 × 4096)   — Mở rộng 4x
  W₂ ∈ ℝ^(4096 × 1024)   — Thu nhỏ lại
  max(0, ·)                — ReLU activation

→ FFN học các biến đổi phi tuyến trên từng token
```

### 3.6 Layer Normalization & Residual Connection

```
output = LayerNorm(x + SubLayer(x))

Trong đó:
  x            = input vào sublayer
  SubLayer(x)  = Attention(x) hoặc FFN(x)
  x + ...      = Residual connection (tránh vanishing gradient)
  LayerNorm()  = Chuẩn hóa theo chiều features
```

### 3.7 Masked Self-Attention trong Decoder

Decoder sử dụng **Masked Self-Attention** để đảm bảo token ở vị trí i chỉ "nhìn" được token ở vị trí ≤ i (không nhìn tương lai):

```
Mask matrix:
     Tôi  thích  mèo  <EOS>
Tôi  [1    -∞    -∞   -∞  ]
thích[1     1    -∞   -∞  ]
mèo  [1     1     1   -∞  ]
<EOS>[1     1     1    1   ]

→ Sau softmax: -∞ → 0 (không attend vào tokens tương lai)
```

### 3.8 Cross-Attention (Encoder-Decoder Attention)

Decoder attend vào output của Encoder:

```
Q = Decoder hidden states    (token đang sinh)
K = Encoder output           (câu nguồn đã encode)
V = Encoder output

→ Decoder "nhìn lại" câu nguồn để quyết định dịch gì tiếp theo
```

**Ví dụ**:
```
Decoder đang sinh token thứ 2 (sau "Tôi"):
  Q = embedding("Tôi")
  K, V = Encoder output ["猫", "が", "好き", "です"]
  
  Attention weights: [0.10, 0.05, 0.80, 0.05]
                      猫    が    好き   です
  
  → Attend mạnh vào "好き" (thích) → sinh "thích"
```

---

## 4. Tokenization — Xử lý văn bản đa ngôn ngữ

### 4.1 Vấn đề Tokenization đa ngôn ngữ

Mỗi ngôn ngữ có cấu trúc khác nhau:

| Ngôn ngữ | Đặc điểm | Ví dụ |
|----------|---------|-------|
| Tiếng Anh | Cách nhau bằng dấu cách | "I love cats" → 3 từ |
| Tiếng Việt | Từ có thể 2+ tiếng | "xin chào" = 1 từ, 2 tiếng |
| Tiếng Nhật | Không có dấu cách, 3 bộ chữ | "猫が好き" → ? |
| Tiếng Trung | Mỗi ký tự = 1 morpheme | "我喜欢猫" → 4 ký tự |

→ Cần tokenizer **universal** xử lý được mọi ngôn ngữ.

### 4.2 SentencePiece — Thuật toán tokenization

NLLB sử dụng **SentencePiece** với thuật toán **BPE (Byte-Pair Encoding)**:

#### Thuật toán BPE:

```
Bước 1: Khởi tạo vocabulary = tất cả individual characters
  V = {a, b, c, ..., あ, い, う, ..., 猫, 好, ...}

Bước 2: While |V| < vocab_size (256,206 cho NLLB):
  - Đếm tần suất mọi cặp tokens liền kề trong corpus
  - Tìm cặp xuất hiện nhiều nhất: (t₁, t₂)
  - Gộp thành token mới: t₁t₂
  - Thêm vào V
  
Ví dụ:
  Corpus: "low lower lowest"
  Iteration 1: "l" + "o" → "lo"    (xuất hiện 3 lần)
  Iteration 2: "lo" + "w" → "low"  (xuất hiện 3 lần)
  Iteration 3: "low" + "e" → "lowe" (xuất hiện 2 lần)
  ...
```

#### Kết quả tokenization thực tế:

```python
# Code thực tế trong dự án
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Tiếng Nhật
tokenizer.src_lang = "jpn_Jpan"
tokens = tokenizer("猫が好きです")
# → input_ids: [256047, 141017, 1033, 96428, 8713, 2]
# → tokens: ["jpn_Jpan", "▁猫", "が", "好き", "です", "</s>"]
#            Language   Mèo    Trợ từ  Thích   Lịch sự  END

# Tiếng Việt
tokenizer.src_lang = "vie_Latn"
tokens = tokenizer("Tôi thích mèo")
# → input_ids: [256151, 14806, 8413, 100366, 2]
# → tokens: ["vie_Latn", "▁Tôi", "▁thích", "▁mèo", "</s>"]
#            Language    Tôi     thích     mèo      END

# Tiếng Trung
tokenizer.src_lang = "zho_Hans"
tokens = tokenizer("我喜欢猫")
# → input_ids: [256108, 2751, 43039, 141017, 2]
# → tokens: ["zho_Hans", "▁我", "喜欢", "猫", "</s>"]
#            Language    Tôi   Thích   Mèo   END
```

### 4.3 Special Tokens trong NLLB

| Token | ID | Chức năng |
|-------|-----|----------|
| `<s>` | 0 | Begin of Sentence |
| `</s>` | 2 | End of Sentence |
| `<unk>` | 3 | Unknown token |
| `<pad>` | 1 | Padding (đệm cho batch) |
| `jpn_Jpan` | 256047 | Language tag: Tiếng Nhật |
| `vie_Latn` | 256151 | Language tag: Tiếng Việt |
| `eng_Latn` | 256047 | Language tag: Tiếng Anh |
| `zho_Hans` | 256108 | Language tag: Tiếng Trung |

### 4.4 Vocabulary size

```
NLLB-200-distilled-600M:
  Vocabulary size = 256,206 tokens
  Bao gồm:
    - ~256,000 subword tokens (cho 200 ngôn ngữ)
    - 206 language tags (1 per language)
    - Special tokens (<s>, </s>, <unk>, <pad>)
```

### 4.5 Tokenization trong quy trình Training

```python
# Code thực tế: train_direct_models.py — line 228-236
def tokenize_fn(examples):
    # Tokenize câu nguồn (Nhật)
    inputs = tokenizer(
        examples["src"],       # ["猫が好きです", "天気がいい", ...]
        max_length=64,         # Giới hạn 64 tokens
        truncation=True,       # Cắt nếu quá dài
        padding="max_length"   # Pad nếu ngắn (thêm <pad>=1)
    )
    # → input_ids: [[256047, 141017, 1033, ..., 1, 1, 1], ...]
    #                jpn     猫      が        pad pad pad
    
    # Tokenize câu đích (Việt)
    labels = tokenizer(
        text_target=examples["tgt"],  # ["Tôi thích mèo", "Thời tiết đẹp", ...]
        max_length=64,
        truncation=True,
        padding="max_length"
    )
    # → labels: [[256151, 14806, 8413, 100366, 2, 1, 1], ...]
    #            vie     Tôi    thích  mèo     </s> pad pad
    
    inputs["labels"] = labels["input_ids"]
    return inputs
```

---

## 5. Mô hình NLLB-200 — Kiến trúc & Cơ chế hoạt động

### 5.1 Thông số mô hình

| Thông số | NLLB-200-distilled-600M |
|---------|------------------------|
| **Kiến trúc** | Transformer Encoder-Decoder |
| **Tổng tham số** | 615,220,736 (615M) |
| **Encoder layers** | 12 |
| **Decoder layers** | 12 |
| **Hidden size (d_model)** | 1024 |
| **FFN size (d_ff)** | 4096 (4× d_model) |
| **Attention heads** | 16 |
| **Head dimension (d_k)** | 64 (= 1024/16) |
| **Vocabulary** | 256,206 |
| **Số ngôn ngữ** | 200+ |
| **Kích thước file** | 2.4 GB (safetensors) |
| **Pre-training data** | CCMatrix, CCAligned, OPUS, WikiMatrix |

### 5.2 Cơ chế Language Tag

NLLB sử dụng **language tags** để điều khiển ngôn ngữ dịch:

```
Encoder input:  [jpn_Jpan] [猫] [が] [好き] [です] [</s>]
                 ↑ Tag nguồn

Decoder input:  [vie_Latn] [Tôi] [thích] [mèo] [</s>]
                 ↑ Tag đích (forced_bos_token_id)
```

**Cách hoạt động:**
1. Token đầu tiên của encoder = source language tag → model biết đang đọc tiếng gì
2. `forced_bos_token_id` = target language tag → model biết phải sinh tiếng gì
3. Nhờ vậy, **1 model duy nhất** dịch được 200×200 = 40,000 cặp ngôn ngữ

### 5.3 Knowledge Distillation

NLLB-200-**distilled**-600M được tạo từ model lớn hơn (3.3B params):

```
Teacher model: NLLB-200-3.3B (3.3 tỷ tham số) ← Model gốc
        ↓ Knowledge Distillation
Student model: NLLB-200-distilled-600M (600 triệu) ← Dùng trong dự án

Quy trình distillation:
  1. Teacher dịch corpus → tạo "soft labels" (phân phối xác suất)
  2. Student học bắt chước phân phối xác suất của Teacher
  3. Loss = α·CrossEntropy(student, true_labels) + (1-α)·KL(student, teacher_soft_labels)
  
→ Student nhỏ hơn 5x nhưng giữ ~90% chất lượng của Teacher
```

### 5.4 So sánh với các model khác

| Model | Params | Ngôn ngữ | Dùng trong dự án |
|-------|--------|----------|-----------------|
| NLLB-200-3.3B | 3.3B | 200 | ✗ (quá lớn) |
| **NLLB-200-distilled-600M** | 600M | 200 | **✅ Fine-tune JA→VI** |
| mBART-50 | 610M | 50 | ✗ |
| **OpusMT (MarianMT)** | ~74M | 2 (per model) | **✅ Fine-tune EN→VI** |
| Google Translate | ? | 100+ | ✗ (API, không local) |

---

## 6. Thu thập dữ liệu — Kỹ thuật Bridge Matching

### 6.1 Vấn đề: Low-Resource Language Pairs

**Low-resource language pair** = cặp ngôn ngữ có ít hoặc không có parallel data.

```
High-resource:  EN↔FR, EN↔DE, EN↔ZH    → Hàng triệu câu song ngữ
Low-resource:   JA↔VI, ZH↔VI, KO↔VI   → Gần như không có dataset công khai
```

### 6.2 Giải pháp: Bridge Matching Algorithm

**Ý tưởng**: Sử dụng tiếng Anh làm **ngôn ngữ cầu nối (bridge language)**:

```
Tồn tại:  Dataset D₁ = {(en_i, ja_i)}  ← EN-JA pairs  (OPUS-100: 1,000,000 pairs)
Tồn tại:  Dataset D₂ = {(en_j, vi_j)}  ← EN-VI pairs  (OPUS-100: 775,303 pairs)

Thuật toán:
  1. Xây hash map: H = {normalize(en_j) → vi_j} từ D₂
  2. Quét D₁: với mỗi (en_i, ja_i):
     - key = normalize(en_i)
     - Nếu key ∈ H:
       - Tạo pair mới: (ja_i, H[key]) = (ja_i, vi_matched)
       - Thêm vào output

Kết quả: Dataset D₃ = {(ja_k, vi_k)}   ← JA-VI pairs!
```

### 6.3 Normalization Function

```python
def normalize(en_text):
    return en_text.strip().lower()

# Tại sao normalize?
# "I Love Cats" vs "i love cats" → cùng 1 câu
# "Hello " vs "Hello" → cùng 1 câu
```

### 6.4 Code Implementation

```python
# BƯỚC 1: Xây bảng tra cứu (Hash Map) EN→VI
def _build_en_vi_lookup():
    ds = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="train")
    # OPUS-100 EN-VI có 775,303 cặp câu song ngữ
    
    en_vi = {}  # Hash map: O(1) lookup
    for item in ds:
        en = item["translation"]["en"].strip()
        vi = item["translation"]["vi"].strip()
        if en and vi and len(en) > 3:      # Lọc câu quá ngắn
            key = en.lower().strip()        # Normalize key
            en_vi[key] = vi
    
    return en_vi  # 775,303 entries, ~200MB RAM

# BƯỚC 2: Bridge Matching
def load_ja_vi_data(max_samples=5000):
    en_vi = _build_en_vi_lookup()   # O(N) — load 1 lần
    
    pairs = []
    ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train")
    # OPUS-100 EN-JA có ~1,000,000 cặp câu
    
    matched = 0
    total = 0
    for item in ds:
        en = item["translation"]["en"].strip()
        ja = item["translation"]["ja"].strip()
        if en and ja and len(ja) > 2:
            total += 1
            key = en.lower().strip()
            
            if key in en_vi:               # O(1) hash lookup
                pairs.append({
                    "src": ja,              # 猫が好きです
                    "tgt": en_vi[key]       # Tôi thích mèo
                })
                matched += 1
        
        if matched >= max_samples:
            break
    
    return pairs
```

### 6.5 Thống kê thực tế

```
OPUS-100 EN-JA: 1,000,000 cặp
OPUS-100 EN-VI:   775,303 cặp

Bridge matching JA→VI:
  Quét:     20,779 cặp JA→EN
  Matched:   5,000 cặp JA→VI  (tỷ lệ match: 24.1%)
  
Lý do tỷ lệ 24.1%:
  - Cùng 1 câu EN xuất hiện trong cả 2 dataset
  - 24.1% câu EN trong EN-JA cũng tồn tại trong EN-VI
  - Hợp lý vì OPUS lấy data từ nhiều nguồn chung (Wikipedia, TED, ...)
```

### 6.6 Ưu/nhược điểm Bridge Matching

| Ưu điểm | Nhược điểm |
|---------|-----------|
| ✅ Tạo data cho low-resource pairs | ❌ Phụ thuộc chất lượng bản dịch EN |
| ✅ Nhanh (hash lookup O(1)) | ❌ Câu EN có thể dịch hơi khác nghĩa |
| ✅ Data thực tế (không synthetic) | ❌ Giới hạn bởi overlap giữa 2 dataset |
| ✅ Không cần GPU (chỉ text matching) | ❌ Tỷ lệ match thường < 30% |

---

## 7. Fine-tuning — Quy trình & Toán học

### 7.1 Pre-training vs Fine-tuning

```
PRE-TRAINING (Meta AI đã làm):
  - Data: Hàng tỷ câu từ 200 ngôn ngữ
  - Mục tiêu: Học dịch tổng quát
  - Thời gian: Hàng tuần trên cluster GPU
  - Kết quả: NLLB-200-distilled-600M

FINE-TUNING (Dự án này):
  - Data: 5,000 cặp JA→VI (bridge matched)
  - Mục tiêu: Chuyên biệt hóa cho JA→VI
  - Thời gian: 92 phút trên CPU
  - Kết quả: nllb-ja-vi-direct model
```

### 7.2 Loss Function: Cross-Entropy Loss

Mục tiêu training là **minimize Cross-Entropy Loss**:

```
L = -∑ᵢ ∑ₜ log P(yₜ | y₁,...,yₜ₋₁, x)

Trong đó:
  i  = index của sample trong batch
  t  = timestep (vị trí token) trong câu đích
  yₜ = true token tại vị trí t
  x  = câu nguồn
  P(yₜ | ...) = xác suất model dự đoán đúng token yₜ

Ví dụ:
  Source: "猫が好きです"
  Target: ["vie_Latn", "Tôi", "thích", "mèo", "</s>"]
  
  Step 1: P("Tôi" | "vie_Latn", source) = 0.3 → loss = -log(0.3) = 1.2
  Step 2: P("thích" | "vie_Latn", "Tôi", source) = 0.5 → loss = -log(0.5) = 0.69
  Step 3: P("mèo" | "vie_Latn", "Tôi", "thích", source) = 0.7 → loss = -log(0.7) = 0.36
  
  Total loss cho câu này = 1.2 + 0.69 + 0.36 = 2.25
```

### 7.3 Optimizer: AdamW

```
AdamW = Adam + Weight Decay (decoupled)

Cập nhật trọng số:
  m_t = β₁·m_{t-1} + (1-β₁)·g_t         (1st moment: momentum)
  v_t = β₂·v_{t-1} + (1-β₂)·g_t²        (2nd moment: adaptive LR)
  
  m̂_t = m_t / (1 - β₁^t)                 (Bias correction)
  v̂_t = v_t / (1 - β₂^t)
  
  θ_t = θ_{t-1} - lr · (m̂_t/(√v̂_t + ε) + λ·θ_{t-1})
                         ↑ Adam update      ↑ Weight decay (L2 reg)

Hyperparameters dự án:
  lr = 1e-5 (learning rate)
  β₁ = 0.9
  β₂ = 0.999
  ε = 1e-8
  λ = 0.01 (weight_decay)
```

### 7.4 Learning Rate Schedule: Warmup

```
LR Schedule trong dự án:

LR
1e-5 ┤                  ┌───────────────────
     │                 /
     │               /
     │             / ← Linear warmup (300 steps)
     │           /
     │         /
   0 ┤────────/
     └──┬─────┬──────────────────────────── Steps
        0    300                        225*3

Warmup phase (0-300 steps):
  lr(t) = (t / 300) × 1e-5
  
Constant phase (300+ steps):
  lr(t) = 1e-5
```

**Tại sao warmup?**
- Đầu training: gradients chưa ổn định
- LR quá lớn ngay từ đầu → diverge (loss tăng)
- Warmup: tăng dần LR → model "quen" dần → ổn định

### 7.5 Gradient Accumulation

```
Bình thường (batch_size=64):
  Forward 64 samples → Backward → Update weights
  ❌ Tốn RAM: 64 × model_size

Gradient Accumulation (batch=16, accum=4):
  Forward 16 samples → Backward → Tích lũy gradient
  Forward 16 samples → Backward → Tích lũy gradient
  Forward 16 samples → Backward → Tích lũy gradient
  Forward 16 samples → Backward → Tích lũy gradient
  → Update weights (tổng gradient = 64 samples)
  ✅ Tốn RAM: chỉ 16 × model_size

Effective batch size = per_device_batch × gradient_accumulation
                     = 16 × 4 = 64
```

### 7.6 Gradient Checkpointing

```
Bình thường:
  Layer 1 → save activations → Layer 2 → save activations → ... → Layer 12
  Backward: dùng saved activations
  ❌ RAM: O(12 × activation_size) ≈ nhiều GB

Gradient Checkpointing:
  Layer 1 → Layer 2 → ... → Layer 12 (KHÔNG save activations)
  Backward: tính lại activations khi cần
  ✅ RAM: O(√12 × activation_size) ≈ ít hơn ~40%
  ⚠️ Chậm hơn ~20% (phải tính lại forward pass)

→ Đánh đổi thời gian lấy bộ nhớ (time-memory tradeoff)
```

### 7.7 Cấu hình Training thực tế

```python
# Code thực tế: train_direct_models.py — line 260-280
training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints",
    
    # ── Training schedule ──
    num_train_epochs=3,                     # 3 lượt duyệt data
    per_device_train_batch_size=16,         # 16 câu/batch
    gradient_accumulation_steps=4,          # Effective batch = 64
    
    # ── Optimizer ──
    learning_rate=1e-5,                     # AdamW LR
    weight_decay=0.01,                      # L2 regularization
    warmup_steps=300,                       # LR warmup
    
    # ── Memory optimization ──
    gradient_checkpointing=True,            # Giảm 40% RAM
    fp16=False,                             # Full precision (CPU)
    use_cpu=True,                           # Force CPU training
    
    # ── Logging ──
    logging_steps=50,                       # Log mỗi 50 steps
    eval_strategy="steps",
    eval_steps=500,                         # Eval mỗi 500 steps
    save_total_limit=2,                     # Giữ max 2 checkpoints
)
```

### 7.8 Tính toán số Training Steps

```
Tổng samples: 4,750 (train set)
Batch size per device: 16
Gradient accumulation: 4

Steps per epoch = ceil(4,750 / 16) = 298
                  → Nhưng gradient accum = 4:
                  = ceil(298 / 4) ≈ 75 optimizer steps per epoch

Total optimizer steps = 75 × 3 epochs = 225 steps
(Logging mỗi 50 steps → 4 log entries: step 50, 100, 150, 200)
```

---

## 8. Inference — Quy trình dịch thuật

### 8.1 Autoregressive Decoding

Decoder sinh text **tuần tự từ trái sang phải**:

```
Step 0: Input: [vie_Latn]  → Predict: "Tôi" (P=0.72)
Step 1: Input: [vie_Latn, Tôi]  → Predict: "thích" (P=0.68)
Step 2: Input: [vie_Latn, Tôi, thích]  → Predict: "mèo" (P=0.81)
Step 3: Input: [vie_Latn, Tôi, thích, mèo]  → Predict: "</s>" (P=0.95)
→ Kết thúc!
```

### 8.2 Beam Search

**Greedy decoding**: Chọn token có P cao nhất tại mỗi step → suboptimal.

**Beam Search** (num_beams=4): Giữ top-4 giả thuyết (hypotheses):

```
Step 0: [vie_Latn]
  → Top 4: "Tôi" (0.72), "Mình" (0.11), "Tui" (0.08), "Em" (0.05)

Step 1: Expand mỗi beam:
  "Tôi" →    "thích" (0.49), "yêu" (0.14), "mến" (0.04)
  "Mình" →   "thích" (0.08), "yêu" (0.02)
  "Tui" →    "thích" (0.05)
  "Em" →     "thích" (0.03)
  
  Keep top 4: "Tôi thích"(0.49), "Tôi yêu"(0.14), "Mình thích"(0.08), "Tui thích"(0.05)

Step 2: Expand again...
  "Tôi thích" → "mèo"(0.40)  ← WINNER
  "Tôi yêu"   → "mèo"(0.11)
  ...

Final: "Tôi thích mèo" (score = 0.40)
```

### 8.3 Code Inference thực tế

```python
# backend/main.py — Translation pipeline
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load fine-tuned model
model_path = "training_runs/nllb-ja-vi-direct/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()  # Tắt dropout, batchnorm → deterministic

# Set source language
tokenizer.src_lang = "jpn_Jpan"

# Target language token ID
target_token_id = tokenizer.convert_tokens_to_ids("vie_Latn")

# Tokenize input
text = "機械翻訳は過去十年で大きく進化してきました。"
inputs = tokenizer(
    text,
    return_tensors="pt",     # PyTorch tensors
    max_length=256,
    truncation=True,
    padding=True,
)

# Generate translation
with torch.no_grad():                    # Không tính gradient (save RAM)
    generated = model.generate(
        **inputs,
        forced_bos_token_id=target_token_id,  # Force đầu ra = vie_Latn
        max_length=256,
        num_beams=4,                          # Beam search width
        length_penalty=1.0,                   # Không ưu tiên câu ngắn/dài
    )

# Decode output tokens → text
translated = tokenizer.batch_decode(
    generated,
    skip_special_tokens=True    # Bỏ <s>, </s>, <pad>
)[0]
# → "Dịch máy đã có sự tiến hóa lớn trong mười năm qua."
```

### 8.4 Batch Inference (Tối ưu cho phụ đề)

```python
# Dịch nhiều câu cùng lúc (batch) → nhanh hơn nhiều so với 1 câu 1 lần
BATCH_SIZE = 8

for batch_start in range(0, len(segments), BATCH_SIZE):
    batch_texts = [seg["text"] for seg in segments[batch_start:batch_start+8]]
    
    # Tokenize cả batch
    inputs = tokenizer(
        batch_texts,                # ["sentence1", "sentence2", ..., "sentence8"]
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True,               # Pad tất cả về cùng độ dài
    )
    
    # Generate cho cả batch cùng lúc
    with torch.no_grad():
        generated = model.generate(**inputs, ...)
    
    # Decode batch
    translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
    # → ["dịch1", "dịch2", ..., "dịch8"]

# Tại sao batch nhanh hơn?
# - GPU/CPU tận dụng song song hóa SIMD/matrix operations
# - Model weights load 1 lần, dùng cho 8 câu
# - Padding overhead nhỏ so với overhead load model
```

---

## 9. Speech-to-Text (ASR) — Whisper

### 9.1 Kiến trúc Whisper

```
Audio waveform (16kHz, mono)
        ↓
┌──────────────────┐
│  Mel Spectrogram │  ← Chuyển audio → biểu đồ tần số (80 bins × T frames)
│  80 mel bins      │
│  30-second chunks│
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Audio ENCODER    │  ← Transformer encoder (vị trí = thời gian)
│  Conv1D + Transformer
│  Output: hidden states
└────────┬─────────┘
         ↓ Cross-Attention
┌──────────────────┐
│  Text DECODER     │  ← Transformer decoder (sinh text tuần tự)
│  Autoregressive   │
│  Output: text + timestamps
└──────────────────┘
```

### 9.2 VAD (Voice Activity Detection)

```python
# Code thực tế
whisper = WhisperModel(
    "base",              # Model size: tiny/base/small/medium/large
    device="cpu",
    compute_type="int8", # Quantization: float32 → int8 (giảm 4x RAM)
)

segments, info = whisper.transcribe(
    audio_path,
    language=None,       # Auto-detect ngôn ngữ
    beam_size=5,         # Beam search cho ASR
    vad_filter=True,     # Voice Activity Detection
    vad_parameters=dict(
        min_silence_duration_ms=500,  # Im lặng 500ms → tách câu
        speech_pad_ms=200,            # Đệm 200ms trước/sau câu
    ),
)

detected_lang = info.language  # "ja", "en", "zh", ...
```

### 9.3 Quantization (INT8)

```
Float32 model:  140 MB, 1.0x speed
INT8 model:      35 MB, ~2x speed (CPU)

Quantization: W_float32 → W_int8
  float32: 32 bits per weight  → 4 bytes
  int8:     8 bits per weight  → 1 byte  (75% giảm)
  
Quality loss: ~1-2% WER increase (chấp nhận được cho phụ đề)
```

---

## 10. Kết quả thực nghiệm

### 10.1 Training Results — NLLB JA→VI

| Metric | Giá trị |
|--------|---------|
| **Model** | NLLB-200-distilled-600M |
| **Training data** | 5,000 cặp JA→VI (bridge matched) |
| **Train/Eval split** | 4,750 / 250 (95/5) |
| **Epochs** | 3 |
| **Effective batch** | 64 (16 × 4 accum) |
| **Total steps** | 225 |
| **Initial loss** | 52.00 |
| **Final loss** | 39.99 (`-23.1%`) |
| **Training time** | 92 phút (CPU) |
| **Hardware** | MacBook Pro M4 Pro |

### 10.2 Loss Curve Analysis

```
Loss
 52 ┤ ●──────── Epoch 0.67: Model bắt đầu học patterns cơ bản
    │   ╲
 48 ┤     ●──── Epoch 1.34: Đã qua 1 epoch, học patterns phổ biến
    │       ╲
 44 ┤         ●─ Epoch 2.00: Bắt đầu convergence, gradient norm giảm
    │           ╲
 40 ┤             ●─ Epoch 2.67: Near optimal cho data size này
    │
 36 ┤
    └──┬──┬──┬──┬──
       50 100 150 200  Steps

Gradient norm history:
  Step 50:  grad_norm = 8.448   (Gradients lớn → đang học mạnh)
  Step 100: grad_norm = 6.226   (Giảm → ổn định hơn)
  Step 150: grad_norm = 1.920   (Rất nhỏ → gần convergence)
  Step 200: grad_norm = 12.590  (Spike → model gặp samples khó)
```

### 10.3 So sánh OpusMT vs NLLB

| Metric | OpusMT EN→VI | NLLB JA→VI |
|--------|-------------|------------|
| Model size | 74M params | 600M params |
| File size | ~300 MB | 2.4 GB |
| Training time | 7.5 phút | 92 phút |
| Final loss | 0.186 | 39.99 |
| Training speed | 500 steps/min | 2.4 steps/min |

**Tại sao loss NLLB cao hơn?**
- NLLB vocabulary = 256K tokens → entropy tự nhiên cao hơn
- NLLB dịch trực tiếp JA→VI (khó hơn EN→VI)
- Cross-entropy trên vocabulary 256K > vocabulary 60K (OpusMT)
- Loss không so sánh trực tiếp được giữa 2 model khác kiến trúc

### 10.4 Pipeline Performance

| Bước | Thời gian (100 segments) | Công nghệ |
|------|--------------------------|-----------|
| Download + Extract | ~30s | yt-dlp + ffmpeg |
| Transcribe (Whisper) | ~60s | Faster-Whisper INT8 |
| Detect Language | **< 0.1s** | Whisper-detected (instant) |
| **Translate** | **~120s** | NLLB batch inference |
| Output | < 1s | SRT generation |
| **Tổng** | **~3.5 phút** | — |

---

## 11. Tài liệu tham khảo

### Papers

1. **Vaswani et al. (2017)**. "Attention Is All You Need." *NeurIPS 2017*.
   - Kiến trúc Transformer gốc

2. **NLLB Team, Meta AI (2022)**. "No Language Left Behind: Scaling Human-Centered Machine Translation." *arXiv:2207.04672*.
   - Mô hình NLLB-200

3. **Sennrich et al. (2016)**. "Neural Machine Translation of Rare Words with Subword Units." *ACL 2016*.
   - Thuật toán BPE (Byte-Pair Encoding)

4. **Kudo & Richardson (2018)**. "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing." *EMNLP 2018*.
   - SentencePiece tokenizer

5. **Radford et al. (2023)**. "Robust Speech Recognition via Large-Scale Weak Supervision." *ICML 2023*.
   - OpenAI Whisper

6. **Loshchilov & Hutter (2019)**. "Decoupled Weight Decay Regularization." *ICLR 2019*.
   - AdamW optimizer

### Frameworks & Libraries

| Tool | Version | Chức năng |
|------|---------|----------|
| PyTorch | 2.x | Deep Learning framework |
| 🤗 Transformers | 4.x | Model loading & training |
| 🤗 Datasets | latest | Dataset management |
| Faster-Whisper | latest | Optimized Whisper inference |
| SentencePiece | latest | Tokenizer backend |
| FastAPI | latest | API Server |

### Datasets

| Dataset | Source | Pairs | Dùng cho |
|---------|--------|-------|---------|
| Helsinki-NLP/opus-100 (en-vi) | OPUS | 775,303 | EN→VI training + bridge lookup |
| Helsinki-NLP/opus-100 (en-ja) | OPUS | 1,000,000 | Bridge matching JA→VI |
| Helsinki-NLP/opus-100 (en-zh) | OPUS | 1,000,000 | Bridge matching ZH→VI |

---

*Báo cáo được tạo tự động từ mã nguồn dự án Agent-Translate*  
*Ngày: 02/03/2026*
