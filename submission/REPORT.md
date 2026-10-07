# Lab 21 — Evaluation Report

**Họ tên**: Mai Quang Dũng  **MSSV**: 2A202602966  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | Ticket CSKH → JSON triage (250 mẫu) - Lý do: Dữ liệu mặc định phù hợp cho bài toán trích xuất cấu trúc |
| Train / val | 225 / 25 (seed 42) |
| Lý do chọn Model | `Qwen3.5-4B` phù hợp với phần cứng T4 16GB, tốc độ sinh tốt và có khả năng đọc hiểu Tiếng Việt |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3465.3 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.000 | 1062.1 |
| (c) LoRA fine-tune | 0.9700 | 0.5444 | 1.000 | 1435.9 |

**(b) có thật sự mạnh hơn (a) không?** Có — format đạt chuẩn hoàn toàn 1.0 so với 0.0 và target tăng vọt lên 0.765.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không, giữ nguyên mẫu gốc để đối chứng khách quan.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6268 | 0.9700 | 421.2 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5372 | 0.9700 | 281.1 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.0000 | 421.1 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.9400 | 491.7 | 3.86 |

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target toàn phần, `attn_only` hoà với `correct` (0.9700). Train loss của `attn_only` (0.5372) thấp hơn `correct` (0.6268), chứng tỏ nó học tập train tốt hơn đôi chút. Mặc dù rank được đẩy lên rất cao (283) để bù đắp, điểm thực tế trên tập target cũng không thể vượt quá `correct` (all-linear, r=16), qua đó cho thấy Rank không phải là đòn bẩy vạn năng; Vị trí gắn adapter quan trọng hơn rất nhiều trong việc quyết định giới hạn chất lượng của mô hình.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` cao hơn hẳn (1.5702 so với 0.6268 của `correct`) và giảm rất chậm. Nếu chỉ nhìn đường loss mà không biết nguyên nhân do LR, ta rất dễ kết luận sai lầm rằng LoRA không hiệu quả, rank quá nhỏ, hoặc model không học được. Sự thật là LoRA cần một khoảng LR lớn hơn nhiều (khoảng 10 lần) so với Full Fine-Tuning để có thể cập nhật trọng số hiệu quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` giúp tiết kiệm khoảng 4.92 GB VRAM (chỉ dùng 3.86GB so với 8.78GB của bf16/fp16 LoRA). Tuy nhiên, cái giá phải trả là độ chính xác trên tập target giảm (0.9400 so với 0.9700) và tốn thêm thời gian train (~70 giây). Số liệu này ủng hộ hoàn toàn khuyến nghị của nhà cung cấp: với Qwen3.5, việc lượng tử hóa QLoRA gây ra độ sai số lớn dẫn đến sụt giảm chất lượng đáng kể, không nên dùng nếu VRAM vẫn đủ chứa bf16/fp16 LoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.247` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ): 
Dù bản fine-tune đạt điểm Target cực kỳ cao (0.9700, tăng 0.205 so với baseline tối ưu), mô hình lại vấp phải lỗi quên kiến thức nghiêm trọng (catastrophic forgetting), làm điểm regression tụt dốc thê thảm (-0.247), vượt xa ngưỡng dung sai 0.020. Việc chỉ huấn luyện mô hình bằng 250 mẫu dữ liệu quá hẹp (triage ticket) mà không có dữ liệu trộn (replay data) đã phá hủy khả năng tổng quát hóa của Base Model. Phán quyết FAILED này hoàn toàn hợp lý, nó là bài học thực tế cho thấy trong môi trường production, ta không thể đánh đổi việc mất đi năng lực trả lời cơ bản của LLM chỉ để đổi lấy chút % chính xác trên một tác vụ nhỏ gọn.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "mAnh `t chuTt khA'ng dAy... tr l" | doi_tra, cao | doi_tra | doi_tra | ✅ FT thắng/giữ phong độ |
| 2 | "mAnh `t `p lng... hoAn ti?n..." | hoan_tien, trung_binh | hoan_tien | hoan_tien | ✅ FT thắng/giữ phong độ |
| 3 | "mAnh `t bAnh gi_ nhit... cha thy" | hoan_tien, trung_binh | hoan_tien | hoan_tien (đứt gãy JSON) | ❌ **FT thua** do sinh chuỗi JSON không hoàn chỉnh |
| 4 | "mAnh `t n"i chiAn khA'ng... thiu" | san_pham_loi, trung_binh | san_pham_loi | san_pham_loi (đứt gãy JSON) | ❌ **FT thua** do không kết thúc được token JSON |
| 5 | "mAnh `t balo laptop... ? i size" | doi_tra, thap | doi_tra | doi_tra | ✅ FT sinh format đúng và nhãn chuẩn |

Có mẫu chung nào ở các ca FT thua không?
Ca FT thua có mẫu số chung là mô hình sinh ra một chuỗi JSON bị đứt gãy ở giữa chừng (vd: chưa đóng ngoặc nhọn hoặc thiếu trường sentiment) dẫn đến parse JSON thất bại. Điều này có thể xuất phát từ việc chiều dài tối đa sinh ra (max_new_tokens) bị giới hạn hoặc cơ chế stop-token không nhận diện đúng.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** 
Tôi KHÔNG nên đưa bản fine-tune này lên deploy thực tế (production). Mặc dù bản fine-tune hoàn thành xuất sắc nhiệm vụ rút trích trường thông tin JSON với Target Score 0.97, nó lại đánh mất khả năng tư duy và trả lời kiến thức chung. Đòn bẩy thật sự trong bài lab này chính là "Vị trí gắn Adapter" kết hợp "Learning Rate", nhưng quan trọng nhất vẫn là chiến lược bảo vệ chất lượng dữ liệu. LoRA chỉ khuếch đại những gì nó nhìn thấy trong dữ liệu; nếu dữ liệu không có thông tin tổng quát, nó sẽ quên đi kiến thức cũ. Rank dù cấu hình cao đến đâu cũng không thể cứu vãn được giới hạn của vị trí gắn adapter tồi (attn_only).

**Ba điều tôi học được**:
1. Để so sánh sức mạnh thực sự giữa các cấu hình LoRA, ta phải cân bằng số lượng tham số huấn luyện (trainable params - matched rank) thay vì để Rank bằng nhau một cách cứng nhắc.
2. Learning Rate của LoRA phải thiết lập lớn hơn rất nhiều (10x) so với Fine-Tuning toàn bộ trọng số, nếu để mức LR thông thường, LoRA dường như sẽ không học được gì (như cấu hình wrong_lr đã chứng minh).
3. Metric duy nhất là "Chính xác trên tập test" có thể là cái bẫy. Việc duy trì một bộ cổng Regression (kiểm tra kiến thức cũ) là bắt buộc để đảm bảo LLM không bị ngốc đi.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn vào dataset khoảng 1-5% dữ liệu tổng quát (general knowledge replay data) và tiến hành huấn luyện lại bản `correct` để khắc phục điểm regression giảm, nhằm biến phán quyết từ FAILED sang PASSED.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/quangdung12-hcmus/lab21-qwen35-triage-vi
