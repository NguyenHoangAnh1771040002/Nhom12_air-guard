# 🌬️ Air Guard: Dự báo PM2.5 & Cảnh báo AQI

> **Mini Project:** Ứng dụng Học bán giám sát (Semi-Supervised Learning) trong dự báo chất lượng không khí.

## 👥 Thông tin Nhóm
- **Nhóm:** 12
- **Thành viên:** 
  - Nguyễn Hoàng Anh
  - Nguyễn Trung Thành
  - Trần Việt Vinh
  - Nguyễn Minh Phượng
- **Chủ đề:** Phân loại chất lượng không khí (AQI) sử dụng Self-Training và Co-Training.
- **Dataset:** Beijing Multi-Site Air Quality (UCI) — Dữ liệu chất lượng không khí 12 trạm quan trắc (2013–2017).

---

## 🎯 Mục tiêu
> Xây dựng pipeline phân loại AQI từ dữ liệu PM2.5, tập trung giải quyết vấn đề **thiếu hụt dữ liệu có nhãn** bằng các kỹ thuật **Học bán giám sát (Semi-Supervised Learning)**.

**Điểm khác biệt:** Thay vì chỉ fine-tune mô hình trên tập dữ liệu nhỏ, chúng tôi áp dụng Self-Training và Co-Training để khai thác tri thức từ lượng lớn dữ liệu chưa gán nhãn, giúp cải thiện khả năng tổng quát hóa của mô hình.

---

## 1. Ý tưởng & Feynman Style

### 🤔 Bài toán đặt ra
Trong thực tế, việc gán nhãn dữ liệu (labeling) rất tốn kém và mất thời gian, trong khi dữ liệu thô (unlabeled) lại rất dồi dào.
Giả sử bạn có 4 năm dữ liệu khí tượng, nhưng chỉ có thông tin AQI chính xác cho 6 tháng đầu. Làm sao tận dụng 3.5 năm dữ liệu còn lại để mô hình thông minh hơn?

Nếu chỉ dùng dữ liệu có nhãn ít ỏi, mô hình sẽ không tổng quát được (overfitting hoặc bias). Nếu bỏ qua dữ liệu chưa nhãn, ta lãng phí một nguồn tri thức khổng lồ về phân phối dữ liệu (data distribution).

### 💡 Giải pháp Học bán giám sát
1.  **Self-Training (Tự học):** Giống như một học sinh tự ôn bài. Mô hình học trên dữ liệu có sẵn, sau đó tự làm bài tập (dự đoán trên dữ liệu chưa nhãn). Những câu nào nó "rất chắc chắn" (Confidence > 90%), nó sẽ coi như đáp án đúng và học tiếp từ đó.
2.  **Co-Training (Đôi bạn cùng tiến):** Sử dụng 2 góc nhìn khác nhau ("views"). 
    - *View 1:* "Nhìn chất ô nhiễm" (PM10, SO2, NO2...).
    - *View 2:* "Nhìn trời đất" (Gió, Mưa, Nhiệt độ, Thời gian...).
    - Hai mô hình độc lập sẽ dạy lẫn nhau những mẫu dữ liệu mà họ tự tin nhất.

---

## 2. Quy trình Thực hiện

```mermaid
graph LR
    A[Raw Data] --> B[Preprocessing]
    B --> C[Feature Engineering]
    C --> D{Split Data\n(Cutoff 2017)}
    D -->|Labeled Set\n(~8.7%)| E[Baseline Model]
    D -->|Unlabeled Set\n(~91.3%)| F[Semi-Supervised Loop]
    F --> G[Self-Training]
    F --> H[Co-Training]
    E & G & H --> I[Test on 2017 Data]
```

**Chi tiết các bước:**
1.  **Tiền xử lý:** Làm sạch, xử lý missing, chuẩn hóa.
2.  **Split Data:** Chia theo thời gian (Time-based split) để tránh data leakage.
    - **Train:** Trước 2017-01-01. Trong đó chỉ giữ lại một phần nhỏ có nhãn.
    - **Test:** Từ 2017-01-01 đến hết (dùng để đánh giá độc lập).
3.  **Modeling:** Chạy các thuật toán Baseline, Self-Training, Co-Training.
4.  **Comparison:** So sánh hiệu quả mở rộng tập nhãn.

---

## 3. Tiền xử lý & Feature Engineering

### Các bước làm sạch
- ✅ **Missing Values:** Linear Interpolation cho các biến liên tục (TEMP, PRES...).
- ✅ **AQI Labeling:** Chuyển đổi PM2.5 thành 6 mức AQI (Good, Moderate... Hazardous).
- ✅ **Sliding Window:** Tạo các features trễ (lag) để bắt tính tự tương quan.

### Feature Engineering
| Loại feature | Chi tiết |
|--------------|----------|
| **Time** | `hour_sin`, `hour_cos` (chu kỳ 24h), `dow`, `is_weekend` |
| **Lag** | `PM2.5_lag1`, `lag3`, `lag24` (tự tương quan quá khứ) |
| **Pollutants** | `PM10`, `SO2`, `NO2`, `CO`, `O3` |
| **Meteo** | `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM` (tốc độ gió) |

---

## 4. Phân tích Khám phá Dữ liệu (EDA)

### Q1.1 — Phân phối dữ liệu Train/Test
Dữ liệu được chia theo thời gian (Cutoff: 2017-01-01) để tránh data leakage.

| Tập dữ liệu | Số lượng mẫu | Tỷ lệ |
|-------------|--------------|-------|
| **Train (Labeled)** | ~34,400 | ~8.7% |
| **Train (Unlabeled)** | ~361,800 | ~91.3% |
| **Test (2017)** | 16,671 | - |
| **Tổng cộng** | ~413,000 | 100% |

### Q1.2 — Mất cân bằng lớp (Class Imbalance)
Thống kê trên tập Test cho thấy sự chênh lệch lớn giữa các lớp AQI:

| Lớp AQI | Số lượng mẫu (Test) | Tỷ lệ |
|---------|---------------------|-------|
| **Moderate** | 4,833 | ~29% |
| **Unhealthy** | 4,286 | ~26% |
| **Very Unhealthy** | 2,499 | ~15% |
| **Unhealthy for Sens.** | 2,166 | ~13% |
| **Hazardous** | 1,855 | ~11% |
| **Good** | **1,032** | **~6%** |

> **Nhận xét:** Lớp **"Good"** (Không khí tốt) là lớp thiểu số (chỉ chiếm 6%). Đây là thách thức lớn cho mô hình Baseline, dễ dẫn đến việc bỏ qua lớp này.

---

## 5. Mô hình Baseline (Supervised)

### Q2.1 — Thiết lập
- **Thuật toán:** HistGradientBoostingClassifier.
- **Dữ liệu:** Chỉ train trên tập Labeled (~8.7%).

### Q2.2 — Kết quả Baseline
| Metric | Giá trị | Nhận xét |
|--------|---------|----------|
| **Accuracy** | 60.22% | Tạm chấp nhận được. |
| **F1-Macro** | 47.15% | Khá thấp do bias. |
| **Good Class F1** | **0.00** | **Critical:** Mô hình không học được gì về lớp Good. |

---

## 6. Mô hình Self-Training

### Q3.1 — Ý tưởng & Cấu hình
- Tận dụng mô hình Baseline để dự đoán nhãn cho tập Unlabeled.
- **Ngưỡng tự tin ($\tau$):** 0.9.
- **Quy trình:** Top-K mẫu tự tin nhất được gán nhãn giả -> Retrain.

### Q3.2 — Kết quả Self-Training
| Metric | Giá trị | So với Baseline |
|--------|---------|-----------------|
| **Accuracy** | 58.90% | Giảm nhẹ (-1.3%) |
| **F1-Macro** | **53.43%** | **Tăng mạnh (+6.3%)** |
| **Good Class F1** | **0.49** | Khôi phục khả năng nhận diện lớp Good. |

> **Insight:** Self-training đã "cứu" các lớp thiểu số bằng cách tìm kiếm thêm mẫu Good trong tập Unlabeled khổng lồ.

---

## 7. Mô hình Co-Training

### Q4.1 — Tách Views (Splitting Views)
- **View 1:** Chất ô nhiễm (PM10, SO2, NO2, CO, O3) + Lags.
- **View 2:** Khí tượng (TEMP, RAIN, WSPM, Station) + Thời gian.

### Q4.2 — Kết quả Co-Training
| Metric | Giá trị | Nhận xét |
|--------|---------|----------|
| **Accuracy** | 53.35% | Giảm đáng kể. |
| **F1-Macro** | 40.44% | Kém hơn cả Baseline. |

> **Lý do thất bại:** View 2 (Khí tượng) có thể không đủ thông tin để phân loại chính xác AQI một mình (Conditional Independence violation), dẫn đến việc gán nhãn sai cho View 1.

---

## 8. So sánh & Đánh giá (Comparison)

### Điều kiện so sánh công bằng
- ✅ **Test Set:** Cố định (Dữ liệu năm 2017).
- ✅ **Metric:** F1-Macro (ưu tiên do mất cân bằng lớp) và Accuracy.
- ✅ **Features:** Baseline và Self-Training dùng chung bộ feature.

### Bảng kết quả tổng quan

| Mô hình | F1-Macro | Accuracy | Lớp "Good" F1 | Nhận xét |
|---------|----------|----------|---------------|----------|
| **Baseline** | 47.15% | **60.22%** | 0.00 | Bias mạnh, bỏ qua lớp hiếm. |
| **Self-Training** | **53.43%** | 58.90% | **0.49** | Cân bằng, nhận diện tốt lớp hiếm. |
| Co-Training | 40.44% | 53.35% | 0.07 | Kém hiệu quả do View yếu. |

### Q1: Mô hình nào tốt nhất cho bài toán này?
**Kết luận:** **Self-Training** là lựa chọn tốt nhất.

**Lý do:**
1.  **Cải thiện F1-Macro (+6.3%):** Chứng tỏ khả năng tổng quát hóa tốt hơn.
2.  **Khôi phục "Tri thức ẩn":** Việc F1 lớp `Good` tăng từ 0 lên 0.49 cho thấy Self-Training đã "khai quật" được các mẫu không khí sạch trong đống dữ liệu chưa nhãn mà Baseline bỏ sót.

### Q2: Tại sao Accuracy của Baseline cao nhất nhưng không được chọn?
> Đây là cạm bẫy **Accuracy Paradox** trong dữ liệu mất cân bằng.
Baseline chỉ tập trung tối ưu cho các lớp đa số (Moderate, Unhealthy) và chấp nhận đoán sai hết các lớp thiểu số. Điều này vô dụng trong thực tế vì ta cần cảnh báo các mức độ nguy hại (Hazardous) hoặc an toàn (Good) chính xác.

### Q3: Tại sao Co-Training lại thất bại (F1 giảm)?
**Nguyên nhân:** Vi phạm giả định **"Sufficient Views"**.
> View 2 (Thời tiết + Thời gian) không đủ thông tin để phân loại AQI một cách độc lập. Khi View 2 đoán sai, nó sẽ gán nhãn giả sai cho View 1 học. Quá trình này tạo ra một vòng lặp nhiễu (feedback loop of noise) khiến cả 2 mô hình cùng đi xuống.

---

## 9. Insights & Khuyến nghị

### Insight #1: Dữ liệu không nhãn là "Mỏ vàng"
> Với tỷ lệ gán nhãn chỉ ~8.7%, Self-Training vẫn cải thiện được hiệu năng đáng kể. Điều này khẳng định chiến lược Semi-supervised là đúng đắn cho các bài toán môi trường thiếu kinh phí gán nhãn.

### Insight #2: Chất lượng nhãn giả > Số lượng (Thresholding)
> Ngưỡng $\tau = 0.9$ là rào chắn quan trọng. Thử nghiệm cho thấy nếu hạ $\tau$ xuống 0.75, lượng nhãn giả tăng gấp đôi nhưng F1-Macro giảm do nhiễu. "Thà bỏ sót còn hơn học sai".

### Insight #3: Thời tiết ảnh hưởng mạnh nhưng chưa đủ
> Nhiệt độ và gió có tương quan với PM2.5, nhưng không thể dùng riêng lẻ để định đoạt AQI. Cần kết hợp thêm dữ liệu không gian (PM2.5 từ các trạm lân cận) để View 2 mạnh hơn trong Co-Training.

### Insight #4: Streamlit Dashboard hỗ trợ ra quyết định
> Việc trực quan hóa các điểm cảnh báo (Alerts) trên Dashboard giúp chuyên gia môi trường nhanh chóng nhận diện các đợt ô nhiễm bất thường mà chỉ nhìn vào số liệu thô sẽ khó thấy.
---

## 10. Cấu trúc Project

```
Nhom12_air-guard/
├── data/
│   ├── raw/                # Dữ liệu gốc
│   └── processed/          # Dữ liệu đã xử lý & metrics
├── notebooks/
│   ├── EDA_Preprocessing.ipynb
│   ├── Training_SemiSupervised.ipynb
│   └── Analysis_Report.ipynb
├── src/                    # Thư viện mã nguồn (utils)
├── app.py                  # Dashboard Streamlit
├── run_papermill.py        # Automation Script
├── requirements.txt
└── README.md
```

---

## 11. Hướng dẫn Chạy

### Cài đặt
```bash
git clone <repo-url>
cd Nhom12_air-guard
pip install -r requirements.txt
```

### Chạy Dashboard
```bash
streamlit run app.py
```
> Mở trình duyệt tại `http://localhost:8501`.

### Chạy Pipeline (Tái hiện kết quả)
```bash
python run_papermill.py
```

---

## 12. Tech Stack

| Công nghệ | Mục đích |
|----------|----------|
| **Python 3.9+** | Ngôn ngữ chính. |
| **Scikit-learn** | Thuật toán HistGradientBoosting. |
| **Streamlit** | Xây dựng Web App tương tác. |
| **Plotly** | Biểu đồ tương tác trên Dashboard. |
| **Papermill** | Tham số hóa và chạy notebook tự động. |

---

## 13. Kết luận
Dự án **Air Guard** đã chứng minh tính hiệu quả của **Self-Training** trong việc giải quyết bài toán thiếu nhãn cho dữ liệu không khí. Dashboard trực quan giúp người dùng dễ dàng theo dõi và ra quyết định.

---