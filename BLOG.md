# 🌬️ AIR GUARD: Dự đoán Chất lượng Không khí với Semi-Supervised Learning

> **Tác giả:** AIR GUARD Team  
> **Ngày:** Tháng 1, 2026  
> **Tags:** Machine Learning, Semi-Supervised Learning, Air Quality, PM2.5, Python

---

## 📌 Tổng quan

Ô nhiễm không khí đang trở thành vấn đề nghiêm trọng tại các thành phố lớn trên thế giới, đặc biệt là ở Trung Quốc. **PM2.5** (bụi mịn có đường kính ≤ 2.5 micromet) là một trong những chất gây ô nhiễm nguy hiểm nhất, có thể xâm nhập sâu vào phổi và gây ra nhiều bệnh hô hấp.

Dự án **AIR GUARD** được xây dựng nhằm:
- 🎯 Dự đoán nồng độ PM2.5 theo thời gian
- 🏷️ Phân loại chỉ số chất lượng không khí (AQI) thành 6 mức
- 🚨 Cảnh báo sớm khi chất lượng không khí đạt ngưỡng nguy hiểm
- 📊 Xây dựng Dashboard trực quan để theo dõi

**Điểm đặc biệt:** Chúng tôi sử dụng **Semi-Supervised Learning** để tận dụng cả dữ liệu có nhãn và không có nhãn, giải quyết bài toán thực tế khi việc gán nhãn dữ liệu tốn kém và mất thời gian.

---

## 📊 Dữ liệu

### Nguồn dữ liệu
Chúng tôi sử dụng bộ dữ liệu **Beijing Multi-Site Air-Quality** từ UCI Machine Learning Repository, bao gồm:

| Thông số | Giá trị |
|----------|---------|
| Số bản ghi | 420,768 |
| Số trạm quan trắc | 12 |
| Khoảng thời gian | 2013-03-01 đến 2017-02-28 |
| Tần suất | Theo giờ |

### Các biến quan trắc
- **Chất ô nhiễm:** PM2.5, PM10, SO2, NO2, CO, O3
- **Khí tượng:** Nhiệt độ (TEMP), Áp suất (PRES), Điểm sương (DEWP), Lượng mưa (RAIN), Tốc độ gió (WSPM), Hướng gió (wd)
- **Thông tin khác:** Trạm quan trắc, Thời gian (năm, tháng, ngày, giờ)

### Phân loại AQI
Dựa trên nồng độ PM2.5, chúng tôi phân loại thành 6 mức AQI:

| Mức AQI | PM2.5 (µg/m³) | Màu sắc |
|---------|---------------|---------|
| 🟢 Good | 0 - 35 | Xanh lá |
| 🟡 Moderate | 35 - 75 | Vàng |
| 🟠 Unhealthy for Sensitive Groups | 75 - 115 | Cam |
| 🔴 Unhealthy | 115 - 150 | Đỏ |
| 🟣 Very Unhealthy | 150 - 250 | Tím |
| 🟤 Hazardous | > 250 | Nâu |

---

## 🔬 Phương pháp

### 1. Tiền xử lý dữ liệu

```
📁 preprocessing_and_eda.ipynb
```

- **Xử lý missing values:** Sử dụng forward fill và backward fill cho các giá trị thiếu
- **Feature engineering:** Tạo các đặc trưng thời gian (year, month, day, hour)
- **Encoding:** One-hot encoding cho hướng gió, Label encoding cho trạm quan trắc
- **Kết quả:** 420,768 bản ghi sạch với 55 cột đặc trưng

### 2. Chuẩn bị dữ liệu Semi-Supervised

```
📁 semi_dataset_preparation.ipynb
```

Để mô phỏng tình huống thực tế khi chỉ có một phần dữ liệu được gán nhãn:
- **Tỷ lệ labeled:** ~8.67% (36,485 mẫu có nhãn)
- **Cutoff date:** 2017-01-01 (train: trước cutoff, test: sau cutoff)
- **Chiến lược:** Chỉ giữ nhãn cho một số trạm và khung giờ nhất định

### 3. Baseline Model (Supervised Learning)

```
📁 classification_modelling.ipynb
```

Sử dụng **HistGradientBoostingClassifier** với 100% dữ liệu có nhãn:
- **51 features** sau khi chuẩn bị
- Train/Test split theo thời gian (cutoff: 2017-01-01)

**Kết quả Baseline:**
| Metric | Giá trị |
|--------|---------|
| Accuracy | 60.22% |
| F1-macro | 47.15% |

### 4. Self-Training

```
📁 semi_self_training.ipynb
```

**Self-Training** là phương pháp semi-supervised đơn giản nhưng hiệu quả:

```
┌─────────────────────────────────────────────────────────┐
│  1. Train model với labeled data                        │
│  2. Predict cho unlabeled data                          │
│  3. Chọn samples có confidence > TAU                    │
│  4. Thêm pseudo-labels vào training set                 │
│  5. Lặp lại cho đến khi hội tụ                         │
└─────────────────────────────────────────────────────────┘
```

**Tham số TAU (threshold)** quyết định độ tin cậy tối thiểu để gán pseudo-label.

#### Thử nghiệm TAU

| TAU | Accuracy | F1-macro | Nhận xét |
|-----|----------|----------|----------|
| 0.80 | 59.41% | 51.67% | TAU thấp → nhiều noise |
| **0.90** | 58.90% | **53.43%** | ⭐ Tốt nhất |
| 0.95 | 59.31% | 53.30% | TAU cao → ít pseudo-labels |

**TAU = 0.9 cho kết quả tối ưu**, cân bằng giữa số lượng và chất lượng pseudo-labels.

### 5. Co-Training

```
📁 semi_co_training.ipynb
```

**Co-Training** sử dụng 2 classifiers với 2 views khác nhau của dữ liệu:

```
┌─────────────────────────────────────────────────────────┐
│  View 1: Features về chất ô nhiễm                       │
│  View 2: Features về khí tượng + thời gian             │
│                                                         │
│  Classifier 1 và 2 "dạy" lẫn nhau bằng cách:           │
│  - Mỗi classifier predict cho unlabeled data            │
│  - Chọn top-k confident samples                         │
│  - Thêm vào training set của classifier kia            │
└─────────────────────────────────────────────────────────┘
```

#### Kết quả Co-Training

| Phương pháp chia views | Accuracy | F1-macro |
|------------------------|----------|----------|
| Auto (random split) | 53.35% | 40.44% |
| Manual (domain-based) | 59.61% | 47.67% |

**Nhận xét:** Co-Training với view chia tự động không hiệu quả do các features có correlation cao. Việc chia views dựa trên domain knowledge cho kết quả tốt hơn.

---

## 📈 Kết quả tổng hợp

### So sánh các phương pháp

| Phương pháp | Labeled Data | Accuracy | F1-macro | Δ F1 vs Baseline |
|-------------|--------------|----------|----------|------------------|
| **Baseline** | 100% | 60.22% | 47.15% | - |
| **Self-Training** (TAU=0.9) | 8.67% | 58.90% | **53.43%** | **+6.28%** ⬆️ |
| **Co-Training** (Manual) | 8.67% | 59.61% | 47.67% | +0.52% |
| **Co-Training** (Auto) | 8.67% | 53.35% | 40.44% | -6.71% ⬇️ |

### Biểu đồ Training Dynamics

**Self-Training:** Số lượng pseudo-labels tăng nhanh trong các iteration đầu, sau đó giảm dần khi model đã "học" hết các samples dễ.

**Co-Training:** Số lượng pseudo-labels ổn định hơn do cơ chế chọn top-k samples.

### Phân tích theo trạm

Top 4 trạm có tần suất cảnh báo AQI cao nhất:

| Trạm | Alert Rate |
|------|------------|
| Dongsi | 55.90% |
| Aotizhongxin | 49.10% |
| Changping | 46.16% |
| Dingling | 39.62% |

---

## 🖥️ Dashboard

Chúng tôi xây dựng **Streamlit Dashboard** với các tính năng:

### 📊 Trang Overview
- Giới thiệu dự án
- Thống kê tổng quan về dữ liệu
- Phân bố các lớp AQI

### 📈 Trang Model Comparison
- So sánh Baseline vs Self-Training vs Co-Training
- Biểu đồ Accuracy và F1-macro
- Confusion Matrix

### 🔄 Trang Training Progress
- Biểu đồ dynamics của quá trình training
- Số lượng pseudo-labels theo iteration
- Validation metrics theo thời gian

### 🚨 Trang AQI Alerts
- Thống kê cảnh báo theo trạm
- Timeline dự đoán AQI
- Filter theo trạm và khoảng thời gian

### 🔮 Trang Predictions
- Dự đoán AQI cho dữ liệu mới
- Upload CSV và nhận kết quả
- Export predictions

**Truy cập Dashboard:** `http://localhost:8502`

---

## 💡 Kết luận

### Những gì đã đạt được

1. ✅ **Self-Training hiệu quả hơn Supervised** khi chỉ có 8.67% dữ liệu có nhãn
2. ✅ **F1-macro cải thiện +6.28%** - quan trọng cho bài toán imbalanced classes
3. ✅ **TAU = 0.9 là threshold tối ưu** cho Self-Training
4. ✅ **Domain knowledge quan trọng** trong việc chia views cho Co-Training

### Bài học kinh nghiệm

1. 📚 **Semi-supervised learning** là giải pháp thực tế khi gán nhãn tốn kém
2. 📚 **Threshold tuning** (TAU) quan trọng - cần thử nghiệm nhiều giá trị
3. 📚 **View independence** trong Co-Training cần được đảm bảo
4. 📚 **F1-macro > Accuracy** khi đánh giá bài toán multi-class imbalanced

### Hướng phát triển

- 🔮 Thử nghiệm **Label Propagation** và **MixMatch**
- 🔮 Tích hợp **LSTM/GRU** cho time series forecasting
- 🔮 Áp dụng **Active Learning** để chọn samples cần gán nhãn
- 🔮 Deploy lên **cloud** với real-time monitoring

---

## 🛠️ Công nghệ sử dụng

| Category | Technologies |
|----------|--------------|
| Language | Python 3.12 |
| ML Framework | scikit-learn 1.8.0 |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, plotly |
| Dashboard | Streamlit 1.53 |
| Environment | venv, pip |

---

## 📁 Cấu trúc dự án

```
air_guard-main/
├── 📁 data/
│   ├── raw/              # Dữ liệu gốc
│   └── processed/        # Dữ liệu đã xử lý + metrics
├── 📁 notebooks/
│   ├── preprocessing_and_eda.ipynb
│   ├── semi_dataset_preparation.ipynb
│   ├── feature_preparation.ipynb
│   ├── classification_modelling.ipynb
│   ├── semi_self_training.ipynb
│   ├── semi_co_training.ipynb
│   └── semi_supervised_report.ipynb
├── 📁 src/
│   ├── classification_library.py
│   ├── regression_library.py
│   ├── semi_supervised_library.py
│   └── timeseries_library.py
├── app.py                # Streamlit Dashboard
├── requirements.txt
└── README.md
```

---

## 📚 Tài liệu tham khảo

1. Yarowsky, D. (1995). "Unsupervised Word Sense Disambiguation Rivaling Supervised Methods"
2. Blum, A., & Mitchell, T. (1998). "Combining Labeled and Unlabeled Data with Co-Training"
3. Zhu, X., & Goldberg, A. B. (2009). "Introduction to Semi-Supervised Learning"
4. UCI Machine Learning Repository - Beijing Multi-Site Air-Quality Data Set

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo Issue hoặc Pull Request trên GitHub.

---

<div align="center">

**🌬️ AIR GUARD - Bảo vệ không khí, bảo vệ sức khỏe 🌬️**

*Made with ❤️ by AIR GUARD Team*

</div>
