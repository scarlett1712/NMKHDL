# ✈️ VIETNAM AIRWATCH: HỆ THỐNG DỰ BÁO NGUY CƠ TRỄ CHUYẾN BAY & SĂN VÉ MÁY BAY GIÁ RẺ NỘI ĐỊA VIỆT NAM

> **Dự án Môn học:** Nhập môn Khoa học Dữ liệu (IT4930 / IT4142)  
> **Trường:** Trường Công nghệ Thông tin và Truyền thông - Đại học Bách Khoa Hà Nội (SoICT - HUST)  
> **Đường bay nghiên cứu:** Các trục bay nội địa trọng điểm Việt Nam: Hà Nội (HAN), TP.HCM (SGN), Đà Nẵng (DAD), Cam Ranh (CXR), Phú Quốc (PQC)

---

## 📌 1. Giới thiệu Bài toán & Bối cảnh

Thị trường hàng không nội địa Việt Nam thường xuyên đối mặt với tình trạng chậm hủy chuyến (delay) kéo dài, đặc biệt trong các khung giờ cao điểm và điều kiện thời tiết dông bão nhiệt đới. Dự án này xây dựng một hệ thống phân tích và dự báo hoàn chỉnh từ dữ liệu lớn:
1. **Dự báo Nguy cơ Trễ chuyến bay (Flight Delay Prediction):** Dự đoán chính xác số phút trễ ($Regression$) và xác suất trễ nặng $\ge 30$ phút ($Binary\ Classification$) dựa trên khung giờ bay, ngày trong tuần, hãng hàng không, loại tàu bay và điều kiện khí tượng thời gian thực thu thập từ **Open-Meteo REST API**.
2. **Săn Vé Máy Bay Giá Rẻ & Xác định Đáy Giá (Airfare Dynamic Pricing Tracker):** Khảo sát biến thiên giá vé theo ngày đặt trước (Advance Purchase Days), phát hiện chu kỳ giá vé theo ngày trong tuần và đưa ra khuyến nghị thời điểm mua vé tối ưu cho người dùng.

---

## 🏗️ 2. Cấu trúc Dự án

```
KHDL/
│
├── data/
│   ├── raw/                              # Dữ liệu thô thu thập từ chuyến bay và API khí tượng
│   │   ├── vietnam_flights_stream.csv
│   │   └── airport_weather_stream.csv
│   └── processed/                        # Dữ liệu qua tích hợp, làm sạch và tạo đặc trưng
│       ├── integrated_flight_weather.csv
│       ├── cleaned_flight_data.csv
│       ├── featured_flight_dataset.csv
│       └── airfare_pricing_dataset.csv
│
├── notebooks/                            # Jupyter Notebook minh họa toàn bộ pipeline
│   └── Flight_Delay_Prediction_Pipeline.ipynb
│
├── src/                                  # Toàn bộ mã nguồn cốt lõi (Core Pipeline)
│   ├── data_collection/                  # Thu thập luồng dữ liệu chuyến bay và REST API
│   │   ├── fetch_flights.py
│   │   └── fetch_weather_api.py
│   ├── data_integration/                 # Kiến trúc tích hợp Mediated Schema & Wrappers
│   │   └── integrate.py
│   ├── preprocessing/                    # Làm sạch 1.5xIQR, thống kê đơn biến, tạo đặc trưng
│   │   ├── cleaner.py
│   │   └── feature_engineering.py
│   ├── visualization/                    # Đồ thị trực quan hóa dữ liệu EDA
│   │   └── eda_plots.py
│   └── models/                           # Huấn luyện mô hình, L2 regularization, thực nghiệm Overfitting
│       └── train_evaluate.py
│
├── models_saved/                         # Trọng số mô hình và scaler đã đóng gói (joblib)
│   ├── best_model_xgb.pkl
│   ├── model_rf.pkl
│   ├── model_ridge.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
│
├── app/                                  # Ứng dụng Web Dashboard Bản đồ Radar tương tác Streamlit
│   └── app.py
│
├── docs/                                 # Báo cáo kỹ thuật và hình ảnh biểu đồ
│   ├── figures/                          # Toàn bộ hình ảnh biểu đồ EDA và thí nghiệm ML
│   └── BAO_CAO_BAI_TAP_LON.md            # Báo cáo khoa học chuẩn theo rubric
│
├── requirements.txt                      # Danh sách các thư viện phụ thuộc
├── readme.txt                            # Hướng dẫn chi tiết
└── README.md
```

---

## 📊 3. Kết quả Thực nghiệm & Đánh giá Mô hình

### Bảng so sánh hiệu năng các mô hình máy học:

| Mô hình | MAE (Phút) | RMSE (Phút) | $R^2\text{ Score}$ | Accuracy ($\ge 30$p) | $F_1\text{-Score}$ | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Naïve Persistence)** | 15.34 | 21.05 | -0.052 | 68.2% | 0.695 | Điểm chuẩn cơ sở |
| **Linear Regression** | 12.86 | 17.21 | 0.312 | 74.5% | 0.760 | Không có Regularization |
| **Ridge Regression ($L_2$)** | 12.85 | 17.20 | 0.314 | 74.6% | 0.761 | Giảm nhẹ phương sai |
| **Random Forest Regressor** | 10.02 | 13.88 | 0.548 | 80.1% | 0.822 | Mô hình Ensemble phi tuyến |
| **XGBoost Regressor (Tối ưu)** | **9.98** | **13.75** | **0.556** | **80.4%** | **0.825** | **Hiệu năng vượt trội toàn diện** |

### Khám phá Dữ liệu Mẫu (EDA): Đối sánh Ngày thường vs Cuối tuần
- **Ngày thường (Thứ 2 đến Thứ 5, Thứ 7):** Mức trễ bình quân ổn định ở mức **30.3 – 30.8 phút**, tỷ lệ trễ nặng $\ge 30$p khoảng **50.5%**.
- **Cuối tuần cao điểm (Thứ 6 & Chủ Nhật):** Mức trễ bình quân tăng vọt lên **38.5 phút** (tăng thêm **+7.8 phút**, tương đương **+25.4%**), và tỷ lệ trễ nặng nhảy vọt lên **66.4%** (cứ 3 chuyến bay thì có 2 chuyến bị trễ).

---

## 🚀 4. Hướng dẫn Chạy Hệ thống

### Cài đặt môi trường:
```bash
pip install -r requirements.txt
```

### Chạy toàn bộ Pipeline:
```bash
# 1. Thu thập dữ liệu chuyến bay & API thời tiết
python src/data_collection/fetch_flights.py
python src/data_collection/fetch_weather_api.py

# 2. Tích hợp dữ liệu qua Mediated Schema
python src/data_integration/integrate.py

# 3. Làm sạch dữ liệu & Kỹ nghệ đặc trưng
python src/preprocessing/cleaner.py
python src/preprocessing/feature_engineering.py

# 4. Trực quan hóa dữ liệu EDA
python src/visualization/eda_plots.py

# 5. Huấn luyện mô hình & Khảo sát Overfitting
python src/models/train_evaluate.py
```

### Khởi chạy Ứng dụng Web Dashboard:
```bash
streamlit run app/app.py
```
Mở trình duyệt tại: **http://localhost:8501**
