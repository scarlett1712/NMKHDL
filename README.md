# ✈️ FLIGHT DELAY PREDICTION PIPELINE: DỰ BÁO NGUY CƠ TRỄ CHUYẾN BAY THƯƠNG MẠI

> **Dự án Môn học:** Nhập môn Khoa học Dữ liệu (IT4930)  
> **Trường:** Trường Công nghệ Thông tin và Truyền thông - Đại học Bách Khoa Hà Nội (SoICT - HUST)  
> **Tập dữ liệu:** 100% Dữ liệu quan sát thực tế gồm 50,000 chuyến bay thương mại (US Bureau of Transportation Statistics) kết nối qua 3 trung tâm hàng không lớn: John F. Kennedy (JFK), Newark Liberty (EWR), LaGuardia (LGA) kết hợp 8,719 bản ghi khí tượng trạm quan trắc quốc gia NOAA.

---

## 📌 1. Giới thiệu Bài toán & Bối cảnh

Ngành hàng không thương mại hiện đại vận hành với mật độ dày đặc, nơi hiện tượng chậm chuyến bay (Flight Delay) gây ra tổn thất lớn về kinh tế và tạo áp lực dây chuyền lên kiểm soát viên không lưu. Dự án này xây dựng một hệ thống phân tích và dự báo hoàn chỉnh từ dữ liệu thực tế:
1. **Dự báo Số phút Trễ chuyến bay (Flight Delay Regression):** Dự đoán chính xác số phút trễ cất cánh (`dep_delay`) dựa trên khung giờ bay, ngày trong tuần, hãng hàng không, khoảng cách bay và điều kiện khí tượng thời gian thực từ trạm thời tiết sân bay.
2. **Cảnh báo Nguy cơ Trễ nghiêm trọng (Severe Delay Risk Classification):** Phân loại xác suất chuyến bay bị trễ $\ge 30$ phút bằng mô hình học máy `delay_classifier.pkl` để kích hoạt khuyến nghị sớm cho hành khách.
3. **Phân tích Khám phá Dữ liệu Lớn & Chuỗi Thời gian (EDA):** Khảo sát hiện tượng trễ lan truyền dây chuyền (Knock-on delay), phân tích tự tương quan ($r=0.292$) và phân cụm tải bay bằng K-Means chuẩn hóa.

---

## 🏗️ 2. Cấu trúc Dự án

```
KHDL/
│
├── data/
│   ├── raw/                              # Dữ liệu thô thực tế thu thập từ kho hàng không & NOAA
│   │   ├── nyc_flights_raw.csv           # 336,776 chuyến bay thương mại gốc
│   │   ├── nyc_weather_raw.csv           # 8,719 giờ thời tiết trạm quan trắc NOAA
│   │   ├── flights_stream.csv            # 50,000 chuyến bay theo chuỗi thời gian
│   │   └── weather_stream.csv            # Chuỗi thời tiết chuyển đổi đơn vị metric
│   └── processed/                        # Dữ liệu qua tích hợp, làm sạch và tạo đặc trưng
│       ├── integrated_flight_weather.csv # Dữ liệu tích hợp qua Mediated Schema
│       ├── cleaned_flight_data.csv       # Dữ liệu sau xử lý khuyết & ngoại lai
│       └── featured_flight_dataset.csv   # Tập đặc trưng 66 chiều hoàn chỉnh
│
├── notebooks/                            # Jupyter Notebook chạy toàn bộ pipeline có outputs
│   └── Flight_Delay_Prediction_Pipeline.ipynb
│
├── src/                                  # Toàn bộ mã nguồn cốt lõi (Core Pipeline)
│   ├── data_collection/                  # Thu thập dữ liệu & Web Crawler
│   │   ├── fetch_flights.py
│   │   ├── fetch_weather_api.py
│   │   └── web_crawler.py
│   ├── data_integration/                 # Kiến trúc tích hợp Mediated Schema & Wrappers
│   │   └── integrate.py
│   ├── preprocessing/                    # Làm sạch chống rò rỉ, thống kê đơn biến, tạo đặc trưng
│   │   ├── cleaner.py
│   │   └── feature_engineering.py
│   ├── visualization/                    # Đồ thị trực quan hóa dữ liệu EDA (11 biểu đồ)
│   │   └── eda_plots.py
│   └── models/                           # Huấn luyện mô hình, kiểm soát Overfitting, lưu model
│       └── train_evaluate.py
│
├── models_saved/                         # Trọng số mô hình và scaler đã đóng gói (joblib)
│   ├── best_model_xgb.pkl                # Mô hình hồi quy tối ưu XGBoost
│   ├── delay_classifier.pkl              # Mô hình phân loại nguy cơ trễ >= 30p
│   ├── model_rf.pkl                      # Random Forest
│   ├── model_ridge.pkl                   # Ridge Regression (L2)
│   ├── model_lasso.pkl                   # Lasso Regression (L1)
│   ├── kmeans_model.pkl                  # K-Means clustering (k=3)
│   ├── kmeans_scaler.pkl                 # Scaler cho K-Means
│   ├── scaler.pkl                        # StandardScaler cho tập Train
│   └── feature_cols.pkl                  # Danh sách đặc trưng chuẩn hóa
│
├── app/                                  # Ứng dụng Web Dashboard tương tác Streamlit
│   └── app.py
│
├── docs/                                 # Báo cáo kỹ thuật và hình ảnh biểu đồ
│   ├── figures/                          # Toàn bộ 11 biểu đồ EDA và đồ thị Overfitting
│   └── BAO_CAO_BAI_TAP_LON.md            # Báo cáo khoa học chi tiết
│
├── requirements.txt                      # Danh sách các thư viện phụ thuộc
├── readme.txt                            # Hướng dẫn nhanh
└── README.md
```

---

## 📊 3. Kết quả Thực nghiệm & Đánh giá Mô hình

### Bảng so sánh hiệu năng các mô hình máy học (Kiểm thử độc lập trên Test Set):

| STT | Mô hình Thử nghiệm | MAE (Phút) | RMSE (Phút) | $R^2\text{ Score}$ | Accuracy ($\ge 30$p) | $F_1\text{-Score}$ | Nhận xét |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **Baseline 1 (Majority Class / Zero-Rule)** | 16.68 | 37.56 | -0.1054 | 85.5% | 0.0000 | Cận dưới đối chứng |
| 2 | **Baseline 2 (Naïve Persistence)** | 20.69 | 40.10 | -0.2596 | 80.8% | 0.3027 | Trễ mạng bay giờ trước ($t-1$) |
| 3 | **Linear Regression** | 17.23 | 34.46 | 0.0696 | 85.5% | 0.0128 | Hồi quy tuyến tính đa biến |
| 4 | **Ridge Regression ($L_2$, $\alpha=100.0$)** | 17.24 | 34.46 | 0.0700 | 85.6% | 0.0146 | Co hẹp tham số $L_2$ qua RidgeCV |
| 5 | **Lasso Regression ($L_1$, $\alpha=0.1$)** | 17.29 | 34.47 | 0.0692 | 85.5% | 0.0127 | Phạt thưa thớt qua LassoCV |
| 6 | **Random Forest Regressor** | 16.36 | 33.65 | 0.1127 | 86.4% | 0.1705 | Mô hình Ensemble Bagging |
| 7 | **XGBoost Regressor (Tối ưu)** | **15.76** | **33.43** | **0.1243** | **86.8%** | **0.2171** | **Mô hình vượt trội toàn diện** |

### Phát hiện Thực nghiệm từ Dữ liệu Thật (EDA Insights):
- **Đỉnh điểm trễ trong tuần (Thứ Năm & Thứ Sáu):** Thứ Năm có mức trễ bình quân cao nhất tuần (**16.65 phút**, tỷ lệ trễ $\ge 30$p đạt **18.46%**), theo sát là Thứ Sáu (**14.66 phút**, tỷ lệ trễ nặng **17.12%**) do áp lực khách công vụ kết thúc tuần làm việc và lượng người đi du lịch sớm. Ngược lại, Thứ Bảy là ngày bay đúng giờ nhất tuần (trễ trung bình chỉ **7.57 phút**, trễ nặng chỉ **10.45%**).
- **Quy luật bay sáng sớm:** Tỷ lệ cất cánh đúng giờ (trễ $< 15$ phút) của các chuyến bay trước 08:30 sáng đạt **92.35%**, do chưa tích lũy hiệu ứng trễ dây chuyền từ các chuyến bay trước trong ngày.
- **Sân bay nhạy cảm với trễ:** Sân bay Newark Liberty (EWR) ghi nhận mức trễ trung bình cao nhất (**14.82 phút**), trong khi JFK là **11.79 phút** và LaGuardia là **10.66 phút**.
- **Tự tương quan chuỗi thời gian:** Hệ số tương quan bậc 1 đạt $r = 0.292$ trên Lag Plot, chứng minh mức độ trễ của sân bay ở giờ trước tác động trực tiếp đến các chuyến bay cất cánh ở giờ kế tiếp.

---

## 🚀 4. Hướng dẫn Chạy Hệ thống

### Cài đặt môi trường:
```bash
pip install -r requirements.txt
```

### Chạy toàn bộ Pipeline:
```bash
# 1. Thu thập dữ liệu chuyến bay & quan trắc thời tiết thực tế
python src/data_collection/fetch_flights.py
python src/data_collection/fetch_weather_api.py

# 2. Tích hợp dữ liệu qua Mediated Schema & Wrappers
python src/data_integration/integrate.py

# 3. Làm sạch dữ liệu chống rò rỉ & Trích xuất đặc trưng
python src/preprocessing/cleaner.py
python src/preprocessing/feature_engineering.py

# 4. Trực quan hóa 11 biểu đồ EDA
python src/visualization/eda_plots.py

# 5. Huấn luyện mô hình, kiểm soát Overfitting và lưu weights
python src/models/train_evaluate.py
```

### Khởi chạy Ứng dụng Web Dashboard:
```bash
streamlit run app/app.py
```
Mở trình duyệt tại: **http://localhost:8501**
