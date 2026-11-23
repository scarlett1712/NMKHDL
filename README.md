# ✈️ FLIGHT DELAY PREDICTION: DỰ BÁO TRỄ CHUYẾN BAY THƯƠNG MẠI TỪ DỮ LIỆU THỰC TẾ

> **Môn học:** Nhập môn Khoa học Dữ liệu (IT4930 / IT4142) - Trường CNTT&TT, Đại học Bách Khoa Hà Nội (SoICT - HUST)
> **Dữ liệu:** 50.000 chuyến bay thương mại thực tế năm 2013 khởi hành từ 3 sân bay New York (JFK, EWR, LGA) - gói nycflights13 (trích từ US Bureau of Transportation Statistics), kết hợp quan trắc thời tiết theo giờ (ASOS/NOAA) và dữ liệu quy mô sân bay cào từ Wikipedia (số liệu FAA).

---

## 📌 1. Bài toán

1. **Hồi quy:** dự báo số phút trễ cất cánh (`dep_delay`) từ thông tin biết trước khi bay: lịch bay, hãng, sân bay đi/đến, khoảng cách, thời tiết tại sân bay đi, mức trễ của sân bay ở các giờ trước.
2. **Phân loại:** cảnh báo chuyến bay có nguy cơ trễ ≥ 30 phút (lớp thiểu số, ~15% số chuyến).
3. **Phân tích khám phá (EDA):** quy luật trễ theo giờ, ngày, tháng, sân bay, hãng bay, thời tiết.

## 🗂️ 2. Ba nguồn dữ liệu

| Nguồn | Cách thu thập | Nội dung |
|---|---|---|
| Chuyến bay | File gốc `nyc_flights_raw.csv` (nycflights13 `flights`, 336.776 chuyến) | Lấy mẫu đều 50.000 chuyến theo thời gian |
| Thời tiết | Tải qua HTTP từ Rdatasets (nycflights13 `weather`, bản CRAN hiện hành) | 26.115 quan trắc theo giờ, đủ cả JFK / EWR / LGA |
| Quy mô sân bay đến | **Web Crawler** (`requests` + `BeautifulSoup`) cào Wikipedia *List of the busiest airports in the United States* | 65 sân bay large/medium hub, số lượt khách lên máy bay (FAA) |

Ba nguồn được tích hợp qua Mediated Schema: thời tiết ghép theo `(origin, year, month, day, hour)`, quy mô sân bay ghép theo `dest`.

## 🏗️ 3. Cấu trúc dự án

```
NMKHDL/
├── data/
│   ├── raw/          nyc_flights_raw.csv, nyc_weather_raw.csv, flights_stream.csv, weather_stream.csv,
│   │                 wiki_busiest_us_airports.html (trang gốc đã cào), crawled_us_airport_traffic.csv
│   └── processed/    integrated_flight_weather.csv, cleaned_flight_data.csv, featured_flight_dataset.csv
├── notebooks/        Flight_Delay_Prediction_Pipeline.ipynb  (chạy toàn bộ pipeline, có output)
├── src/
│   ├── data_collection/   fetch_flights.py, fetch_weather_api.py, web_crawler.py
│   ├── data_integration/  integrate.py          (Wrappers + Mediated Schema)
│   ├── preprocessing/     splits.py, cleaner.py, feature_engineering.py
│   ├── visualization/     eda_plots.py          (11 biểu đồ EDA)
│   └── models/            train_evaluate.py     (huấn luyện, chọn tham số trên Val, đánh giá Test)
├── models_saved/     best_model_xgb.pkl, delay_classifier.pkl, classifier_calibrator.pkl, ...
├── app/app.py        Web Dashboard Streamlit
└── docs/             BAO_CAO_BAI_TAP_LON.md, figures/ (biểu đồ + các bảng kết quả .csv)
```

## 📊 4. Kết quả (tập Test: 7.500 chuyến, 06/11 - 31/12/2013, đánh giá một lần)

Chia dữ liệu tuần tự theo thời gian Train 70% / Validation 15% / Test 15%. Mọi siêu tham số được chọn trên Validation. Median điền khuyết, K-Means, StandardScaler, biên IQR chỉ được fit trên Train.

### Hồi quy số phút trễ (`docs/figures/model_comparison_metrics.csv`)

| Mô hình | MAE (phút) | RMSE (phút) | R² |
|---|:---:|:---:|:---:|
| Baseline 1: hằng số = median Train (−1 phút) | 16.63 | 37.88 | −0.124 |
| Baseline 2: Naïve Persistence (trễ TB sân bay giờ trước) | 18.68 | 36.61 | −0.050 |
| Linear / Ridge (α=0.1) / Lasso (α=0.001) | 16.27 | 34.44 | 0.071 |
| Random Forest | 16.14 | 33.47 | 0.122 |
| XGBoost (squared error) | 15.52 | 33.56 | 0.118 |
| **XGBoost (absolute error)** | **14.76** | **31.82** | **0.207** |

### Phân loại trễ ≥ 30 phút (`docs/figures/classification_metrics.csv`)

| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline: luôn đoán "không trễ" | 85.5% | 0 | 0 | 0 | - |
| Baseline: Persistence ≥ 30 phút | 82.8% | 0.370 | 0.269 | 0.312 | 0.677 |
| **XGBClassifier (cân bằng lớp, ngưỡng 0.54 chọn trên Val)** | 82.2% | 0.411 | 0.529 | **0.463** | **0.792** |

Lớp "trễ ≥ 30 phút" chỉ chiếm 14.5% tập Test nên Accuracy không phản ánh đúng chất lượng; nên đọc F1 và ROC-AUC. Xác suất của bộ phân loại được hiệu chỉnh bằng Isotonic Regression trên Validation (Brier score 0.145 → 0.101).

### Một số phát hiện từ dữ liệu
- Chuyến cất cánh trước 08:30 đúng giờ (trễ < 15 phút) **93.0%**, so với 77.8% của toàn bộ dữ liệu; trễ tăng dần theo giờ trong ngày.
- Thứ Năm trễ nhiều nhất (16.65 phút, 18.5% trễ ≥ 30 phút), Thứ Bảy ít nhất (7.57 phút, 10.4%).
- Mùa vụ rõ rệt: tháng 6-7 và tháng 12 trễ cao nhất, tháng 9-11 thấp nhất.
- Thời tiết: chuyến bay khi trời mưa trễ TB 27.2 phút, mưa dông 34.8 phút, so với 11.3 phút khi trời quang.
- EWR trễ trung bình cao nhất (14.82 phút), sau đó JFK (11.79) và LGA (10.66).

### Giới hạn
Dữ liệu chỉ có năm 2013 và 3 sân bay NYC; R² thấp vì phân phối trễ có đuôi rất dài (tối đa 960 phút); tập Test (tháng 11-12) khác mùa với tập Train (tháng 1-9); bảng Wikipedia chỉ có số liệu từ 2016 nên dùng năm 2016 làm đại diện quy mô sân bay đến.

## 🚀 5. Hướng dẫn chạy

```bash
pip install -r requirements.txt

# 1. Thu thập dữ liệu
python src/data_collection/fetch_flights.py
python src/data_collection/fetch_weather_api.py        # tự tải thời tiết nếu chưa có; --refresh để tải lại
python src/data_collection/web_crawler.py              # cào Wikipedia; --offline để dùng HTML đã lưu

# 2. Tích hợp, làm sạch, tạo đặc trưng
python src/data_integration/integrate.py
python src/preprocessing/cleaner.py
python src/preprocessing/feature_engineering.py

# 3. Biểu đồ EDA và huấn luyện mô hình
python src/visualization/eda_plots.py
python src/models/train_evaluate.py

# 4. Web Dashboard
streamlit run app/app.py                               # http://localhost:8501
```

Hoặc mở `notebooks/Flight_Delay_Prediction_Pipeline.ipynb` để chạy toàn bộ pipeline trong một notebook (cần thêm `jupyter`).
