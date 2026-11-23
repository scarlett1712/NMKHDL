========================================================================================
DỰ BÁO TRỄ CHUYẾN BAY THƯƠNG MẠI TỪ DỮ LIỆU THỰC TẾ (FLIGHT DELAY PREDICTION)
BÀI TẬP LỚN MÔN HỌC: NHẬP MÔN KHOA HỌC DỮ LIỆU (IT4930 / IT4142)
TRƯỜNG CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG - ĐẠI HỌC BÁCH KHOA HÀ NỘI (SoICT - HUST)
========================================================================================

1. GIỚI THIỆU CHUNG
Dự án dự báo số phút trễ cất cánh và cảnh báo nguy cơ trễ >= 30 phút cho 50,000 chuyến bay
thương mại thực tế năm 2013 khởi hành từ 3 sân bay New York (JFK, EWR, LGA).

Ba nguồn dữ liệu:
- Chuyến bay: gói nycflights13 (Wickham), trích từ US Bureau of Transportation Statistics.
- Thời tiết theo giờ tại 3 sân bay: bảng weather của nycflights13 (trạm ASOS / NOAA),
  tải qua HTTP từ kho Rdatasets.
- Quy mô sân bay đến: Web Crawler (requests + BeautifulSoup) cào trang Wikipedia
  "List of the busiest airports in the United States" (số liệu FAA).

Các bước chính:
- Tích hợp 3 nguồn qua Wrappers và Mediated Schema.
- Chia dữ liệu tuần tự theo thời gian Train 70% / Validation 15% / Test 15% (src/preprocessing/splits.py).
  Median điền khuyết, K-Means, StandardScaler, biên IQR chỉ được fit trên Train.
- Loại air_time (thời gian bay thực tế, chỉ biết sau khi hạ cánh) để tránh rò rỉ dữ liệu.
- Thống kê mô tả đơn biến; 11 biểu đồ EDA (Run Sequence, Lag Plot, Heatmap 24x7, Correlation,
  Parallel Coordinates, Mùa vụ & quy mô sân bay, Histogram, Boxplot, Violin, PCA, K-Means Elbow).
- Mô hình hồi quy: 2 baseline, Linear, Ridge (L2), Lasso (L1), Random Forest, XGBoost
  (squared error và absolute error). Siêu tham số chọn trên Validation.
- Mô hình phân loại trễ >= 30 phút: XGBClassifier cân bằng lớp, ngưỡng chọn trên Validation,
  xác suất hiệu chỉnh bằng Isotonic Regression.
- Ứng dụng Web Dashboard Streamlit + bản đồ Folium.

Kết quả trên tập Test (7,500 chuyến, 06/11 - 31/12/2013):
- Hồi quy: XGBoost (absolute error) MAE = 14.76 phút, R2 = 0.207 (baseline hằng số: MAE 16.63).
- Phân loại: XGBClassifier F1 = 0.463, ROC-AUC = 0.792 (baseline persistence: F1 0.312).
Chi tiết trong docs/figures/*.csv và docs/BAO_CAO_BAI_TAP_LON.md.

----------------------------------------------------------------------------------------
2. CẤU TRÚC THƯ MỤC
----------------------------------------------------------------------------------------
NMKHDL/
├── data/
│   ├── raw/
│   │   ├── nyc_flights_raw.csv               # 336,776 chuyến bay gốc (nycflights13)
│   │   ├── nyc_weather_raw.csv               # 26,115 quan trắc thời tiết theo giờ (3 sân bay)
│   │   ├── flights_stream.csv                # 50,000 chuyến bay đã lấy mẫu
│   │   ├── weather_stream.csv                # Thời tiết đã đổi đơn vị mét
│   │   ├── wiki_busiest_us_airports.html     # Trang Wikipedia gốc do crawler tải về
│   │   └── crawled_us_airport_traffic.csv    # 65 sân bay hub cào được
│   └── processed/
│       ├── integrated_flight_weather.csv     # Sau tích hợp 3 nguồn
│       ├── cleaned_flight_data.csv           # Sau làm sạch
│       └── featured_flight_dataset.csv       # Sau tạo đặc trưng (51 đặc trưng đầu vào mô hình)
├── notebooks/
│   └── Flight_Delay_Prediction_Pipeline.ipynb
├── src/
│   ├── data_collection/     fetch_flights.py, fetch_weather_api.py, web_crawler.py
│   ├── data_integration/    integrate.py
│   ├── preprocessing/       splits.py, cleaner.py, feature_engineering.py
│   ├── visualization/       eda_plots.py
│   └── models/              train_evaluate.py
├── models_saved/            Mô hình, scaler, K-Means, bộ hiệu chỉnh xác suất (joblib)
├── app/app.py               Web Dashboard Streamlit
├── docs/
│   ├── figures/             Biểu đồ .png và bảng kết quả .csv
│   └── BAO_CAO_BAI_TAP_LON.md
├── requirements.txt
├── readme.txt
└── README.md

----------------------------------------------------------------------------------------
3. CÀI ĐẶT
----------------------------------------------------------------------------------------
Mở Terminal / PowerShell tại thư mục gốc dự án và chạy:
    pip install -r requirements.txt

Thư viện chính: pandas, numpy, scipy, scikit-learn, xgboost, joblib, matplotlib, seaborn,
requests, beautifulsoup4, streamlit, folium, streamlit-folium.
Windows PowerShell nên đặt:  $env:PYTHONIOENCODING="utf-8"

----------------------------------------------------------------------------------------
4. CHẠY PIPELINE TỪNG BƯỚC
----------------------------------------------------------------------------------------
Bước 1: Thu thập dữ liệu
    python src/data_collection/fetch_flights.py
    python src/data_collection/fetch_weather_api.py      (thêm --refresh để tải lại thời tiết)
    python src/data_collection/web_crawler.py            (thêm --offline để dùng HTML đã lưu)

Bước 2: Tích hợp dữ liệu
    python src/data_integration/integrate.py

Bước 3: Làm sạch và tạo đặc trưng
    python src/preprocessing/cleaner.py
    python src/preprocessing/feature_engineering.py

Bước 4: Sinh 11 biểu đồ EDA
    python src/visualization/eda_plots.py

Bước 5: Huấn luyện, chọn tham số trên Validation, đánh giá trên Test
    python src/models/train_evaluate.py

Hoặc chạy toàn bộ trong notebooks/Flight_Delay_Prediction_Pipeline.ipynb.

----------------------------------------------------------------------------------------
5. WEB DASHBOARD
----------------------------------------------------------------------------------------
    streamlit run app/app.py
Mở trình duyệt tại http://localhost:8501
- Tab 1: chọn hãng, tuyến, ngày, giờ bay và kịch bản thời tiết (hoặc lấy dự báo Open-Meteo);
  app hiển thị dự báo trễ, xác suất trễ >= 30 phút, cảnh báo và thống kê lịch sử của các nhóm
  chuyến bay tương tự.
- Tab 2: toàn bộ biểu đồ EDA, bảng kết quả hồi quy / phân loại, hiệu chỉnh xác suất, Overfitting.
Lưu ý: mô hình học từ dữ liệu 2013; thời tiết trực tiếp chỉ dùng để minh họa.
========================================================================================
