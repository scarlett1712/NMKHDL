========================================================================================
HỆ THỐNG DỰ BÁO NGUY CƠ TRỄ CHUYẾN BAY THƯƠNG MẠI (FLIGHT DELAY PREDICTION PIPELINE)
BÀI TẬP LỚN MÔN HỌC: KHOA HỌC DỮ LIỆU (IT4930 / IT4142)
TRƯỜNG CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG - ĐẠI HỌC BÁCH KHOA HÀ NỘI (SoICT - HUST)
========================================================================================

1. GIỚI THIỆU CHUNG
Dự án xây dựng hệ thống dự báo nguy cơ trễ chuyến bay (delay) hoàn chỉnh dựa trên 100% dữ liệu
quan sát thực tế từ 50,000 chuyến bay thương mại (Cục Thống kê Giao thông Vận tải Hoa Kỳ - BTS)
kết nối qua 3 trung tâm hàng không nhộn nhịp nhất: John F. Kennedy (JFK), Newark Liberty (EWR),
LaGuardia (LGA) kết hợp chuỗi dữ liệu khí tượng trạm quan trắc quốc gia NOAA.

Dự án được hoàn thành toàn diện qua các giai đoạn chuẩn mực:
- Thu thập và tích hợp dữ liệu đa nguồn qua kiến trúc Mediated Schema & Wrappers.
- Xây dựng Web Crawler (BeautifulSoup) cào thông tin đường bay và đội tàu bay thương mại.
- Làm sạch dữ liệu, xử lý ngoại lai trễ chuyến theo dải phân vị 1.5xIQR trên Train (No Data Leakage).
- Tính toán đầy đủ các chỉ số thống kê mô tả đơn biến (Mean, Median, Mode, Variance, Std, Skewness, Kurtosis).
- Trích xuất đặc trưng chuỗi thời gian: Lags mạng bay, Rolling 3h, Cyclical sin/cos encoding, K-Means clustering.
- Trực quan hóa 11 biểu đồ EDA chuyên sâu: Run Sequence Plot, Lag Plot tự tương quan (r=0.292), Heatmap 24x7,
  Correlation Heatmap, Parallel Coordinates, Histogram & KDE, Boxplot, Violin Plot, PCA 2D, K-Means Elbow.
- Huấn luyện 7 mô hình máy học: 2 Baseline đối chứng (Zero-Rule & Naïve Persistence), Linear Regression,
  Ridge (L2 Regularization), Lasso (L1 Regularization), Random Forest và XGBoost Regressor/Classifier.
- Thực nghiệm chuyên sâu về hiện tượng Overfitting trên tập Train và Validation.
- Triển khai ứng dụng Web Dashboard Bản đồ Radar tương tác (Streamlit & Folium).

----------------------------------------------------------------------------------------
2. CẤU TRÚC THƯ MỤC DỰ ÁN
----------------------------------------------------------------------------------------
KHDL/
│
├── data/
│   ├── raw/                              # Dữ liệu thô thực tế thu thập từ kho hàng không & NOAA
│   │   ├── nyc_flights_raw.csv           # 336,776 chuyến bay thương mại gốc
│   │   ├── nyc_weather_raw.csv           # 8,719 giờ thời tiết trạm quan trắc NOAA
│   │   ├── flights_stream.csv            # 50,000 chuyến bay theo chuỗi thời gian
│   │   ├── weather_stream.csv            # Chuỗi thời tiết chuyển đổi đơn vị metric
│   │   ├── crawled_airlines_info.csv     # Dữ liệu Web Crawler cào hãng bay
│   │   └── crawled_route_metadata.csv    # Dữ liệu Web Crawler cào mạng bay
│   └── processed/                        # Dữ liệu qua tích hợp, làm sạch và tạo đặc trưng
│       ├── integrated_flight_weather.csv # Dữ liệu tích hợp qua Mediated Schema
│       ├── cleaned_flight_data.csv       # Dữ liệu sau xử lý khuyết & ngoại lai
│       └── featured_flight_dataset.csv   # Tập đặc trưng 66 chiều hoàn chỉnh
│
├── notebooks/                            # Jupyter Notebook minh họa toàn bộ pipeline (có outputs)
│   └── Flight_Delay_Prediction_Pipeline.ipynb
│
├── src/                                  # Toàn bộ mã nguồn cốt lõi (Core pipeline)
│   ├── data_collection/                  # Thu thập dữ liệu & Web Crawler
│   │   ├── fetch_flights.py
│   │   ├── fetch_weather_api.py
│   │   └── web_crawler.py
│   ├── data_integration/                 # Kiến trúc tích hợp Mediated Schema & Wrappers
│   │   └── integrate.py
│   ├── preprocessing/                    # Làm sạch 1.5xIQR, thống kê đơn biến, tạo đặc trưng
│   │   ├── cleaner.py
│   │   └── feature_engineering.py
│   ├── visualization/                    # Đồ thị trực quan hóa dữ liệu EDA
│   │   └── eda_plots.py
│   └── models/                           # Huấn luyện mô hình, kiểm soát Overfitting, lưu model
│       └── train_evaluate.py
│
├── models_saved/                         # Trọng số mô hình và scaler đã đóng gói (joblib)
│   ├── best_model_xgb.pkl                # Mô hình hồi quy tối ưu XGBoost
│   ├── delay_classifier.pkl              # Mô hình phân loại trễ >= 30p
│   ├── model_rf.pkl                      # Random Forest
│   ├── model_ridge.pkl                   # Ridge Regression (L2)
│   ├── model_lasso.pkl                   # Lasso Regression (L1)
│   ├── kmeans_model.pkl                  # K-Means clustering (k=3)
│   ├── kmeans_scaler.pkl                 # Scaler cho K-Means
│   ├── scaler.pkl                        # StandardScaler cho tập Train
│   └── feature_cols.pkl                  # Danh sách thuộc tính
│
├── app/                                  # Ứng dụng Web Dashboard tương tác Streamlit
│   └── app.py
│
├── docs/                                 # Báo cáo kỹ thuật và hình ảnh trích xuất
│   ├── figures/                          # Toàn bộ hình ảnh biểu đồ EDA và thí nghiệm ML
│   └── BAO_CAO_BAI_TAP_LON.md            # Báo cáo kỹ thuật chi tiết theo rubric SoICT
│
├── requirements.txt                      # Danh sách các thư viện phụ thuộc
├── readme.txt                            # Hướng dẫn chi tiết
└── README.md

----------------------------------------------------------------------------------------
3. HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG
----------------------------------------------------------------------------------------
Bước 1: Mở Terminal / PowerShell tại thư mục gốc của dự án.
Bước 2: Cài đặt các thư viện cần thiết bằng lệnh pip:
    pip install -r requirements.txt

----------------------------------------------------------------------------------------
4. QUY TRÌNH CHẠY HỆ THỐNG TỪNG BƯỚC (END-TO-END PIPELINE)
----------------------------------------------------------------------------------------
Bước 1: Chuẩn bị luồng dữ liệu chuyến bay thực tế & Web Crawler:
    python src/data_collection/fetch_flights.py
    python src/data_collection/fetch_weather_api.py
    python src/data_collection/web_crawler.py

Bước 2: Tích hợp dữ liệu qua Mediated Schema & Wrappers:
    python src/data_integration/integrate.py

Bước 3: Làm sạch dữ liệu chống rò rỉ & trích xuất đặc trưng:
    python src/preprocessing/cleaner.py
    python src/preprocessing/feature_engineering.py

Bước 4: Xuất toàn bộ 11 biểu đồ trực quan hóa dữ liệu EDA:
    python src/visualization/eda_plots.py

Bước 5: Huấn luyện mô hình, kiểm soát Overfitting và xuất bảng đối sánh:
    python src/models/train_evaluate.py

----------------------------------------------------------------------------------------
5. KHỞI CHẠY ỨNG DỤNG WEB DASHBOARD
----------------------------------------------------------------------------------------
Khởi chạy ứng dụng Streamlit:
    streamlit run app/app.py

Mở trình duyệt tại địa chỉ: http://localhost:8501
========================================================================================
