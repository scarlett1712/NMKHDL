========================================================================================
HỆ THỐNG DỰ BÁO NGUY CƠ TRỄ CHUYẾN BAY (FLIGHT DELAY) & SĂN VÉ MÁY BAY GIÁ RẺ NỘI ĐỊA VIỆT NAM
BÀI TẬP LỚN MÔN HỌC: KHOA HỌC DỮ LIỆU (IT4930 / IT4142)
TRƯỜNG CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG - ĐẠI HỌC BÁCH KHOA HÀ NỘI (SoICT - HUST)
========================================================================================

1. GIỚI THIỆU CHUNG
Dự án xây dựng hệ thống dự báo nguy cơ trễ chuyến bay (delay) và phân tích xu hướng biến động
giá vé máy bay (Airfare Dynamic Pricing Tracker) cho các chặng bay nội địa trọng điểm Việt Nam
(Hà Nội HAN, TP.HCM SGN, Đà Nẵng DAD, Cam Ranh CXR, Phú Quốc PQC) kết hợp dữ liệu khí tượng
thời gian thực thu thập từ Open-Meteo REST API.

Dự án được hoàn thành toàn diện qua các giai đoạn chuẩn mực:
- Thu thập và tích hợp dữ liệu đa nguồn qua kiến trúc Mediated Schema & Wrappers.
- Làm sạch dữ liệu, xử lý ngoại lai trễ chuyến theo dải phân vị 1.5xIQR, tính toán thống kê đơn biến.
- Trích xuất đặc trưng chuỗi thời gian: Lags, Rolling statistics, Cyclical sin/cos encoding, K-Means clustering.
- Trực quan hóa đa biến: Run Sequence Plot, Lag Plot tự tương quan mạng bay, Heatmap 24x7, Correlation Heatmap,
  Parallel Coordinates Plot và Biểu đồ đường cong giá vé (Dynamic Pricing Curve).
- Huấn luyện mô hình học máy: Baseline (Naïve Persistence), Linear Regression, Ridge (L2 Regularization),
  Random Forest và XGBoost.
- Thực nghiệm chuyên sâu về hiện tượng Overfitting (Train/Test Error vs Tree Depth).
- Triển khai ứng dụng Web Dashboard Bản đồ Radar tương tác (Streamlit & Folium).

----------------------------------------------------------------------------------------
2. CẤU TRÚC THƯ MỤC DỰ ÁN
----------------------------------------------------------------------------------------
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
├── src/                                  # Toàn bộ mã nguồn cốt lõi (Core pipeline)
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
├── docs/                                 # Báo cáo kỹ thuật và hình ảnh trích xuất
│   ├── figures/                          # Toàn bộ hình ảnh biểu đồ EDA và thí nghiệm ML
│   └── BAO_CAO_BAI_TAP_LON.md            # Báo cáo chuẩn 6 mục La Mã (I -> VI) theo rubric của SoICT
│
├── requirements.txt                      # Danh sách các thư viện phụ thuộc
└── readme.txt                            # Hướng dẫn cài đặt và sử dụng hệ thống

----------------------------------------------------------------------------------------
3. HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG
----------------------------------------------------------------------------------------
Bước 1: Mở Terminal / PowerShell tại thư mục gốc của dự án.
Bước 2: Cài đặt các thư viện cần thiết bằng lệnh pip:
    pip install -r requirements.txt

Các thư viện chính được sử dụng:
- pandas, numpy, scipy
- scikit-learn, xgboost, joblib
- matplotlib, seaborn, folium, streamlit-folium
- streamlit, requests

----------------------------------------------------------------------------------------
4. QUY TRÌNH CHẠY HỆ THỐNG TỪNG BƯỚC (END-TO-END PIPELINE)
----------------------------------------------------------------------------------------
Thiết lập encoding UTF-8 (dành cho Windows PowerShell nếu cần):
    $env:PYTHONIOENCODING="utf-8"

Bước 1: Sinh luồng dữ liệu chuyến bay nội địa Việt Nam:
    python src/data_collection/fetch_flights.py

Bước 2: Thu thập dữ liệu trạm khí tượng sân bay & kiểm tra kết nối REST API:
    python src/data_collection/fetch_weather_api.py

Bước 3: Tích hợp dữ liệu qua Mediated Schema & Wrappers:
    python src/data_integration/integrate.py

Bước 4: Làm sạch dữ liệu, xử lý ngoại lai 1.5xIQR và tính thống kê đơn biến:
    python src/preprocessing/cleaner.py

Bước 5: Kỹ nghệ đặc trưng chuỗi thời gian, K-Means clustering và tạo dữ liệu giá vé:
    python src/preprocessing/feature_engineering.py

Bước 6: Sinh toàn bộ đồ thị trực quan hóa dữ liệu đa biến:
    python src/visualization/eda_plots.py

Bước 7: Huấn luyện mô hình, kiểm chứng Overfitting và xuất bảng số liệu:
    python src/models/train_evaluate.py

----------------------------------------------------------------------------------------
5. KHỞI CHẠY GIAO DIỆN WEB DASHBOARD (STREAMLIT)
----------------------------------------------------------------------------------------
Chạy lệnh sau tại thư mục gốc:
    streamlit run app/app.py

Giao diện sẽ hiển thị tại http://localhost:8501 với 3 phân hệ hoàn chỉnh:
- Tab 1: "✈️ DỰ BÁO NGUY CƠ TRỄ CHUYẾN BAY": Nhập thông tin chuyến bay của bạn, gọi thời tiết
  thời gian thực từ Open-Meteo REST API, hiển thị % xác suất trễ chuyến, số phút trễ dự kiến,
  giải thích nguyên nhân bằng AI (XAI), bản đồ Radar đường bay tương tác Folium và biểu đồ 24h.
- Tab 2: "💰 SĂN VÉ MÁY BAY GIÁ RẺ": Mô phỏng đường cong định giá động (Dynamic Pricing),
  xác định chính xác "Vùng Đáy Giá Vé" (mua trước 14 - 28 ngày) và đưa ra lời khuyên săn vé.
- Tab 3: "📊 PHÂN TÍCH CHUYÊN SÂU & ĐÁNH GIÁ MÔ HÌNH": Trình bày đầy đủ các biểu đồ Run Sequence,
  Lag Plot tự tương quan, Heatmap 24x7, bảng so sánh MAE/RMSE/R2/Accuracy/F1 và đồ thị Overfitting.

----------------------------------------------------------------------------------------
6. PHÂN CÔNG TRÁCH NHIỆM THÀNH VIÊN TRONG NHÓM 4 NGƯỜI
----------------------------------------------------------------------------------------
- Thành viên 1: Kỹ sư Tích hợp Dữ liệu (Data Integration & Pipeline Engineer)
- Thành viên 2: Chuyên viên Làm sạch & Thống kê (Data Cleaning & Statistical Analyst)
- Thành viên 3: Chuyên viên Trực quan hóa Dữ liệu (Data Visualization & EDA Specialist)
- Thành viên 4: Trưởng nhóm Mô hình & Phát triển Hệ thống (Machine Learning & System Lead)
========================================================================================