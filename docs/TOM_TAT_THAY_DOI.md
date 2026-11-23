# TÓM TẮT CÁC THAY ĐỔI TRÊN NHÁNH `fix/data-and-leakage`

Nhánh được tạo từ `main` (commit `ee5866f - fix: audit code cleanup`) để sửa các lỗi về dữ liệu, rò rỉ dữ liệu, đánh giá mô hình, app và tài liệu. Toàn bộ pipeline đã được chạy lại từ đầu; mọi con số dưới đây lấy từ các file CSV trong `docs/figures/`.

---

## 1. Kết quả trước và sau khi sửa (tập Test)

| Chỉ số | Trước (`main`) | Sau (nhánh này) |
|---|:---:|:---:|
| Hồi quy - MAE mô hình tốt nhất | 15,76 phút | **14,76 phút** (XGBoost absolute error) |
| Hồi quy - R² | 0,124 | **0,207** |
| Phân loại trễ ≥ 30 phút - F1 | 0,217 (hồi quy rồi cắt ngưỡng) | **0,463** (XGBClassifier) |
| Phân loại - ROC-AUC | chưa đánh giá | **0,792** |
| Xác suất trễ trong app | điểm thô, trung bình 33,8% | **đã hiệu chỉnh**, trung bình 12,2% (thực tế 14,5%) |
| Brier score | - | 0,145 → **0,101** |

Phần cải thiện chủ yếu đến từ việc có đủ dữ liệu thời tiết cho JFK và LGA (trước đây khoảng 64% số chuyến không có thời tiết).

---

## 2. Dữ liệu

| Vấn đề | Cách sửa | File |
|---|---|---|
| File thời tiết gần như chỉ có EWR (JFK 9 dòng, LGA 1 dòng), là bản nycflights13 đời đầu (`wind_gust` luôn = `wind_speed` × 1,15) | Tải bảng `weather` đầy đủ của bản CRAN hiện hành qua HTTP (Rdatasets): 8.703 EWR, 8.706 JFK, 8.706 LGA. Có kiểm tra đủ 3 sân bay khi tải | `src/data_collection/fetch_weather_api.py`, `data/raw/nyc_weather_raw.csv` |
| Giá trị gió phi vật lý 1.687 km/h (lỗi cảm biến đã biết) | Gió > 200 km/h được đặt thành giá trị khuyết | `fetch_weather_api.py` |
| Gió giật trống bị điền bằng `wind_speed + 5` | Gió giật trống = không có giật → lấy bằng tốc độ gió | `fetch_weather_api.py` |
| `web_crawler.py` không cào gì, chỉ ghi danh sách tuyến bay Việt Nam gõ tay | Viết lại crawler thật bằng `requests` + `BeautifulSoup`, cào Wikipedia *List of the busiest airports in the United States* (65 sân bay hub, số liệu FAA); lưu HTML gốc, có chế độ `--offline` | `src/data_collection/web_crawler.py`, `data/raw/crawled_us_airport_traffic.csv`, `data/raw/wiki_busiest_us_airports.html` |
| Dữ liệu crawler không liên quan đến bài toán | Tích hợp làm nguồn thứ 3, ghép theo `dest` → 2 đặc trưng mới `dest_hub_tier`, `dest_enplanements_m` (89,2% chuyến có sân bay đến là hub) | `src/data_integration/integrate.py` |
| 2 file crawler giả | Đã xóa | `data/raw/crawled_airlines_info.csv`, `data/raw/crawled_route_metadata.csv` |

---

## 3. Chống rò rỉ dữ liệu

| Vấn đề | Cách sửa | File |
|---|---|---|
| `air_time` (thời gian bay **thực tế**, chỉ biết sau khi hạ cánh) được dùng làm đặc trưng | Loại khỏi dữ liệu ở bước làm sạch; thêm `assert` chặn dùng lại | `cleaner.py`, `train_evaluate.py`, `app.py` |
| Median điền khuyết tính trên toàn bộ dữ liệu (kể cả Test) | Chia Train/Val/Test trước; median theo (sân bay, tháng) chỉ tính trên Train | `src/preprocessing/cleaner.py` |
| K-Means và scaler của nó fit trên toàn bộ dữ liệu | Chỉ fit trên Train, predict cho Val/Test | `feature_engineering.py` |
| Median điền lag, danh sách top hãng / top sân bay đến tính trên toàn bộ dữ liệu | Chỉ tính trên Train | `feature_engineering.py` |
| Mỗi file tự chia dữ liệu theo cách riêng | Thêm module dùng chung quy định ranh giới Train 70% / Val 15% / Test 15% theo thời gian | `src/preprocessing/splits.py` (mới) |

---

## 4. Đặc trưng

| Vấn đề | Cách sửa |
|---|---|
| `network_delay_rolling_3h` dùng `.rolling(3)` ngoài `groupby` → cửa sổ tràn qua sân bay khác | Tính riêng cho từng sân bay |
| Lag dịch theo **số dòng** (giờ trước có thể là nhiều giờ trước, qua đêm) | Dịch theo **giờ thực tế** (t−1, t−2, t−3) bằng `asfreq('h')` |
| Hàm Min-Max / Z-score có viết nhưng không dùng | Biểu đồ tọa độ song song dùng `DataCleaner.min_max_scale` |
| Cảnh báo deprecated của pandas 3.x (`inplace`, `dt.dayofweek`) | Đã sửa |

Sau khi sửa lag, hệ số tương quan của lag plot đổi từ 0,292 thành **0,362**.

---

## 5. Mô hình và đánh giá (`src/models/train_evaluate.py`)

| Vấn đề | Cách sửa |
|---|---|
| Ridge/Lasso dùng RidgeCV/LassoCV (cross-validation không theo thời gian) | Chọn α trên tập **Validation** |
| Random Forest, XGBoost đặt cứng tham số | Grid nhỏ chọn trên Validation; XGBoost dùng early stopping |
| Luôn nén ngoại lai `y_train` | Có/không nén được coi là siêu tham số, chọn trên Validation |
| Chỉ có XGBoost squared error | Thêm XGBoost **absolute error** (tối ưu trực tiếp MAE) → mô hình tốt nhất |
| Baseline 1 dự báo 0 hoặc 35 phút | Dự báo median của Train (−1 phút) |
| `delay_classifier.pkl` được dùng trong app nhưng **chưa từng được đánh giá** | Đánh giá trên Test bằng Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC; cân bằng lớp bằng `scale_pos_weight`; ngưỡng chọn trên Validation (0,54) |
| Xác suất bị đẩy cao do cân bằng lớp | Hiệu chỉnh bằng Isotonic Regression fit trên Validation |

**File kết quả mới trong `docs/figures/`:**

| File | Nội dung |
|---|---|
| `classification_metrics.csv` | Kết quả phân loại trễ ≥ 30 phút |
| `probability_calibration.csv` | Xác suất trước/sau hiệu chỉnh, Brier score |
| `hyperparameter_tuning_val.csv` | Toàn bộ cấu hình đã thử và điểm trên Validation |
| `overfitting_curve.csv` | Train/Val MAE theo độ sâu cây |
| `feature_importance_top12.csv` | 12 đặc trưng quan trọng nhất |

**Model mới trong `models_saved/`:** `classifier_calibrator.pkl`, `classifier_threshold.pkl`, `model_info.pkl`.

---

## 6. Biểu đồ EDA (`src/visualization/eda_plots.py`)

- Xóa `task_c6_pricing_curve.png` (ảnh giá vé Việt Nam cũ còn sót, không do code sinh ra).
- Thêm `task_c6_monthly_hub_delay.png`: trễ theo tháng và theo quy mô sân bay đến.
- Cố định seed cho PCA và K-Means Elbow để mỗi lần chạy ra cùng một hình.
- Toàn bộ 11 biểu đồ được sinh lại trên dữ liệu mới.

---

## 7. App (`app/app.py`)

| Trước | Sau |
|---|---|
| Tiêu đề "Vietnam AirWatch - Săn vé", code còn `SGN`, chú thích CAAV / Nội Bài | Đổi thành "Flight Delay Predictor", bỏ nội dung Việt Nam |
| Import OCR (`rapidocr`, `PIL`) không dùng | Đã bỏ |
| Ô chọn loại máy bay (mô hình không dùng) | Đã bỏ |
| Công suất sân bay, OTP hãng bay tự đặt | Bỏ; thay bằng quy mô sân bay từ dữ liệu crawler |
| Mục "XAI" với trọng số tự đặt (+25%, −20%…) | Bảng thống kê lịch sử 2013 của các nhóm chuyến bay tương tự, tính từ dữ liệu |
| Số liệu theo ngày, tỷ lệ đúng giờ, MAE… gõ tay | Tính trực tiếp từ CSV |
| Xác suất trễ là điểm thô | Xác suất đã hiệu chỉnh + cảnh báo theo ngưỡng chọn trên Validation |
| Số phút trễ không ghi rõ là gì | Ghi rõ "trung vị" và có chú thích |
| Không có lưu ý về dữ liệu 2013 | Thêm lưu ý: thời tiết trực tiếp chỉ để minh họa |
| Tab 2 chỉ có bảng hồi quy | Thêm bảng phân loại, hiệu chỉnh xác suất, đường cong overfitting |

---

## 8. Notebook, tài liệu, thư viện

- **Notebook** `notebooks/Flight_Delay_Prediction_Pipeline.ipynb`: viết lại theo pipeline mới (trước đó vẫn ghi đề tài Việt Nam, lỗi import ở cell 3, số liệu MAE 8,41 / F1 0,838 không đúng); đã chạy lại toàn bộ, có output, không lỗi.
- **README.md, readme.txt:** viết lại theo dữ liệu, kết quả và cấu trúc thư mục hiện tại (`NMKHDL/`).
- **Báo cáo** `docs/BAO_CAO_BAI_TAP_LON.md`: viết lại đủ 6 mục; mọi số liệu lấy từ CSV; ghi nguồn chính xác; bỏ các câu "vượt trội toàn diện", "100% tiêu chí"; thêm mục thư viện kế thừa, khó khăn - giải pháp, giới hạn.
- **Báo cáo PDF** `docs/BAO_CAO_BAI_TAP_LON.pdf` (mới).
- **requirements.txt:** bỏ `pydeck`, `Pillow`, `rapidocr-onnxruntime` (không dùng).

---

## 9. Việc còn lại

- **Điền bảng phân công** trong báo cáo (họ tên, MSSV, % hoàn thành) rồi xuất lại PDF.
- **Giới hạn chưa sửa:** đặc trưng "trễ của sân bay giờ trước" dùng trễ của cả những chuyến chưa cất cánh tại thời điểm dự báo, nên kết quả có thể hơi lạc quan. Đã nêu trong mục VI của báo cáo.
- Chưa có bộ test tự động (`tests/`, pytest).

---

## 10. Cách chạy lại để kiểm tra

```powershell
git switch fix/data-and-leakage
pip install -r requirements.txt
$env:PYTHONIOENCODING="utf-8"

python src/data_collection/fetch_flights.py
python src/data_collection/fetch_weather_api.py
python src/data_collection/web_crawler.py --offline
python src/data_integration/integrate.py
python src/preprocessing/cleaner.py
python src/preprocessing/feature_engineering.py
python src/visualization/eda_plots.py
python src/models/train_evaluate.py

streamlit run app/app.py
```

Kết quả đúng: bảng hồi quy có `XGBoost (absolute error)` MAE **14.76**, R² **0.2066**; bảng phân loại có `XGBClassifier (ngưỡng 0.54 …)` F1 **0.4628**, ROC-AUC **0.7923**.
