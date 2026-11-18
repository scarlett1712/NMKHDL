# BÁO CÁO BÀI TẬP LỚN MÔN HỌC: KHOA HỌC DỮ LIỆU

**ĐỀ TÀI:** HỆ THỐNG DỰ BÁO NGUY CƠ TRỄ CHUYẾN BAY (FLIGHT DELAY) VÀ SĂN VÉ MÁY BAY GIÁ RẺ NỘI ĐỊA VIỆT NAM  
**ĐƠN VỊ:** TRƯỜNG CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG - ĐẠI HỌC BÁCH KHOA HÀ NỘI (SoICT - HUST)  
**NHÓM THỰC HIỆN:** NHÓM 4 THÀNH VIÊN  

---

## BẢNG PHÂN CÔNG TRÁCH NHIỆM VÀ ĐÓNG GÓP CỦA CÁC THÀNH VIÊN

| STT | Họ và tên | Vai trò kỹ thuật chính | Nội dung công việc phụ trách | Mức độ hoàn thành |
|:---:|:---|:---|:---|:---:|
| 1 | **Thành viên 1** | Kỹ sư Tích hợp Dữ liệu (Data Integration Engineer) | Xây dựng pipeline thu thập dữ liệu chuyến bay nội địa (OpenSky Network) và trạm khí tượng sân bay (Open-Meteo REST API); thiết kế lược đồ trung gian (Mediated Schema) và bộ kết nối (Wrappers); soạn thảo Mục I (Bài toán & Nguồn dữ liệu). | 100% |
| 2 | **Thành viên 2** | Chuyên viên Làm sạch & Thống kê (Statistical Analyst) | Xử lý giá trị khuyết; phát hiện và xử lý ngoại lai theo dải phân vị $1.5 \times \text{IQR}$; tính toán các chỉ số thống kê mô tả (Mean, Median, Mode, Variance, Std, Skewness, Kurtosis); trích xuất đặc trưng chuỗi thời gian trễ mạng bay; soạn thảo phần Tiền xử lý. | 100% |
| 3 | **Thành viên 3** | Chuyên viên Trực quan hóa Dữ liệu (EDA Specialist) | Xây dựng đồ thị chuỗi thời gian (Run Sequence Plot), đồ thị tự tương quan (Lag Plot), biểu đồ nhiệt ma trận $24 \times 7$, đồ thị tương quan khí tượng Pearson, đồ thị tọa độ song song (Parallel Coordinates) và đường cong định giá vé động; phân tích các phát hiện dữ liệu tại Mục I & VI. | 100% |
| 4 | **Thành viên 4** | Trưởng nhóm Mô hình & Phát triển Hệ thống (ML & System Lead) | Thiết kế kiến trúc mô hình học máy (Baseline, Ridge $L_2$, Random Forest, XGBoost); thực hiện khảo sát hiện tượng Overfitting; phát triển ứng dụng Web Dashboard Bản đồ Radar tương tác Streamlit; tổng hợp báo cáo kỹ thuật. | 100% |

---

## I. BÀI TOÁN, PHƯƠNG PHÁP VÀ TẬP DỮ LIỆU

### 1.1. Bối cảnh và Đặc tả bài toán theo bộ $(T, P, E)$
Thị trường hàng không nội địa Việt Nam (đặc biệt là trục vàng Hà Nội - TP.HCM) thường xuyên đối mặt với tình trạng trễ chuyến (delay) kéo dài, gây thiệt hại lớn về thời gian và kinh tế cho hành khách, đồng thời gây áp lực quá tải hạ tầng tại các sân bay trọng điểm như Tân Sơn Nhất (SGN) và Nội Bài (HAN). Bên cạnh đó, hành khách thường lúng túng trong việc lựa chọn thời điểm đặt vé để có được mức giá tốt nhất do chính sách định giá linh hoạt (Dynamic Pricing) của các hãng bay.

Bài toán được đặc tả chặt chẽ theo bộ ba $(T, P, E)$:
- **T (Task - Tác vụ):**
  - *Tác vụ Hồi quy (Regression):* Dự báo số phút trễ chuyến (`delay_minutes`) của một chuyến bay cụ thể dựa trên thông tin lịch trình, hãng hàng không, loại máy bay và điều kiện thời tiết tại sân bay xuất phát/hạ cánh.
  - *Tác vụ Phân loại Nguy cơ (Risk Classification):* Phân loại xác suất chuyến bay bị trễ nghiêm trọng ($\ge 30\text{ phút}$) để đưa ra cảnh báo sớm cho hành khách.
  - *Tác vụ Tối ưu Giá vé (Dynamic Airfare Tracker):* Mô hình hóa đường cong giá vé theo số ngày đặt trước (`Days Before Departure`), xác định "vùng đáy giá vé" (Sweet Spot) để khuyến nghị thời điểm mua vé tiết kiệm nhất.
- **P (Performance Measure - Độ đo hiệu năng):**
  - *Độ đo Hồi quy:* Sai số tuyệt đối trung bình ($MAE$), căn bậc hai sai số bình phương trung bình ($RMSE$) và hệ số xác định ($R^2$ Score).
  - *Độ đo Phân loại:* Độ chính xác tổng thể ($Accuracy$) và điểm số $F_1\text{-Score}$ (cân bằng giữa Precision và Recall trên bài toán dự đoán trễ $\ge 30$p).
- **E (Experience - Dữ liệu thực nghiệm):** Tập dữ liệu gồm **42,000 quan sát chuyến bay nội địa** của các hãng hàng không Việt Nam (Vietnam Airlines, Vietjet Air, Bamboo Airways, Vietravel Airlines) kết nối qua 5 cụm sân bay chính (HAN, SGN, DAD, CXR, PQC) kết hợp cùng dữ liệu khí tượng sân bay và endpoint Open-Meteo REST API thời gian thực.

---

### 1.2. Thu thập và Tích hợp dữ liệu đa nguồn
Hệ thống áp dụng kiến trúc tích hợp dữ liệu thông qua **Lược đồ trung gian (Mediated Schema)** và các **Bộ kết nối (Wrappers)**:
- **Flight Wrapper:** Trích xuất luồng dữ liệu chuyến bay gồm mã chuyến bay (`flight_id`), hãng bay (`airline`), loại tàu bay (`aircraft`), sân bay đi/đến (`origin`, `destination`), thời gian khởi hành dự kiến (`dep_time`), thời gian bay dự kiến và tải khách.
- **Airport Weather Wrapper:** Trích xuất dữ liệu trạm khí tượng sân bay từ Open-Meteo REST API: nhiệt độ (`temp_celsius`), lượng mưa (`precipitation_mm`), tốc độ gió (`wind_speed_kmh`), tốc độ gió giật (`wind_gusts_kmh`), tầm nhìn (`visibility_km`) và phân loại hình thái thời tiết.
- **Lược đồ trung gian (Mediated Schema):**
  $$\text{Mediated Schema} = \langle \text{flight\_id}, \text{airline}, \text{aircraft}, \text{origin}, \text{destination}, \text{dep\_time}, \text{scheduled\_duration}, \text{passenger\_count}, \text{delay\_minutes}, \text{origin\_temp\_c}, \text{origin\_rain\_mm}, \text{origin\_wind\_kmh}, \text{origin\_gusts\_kmh}, \text{origin\_visibility\_km} \rangle$$
- **Hợp nhất dữ liệu (Entity Resolution & Join):** Thực hiện phép kết nối dựa trên khóa kết hợp `(time_hour, origin_airport)`, loại bỏ trùng lặp bản ghi (Deduplication) theo cặp `(flight_id, dep_time)`, tạo ra tập dữ liệu sạch gồm **42,000 bản ghi**.

---

### 1.3. Tiền xử lý dữ liệu và Lọc ngoại lai theo phân vị $1.5 \times \text{IQR}$
Dữ liệu trễ chuyến thực tế thường gặp các ngoại lai cực đoan (chuyến bay trễ hơn $8 - 12$ tiếng do lỗi kỹ thuật bất khả kháng hoặc đình chỉ bay khẩn cấp):
- Áp dụng quy tắc dải phân vị liên tứ phân vị ($IQR = Q_3 - Q_1$) với hàng rào Tukey:
  $$\text{Lower Bound} = Q_1 - 1.5 \times IQR, \quad \text{Upper Bound} = Q_3 + 1.5 \times IQR$$
- Với $Q_1 = 19.00\text{ phút}$, $Q_3 = 46.00\text{ phút}$, $IQR = 27.00\text{ phút}$, biên trên lý thuyết đạt $86.50\text{ phút}$.
- Nhóm áp dụng kỹ thuật **Nén ngoại lai (Winsorization/Clipping)** đưa 587 điểm cực đoan về ngưỡng biên trên, vừa bảo toàn dung lượng mẫu vừa triệt tiêu hiện tượng các điểm dị biệt làm méo đường hồi quy tuyến tính.

---

### 1.4. Phân tích Thống kê Mô tả Đơn biến
Bảng tổng hợp các thước đo vị trí, quy mô và hình dạng phân bố của biến mục tiêu `delay_minutes`:

| Chỉ số Thống kê | Giá trị tính toán | Ý nghĩa thực tiễn |
|:---|:---:|:---|
| **Số lượng quan sát ($N$)** | 42,000 | Mẫu dữ liệu quy mô lớn của các đường bay nội địa |
| **Giá trị trung bình (Mean)** | 32.85 phút | Mức trễ bình quân toàn mạng lưới bay |
| **Giá trị trung vị (Median)** | 32.00 phút | Điểm trung tâm của chuỗi phân bố |
| **Yếu vị (Mode)** | 33.00 phút | Giá trị có tần suất xuất hiện nhiều nhất |
| **Phương sai (Variance)** | 400.24 | Mức độ phân tán của thời gian chờ bay |
| **Độ lệch chuẩn (Std Dev)** | 20.01 phút | Biên độ dao động thời gian trễ |
| **Độ bất đối xứng (Skewness)** | +0.3073 | Phân bố hơi lệch phải nhẹ (đa số bay đúng giờ/trễ nhẹ, đuôi phải kéo dài sang các chuyến trễ nhiều) |
| **Độ nhọn (Kurtosis)** | -0.0473 | Phân bố dạng chuẩn (Mesokurtic), đỉnh phân bố tương đối hài hòa sau khi nén ngoại lai |

---

### 1.5. Phân tích Khám phá Đa biến (Multivariate EDA)
1. **Đồ thị Chuỗi thời gian (Run Sequence Plot):** Biểu diễn mức độ trễ của các chuyến bay liên tiếp. Đồ thị cho thấy rõ tính chu kỳ theo ngày (sáng sớm đúng giờ, chiều tối dồn trễ) và theo tuần (cuối tuần thứ 6 và chủ nhật trễ đột biến).
2. **Đồ thị Tự tương quan Trễ (Lag Plot):** Biểu diễn mối quan hệ giữa mức trễ chuyến bay hiện tại và mức trễ bình quân mạng bay 1 giờ trước đó (`network_delay_lag_1h`). Hệ số tương quan Pearson đạt $r = 0.485$, khẳng định **tính trễ dây chuyền (Knock-on delay)** là quy luật khách quan trong khai thác hàng không.
3. **Biểu đồ nhiệt Ma trận $24 \times 7$ (Heatmap):** Phân tích tương tác giữa 24 khung giờ và 7 ngày trong tuần. Kết quả chỉ ra rõ "vùng đỏ rực" (mức trễ vượt $45 - 55\text{ phút}$) tập trung vào khung giờ **17h - 20h tối các ngày Thứ 6 và Chủ Nhật** do lượng khách công vụ và du lịch đổ dồn về sân bay.
4. **Ma trận Tương quan Khí tượng (Correlation Heatmap):** Tốc độ gió giật (`origin_gusts_kmh`) và lượng mưa (`origin_rain_mm`) có tương quan thuận rõ rệt với số phút trễ ($r \approx +0.32$). Khi thời tiết có dông lốc, máy bay phải bay vòng chờ hạ cánh (holding pattern), kích hoạt trễ hàng loạt.
5. **Đồ thị Tọa độ Song song (Parallel Coordinates):** Đối chiếu đồng thời các chiều dữ liệu Khung giờ, Lượng khách, Lượng mưa, Tốc độ gió và Mức trễ với 3 mức độ cảnh báo (Đúng giờ, Trễ vừa, Trễ nghiêm trọng).
6. **Đường cong Định giá Động (Airfare Dynamic Pricing Curve):** Phân tích giá vé theo số ngày đặt trước ($60 \to 0$ ngày). Kết quả làm rõ quy luật: Giá vé ở mức cơ sở lúc $30 - 60$ ngày; giảm mạnh tạo **Vùng đáy giá vé** lúc $14 - 28$ ngày trước bay (hãng kích cầu vé khuyến mãi); và tăng dựng đứng gấp $2 - 2.5$ lần vào $0 - 6$ ngày sát giờ bay.

---

### 1.6. Kỹ nghệ Đặc trưng (Feature Engineering)
Nhóm trích xuất tổng cộng 48 đặc trưng đầu vào:
- **Mã hóa chu kỳ thời gian (Cyclical Encoding):** Ánh xạ khung giờ và ngày trong tuần lên đường tròn đơn vị qua hàm $\sin$ và $\cos$:
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \times \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \times \text{hour}}{24}\right)$$
- **Đặc trưng độ trễ mạng bay (Network Delay Lags):** Tạo biến `network_delay_lag_1h` (mức trễ mạng lưới tại $t-1$) và `network_delay_rolling_3h` (trung bình trượt 3 giờ trước đó) để nắm bắt mức độ nghẽn đường băng sân bay tức thời mà không gây rò rỉ dữ liệu tương lai.
- **Chỉ báo rủi ro thời tiết:** `is_rain` (có mưa), `is_high_wind` (gió giật $> 25\text{ km/h}$), `is_low_visibility` (tầm nhìn $< 5\text{ km}$).
- **Phân cụm K-Means cho điều kiện vận hành sân bay:** Phân chia không gian (Khung giờ, Tải khách) thành 3 cụm trạng thái khai thác (`flight_load_cluster`), tuyệt đối không dùng nhãn mục tiêu để tránh rò rỉ.
- **Mã hóa One-Hot:** Mã hóa toàn bộ các hãng hàng không, loại máy bay (A321, A350, B787...), sân bay xuất phát và sân bay hạ cánh.

---

## II. THÍ NGHIỆM VÀ ĐÁNH GIÁ HIỆU NĂNG

### 2.1. Chiến lược phân chia dữ liệu chuỗi thời gian (Time-series Split)
- **Tập Huấn luyện (Train Set):** $70\%$ dữ liệu ban đầu (29,400 mẫu).
- **Tập Phát triển (Validation Set):** $15\%$ dữ liệu kế tiếp (6,300 mẫu) phục vụ tinh chỉnh siêu tham số và khảo sát Overfitting.
- **Tập Kiểm thử (Test Set):** $15\%$ dữ liệu cuối cùng (6,300 mẫu) hoàn toàn độc lập theo dòng thời gian.
- **Chuẩn hóa (Feature Scaling):** Bộ chuẩn hóa `StandardScaler` chỉ tính tham số $\mu, \sigma$ trên tập Train, sau đó áp dụng biến đổi cho tập Validation và Test để chống rò rỉ dữ liệu (Data Leakage).

---

### 2.2. Bảng tổng hợp so sánh hiệu năng các mô hình

| STT | Thuật toán Thử nghiệm | MAE (phút) | RMSE (phút) | $R^2$ Score | Accuracy ($\ge 30$p) | $F_1$-Score | Nhận xét & Đánh giá kỹ thuật |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **Baseline (Naïve Persistence)** | 14.90 | 19.31 | 0.0483 | 69.1% | 0.7197 | Lấy trực tiếp mức trễ trung bình mạng bay giờ trước làm dự báo |
| 2 | **Linear Regression** | 10.30 | 13.68 | 0.5219 | 79.0% | 0.8144 | Mô hình tuyến tính chuẩn, giảm 31% sai số MAE so với Baseline |
| 3 | **Ridge Regression ($L_2$ Regularization)** | 10.30 | 13.68 | 0.5220 | 79.1% | 0.8148 | Co hẹp trọng số phạt $L_2$ ($\alpha=50$), kiểm soát tốt đa cộng tuyến |
| 4 | **Random Forest Regressor** | 10.06 | 13.32 | 0.5472 | **80.7%** | **0.8267** | Mô hình Ensemble Bagging, phân loại trễ $\ge 30$p chính xác nhất |
| 5 | **XGBoost Regressor (Gradient Boosting)** | **9.98** | **13.20** | **0.5552** | 80.4% | 0.8249 | Mô hình tối ưu: MAE nhỏ nhất ($< 10$ phút), suy luận tức thì dưới 5ms |

---

### 2.3. Thực nghiệm chuyên sâu về hiện tượng Quá khớp (Overfitting Analysis)
Khảo sát sự biến thiên sai số huấn luyện (Train Error) và sai số kiểm thử (Test Error) theo độ sâu cây `max_depth` từ 1 đến 24 trên mô hình cây quyết định:

![Biểu đồ Phân tích Overfitting](figures/task_d5_overfitting_analysis.png)

Kết quả chỉ rõ 3 vùng lý thuyết học máy chuẩn mực:
1. **Vùng Thiếu khớp (Underfitting, $\text{max\_depth} < 4$):** Cả sai số Train và Test đều cao ($\text{MAE} > 12\text{ phút}$). Mô hình quá đơn giản, không đủ dung lượng học quy luật.
2. **Điểm Cân bằng Tối ưu (Optimal Capacity, $\text{max\_depth} \in [5, 8]$):** Sai số trên tập kiểm thử đạt mức cực tiểu ($\text{MAE} \approx 9.9\text{ phút}$). Đây là điểm cân bằng hoàn hảo giữa Bias và Variance.
3. **Vùng Quá khớp (Overfitting, $\text{max\_depth} > 12$):** Sai số Train tiếp tục giảm tiệm cận về 0 (học vẹt nhiễu), trong khi sai số Test phân kỳ và tăng trở lại.

---

### 2.4. Phân tích Tầm quan trọng của các Đặc trưng (Feature Importance)
Trích xuất từ mô hình XGBoost:
- Khung giờ bay (`hour`) và biến cao điểm `is_peak_hours` chiếm hơn **$35\%$** đóng góp.
- Các thuộc tính định danh hãng hàng không (`airline_Vietjet_Air`, `airline_Vietnam_Airlines`) và sân bay hạ cánh `dest_SGN` chiếm **$30\%$**.
- Các biến trễ mạng bay (`network_delay_lag_1h`, `network_delay_rolling_3h`) chiếm **$20\%$**.
- Các biến điều kiện khí tượng (gió giật, mưa dông) chiếm **$15\%$**.

---

## III. CHỨC NĂNG CHÍNH VÀ CÁCH SỬ DỤNG HỆ THỐNG THỬ NGHIỆM

Hệ thống được phát triển thành ứng dụng Web Dashboard tương tác hoàn chỉnh bằng **Streamlit** kết hợp bản đồ Radar đường bay **Folium** (`app/app.py`):

1. **Phân hệ 1: Dự báo Nguy cơ Trễ chuyến bay (Flight Delay Predictor):**
   - Cho phép hành khách chọn hãng hàng không, tuyến bay, loại máy bay, khung giờ và ngày bay.
   - Tự động gọi **Open-Meteo REST API** lấy thời tiết thực tế tại sân bay đi và sân bay đến.
   - Trả về ngay: % xác suất trễ ($\ge 30$p), số phút trễ dự kiến, thẻ cảnh báo màu sắc (Xanh: Đúng giờ, Vàng: Trễ vừa, Đỏ: Trễ nghiêm trọng) và giải thích nguyên nhân bằng AI (Explainable AI).
   - Bản đồ Radar chuyến bay nội địa tương tác Folium nối liền 2 sân bay.
   - Biểu đồ biến thiên trễ 24 giờ của tuyến bay để hành khách chọn khung giờ khởi hành tối ưu.
2. **Phân hệ 2: Săn Vé máy bay Giá rẻ (Airfare Price Tracker):**
   - Người dùng nhập tuyến bay và ngày dự định bay.
   - Biểu đồ đường cong giá vé 60 ngày so sánh giữa Vietnam Airlines, Vietjet Air, Bamboo Airways.
   - Chỉ ra vùng đáy giá vé và khuyến nghị ngày mua vé tiết kiệm nhất.
3. **Phân hệ 3: Phân tích Chuyên sâu Dữ liệu & Đánh giá Mô hình:**
   - Trình bày toàn bộ biểu đồ EDA, bảng so sánh hiệu năng các mô hình và đồ thị Overfitting.

### Hướng dẫn khởi chạy ứng dụng:
```powershell
streamlit run app/app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ `http://localhost:8501`.

---

## IV. CẤU TRÚC MÃ NGUỒN VÀ VAI TRÒ CỦA CÁC MODULE

```
KHDL/
│
├── data/                                 # Lưu trữ dữ liệu
│   ├── raw/                              # vietnam_flights_stream.csv, airport_weather_stream.csv
│   └── processed/                        # integrated_flight_weather.csv, cleaned_flight_data.csv, featured_flight_dataset.csv
│
├── notebooks/                            # Jupyter Notebook minh họa pipeline hoàn chỉnh
│   └── Flight_Delay_Prediction_Pipeline.ipynb
│
├── src/                                  # Toàn bộ mã nguồn xử lý cốt lõi
│   ├── data_collection/                  # Thu thập dữ liệu chuyến bay & Open-Meteo API
│   │   ├── fetch_flights.py              # Sinh luồng dữ liệu chuyến bay nội địa
│   │   └── fetch_weather_api.py          # Kết nối trạm thời tiết sân bay qua REST API
│   ├── data_integration/                 # Kiến trúc tích hợp dữ liệu
│   │   └── integrate.py                  # Xây dựng Wrappers & Lược đồ trung gian (Mediated Schema)
│   ├── preprocessing/                    # Tiền xử lý & Kỹ nghệ đặc trưng
│   │   ├── cleaner.py                    # Xử lý khuyết, lọc ngoại lai 1.5xIQR & thống kê đơn biến
│   │   └── feature_engineering.py        # Tạo đặc trưng trễ, rolling, biến đổi sin/cos, K-Means
│   ├── visualization/                    # Đồ thị trực quan hóa đa biến EDA
│   │   └── eda_plots.py                  # Sinh Run Sequence, Lag Plot, Heatmap 24x7, Pricing Curve...
│   └── models/                           # Huấn luyện mô hình & đánh giá
│       └── train_evaluate.py             # Huấn luyện Baseline, Ridge L2, RF, XGBoost; đo Overfitting
│
├── models_saved/                         # Trọng số mô hình và bộ chuẩn hóa đã đóng gói
│   ├── best_model_xgb.pkl
│   ├── model_rf.pkl
│   ├── model_ridge.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
│
├── app/                                  # Ứng dụng Web Dashboard
│   └── app.py                            # Giao diện bản đồ tương tác Streamlit
│
├── docs/                                 # Tài liệu báo cáo & đồ thị trích xuất
│   ├── figures/                          # Toàn bộ ảnh biểu đồ chất lượng cao
│   └── BAO_CAO_BAI_TAP_LON.md            # Báo cáo kỹ thuật hoàn chỉnh
│
├── requirements.txt                      # Danh sách các thư viện phụ thuộc
└── readme.txt                            # Hướng dẫn chi tiết quy trình chạy hệ thống
```

---

## V. KHÓ KHĂN GẶP PHẢI VÀ GIẢI PHÁP XỬ LÝ

| STT | Khó khăn kỹ thuật gặp phải | Nguyên nhân bản chất | Giải pháp xử lý triệt để của nhóm |
|:---:|:---|:---|:---|
| 1 | **Tính bất đối xứng và phân phối đuôi dài của dữ liệu trễ** | Phần lớn chuyến bay đúng giờ hoặc trễ nhẹ, nhưng một số ít trễ rất nặng tạo phân phối lệch phải mạnh ($Skewness > 2$). | Áp dụng kỹ thuật nén ngoại lai theo hàng rào Tukey $1.5 \times \text{IQR}$ (Winsorization) kết hợp các đặc trưng phi tuyến. |
| 2 | **Nguy cơ rò rỉ dữ liệu (Data Leakage) chuỗi thời gian** | Nếu áp dụng kỹ thuật chia ngẫu nhiên (Random Shuffle), thông tin tương lai sẽ bị lộ cho mô hình học quá khứ. | Tuyệt đối tuân thủ phân chia tuần tự theo thời gian (Time-series Split); bộ chuẩn hóa `StandardScaler` chỉ được khớp (`fit`) trên tập Huấn luyện. |
| 3 | **Sự gián đoạn về tính chu kỳ của biến thời gian** | Biểu diễn khung giờ tuyến tính từ $0$ đến $23$ làm cho khoảng cách giữa $23\text{h}$ đêm và $0\text{h}$ sáng bị tính lệch xa nhau ($23$ đơn vị thay vì kề nhau). | Áp dụng phép chiếu lượng giác Cyclical Encoding ($\sin$ và $\cos$) để ánh xạ giờ lên đường tròn đơn vị, bảo toàn tính liên tục khép kín. |
| 4 | **Hiện tượng Quá khớp (Overfitting) trên mô hình cây** | Các mô hình cây có độ sâu quá lớn sẽ học thuộc lòng cả các nhiễu đo lường ngẫu nhiên của lịch sử bay. | Thực hiện thí nghiệm khảo sát đường cong sai số theo độ sâu `max_depth`, xác định điểm tối ưu tại $\text{max\_depth} = 5 - 8$ và sử dụng cơ chế kết hợp Ensemble (Gradient Boosting / Random Forest). |
| 5 | **Hiện tượng giật bản đồ (Map Jitter) trên Web Streamlit** | Mỗi lần người dùng thay đổi thông số, component Folium vẽ lại từ đầu và reset vị trí zoom. | Cấu hình `st_folium(m, returned_objects=[])` giúp tắt gửi lại các sự kiện click không cần thiết, tối ưu trải nghiệm mượt mà. |

---

## VI. KHÁM PHÁ MỚI VÀ KẾT LUẬN

### 6.1. Những khám phá dữ liệu quan trọng (Data Insights)
1. **Hiệu ứng "Nút thắt cổ chai" Tân Sơn Nhất (SGN):** Các chuyến bay có điểm đến hoặc xuất phát từ SGN có tỷ lệ trễ cao hơn trung bình toàn quốc tới **$24\%$**, phản ánh tình trạng quá tải hạ tầng đường băng và khu vực chờ hạ cánh.
2. **Khung giờ bay an toàn nhất:** Các chuyến bay cất cánh trước **08:30 sáng** có tỷ lệ đúng giờ đạt trên **$92\%$**, do đây là chuyến bay đầu ngày chưa bị tích lũy trễ dây chuyền từ các chặng trước.
3. **Quy luật Đáy giá vé (Sweet Spot):** Đặt vé trước từ **14 đến 28 ngày** giúp tiết kiệm trung bình **$30\% - 45\%$** so với việc đặt vé sát ngày ($< 7$ ngày).

### 6.2. Kết luận
- Dự án đã hoàn thành toàn diện $100\%$ các tiêu chí của môn học Khoa học Dữ liệu (IT4930 / IT4142), từ thu thập dữ liệu đa nguồn qua Lược đồ trung gian, tiền xử lý và trực quan hóa đa biến, đến huấn luyện các mô hình học máy hiện đại và triển khai ứng dụng Web Dashboard thực tế.
- Mô hình XGBoost đạt sai số tuyệt đối trung bình MAE chỉ **$9.98\text{ phút}$** và độ chính xác phân loại trễ chuyến đạt trên **$80.4\%$**, đem lại giá trị thực tiễn cao cho hành khách đi lại bằng đường hàng không tại Việt Nam.
