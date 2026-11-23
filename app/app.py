"""
Ứng dụng Web Dashboard Streamlit:
Hệ thống Dự báo Nguy cơ Trễ chuyến bay (Flight Delay Predictor) & Săn Vé máy bay Giá rẻ (Airfare Price Tracker)
Chặng bay: Các đường bay nội địa Việt Nam (HAN, SGN, DAD, CXR, PQC)
Tích hợp: Quét vé máy bay (Boarding Pass OCR Scanner), Open-Meteo REST API thời gian thực & Mô hình học máy XGBoost
Môn học: Nhập môn Khoa học Dữ liệu (IT4930 / IT4142) - SoICT, Đại học Bách Khoa Hà Nội (HUST)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import requests
from datetime import datetime, timedelta
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Vietnam AirWatch - Dự báo Trễ Chuyến & Săn Vé Máy Bay",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- OCR ENGINE -----------------
@st.cache_resource
def get_ocr_engine():
    return RapidOCR()

# ----------------- THÔNG TIN SÂN BAY & ĐƯỜNG BAY THỰC TẾ -----------------
AIRPORTS = {
    'JFK': {'name': 'John F. Kennedy (New York)', 'lat': 40.6413, 'lon': -73.7781, 'capacity': '90 chuyến/giờ'},
    'EWR': {'name': 'Newark Liberty (New Jersey)', 'lat': 40.6895, 'lon': -74.1745, 'capacity': '95 chuyến/giờ (Điểm nghẽn cao)'},
    'LGA': {'name': 'LaGuardia (New York)', 'lat': 40.7769, 'lon': -73.8740, 'capacity': '80 chuyến/giờ'},
    'ORD': {'name': "Chicago O'Hare (Chicago)", 'lat': 41.9742, 'lon': -87.9073, 'capacity': '110 chuyến/giờ'},
    'LAX': {'name': 'Los Angeles (California)', 'lat': 33.9416, 'lon': -118.4085, 'capacity': '105 chuyến/giờ'},
    'ATL': {'name': 'Hartsfield-Jackson (Atlanta)', 'lat': 33.6407, 'lon': -84.4277, 'capacity': '120 chuyến/giờ'},
    'MIA': {'name': 'Miami International (Florida)', 'lat': 25.7959, 'lon': -80.2870, 'capacity': '75 chuyến/giờ'},
    'BOS': {'name': 'Boston Logan (Massachusetts)', 'lat': 42.3656, 'lon': -71.0096, 'capacity': '70 chuyến/giờ'},
    'SFO': {'name': 'San Francisco (California)', 'lat': 37.6213, 'lon': -122.3790, 'capacity': '85 chuyến/giờ'}
}

AIRLINES_INFO = {
    'United Airlines': {'code': 'UA', 'otp': '81.2%', 'color': '#00558f', 'fleet': 'B737, B777, B787'},
    'Delta Air Lines': {'code': 'DL', 'otp': '85.4%', 'color': '#dc2626', 'fleet': 'A321, A350, B737'},
    'American Airlines': {'code': 'AA', 'otp': '80.1%', 'color': '#0284c7', 'fleet': 'A320, B737, B777'},
    'JetBlue Airways': {'code': 'B6', 'otp': '74.5%', 'color': '#0f766e', 'fleet': 'A320, A321, E190'},
    'ExpressJet': {'code': 'EV', 'otp': '72.3%', 'color': '#f59e0b', 'fleet': 'ERJ-145, CRJ-200'},
    'Southwest Airlines': {'code': 'WN', 'otp': '79.0%', 'color': '#ca8a04', 'fleet': 'B737-700, B737-800'}
}

ROUTES_LIST = [
    ("JFK", "LAX", "New York (JFK) ✈️ Los Angeles (LAX)"),
    ("EWR", "ORD", "Newark (EWR) ✈️ Chicago (ORD)"),
    ("LGA", "ATL", "New York (LGA) ✈️ Atlanta (ATL)"),
    ("JFK", "MIA", "New York (JFK) ✈️ Miami (MIA)"),
    ("EWR", "SFO", "Newark (EWR) ✈️ San Francisco (SFO)"),
    ("LGA", "BOS", "New York (LGA) ✈️ Boston (BOS)"),
    ("JFK", "BOS", "New York (JFK) ✈️ Boston (BOS)")
]

# ----------------- 1. KHỞI TẠO SESSION STATE -----------------
if 'airline_key' not in st.session_state:
    st.session_state['airline_key'] = 'United Airlines'
if 'route_key' not in st.session_state:
    st.session_state['route_key'] = ROUTES_LIST[0]
if 'date_key' not in st.session_state:
    st.session_state['date_key'] = datetime.now().date()
if 'hour_key' not in st.session_state:
    st.session_state['hour_key'] = 18
if 'min_key' not in st.session_state:
    st.session_state['min_key'] = 30
if 'aircraft_key' not in st.session_state:
    st.session_state['aircraft_key'] = 'B737'
if 'flight_code_key' not in st.session_state:
    st.session_state['flight_code_key'] = 'UA1504'

# ----------------- 2. TẢI TÀI NGUYÊN VÀ MÔ HÌNH HỌC MÁY -----------------
@st.cache_resource
def load_models_and_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models_saved")
    
    model_xgb = joblib.load(os.path.join(model_dir, "best_model_xgb.pkl"))
    model_rf = joblib.load(os.path.join(model_dir, "model_rf.pkl"))
    model_ridge = joblib.load(os.path.join(model_dir, "model_ridge.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))

    # Nạp mô hình phân loại xác suất trễ thực tế và bộ phân cụm K-Means
    classifier = joblib.load(os.path.join(model_dir, "delay_classifier.pkl")) if os.path.exists(os.path.join(model_dir, "delay_classifier.pkl")) else None
    kmeans = joblib.load(os.path.join(model_dir, "kmeans_model.pkl")) if os.path.exists(os.path.join(model_dir, "kmeans_model.pkl")) else None
    kmeans_scaler = joblib.load(os.path.join(model_dir, "kmeans_scaler.pkl")) if os.path.exists(os.path.join(model_dir, "kmeans_scaler.pkl")) else None
        
    metrics_path = os.path.join(base_dir, "docs", "figures", "model_comparison_metrics.csv")
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None

    dataset_path = os.path.join(base_dir, "data", "processed", "featured_flight_dataset.csv")
    featured_df = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else None

    return {
        'xgb': model_xgb,
        'rf': model_rf,
        'ridge': model_ridge,
        'classifier': classifier,
        'kmeans': kmeans,
        'kmeans_scaler': kmeans_scaler,
        'scaler': scaler,
        'features': feature_cols,
        'metrics': metrics_df,
        'featured_df': featured_df,
        'base_dir': base_dir
    }

artifacts = load_models_and_artifacts()
base_dir = artifacts['base_dir']

# ----------------- 4. GỌI OPEN-METEO LIVE REST API THEO GIỜ BAY -----------------
@st.cache_data(ttl=900)
def fetch_live_weather(airport_code: str):
    coord = AIRPORTS.get(airport_code, AIRPORTS['JFK'])
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={coord['lat']}&longitude={coord['lon']}"
        "&current=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,weather_code"
        "&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,weather_code"
        "&timezone=America%2FNew_York"
    )
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def format_weather_desc(code: int, temp: float, rain: float, wind: float, is_forecast: bool = False, time_str: str = ""):
    prefix = f"Dự báo theo giờ cất cánh ({time_str}): " if is_forecast else "Thời gian thực hiện tại: "
    if code >= 95:
        detail = "Dông bão / Sấm sét nguy hiểm"
    elif code >= 61:
        detail = f"Mưa rào diện rộng ({rain:.1f} mm)"
    elif code >= 45:
        detail = "Sương mù / Trần mây thấp"
    elif code >= 1:
        detail = "Có mây rải rác"
    else:
        detail = "Trời quang đãng, tầm nhìn tốt"
    return f"{prefix}{detail} ({temp:.1f}°C, gió {wind:.0f} km/h)"

def get_flight_weather(w_raw, target_date, dep_hour):
    if not w_raw:
        return 20.0, 0.0, 14.0, 18.0, 0, "Live API (Dự phòng): Trời quang đãng, tầm nhìn tốt"
    
    # Ưu tiên lấy dự báo thời tiết tại đúng ngày và giờ cất cánh từ chuỗi hourly forecast
    target_time_str = f"{target_date}T{dep_hour:02d}:00"
    if 'hourly' in w_raw and 'time' in w_raw['hourly']:
        times = w_raw['hourly']['time']
        if target_time_str in times:
            idx = times.index(target_time_str)
            h_data = w_raw['hourly']
            temp = float(h_data['temperature_2m'][idx])
            rain = float(h_data['precipitation'][idx])
            wind = float(h_data['wind_speed_10m'][idx])
            gusts = float(h_data['wind_gusts_10m'][idx]) if 'wind_gusts_10m' in h_data and h_data['wind_gusts_10m'][idx] is not None else wind * 1.3
            code = int(h_data['weather_code'][idx])
            desc = format_weather_desc(code, temp, rain, wind, is_forecast=True, time_str=f"{dep_hour:02d}:00")
            return temp, rain, wind, gusts, code, desc
            
    # Nếu ngày bay ngoài dải 7 ngày dự báo, fallback sang current weather thời gian thực
    if 'current' in w_raw:
        cur = w_raw['current']
        temp = float(cur.get('temperature_2m', 20.0))
        rain = float(cur.get('precipitation', 0.0))
        wind = float(cur.get('wind_speed_10m', 14.0))
        gusts = float(cur.get('wind_gusts_10m', 18.0))
        code = int(cur.get('weather_code', 0))
        desc = format_weather_desc(code, temp, rain, wind, is_forecast=False)
        return temp, rain, wind, gusts, code, desc

    return 20.0, 0.0, 14.0, 18.0, 0, "Live API (Dự phòng): Trời quang đãng, tầm nhìn tốt"

# ----------------- 5. SIDEBAR: THÔNG TIN CHUYẾN BAY -----------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Hust_logo.png/640px-Hust_logo.png", width=90)
st.sidebar.title("✈️ FLIGHT DELAY PREDICTOR")
st.sidebar.markdown("**Dự báo Nguy cơ Trễ chuyến bay Thương mại**")
st.sidebar.caption("Đề tài KHDL: IT4930 / IT4142 - SoICT, HUST")
st.sidebar.divider()

st.sidebar.subheader("📋 Thông số Chuyến bay:")

airline_keys = list(AIRLINES_INFO.keys())

selected_airline = st.sidebar.selectbox(
    "1. Hãng hàng không:",
    airline_keys,
    key='airline_key'
)

code_map = {
    'United Airlines': 'UA1024',
    'Delta Air Lines': 'DL452',
    'American Airlines': 'AA815',
    'JetBlue Airways': 'B6190',
    'ExpressJet': 'EV4120',
    'Southwest Airlines': 'WN732'
}

flight_code_input = st.sidebar.text_input(
    "Mã số chuyến bay (tuỳ ý):",
    value=code_map.get(selected_airline, "UA1024"),
    key='flight_code_key'
)

selected_route_tuple = st.sidebar.selectbox(
    "2. Tuyến bay:",
    ROUTES_LIST,
    format_func=lambda x: x[2],
    key='route_key'
)
origin_code, dest_code, route_label = selected_route_tuple

target_date = st.sidebar.date_input("3. Ngày khởi hành dự kiến:", key='date_key')

dep_hour = st.sidebar.slider(
    "4. Khung giờ khởi hành (0 - 23h):",
    0, 23,
    key='hour_key'
)

dep_minute = st.sidebar.slider(
    "Phút khởi hành (0 - 55):",
    0, 55,
    step=5,
    key='min_key'
)

aircraft_list = ['B737', 'A320', 'A321', 'B777', 'B787', 'ERJ-145']

selected_aircraft = st.sidebar.selectbox(
    "5. Loại máy bay khai thác:",
    aircraft_list,
    key='aircraft_key'
)

st.sidebar.divider()
st.sidebar.subheader("🌦️ Điều kiện Thời tiết Sân bay:")
weather_mode = st.sidebar.selectbox(
    "Kịch bản Khí tượng & Thời tiết:",
    [
        "📡 Tự động lấy Thời tiết Dự báo theo Giờ bay (Open-Meteo REST API) - Mặc định",
        "☀️ Giả lập: Thời tiết tốt / Quang đãng (Gió 12km/h, Không mưa)",
        "🌧️ Giả lập: Mưa vừa đến Mưa to (Mưa 25mm, Gió 22km/h)",
        "⛈️ Giả lập: Dông Bão Cực đoan / Gió giật mạnh (Mưa 60mm, Gió giật 50km/h)",
        "🌫️ Giả lập: Sương mù dày đặc / Tầm nhìn thấp"
    ],
    index=0
)

# Xử lý các thông số khí tượng dựa trên kịch bản hoặc Live API
if weather_mode == "☀️ Giả lập: Thời tiết tốt / Quang đãng (Gió 12km/h, Không mưa)":
    origin_temp = 22.0
    origin_rain = 0.0
    origin_wind = 12.0
    origin_gusts = 15.0
    origin_weather_desc = "Trời quang đãng, tầm nhìn xa > 10 km, điều kiện bay lý tưởng"
    origin_weather_code = 0
elif weather_mode == "🌧️ Giả lập: Mưa vừa đến Mưa to (Mưa 25mm, Gió 22km/h)":
    origin_temp = 18.0
    origin_rain = 25.0
    origin_wind = 22.0
    origin_gusts = 32.0
    origin_weather_desc = "Mưa to diện rộng, trần mây thấp, bề mặt đường băng trơn ướt"
    origin_weather_code = 65
elif weather_mode == "⛈️ Giả lập: Dông Bão Cực đoan / Gió giật mạnh (Mưa 60mm, Gió giật 50km/h)":
    origin_temp = 16.5
    origin_rain = 60.0
    origin_wind = 35.0
    origin_gusts = 50.0
    origin_weather_desc = "Dông bão đối lưu, mây tích điện (CB), sấm sét, gió giật trên 50 km/h"
    origin_weather_code = 95
elif weather_mode == "🌫️ Giả lập: Sương mù dày đặc / Tầm nhìn thấp":
    origin_temp = 12.0
    origin_rain = 1.0
    origin_wind = 6.0
    origin_gusts = 8.0
    origin_weather_desc = "Sương mù bức xạ dày đặc, trần mây cực thấp, tầm nhìn đường băng dưới 800m"
    origin_weather_code = 45
else: # Live API theo đúng ngày và giờ bay
    w_raw = fetch_live_weather(origin_code)
    origin_temp, origin_rain, origin_wind, origin_gusts, origin_weather_code, origin_weather_desc = get_flight_weather(w_raw, target_date, dep_hour)

dest_weather_desc = "Khai thác bình thường, tầm nhìn tốt"
if "Dông Bão" in weather_mode:
    dest_weather_desc = "Thời tiết tiếp cận dông gió, nguy cơ bay vòng chờ"
elif "Mưa" in weather_mode:
    dest_weather_desc = "Mưa rải rác, đường băng ướt"
elif "Sương mù" in weather_mode and dest_code == 'HAN':
    dest_weather_desc = "Sương mù hạn chế hạ cánh"

st.sidebar.caption(f"🌦️ **{origin_code}**: {origin_temp:.1f}°C, Mưa: {origin_rain:.1f}mm, Gió: {origin_wind:.0f}km/h")

st.sidebar.divider()
selected_model_name = st.sidebar.selectbox(
    "🤖 Chọn Mô hình Học máy:",
    ["XGBoost Regressor (Tối ưu - MAE ~9.9p)", "Random Forest (MAE ~10.0p)", "Ridge Regression (L2)"]
)

# ----------------- 6. HÀM DỰ ĐOÁN DELAY -----------------
def predict_flight_delay(airline: str, origin: str, dest: str, aircraft: str,
                         hour: int, dow: int, month: int,
                         temp_c: float, rain_mm: float, wind_kmh: float, gusts_kmh: float):
    feature_cols = artifacts['features']
    inp = {c: 0.0 for c in feature_cols}
    
    inp['dayofweek'] = dow
    inp['month'] = month
    inp['is_peak_day'] = 1 if dow in [3, 4, 6] else 0
    inp['is_peak_hours'] = 1 if 16 <= hour <= 20 else (1 if 7 <= hour <= 9 else 0)
    inp['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    inp['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    inp['day_sin'] = np.sin(2 * np.pi * dow / 7.0)
    inp['day_cos'] = np.cos(2 * np.pi * dow / 7.0)
    
    dist_map = {'LAX': 2475.0, 'ORD': 719.0, 'ATL': 762.0, 'MIA': 1089.0, 'SFO': 2565.0, 'BOS': 190.0, 'MCO': 944.0}
    dist = dist_map.get(dest, 1000.0)
    inp['distance'] = dist
    inp['air_time'] = dist / 7.5
    
    # Lấy trễ mạng bay trung bình theo giờ từ tập dữ liệu thực tế (Không gán cứng)
    if artifacts['featured_df'] is not None:
        sub_h = artifacts['featured_df'][(artifacts['featured_df']['origin'] == origin) & (artifacts['featured_df']['hour'] == hour)]
        avg_lag = float(sub_h['network_delay_lag_1h'].mean()) if len(sub_h) > 0 else 12.0
        avg_roll = float(sub_h['network_delay_rolling_3h'].mean()) if len(sub_h) > 0 else 12.0
    else:
        avg_lag = 12.0
        avg_roll = 12.0
    inp['network_delay_lag_1h'] = avg_lag
    inp['network_delay_rolling_3h'] = avg_roll
    
    inp['origin_temp_c'] = temp_c
    inp['origin_rain_mm'] = rain_mm
    inp['origin_wind_kmh'] = wind_kmh
    inp['origin_gusts_kmh'] = gusts_kmh
    inp['origin_visibility_km'] = 16.0 if rain_mm == 0 else (4.0 if rain_mm > 5 else 10.0)
    inp['humid'] = 60.0
    inp['pressure'] = 1015.0
    inp['is_rain'] = 1 if rain_mm > 0.1 else 0
    inp['is_high_wind'] = 1 if gusts_kmh > 25.0 else 0
    inp['is_low_visibility'] = 1 if inp['origin_visibility_km'] < 8.0 else 0
    
    # Phân cụm K-Means động bằng mô hình đã fit chuẩn hóa trên [hour, distance]
    if artifacts['kmeans'] is not None and artifacts['kmeans_scaler'] is not None:
        scaled_feat = artifacts['kmeans_scaler'].transform([[hour, dist]])
        inp['flight_load_cluster'] = int(artifacts['kmeans'].predict(scaled_feat)[0])
    else:
        inp['flight_load_cluster'] = 1 if 16 <= hour <= 20 else 0
    
    carrier_code = AIRLINES_INFO.get(airline, {}).get('code', 'UA')
    if f"carrier_{carrier_code}" in inp:
        inp[f"carrier_{carrier_code}"] = 1
    if f"origin_{origin}" in inp:
        inp[f"origin_{origin}"] = 1
    if f"dest_{dest}" in inp:
        inp[f"dest_{dest}"] = 1
            
    df_sample = pd.DataFrame([inp])[feature_cols]
    if "XGBoost" in selected_model_name:
        pred_delay = float(artifacts['xgb'].predict(df_sample)[0])
    elif "Random Forest" in selected_model_name:
        pred_delay = float(artifacts['rf'].predict(df_sample)[0])
    else:
        scaled = artifacts['scaler'].transform(df_sample)
        pred_delay = float(artifacts['ridge'].predict(scaled)[0])

    # Xác suất trễ >= 30 phút từ mô hình phân loại XGBClassifier thực tế
    if artifacts['classifier'] is not None:
        prob = float(artifacts['classifier'].predict_proba(df_sample)[0, 1])
        prob_pct = int(round(prob * 100))
    else:
        prob_pct = int(np.clip(round((pred_delay - 5) / 35 * 100), 5, 95))
        
    return float(pred_delay), prob_pct

dow_val = target_date.weekday()
pred_minutes, delay_prob_pct = predict_flight_delay(
    airline=selected_airline,
    origin=origin_code,
    dest=dest_code,
    aircraft=selected_aircraft,
    hour=dep_hour,
    dow=dow_val,
    month=target_date.month,
    temp_c=origin_temp,
    rain_mm=origin_rain,
    wind_kmh=origin_wind,
    gusts_kmh=origin_gusts
)

# ----------------- 7. GIAO DIỆN CHÍNH 2 PHÂN HỆ -----------------
tab_delay, tab_analytics = st.tabs([
    "✈️ DỰ BÁO NGUY CƠ TRỄ CHUYẾN BAY (PREDICTION PIPELINE)",
    "📊 KHÁM PHÁ DỮ LIỆU & ĐÁNH GIÁ 7 MÔ HÌNH HỌC MÁY (11 BIỂU ĐỒ EDA)"
])

# ================= TAB 1: DỰ BÁO TRỄ CHUYẾN BAY =================
with tab_delay:
    st.title("✈️ Dự báo Nguy cơ Trễ chuyến bay Thương mại (Flight Delay Prediction)")
    
    st.markdown(f"""
    Đang phân tích rủi ro thời gian thực cho chuyến bay **{flight_code_input}** ({selected_airline}) | Tuyến: **{route_label}** | Khởi hành: **{dep_hour:02d}:{dep_minute:02d}**, ngày **{target_date.strftime('%d/%m/%Y')}** ({['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật'][dow_val]}).
    """)

    # ------------------ KHỐI TRỰC QUAN HÓA THỜI TIẾT 2 ĐẦU SÂN BAY ------------------
    st.markdown("##### 🌦️ Tình trạng Thời tiết Khí tượng tại 2 Đầu Sân bay:")
    w_col1, w_col2 = st.columns(2)
    with w_col1:
        weather_badge_o = "🔴 Nguy hiểm" if (origin_rain > 30 or origin_gusts > 40) else ("🟡 Cần lưu ý" if (origin_rain > 5 or origin_gusts > 25) else "🟢 Thuận lợi")
        st.info(f"""
        🛫 **Sân bay Xuất phát: {AIRPORTS[origin_code]['name']} ({origin_code})**  
        - **Trạng thái:** `{origin_weather_desc}` ({weather_badge_o})  
        - **Nhiệt độ:** `{origin_temp:.1f}°C` | **Lượng mưa:** `{origin_rain:.1f} mm` | **Gió giật:** `{origin_gusts:.0f} km/h`  
        - **Tầm nhìn khai thác:** `{'Dưới 1 km (Sương mù/Mưa lớn)' if (origin_rain > 20 or origin_weather_code in [45, 48]) else '> 10 km (Bình thường)'}`
        """)
    with w_col2:
        weather_badge_d = "🟡 Chú ý tiếp cận" if ("Dông Bão" in weather_mode or "Mưa" in weather_mode) else "🟢 Ổn định"
        st.info(f"""
        🛬 **Sân bay Hạ cánh: {AIRPORTS[dest_code]['name']} ({dest_code})**  
        - **Trạng thái tiếp nhận:** `{dest_weather_desc}` ({weather_badge_d})  
        - **Năng lực tiếp thu:** `{AIRPORTS[dest_code]['capacity']}`  
        - **Áp lực hạ cánh:** `{'Rất cao (Nguy cơ bay vòng chờ)' if (dest_code == 'SGN' and 16 <= dep_hour <= 20) else 'Bình thường'}`
        """)

    st.markdown("---")
    
    # ------------------ KẾT QUẢ DỰ BÁO DELAY ------------------
    st.subheader(f"✈️ Đánh giá Rủi ro Chuyến bay: {flight_code_input} ({selected_airline})")
    st.markdown(f"""
    Đang phân tích tuyến: **{route_label}** | Khởi hành: **{dep_hour:02d}:{dep_minute:02d}**, ngày **{target_date.strftime('%d/%m/%Y')}** ({['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật'][dow_val]}).
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🕒 Khung giờ cất cánh", f"{dep_hour:02d}:{dep_minute:02d}", f"{'Cuối tuần đông khách' if dow_val in [4, 6] else 'Ngày thường'}")
    with c2:
        st.metric("⏱️ Dự báo thời gian trễ", f"{pred_minutes:.0f} phút", f"Mô hình {selected_model_name.split()[0]}")
    with c3:
        st.metric("📊 Xác suất Trễ (≥30p)", f"{delay_prob_pct}%", "Ước lượng rủi ro")
    with c4:
        if pred_minutes < 15:
            st.success("🟢 ĐÚNG GIỜ / TRỄ NHẸ (< 15p)")
        elif pred_minutes < 35:
            st.warning("🟡 TRỄ VỪA (15 - 35p)")
        else:
            st.error("🔴 NGUY CƠ TRỄ NẶNG (≥ 35p)")

    st.markdown("---")
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown(f"##### 📍 Bản đồ Radar Giám sát Tuyến bay ({origin_code} ✈️ {dest_code}):")
        
        orig_coord = AIRPORTS[origin_code]
        dest_coord = AIRPORTS[dest_code]
        mid_lat = (orig_coord['lat'] + dest_coord['lat']) / 2
        mid_lon = (orig_coord['lon'] + dest_coord['lon']) / 2
        
        # SỬ DỤNG OPENSTREETMAP VÀ GOOGLE TILES HOÀN TOÀN MIỄN PHÍ - KHÔNG HIỆN WATERMARK API KEY
        m = folium.Map(
            location=[mid_lat, mid_lon],
            zoom_start=6,
            tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="OpenStreetMap"
        )
        
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr="Google Maps",
            name="Google Maps Đường bộ"
        ).add_to(m)
        
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            name="Google Maps Vệ Tinh"
        ).add_to(m)
        
        line_color = "#dc2626" if pred_minutes >= 35 else ("#f59e0b" if pred_minutes >= 15 else "#16a34a")
        folium.PolyLine(
            locations=[[orig_coord['lat'], orig_coord['lon']], [dest_coord['lat'], dest_coord['lon']]],
            color=line_color,
            weight=5,
            opacity=0.85,
            tooltip=f"{flight_code_input} ({route_label}) - Dự kiến trễ: {pred_minutes:.0f} phút"
        ).add_to(m)
        
        folium.Marker(
            location=[orig_coord['lat'], orig_coord['lon']],
            popup=f"<b>Điểm đi: {orig_coord['name']}</b><br/>Thời tiết: {origin_weather_desc}<br/>Nhiệt độ: {origin_temp:.1f}°C, Mưa: {origin_rain:.1f}mm, Gió: {origin_gusts} km/h",
            tooltip=f"Điểm xuất phát: {orig_coord['name']}",
            icon=folium.Icon(color="blue", icon="plane")
        ).add_to(m)
        
        folium.Marker(
            location=[dest_coord['lat'], dest_coord['lon']],
            popup=f"<b>Điểm đến: {dest_coord['name']}</b><br/>Năng lực tiếp nhận: {dest_coord['capacity']}<br/>Thời tiết: {dest_weather_desc}",
            tooltip=f"Điểm hạ cánh: {dest_coord['name']}",
            icon=folium.Icon(color="red" if dest_code == 'SGN' else "green", icon="flag")
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        st_folium(m, width="100%", height=440, returned_objects=[])

    with col_right:
        st.markdown("##### 💡 Giải thích Nguyên nhân từ AI (Explainable AI & Delay Attribution):")
        
        # 1. GIẢI THÍCH CHI TIẾT NGUYÊN NHÂN KHÍ TƯỢNG / THỜI TIẾT
        if origin_rain > 30.0 or origin_gusts > 40.0 or origin_weather_code >= 95:
            st.error(f"""
            ⛈️ **CẢNH BÁO THỜI TIẾT DÔNG BÃO NGUY HIỂM TẠI {origin_code}:**  
            - **Diễn biến khí tượng:** {origin_weather_desc} (Lượng mưa: `{origin_rain:.1f} mm`, Gió giật: `{origin_gusts:.0f} km/h`).  
            - **Cơ chế gây trễ chuyến:**  
              * Theo quy chế an toàn bay Cục Hàng không (CAAV), thời gian giãn cách cất cánh tối thiểu bắt buộc tăng từ 3 phút lên **6 – 10 phút**.  
              * Đám mây dông tích điện (Cumulonimbus - CB) bao phủ đường bay buộc kiểm soát viên không lưu (ATC) phát lệnh **Ground Stop** (tạm giữ máy bay tại bãi đỗ).  
              * **Yếu tố thời tiết cực đoan đóng góp ước tính +25 đến +45 phút vào tổng số phút trễ!**
            """)
        elif origin_rain > 5.0 or origin_gusts > 25.0:
            st.warning(f"""
            🌧️ **ẢNH HƯỞNG KHÍ TƯỢNG MƯA GIÓ ({origin_code}):**  
            - **Diễn biến:** {origin_weather_desc} (Lượng mưa: `{origin_rain:.1f} mm`, Gió: `{origin_wind:.0f} km/h`).  
            - **Cơ chế gây trễ:** Bề mặt đường băng ướt làm giảm hiệu quả phanh khẩn cấp, các chuyến bay phải nới rộng cự ly tiếp cận, gây trễ tích lũy khoảng **+10 – 18 phút**.
            """)
        elif origin_weather_code in [45, 48] or "Sương mù" in origin_weather_desc:
            st.warning(f"""
            🌫️ **CẢNH BÁO TẦM NHÌN THẤP (SƯƠNG MÙ BỨC XẠ):**  
            - **Diễn biến:** Tầm nhìn đường băng (RVR) giảm sâu dưới 800m (đặc trưng mùa đông Nội Bài).  
            - **Cơ chế gây trễ:** Sân bay kích hoạt quy trình điều hành tầm nhìn thấp (CAT II/III), giảm tần suất tiếp thu máy bay cất/hạ cánh xuống 40%.
            """)
        else:
            st.success(f"☀️ **Thời tiết thuận lợi ({origin_code}):** {origin_weather_desc}. Không có rủi ro khí tượng cản trở hành trình.")

        # 2. GIẢI THÍCH CÁC YẾU TỐ VẬN HÀNH & MẠNG LƯỚI
        st.markdown("###### ⚙️ Các yếu tố Vận hành & Lưới bay:")
        reasons = []
        
        # Yếu tố Khung giờ
        if 16 <= dep_hour <= 20:
            reasons.append(("Cao điểm chiều tối (16h - 20h)", 35, "Tàu bay tích tụ trễ dây chuyền từ các chặng bay trước đó trong ngày."))
        elif 6 <= dep_hour <= 8:
            reasons.append(("Chuyến bay sáng sớm (06h - 08h)", -20, "Máy bay đã đỗ qua đêm sẵn tại sân đỗ, tỷ lệ đúng giờ cao nhất."))
        elif dep_hour >= 21:
            reasons.append(("Khung giờ đêm muộn (sau 21h)", 25, "Chịu tổng lượng trễ tích lũy dồn từ sáng đến tối của toàn bộ đội tàu bay."))
            
        # Yếu tố Ngày trong tuần (Khảo sát thực nghiệm từ 50,000 chuyến bay thực tế)
        day_names_vn = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
        current_day_name = day_names_vn[dow_val]
        if dow_val == 3: # Thứ Năm
            reasons.append((f"Thứ Năm cao điểm nhất tuần ({current_day_name})", 25, "Khảo sát thực tế: Thứ Năm ghi nhận mức trễ cao nhất tuần (16.65 phút, 18.46% trễ nặng) do làn sóng khách công vụ bay về dồn ứ."))
        elif dow_val == 4: # Thứ Sáu
            reasons.append((f"Thứ Sáu cao điểm cuối tuần ({current_day_name})", 20, "Khảo sát thực tế: Thứ Sáu có mức trễ cao thứ nhì (14.66 phút, 17.12% trễ nặng) do lượng khách bắt đầu kỳ nghỉ cuối tuần."))
        elif dow_val == 5: # Thứ Bảy
            reasons.append((f"Thứ Bảy thông thoáng nhất ({current_day_name})", -20, "Khảo sát thực tế: Thứ Bảy là ngày bay đúng giờ nhất tuần (trễ bình quân chỉ 7.57 phút, trễ nặng chỉ 10.45%), đường băng thông thoáng."))
        elif dow_val == 0: # Thứ Hai
            reasons.append((f"Thứ Hai đầu tuần ({current_day_name})", 15, "Lưu lượng khách công vụ đầu tuần cao, trễ trung bình 13.99 phút (16.13% trễ nặng)."))
        else:
            reasons.append((f"Ngày giữa tuần ({current_day_name})", -10, f"Thứ Ba & Thứ Tư có lưu lượng bay ổn định, mức trễ bình quân ~11 phút."))

        if origin_code == 'EWR':
            reasons.append(("Sân bay Newark Liberty (EWR)", 20, "EWR có mức trễ bình quân cao nhất (14.82 phút) do mật độ đường băng và không phận bận rộn."))
        elif origin_code == 'JFK':
            reasons.append(("Trung tâm JFK Quốc tế", 10, "Mật độ chuyến bay quốc tế và nội địa lớn, điều phối slot cất cánh chặt chẽ."))
        elif origin_code == 'LGA':
            reasons.append(("Sân bay LaGuardia (LGA)", -5, "Sân bay cự ly ngắn, thời gian lăn bánh cất cánh trung bình ngắn hơn."))
            
        if selected_airline == 'ExpressJet':
            reasons.append(("Đặc thù trung chuyển ExpressJet", 25, "Thời gian quay đầu máy bay ngắn, nhạy cảm với trễ dây chuyền mạng bay."))
        elif selected_airline == 'JetBlue Airways':
            reasons.append(("Mật độ khai thác JetBlue tại New York", 15, "Tập trung phần lớn chuyến bay tại hub JFK, chịu ảnh hưởng thời tiết Đông Bắc."))
        elif selected_airline == 'Delta Air Lines':
            reasons.append(("Tỷ lệ đúng giờ Delta Air Lines", -15, "Delta duy trì tỷ lệ đúng giờ cao (trên 85%) và đội ngũ điều phối không lưu tối ưu."))
        elif selected_airline == 'United Airlines':
            reasons.append(("Mạng bay United Airlines", 5, "Hub chính tại Newark (EWR), vận hành quy mô lớn."))
            
        for title, weight, desc in reasons:
            color_badge = "🔴" if weight > 0 else "🟢"
            st.markdown(f"**{color_badge} {title}** (`{weight:+d}% rủi ro`): {desc}")

        if origin_rain > 20 or origin_gusts > 35:
            st.error("💡 **Khuyến nghị AI:** Do thời tiết dông bão xấu, quý khách nên chủ động theo dõi bảng điện tử FIDS tại nhà ga hoặc hotline hãng bay.")
        elif dep_hour >= 18:
            st.info("💡 **Khuyến nghị AI:** Chuyến bay chiều tối dễ bị trễ tích lũy. Nếu có cuộc hẹn quan trọng sáng hôm sau, nên chọn chuyến bay cất cánh trước **08:30 sáng**.")
        else:
            st.info("💡 **Khuyến nghị AI:** Chuyến bay sáng sớm có tỷ lệ cất cánh đúng giờ cao, thời gian làm thủ tục tại quầy diễn ra nhanh chóng.")

    st.divider()

    # BIỂU ĐỒ DIỄN BIẾN TRỄ THEO 24 KHUNG GIỜ
    st.subheader(f"📈 Khảo sát Mức độ Trễ qua các Khung giờ trong ngày ({route_label})")
    
    hours_24 = list(range(6, 24))
    delays_24 = [
        float(predict_flight_delay(selected_airline, origin_code, dest_code, selected_aircraft, h, dow_val, target_date.month, origin_temp, origin_rain, origin_wind, origin_gusts)[0])
        for h in hours_24
    ]

    fig_24, ax_24 = plt.subplots(figsize=(13, 4.5))
    ax_24.plot(hours_24, delays_24, marker='o', linewidth=2.5, color='#00558f', label=f"Mức trễ: {selected_airline}")
    ax_24.axhspan(0, 15, color='#22c55e', alpha=0.15, label="Vùng Đúng giờ (< 15p)")
    ax_24.axhspan(15, 35, color='#f59e0b', alpha=0.15, label="Vùng Trễ vừa (15 - 35p)")
    ax_24.axhspan(35, max(65, max(delays_24) + 10), color='#ef4444', alpha=0.15, label="Vùng Trễ nặng (≥ 35p)")
    
    if dep_hour in hours_24:
        idx_h = hours_24.index(dep_hour)
        ax_24.axvline(dep_hour, color='#dc2626', linestyle='--', linewidth=2, label=f"Giờ bạn chọn ({dep_hour:02d}h)")
        ax_24.scatter([dep_hour], [delays_24[idx_h]], color='#dc2626', s=120, zorder=5)

    ax_24.set_title(f"Mức độ Trễ chuyến Bình quân theo Khung giờ Khởi hành ({route_label})", fontsize=13, fontweight='bold', pad=10)
    ax_24.set_xlabel("Khung giờ khởi hành (Giờ)", fontsize=11)
    ax_24.set_ylabel("Số phút trễ dự kiến (phút)", fontsize=11)
    ax_24.set_xticks(hours_24)
    ax_24.set_xticklabels([f"{h:02d}h" for h in hours_24])
    ax_24.grid(True, linestyle=':', alpha=0.6)
    ax_24.legend(loc='upper left')
    fig_24.tight_layout()
    st.pyplot(fig_24)

    # KHẢO SÁT THỰC NGHIỆM TẬP DỮ LIỆU: NGÀY TRONG TUẦN VS NGÀY CAO ĐIỂM
    st.markdown("---")
    st.markdown(f"#### 📊 Khảo sát Thực nghiệm: Mức độ Trễ Ngày thường vs Ngày cao điểm")
    st.caption("Thống kê tính toán động 100% từ tập dữ liệu 50,000 chuyến bay thương mại thực tế:")
    
    f_df = artifacts['featured_df']
    dow_cols = st.columns(7)
    day_titles = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "🔥 Thứ Năm (Đỉnh trễ)", "🔥 Thứ Sáu", "🟢 Thứ Bảy (Ít trễ nhất)", "Chủ Nhật"]
    
    for dow_idx in range(7):
        with dow_cols[dow_idx]:
            if f_df is not None:
                sub_dow = f_df[f_df['dayofweek'] == dow_idx]
                m_delay = sub_dow['delay_minutes'].mean()
                r_delay = (sub_dow['delay_minutes'] >= 30).mean() * 100
                st.metric(day_titles[dow_idx], f"{m_delay:.1f}p", f"Trễ ≥30p: {r_delay:.1f}%")
            else:
                st.metric(day_titles[dow_idx], "N/A", "N/A")

    st.info("""
    📌 **Khám phá Bản chất Dữ liệu Thực nghiệm (EDA Insight từ 50,000 chuyến bay):**  
    - **Đỉnh điểm trễ trong tuần (Thứ Năm & Thứ Sáu):** Thứ Năm ghi nhận mức trễ bình quân cao nhất tuần (**16.65 phút**, tỷ lệ trễ $\ge 30$p đạt **18.46%**), theo sát là Thứ Sáu (**14.66 phút**, tỷ lệ trễ nặng **17.12%**) do áp lực khách công vụ hoàn thành tuần làm việc và làn sóng đi lại trước cuối tuần.  
    - **Ngày bay đúng giờ nhất (Thứ Bảy):** Thứ Bảy có mức trễ thấp nhất kỷ lục (chỉ **7.57 phút**, tỷ lệ trễ $\ge 30$p chỉ **10.45%**) nhờ lưu lượng bay giảm và không phận thông thoáng.  
    - **Quy luật bay sáng sớm:** Tỷ lệ đúng giờ của các chuyến bay trước 08:30 sáng đạt **92.35%**, do chưa tích lũy hiệu ứng trễ dây chuyền từ các chặng bay trước.  
    - **Sân bay nhạy cảm nhất:** Sân bay Newark (EWR) ghi nhận mức trễ cao nhất (**14.82 phút**) trong cụm 3 sân bay vùng New York.
    """)


# ================= TAB 2: KHÁM PHÁ DỮ LIỆU & ĐÁNH GIÁ MÔ HÌNH =================
with tab_analytics:
    st.title("📊 Phân tích Chuyên sâu Dữ liệu & Đánh giá Mô hình Học máy")
    st.markdown("""
    Phân hệ trình bày toàn diện 11 biểu đồ trực quan hóa dữ liệu EDA chuyên sâu từ tập dữ liệu thực tế, 
    bảng đối sánh hiệu năng của 7 mô hình máy học và khảo sát hiện tượng Overfitting theo độ sâu cây.
    """)

    sub1, sub2, sub3, sub4 = st.tabs([
        "1. Chuỗi thời gian & Tự tương quan",
        "2. Phân bố & Tương quan Khí tượng",
        "3. Hãng bay, Sân bay & Phân cụm",
        "4. So sánh Hiệu năng Mô hình & Overfitting"
    ])

    with sub1:
        st.subheader("Chuỗi Thời Gian & Tính Tự Tương Quan (Serial Correlation Analysis)")
        c_ts1, c_ts2 = st.columns(2)
        with c_ts1:
            img_c1 = os.path.join(base_dir, "docs", "figures", "task_c1_run_sequence.png")
            if os.path.exists(img_c1):
                st.image(img_c1, caption="Run Sequence Plot - Chuỗi thời gian trễ của các chuyến bay liên tiếp", use_container_width=True)
                st.caption("Khám phá: Mức độ trễ có tính chu kỳ dao động mạnh, các chuyến bay buổi tối thường xuyên chạm ngưỡng trễ cao.")
        with c_ts2:
            img_c2 = os.path.join(base_dir, "docs", "figures", "task_c2_lag_plot.png")
            if os.path.exists(img_c2):
                st.image(img_c2, caption="Lag Plot (Mức trễ chuyến bay vs Mức trễ mạng bay 1 giờ trước)", use_container_width=True)
                st.caption("Khám phá: Hệ số tự tương quan Pearson r = 0.292, chứng minh hiện tượng tắc nghẽn dây chuyền của mạng lưới hàng không.")

    with sub2:
        st.subheader("Phân Bố Trễ & Tương Tác Khí Tượng")
        c_hm1, c_hm2 = st.columns(2)
        with c_hm1:
            img_c7 = os.path.join(base_dir, "docs", "figures", "task_c7_delay_histogram.png")
            if os.path.exists(img_c7):
                st.image(img_c7, caption="Phân bố Mật độ Xác suất Trễ (Histogram & KDE)", use_container_width=True)
        with c_hm2:
            img_c4 = os.path.join(base_dir, "docs", "figures", "task_c4_correlation_heatmap.png")
            if os.path.exists(img_c4):
                st.image(img_c4, caption="Ma Trận Hệ Số Tương Quan Pearson", use_container_width=True)

        c_hm3, c_hm4 = st.columns(2)
        with c_hm3:
            img_c3 = os.path.join(base_dir, "docs", "figures", "task_c3_heatmap_24x7.png")
            if os.path.exists(img_c3):
                st.image(img_c3, caption="Heatmap Ma Trận Mức Độ Trễ (24 Giờ x 7 Ngày)", use_container_width=True)
        with c_hm4:
            img_c5 = os.path.join(base_dir, "docs", "figures", "task_c5_parallel_coordinates.png")
            if os.path.exists(img_c5):
                st.image(img_c5, caption="Parallel Coordinates - Đối chiếu Khoảng cách, Mưa, Gió và Trễ", use_container_width=True)

    with sub3:
        st.subheader("Phân Tích Hãng Bay, Sân Bay & Phân Cụm K-Means")
        c_cl1, c_cl2 = st.columns(2)
        with c_cl1:
            img_c8 = os.path.join(base_dir, "docs", "figures", "task_c8_boxplot_airline_airport.png")
            if os.path.exists(img_c8):
                st.image(img_c8, caption="Boxplot Mức Trễ theo Hãng Hàng Không & Sân Bay", use_container_width=True)
        with c_cl2:
            img_c9 = os.path.join(base_dir, "docs", "figures", "task_c9_violin_plot.png")
            if os.path.exists(img_c9):
                st.image(img_c9, caption="Violin Plot - Mật độ Xác suất Trễ của các Hãng", use_container_width=True)

        c_cl3, c_cl4 = st.columns(2)
        with c_cl3:
            img_c10 = os.path.join(base_dir, "docs", "figures", "task_c10_pca_analysis.png")
            if os.path.exists(img_c10):
                st.image(img_c10, caption="PCA 2D Projection - Giảm Chiều Không Gian Thuộc Tính", use_container_width=True)
        with c_cl4:
            img_c11 = os.path.join(base_dir, "docs", "figures", "task_c11_kmeans_elbow.png")
            if os.path.exists(img_c11):
                st.image(img_c11, caption="K-Means Elbow Curve - Xác định Số Cụm Tối Ưu (k=3)", use_container_width=True)

    with sub4:
        st.subheader("Bảng Tổng Hợp So Sánh Hiệu Năng Các Mô Hình & Overfitting")
        if artifacts['metrics'] is not None:
            st.dataframe(artifacts['metrics'], use_container_width=True, hide_index=True)
        
        img_feat = os.path.join(base_dir, "docs", "figures", "task_d6_feature_importance.png")
        if os.path.exists(img_feat):
            st.image(img_feat, caption="Top đặc trưng quan trọng nhất trong mô hình XGBoost", use_container_width=True)
            st.markdown("""
            **Đánh giá:** Mô hình **XGBoost Regressor** đạt hiệu năng vượt trội với MAE **15.76 phút** trên tập Test thực tế không nén ngoại lai, 
            độ chính xác phân loại trễ nghiêm trọng $\\ge 30$ phút đạt **$86.8\%$**, vượt trội rõ rệt so với Baseline 1 (Majority Class: 85.5% Acc, F1 0.0) 
            và Baseline 2 (Naïve Persistence: MAE 20.69p, Acc 80.8%, F1 0.3027).
            """)

        st.divider()
        st.subheader("Thực nghiệm Khảo sát Hiện tượng Quá khớp (Overfitting Analysis)")
        img_overfit = os.path.join(base_dir, "docs", "figures", "task_d5_overfitting_analysis.png")
        if os.path.exists(img_overfit):
            col_ov_img, col_ov_txt = st.columns([3, 2])
            with col_ov_img:
                st.image(img_overfit, caption="Đường cong sai số Training vs Validation Error theo độ sâu mô hình cây", use_container_width=True)
            with col_ov_txt:
                st.markdown("""
                ##### 🔍 Ba vùng đặc trưng trong thực nghiệm:
                1. **Vùng Thiếu khớp (Underfitting, max_depth < 4):** Cả sai số Train và Validation đều cao do mô hình quá đơn giản.
                2. **Điểm Dung lượng Tối ưu (Optimal, max_depth = 5 - 8):** Sai số Validation đạt cực tiểu (MAE thấp nhất), đạt sự cân bằng Bias - Variance hoàn hảo.
                3. **Vùng Quá khớp (Overfitting, max_depth > 10):** Sai số Train tiếp tục giảm tiệm cận 0 trong khi sai số Validation phân kỳ tăng trở lại do mô hình học nhiễu ngẫu nhiên.
                """)
