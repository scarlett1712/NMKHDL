"""
Ứng dụng Web Dashboard Streamlit:
Hệ thống Dự báo Nguy cơ Trễ chuyến bay (Flight Delay Predictor)
Chặng bay: Các đường bay thương mại nội địa Mỹ từ cụm sân bay NYC (JFK / EWR / LGA) - US Airlines
Tích hợp: Open-Meteo REST API thời gian thực, dữ liệu Web Crawler (Wikipedia/FAA) & Mô hình học máy XGBoost
Môn học: Nhập môn Khoa học Dữ liệu (IT4930 / IT4142) - SoICT, Đại học Bách Khoa Hà Nội (HUST)

Mọi con số thống kê hiển thị trong app đều được tính từ data/processed/featured_flight_dataset.csv
và các file kết quả trong docs/figures/, không gõ tay.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Flight Delay Predictor - Dự báo Trễ chuyến bay NYC",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- THÔNG TIN SÂN BAY & ĐƯỜNG BAY THỰC TẾ -----------------
AIRPORTS = {
    'JFK': {'name': 'John F. Kennedy (New York)', 'lat': 40.6413, 'lon': -73.7781},
    'EWR': {'name': 'Newark Liberty (New Jersey)', 'lat': 40.6895, 'lon': -74.1745},
    'LGA': {'name': 'LaGuardia (New York)', 'lat': 40.7769, 'lon': -73.8740},
    'ORD': {'name': "Chicago O'Hare (Chicago)", 'lat': 41.9742, 'lon': -87.9073},
    'LAX': {'name': 'Los Angeles (California)', 'lat': 33.9416, 'lon': -118.4085},
    'ATL': {'name': 'Hartsfield-Jackson (Atlanta)', 'lat': 33.6407, 'lon': -84.4277},
    'MIA': {'name': 'Miami International (Florida)', 'lat': 25.7959, 'lon': -80.2870},
    'BOS': {'name': 'Boston Logan (Massachusetts)', 'lat': 42.3656, 'lon': -71.0096},
    'SFO': {'name': 'San Francisco (California)', 'lat': 37.6213, 'lon': -122.3790}
}

AIRLINES_INFO = {
    'United Airlines': {'code': 'UA', 'color': '#00558f'},
    'Delta Air Lines': {'code': 'DL', 'color': '#dc2626'},
    'American Airlines': {'code': 'AA', 'color': '#0284c7'},
    'JetBlue Airways': {'code': 'B6', 'color': '#0f766e'},
    'ExpressJet': {'code': 'EV', 'color': '#f59e0b'},
    'Southwest Airlines': {'code': 'WN', 'color': '#ca8a04'}
}
DAY_NAMES_VN = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']

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

    traffic_path = os.path.join(base_dir, "data", "raw", "crawled_us_airport_traffic.csv")
    airport_traffic = pd.read_csv(traffic_path).set_index('iata') if os.path.exists(traffic_path) else None

    def _load_opt(name):
        p = os.path.join(model_dir, name)
        return joblib.load(p) if os.path.exists(p) else None

    def _read_fig_csv(name):
        p = os.path.join(base_dir, "docs", "figures", name)
        return pd.read_csv(p) if os.path.exists(p) else None

    thr = _load_opt("classifier_threshold.pkl")
    classifier_threshold = float(thr) if thr is not None else 0.5
    calibrator = _load_opt("classifier_calibrator.pkl")
    model_info = _load_opt("model_info.pkl") or {}

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
        'airport_traffic': airport_traffic,
        'classifier_threshold': classifier_threshold,
        'calibrator': calibrator,
        'model_info': model_info,
        'cls_metrics': _read_fig_csv("classification_metrics.csv"),
        'calibration': _read_fig_csv("probability_calibration.csv"),
        'overfit_curve': _read_fig_csv("overfitting_curve.csv"),
        'base_dir': base_dir
    }

artifacts = load_models_and_artifacts()
base_dir = artifacts['base_dir']
FDF = artifacts['featured_df']
OVERALL_MEAN = float(FDF['delay_minutes'].mean()) if FDF is not None else float('nan')
OVERALL_SEVERE = float((FDF['delay_minutes'] >= 30).mean()) if FDF is not None else float('nan')

def group_effect(mask: pd.Series):
    """Thống kê lịch sử 2013 của một nhóm chuyến bay so với toàn bộ dữ liệu: (số chuyến, trễ TB, % trễ ≥30p)."""
    sub = FDF.loc[mask, 'delay_minutes']
    return len(sub), float(sub.mean()), float((sub >= 30).mean())

def hour_bucket(h: int):
    if h < 9:
        return "Sáng sớm (trước 9h)", (FDF['hour'] < 9)
    if h < 13:
        return "Buổi sáng (9h - 12h)", FDF['hour'].between(9, 12)
    if h < 17:
        return "Buổi chiều (13h - 16h)", FDF['hour'].between(13, 16)
    if h < 21:
        return "Chiều tối (17h - 20h)", FDF['hour'].between(17, 20)
    return "Đêm muộn (từ 21h)", (FDF['hour'] >= 21)

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

selected_aircraft = None  # Mô hình không dùng loại máy bay làm đặc trưng

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
elif "Sương mù" in weather_mode:
    dest_weather_desc = "Sương mù hạn chế hạ cánh"

st.sidebar.caption(f"🌦️ **{origin_code}**: {origin_temp:.1f}°C, Mưa: {origin_rain:.1f}mm, Gió: {origin_wind:.0f}km/h")
st.sidebar.caption("⚠️ Mô hình học từ dữ liệu năm 2013. Thời tiết trực tiếp/giả lập chỉ dùng để minh họa cách mô hình phản ứng.")

st.sidebar.divider()
XGB_LABEL = artifacts['model_info'].get('objective_label', 'XGBoost')
selected_model_name = st.sidebar.selectbox(
    "🤖 Chọn Mô hình Học máy:",
    [f"{XGB_LABEL} - MAE thấp nhất", "Random Forest", "Ridge Regression (L2)"]
)
PREDICTS_MEDIAN = "absolute error" in selected_model_name

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
    
    dist_map = {'LAX': 2475.0, 'ORD': 719.0, 'ATL': 762.0, 'MIA': 1089.0, 'SFO': 2565.0, 'BOS': 190.0}
    dist = dist_map.get(dest, 1000.0)
    inp['distance'] = dist

    # Quy mô sân bay đến - lấy từ dữ liệu Web Crawler (Wikipedia / FAA) đã tích hợp
    traffic = artifacts.get('airport_traffic')
    if traffic is not None and dest in traffic.index:
        inp['dest_hub_tier'] = float(traffic.loc[dest, 'hub_tier'])
        inp['dest_enplanements_m'] = float(traffic.loc[dest, 'enplanements_2016']) / 1e6

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

    # Xác suất trễ >= 30 phút: điểm thô XGBClassifier -> hiệu chỉnh Isotonic; cảnh báo theo ngưỡng chọn trên Val
    prob_pct, alert = None, False
    if artifacts['classifier'] is not None:
        raw = float(artifacts['classifier'].predict_proba(df_sample)[0, 1])
        alert = raw >= artifacts['classifier_threshold']
        prob = float(artifacts['calibrator'].predict([raw])[0]) if artifacts['calibrator'] is not None else raw
        prob_pct = int(round(prob * 100))

    return float(pred_delay), prob_pct, alert

dow_val = target_date.weekday()
pred_minutes, delay_prob_pct, severe_alert = predict_flight_delay(
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
N_MODELS = len(artifacts['metrics']) if artifacts['metrics'] is not None else 0
N_FIGS = len([f for f in os.listdir(os.path.join(base_dir, "docs", "figures")) if f.startswith("task_c") and f.endswith(".png")])
tab_delay, tab_analytics = st.tabs([
    "✈️ DỰ BÁO NGUY CƠ TRỄ CHUYẾN BAY",
    f"📊 KHÁM PHÁ DỮ LIỆU ({N_FIGS} BIỂU ĐỒ EDA) & ĐÁNH GIÁ {N_MODELS} MÔ HÌNH (GỒM 2 BASELINE)"
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
        traffic = artifacts['airport_traffic']
        if traffic is not None and dest_code in traffic.index:
            hub_text = f"{traffic.loc[dest_code, 'hub_category']} - {traffic.loc[dest_code, 'enplanements_2016'] / 1e6:.1f} triệu lượt khách lên máy bay (FAA 2016, Wikipedia)"
        else:
            hub_text = "Không thuộc nhóm large/medium hub của FAA"
        st.info(f"""
        🛬 **Sân bay Hạ cánh: {AIRPORTS[dest_code]['name']} ({dest_code})**
        - **Trạng thái tiếp nhận:** `{dest_weather_desc}` ({weather_badge_d})
        - **Quy mô sân bay (dữ liệu Web Crawler):** `{hub_text}`
        """)

    st.markdown("---")
    
    # ------------------ KẾT QUẢ DỰ BÁO DELAY ------------------
    st.subheader(f"✈️ Đánh giá Rủi ro Chuyến bay: {flight_code_input} ({selected_airline})")
    st.markdown(f"""
    Đang phân tích tuyến: **{route_label}** | Khởi hành: **{dep_hour:02d}:{dep_minute:02d}**, ngày **{target_date.strftime('%d/%m/%Y')}** ({['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật'][dow_val]}).
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🕒 Khung giờ cất cánh", f"{dep_hour:02d}:{dep_minute:02d}", DAY_NAMES_VN[dow_val], delta_color="off")
    with c2:
        st.metric("⏱️ Dự báo trễ (trung vị)" if PREDICTS_MEDIAN else "⏱️ Dự báo trễ (trung bình)",
                  f"{pred_minutes:.0f} phút", selected_model_name.split(' - ')[0], delta_color="off")
    with c3:
        if delay_prob_pct is not None:
            st.metric("📊 Xác suất trễ ≥ 30 phút", f"{delay_prob_pct}%",
                      f"Trung bình lịch sử: {OVERALL_SEVERE * 100:.1f}%", delta_color="off")
        else:
            st.metric("📊 Xác suất trễ ≥ 30 phút", "N/A")
    with c4:
        if severe_alert:
            st.error(f"🔴 CẢNH BÁO NGUY CƠ TRỄ ≥ 30 PHÚT\n\n(XGBClassifier vượt ngưỡng {artifacts['classifier_threshold']:.2f} chọn trên tập Validation)")
        elif delay_prob_pct is not None and delay_prob_pct / 100 > OVERALL_SEVERE:
            st.warning("🟡 RỦI RO CAO HƠN MỨC TRUNG BÌNH")
        else:
            st.success("🟢 RỦI RO THẤP / BÌNH THƯỜNG")
    if PREDICTS_MEDIAN:
        st.caption("Mô hình XGBoost tối ưu hàm mất mát absolute error nên dự báo **trung vị** số phút trễ "
                   f"(trung vị lịch sử ≈ {FDF['delay_minutes'].median():.0f} phút: đa số chuyến cất cánh đúng giờ). "
                   "Nguy cơ trễ nặng nên đọc theo cột xác suất và cảnh báo.")

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
        
        line_color = "#dc2626" if severe_alert else ("#f59e0b" if (delay_prob_pct or 0) / 100 > OVERALL_SEVERE else "#16a34a")
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
            popup=f"<b>Điểm đến: {dest_coord['name']}</b><br/>Thời tiết: {dest_weather_desc}",
            tooltip=f"Điểm hạ cánh: {dest_coord['name']}",
            icon=folium.Icon(color="green", icon="flag")
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        st_folium(m, width="100%", height=440, returned_objects=[])

    with col_right:
        st.markdown("##### 🌦️ Bối cảnh thời tiết (mô tả định tính):")
        if origin_rain > 30.0 or origin_gusts > 40.0 or origin_weather_code >= 95:
            st.error(f"⛈️ **Dông / gió giật mạnh tại {origin_code}:** {origin_weather_desc} "
                     f"(mưa `{origin_rain:.1f} mm`, gió giật `{origin_gusts:.0f} km/h`). Khi có dông, FAA có thể giãn cách cất cánh "
                     "hoặc tạm giữ máy bay tại sân đỗ (ground stop).")
        elif origin_rain > 5.0 or origin_gusts > 25.0:
            st.warning(f"🌧️ **Mưa / gió tại {origin_code}:** {origin_weather_desc} (mưa `{origin_rain:.1f} mm`, gió `{origin_wind:.0f} km/h`).")
        elif origin_weather_code in [45, 48] or "Sương mù" in origin_weather_desc:
            st.warning("🌫️ **Tầm nhìn thấp (sương mù):** sân bay có thể giảm tần suất cất/hạ cánh.")
        else:
            st.success(f"☀️ **Thời tiết thuận lợi ({origin_code}):** {origin_weather_desc}.")

        # Thống kê lịch sử 2013 của từng yếu tố, so với trung bình toàn bộ dữ liệu (tính trực tiếp từ CSV)
        st.markdown("###### 📊 Thống kê lịch sử 2013 của các nhóm chuyến bay tương tự:")
        if FDF is not None:
            carrier_code = AIRLINES_INFO.get(selected_airline, {}).get('code', '')
            hb_label, hb_mask = hour_bucket(dep_hour)
            factors = [
                (f"Khung giờ: {hb_label}", hb_mask),
                (f"Ngày: {DAY_NAMES_VN[dow_val]}", FDF['dayofweek'] == dow_val),
                (f"Sân bay đi: {origin_code}", FDF['origin'] == origin_code),
                (f"Hãng: {selected_airline} ({carrier_code})", FDF['airline'] == carrier_code),
                (f"Sân bay đến: {dest_code}", FDF['dest'] == dest_code),
            ]
            rows = []
            for label, mask in factors:
                n, mean_d, severe = group_effect(mask)
                if n == 0:
                    continue
                rows.append({'Yếu tố': label, 'Số chuyến': n, 'Trễ TB (phút)': round(mean_d, 1),
                             'Chênh lệch (phút)': round(mean_d - OVERALL_MEAN, 1), '% trễ ≥30p': round(severe * 100, 1)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            st.caption(f"Mốc so sánh: toàn bộ {len(FDF):,} chuyến có trễ TB {OVERALL_MEAN:.1f} phút, "
                       f"{OVERALL_SEVERE * 100:.1f}% trễ ≥ 30 phút. Đây là thống kê mô tả (tương quan), "
                       "không phải mức đóng góp nhân quả của từng yếu tố vào dự báo.")

        if severe_alert:
            st.error("💡 **Khuyến nghị:** mô hình cảnh báo nguy cơ trễ ≥ 30 phút - nên theo dõi thông báo của hãng bay.")
        elif FDF is not None:
            early_ontime = (FDF.loc[FDF['hour'] < 9, 'delay_minutes'] < 15).mean() * 100
            st.info(f"💡 **Khuyến nghị:** dữ liệu 2013 cho thấy chuyến cất cánh trước 9h đúng giờ (trễ < 15 phút) "
                    f"tới {early_ontime:.1f}%, cao nhất trong ngày.")

    st.divider()

    # BIỂU ĐỒ DIỄN BIẾN TRỄ THEO 24 KHUNG GIỜ
    st.subheader(f"📈 Dự báo trễ nếu đổi giờ khởi hành ({route_label})")
    
    hours_24 = list(range(6, 24))
    preds_24 = [
        predict_flight_delay(selected_airline, origin_code, dest_code, selected_aircraft, h, dow_val, target_date.month, origin_temp, origin_rain, origin_wind, origin_gusts)
        for h in hours_24
    ]
    delays_24 = [p[0] for p in preds_24]
    probs_24 = [p[1] if p[1] is not None else np.nan for p in preds_24]

    fig_24, ax_24 = plt.subplots(figsize=(13, 4.5))
    ax_24.plot(hours_24, delays_24, marker='o', linewidth=2.5, color='#00558f',
               label="Dự báo số phút trễ (" + ("trung vị" if PREDICTS_MEDIAN else "trung bình") + ")")
    ax_24.axhline(0, color='gray', linewidth=1)
    ax_24.set_ylabel("Số phút trễ dự báo", fontsize=11)
    ax_p = ax_24.twinx()
    ax_p.plot(hours_24, probs_24, marker='s', linewidth=2, color='#dc2626', label="Xác suất trễ ≥ 30 phút (%)")
    ax_p.axhline(OVERALL_SEVERE * 100, color='#dc2626', linestyle=':', alpha=0.6, label=f"Trung bình lịch sử ({OVERALL_SEVERE * 100:.1f}%)")
    ax_p.set_ylabel("Xác suất trễ ≥ 30 phút (%)", fontsize=11, color='#dc2626')
    ax_p.set_ylim(0, max(60, np.nanmax(probs_24) + 10) if not np.all(np.isnan(probs_24)) else 60)
    ax_p.grid(False)

    if dep_hour in hours_24:
        ax_24.axvline(dep_hour, color='#6b7280', linestyle='--', linewidth=1.5, label=f"Giờ bạn chọn ({dep_hour:02d}h)")

    ax_24.set_title(f"Dự báo của mô hình theo giờ khởi hành ({route_label}; các yếu tố khác giữ nguyên)", fontsize=13, fontweight='bold', pad=10)
    ax_24.set_xlabel("Khung giờ khởi hành (Giờ)", fontsize=11)
    ax_24.set_xticks(hours_24)
    ax_24.set_xticklabels([f"{h:02d}h" for h in hours_24])
    ax_24.grid(True, linestyle=':', alpha=0.6)
    h1, l1 = ax_24.get_legend_handles_labels()
    h2, l2 = ax_p.get_legend_handles_labels()
    ax_24.legend(h1 + h2, l1 + l2, loc='upper left')
    fig_24.tight_layout()
    st.pyplot(fig_24)

    # THỐNG KÊ THEO NGÀY TRONG TUẦN - TÍNH TRỰC TIẾP TỪ DỮ LIỆU
    st.markdown("---")
    st.markdown("#### 📊 Mức độ trễ theo ngày trong tuần (dữ liệu 2013)")
    if FDF is not None:
        by_dow = FDF.groupby('dayofweek')['delay_minutes'].agg(['mean', lambda s: (s >= 30).mean() * 100, 'size'])
        by_dow.columns = ['mean', 'severe', 'n']
        worst, best = int(by_dow['mean'].idxmax()), int(by_dow['mean'].idxmin())
        st.caption(f"Tính trực tiếp từ {len(FDF):,} chuyến bay trong data/processed/featured_flight_dataset.csv")
        dow_cols = st.columns(7)
        for dow_idx in range(7):
            tag = " 🔺" if dow_idx == worst else (" 🟢" if dow_idx == best else "")
            dow_cols[dow_idx].metric(DAY_NAMES_VN[dow_idx] + tag, f"{by_dow.loc[dow_idx, 'mean']:.1f}p",
                                     f"Trễ ≥30p: {by_dow.loc[dow_idx, 'severe']:.1f}%", delta_color="off")

        before_830 = FDF[(FDF['hour'] * 60 + FDF['minute']) < 8 * 60 + 30]
        by_origin = FDF.groupby('origin')['delay_minutes'].mean().sort_values(ascending=False)
        st.info(
            f"📌 **Nhận xét từ dữ liệu:** {DAY_NAMES_VN[worst]} có mức trễ trung bình cao nhất "
            f"({by_dow.loc[worst, 'mean']:.2f} phút, {by_dow.loc[worst, 'severe']:.1f}% trễ ≥ 30 phút); "
            f"{DAY_NAMES_VN[best]} thấp nhất ({by_dow.loc[best, 'mean']:.2f} phút, {by_dow.loc[best, 'severe']:.1f}%). "
            f"Chuyến cất cánh trước 08:30 đúng giờ (trễ < 15 phút) {(before_830['delay_minutes'] < 15).mean() * 100:.1f}%. "
            f"Theo sân bay đi: " + ", ".join(f"{o} {v:.2f} phút" for o, v in by_origin.items()) + "."
        )


# ================= TAB 2: KHÁM PHÁ DỮ LIỆU & ĐÁNH GIÁ MÔ HÌNH =================
FIG_DIR = os.path.join(base_dir, "docs", "figures")

def show_fig(name: str, caption: str):
    path = os.path.join(FIG_DIR, name)
    if os.path.exists(path):
        st.image(path, caption=caption, width='stretch')
    else:
        st.caption(f"(Chưa có {name} - chạy src/visualization/eda_plots.py)")

with tab_analytics:
    st.title("📊 Phân tích Dữ liệu & Đánh giá Mô hình Học máy")
    st.markdown(f"{N_FIGS} biểu đồ EDA từ dữ liệu thật và kết quả đánh giá {N_MODELS} mô hình hồi quy + bài toán phân loại "
                "trễ ≥ 30 phút. Mọi số liệu đọc trực tiếp từ các file CSV trong `docs/figures/`.")

    sub1, sub2, sub3, sub4 = st.tabs([
        "1. Chuỗi thời gian & Mùa vụ",
        "2. Phân bố & Tương quan",
        "3. Hãng bay, Sân bay & Phân cụm",
        "4. Hiệu năng Mô hình & Overfitting"
    ])

    with sub1:
        c_ts1, c_ts2 = st.columns(2)
        with c_ts1:
            show_fig("task_c1_run_sequence.png", "Run Sequence Plot - chuỗi trễ của các chuyến bay liên tiếp")
        with c_ts2:
            show_fig("task_c2_lag_plot.png", "Lag Plot - trễ của chuyến bay so với trễ TB của sân bay 1 giờ trước")
            if FDF is not None:
                r_lag = FDF[['network_delay_lag_1h', 'delay_minutes']].corr().iloc[0, 1]
                st.caption(f"Hệ số tương quan Pearson r = {r_lag:.3f}: trễ có tính lan truyền theo giờ tại cùng sân bay (tương quan, không khẳng định nhân quả).")
        show_fig("task_c6_monthly_hub_delay.png", "Mức trễ theo tháng và theo quy mô sân bay đến (dữ liệu Web Crawler)")

    with sub2:
        c_hm1, c_hm2 = st.columns(2)
        with c_hm1:
            show_fig("task_c7_delay_histogram.png", "Histogram & KDE số phút trễ")
        with c_hm2:
            show_fig("task_c4_correlation_heatmap.png", "Ma trận tương quan Pearson")
        c_hm3, c_hm4 = st.columns(2)
        with c_hm3:
            show_fig("task_c3_heatmap_24x7.png", "Heatmap trễ trung bình (24 giờ x 7 ngày)")
        with c_hm4:
            show_fig("task_c5_parallel_coordinates.png", "Parallel Coordinates (chuẩn hóa Min-Max)")

    with sub3:
        c_cl1, c_cl2 = st.columns(2)
        with c_cl1:
            show_fig("task_c8_boxplot_airline_airport.png", "Boxplot trễ theo hãng bay & sân bay đi")
        with c_cl2:
            show_fig("task_c9_violin_plot.png", "Violin plot trễ theo hãng bay")
        c_cl3, c_cl4 = st.columns(2)
        with c_cl3:
            show_fig("task_c10_pca_analysis.png", "PCA 2D")
        with c_cl4:
            show_fig("task_c11_kmeans_elbow.png", "K-Means Elbow (chọn k=3)")

    with sub4:
        st.subheader("Hồi quy số phút trễ - đánh giá 1 lần trên tập Test")
        mt = artifacts['metrics']
        if mt is not None:
            st.dataframe(mt, width='stretch', hide_index=True)
            best_row = mt.loc[mt['MAE (phút)'].idxmin()]
            base_row = mt[mt['Mô hình'].str.startswith('Baseline 1')].iloc[0]
            st.markdown(
                f"**Nhận xét:** {best_row['Mô hình']} có MAE thấp nhất ({best_row['MAE (phút)']:.2f} phút, "
                f"R² = {best_row['R2 Score']:.3f}), tốt hơn baseline hằng số ({base_row['MAE (phút)']:.2f} phút). "
                "R² vẫn thấp vì phân phối trễ có đuôi rất dài (một số chuyến trễ hàng trăm phút) mà các đặc trưng hiện có không giải thích được.")

        st.subheader("Phân loại trễ ≥ 30 phút - tập Test")
        if artifacts['cls_metrics'] is not None:
            st.dataframe(artifacts['cls_metrics'], width='stretch', hide_index=True)
            st.caption("Lớp 'trễ ≥ 30 phút' chỉ chiếm khoảng 14.5% tập Test nên Accuracy dễ gây hiểu nhầm "
                       "(luôn đoán 'không trễ' đã đạt ~85%); nên so sánh bằng F1, ROC-AUC và PR-AUC.")
        if artifacts['calibration'] is not None:
            st.markdown("**Hiệu chỉnh xác suất (Isotonic, fit trên Validation):**")
            st.dataframe(artifacts['calibration'], width='stretch', hide_index=True)

        show_fig("task_d6_feature_importance.png", "Top đặc trưng quan trọng của mô hình XGBoost đã chọn")

        st.divider()
        st.subheader("Thực nghiệm Overfitting (Train vs. Validation theo độ sâu cây)")
        col_ov_img, col_ov_txt = st.columns([3, 2])
        with col_ov_img:
            show_fig("task_d5_overfitting_analysis.png", "MAE Train vs. Validation theo max_depth")
        with col_ov_txt:
            oc = artifacts['overfit_curve']
            if oc is not None:
                best = oc.loc[oc['val_mae'].idxmin()]
                deepest = oc.iloc[-1]
                st.markdown(
                    f"- Validation MAE thấp nhất tại **max_depth = {int(best['max_depth'])}** ({best['val_mae']:.2f} phút).\n"
                    f"- Độ sâu nhỏ hơn: cả Train và Val đều cao (underfitting).\n"
                    f"- Tại max_depth = {int(deepest['max_depth'])}: Train MAE {deepest['train_mae']:.2f} nhưng Val MAE {deepest['val_mae']:.2f} "
                    "- khoảng cách Train/Val lớn là dấu hiệu overfitting.")
                st.dataframe(oc, hide_index=True, width='stretch')
