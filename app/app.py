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

# ----------------- THÔNG TIN SÂN BAY & ĐƯỜNG BAY NỘI ĐỊA -----------------
AIRPORTS = {
    'HAN': {'name': 'Nội Bài (Hà Nội)', 'lat': 21.2212, 'lon': 105.8072, 'capacity': '90 chuyến/giờ'},
    'SGN': {'name': 'Tân Sơn Nhất (TP.HCM)', 'lat': 10.8188, 'lon': 106.6518, 'capacity': '100 chuyến/giờ (Điểm nghẽn cao)'},
    'DAD': {'name': 'Đà Nẵng', 'lat': 16.0439, 'lon': 108.1994, 'capacity': '60 chuyến/giờ'},
    'CXR': {'name': 'Cam Ranh (Khánh Hòa)', 'lat': 11.9981, 'lon': 109.2194, 'capacity': '45 chuyến/giờ'},
    'PQC': {'name': 'Phú Quốc', 'lat': 10.1698, 'lon': 103.9931, 'capacity': '40 chuyến/giờ'}
}

AIRLINES_INFO = {
    'Vietnam Airlines': {'code': 'VN', 'otp': '88.5%', 'color': '#00558f', 'fleet': 'A321, A350, B787'},
    'Vietjet Air': {'code': 'VJ', 'otp': '74.2%', 'color': '#dc2626', 'fleet': 'A320, A321, A330'},
    'Bamboo Airways': {'code': 'QH', 'otp': '90.1%', 'color': '#16a34a', 'fleet': 'A320, A321, E190'},
    'Vietravel Airlines': {'code': 'VU', 'otp': '82.0%', 'color': '#f59e0b', 'fleet': 'A321'}
}

ROUTES_LIST = [
    ("HAN", "SGN", "Hà Nội (HAN) ✈️ TP.HCM (SGN)"),
    ("SGN", "HAN", "TP.HCM (SGN) ✈️ Hà Nội (HAN)"),
    ("HAN", "DAD", "Hà Nội (HAN) ✈️ Đà Nẵng (DAD)"),
    ("DAD", "HAN", "Đà Nẵng (DAD) ✈️ Hà Nội (HAN)"),
    ("SGN", "DAD", "TP.HCM (SGN) ✈️ Đà Nẵng (DAD)"),
    ("DAD", "SGN", "Đà Nẵng (DAD) ✈️ TP.HCM (SGN)"),
    ("HAN", "CXR", "Hà Nội (HAN) ✈️ Cam Ranh (CXR)"),
    ("SGN", "PQC", "TP.HCM (SGN) ✈️ Phú Quốc (PQC)")
]

# ----------------- 1. KHỞI TẠO SESSION STATE -----------------
if 'airline_key' not in st.session_state:
    st.session_state['airline_key'] = 'Vietnam Airlines'
if 'route_key' not in st.session_state:
    st.session_state['route_key'] = ROUTES_LIST[0] # HAN -> SGN
if 'date_key' not in st.session_state:
    st.session_state['date_key'] = datetime.now().date()
if 'hour_key' not in st.session_state:
    st.session_state['hour_key'] = 18
if 'min_key' not in st.session_state:
    st.session_state['min_key'] = 30
if 'aircraft_key' not in st.session_state:
    st.session_state['aircraft_key'] = 'A321'
if 'flight_code_key' not in st.session_state:
    st.session_state['flight_code_key'] = 'VN218'

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
        
    metrics_path = os.path.join(base_dir, "docs", "figures", "model_comparison_metrics.csv")
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None

    pricing_path = os.path.join(base_dir, "data", "processed", "airfare_pricing_dataset.csv")
    pricing_df = pd.read_csv(pricing_path) if os.path.exists(pricing_path) else None

    return {
        'xgb': model_xgb,
        'rf': model_rf,
        'ridge': model_ridge,
        'scaler': scaler,
        'features': feature_cols,
        'metrics': metrics_df,
        'pricing': pricing_df,
        'base_dir': base_dir
    }

artifacts = load_models_and_artifacts()
base_dir = artifacts['base_dir']

# ----------------- 4. GỌI OPEN-METEO LIVE REST API -----------------
@st.cache_data(ttl=900)
def fetch_live_weather(airport_code: str):
    coord = AIRPORTS.get(airport_code, AIRPORTS['HAN'])
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={coord['lat']}&longitude={coord['lon']}"
        "&current=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,cloud_cover,weather_code"
        "&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m"
        "&timezone=Asia%2FBangkok"
    )
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

# ----------------- 5. SIDEBAR: THÔNG TIN CHUYẾN BAY -----------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Hust_logo.png/640px-Hust_logo.png", width=90)
st.sidebar.title("✈️ VIETNAM AIRWATCH")
st.sidebar.markdown("**Dự báo Nguy cơ Trễ chuyến & Săn Vé Giá rẻ**")
st.sidebar.caption("Đề tài KHDL: IT4930 / IT4142 - SoICT, HUST")
st.sidebar.divider()

st.sidebar.subheader("📋 Thông số Chuyến bay:")

airline_keys = list(AIRLINES_INFO.keys())

selected_airline = st.sidebar.selectbox(
    "1. Hãng hàng không:",
    airline_keys,
    key='airline_key'
)

flight_code_input = st.sidebar.text_input(
    "Mã số chuyến bay (tuỳ ý):",
    value="VN218" if selected_airline == "Vietnam Airlines" else ("VJ130" if selected_airline == "Vietjet Air" else ("QH152" if selected_airline == "Bamboo Airways" else "VU641")),
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

aircraft_list = ['A321', 'A320', 'A350', 'B787', 'A330']

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
        "☀️ Thời tiết tốt / Quang đãng (Mặc định)",
        "🌧️ Mưa vừa đến Mưa to (Mưa 25mm, Gió 22km/h)",
        "⛈️ Dông Bão Nhiệt Đới / Gió Giật Mạnh (Mưa 60mm, Gió giật 50km/h - Cực đoan)",
        "🌫️ Sương mù dày đặc / Trần mây thấp (Mùa đông miền Bắc)",
        "📡 Tự động lấy Thời gian thực từ Open-Meteo REST API"
    ],
    index=0
)

# Xử lý các thông số khí tượng dựa trên kịch bản hoặc Live API
if weather_mode == "☀️ Thời tiết tốt / Quang đãng (Mặc định)":
    origin_temp = 29.5
    origin_rain = 0.0
    origin_wind = 12.0
    origin_gusts = 15.0
    origin_weather_desc = "Trời quang đãng, tầm nhìn xa > 10 km, điều kiện bay lý tưởng"
    origin_weather_code = 0
elif weather_mode == "🌧️ Mưa vừa đến Mưa to (Mưa 25mm, Gió 22km/h)":
    origin_temp = 24.0
    origin_rain = 25.0
    origin_wind = 22.0
    origin_gusts = 32.0
    origin_weather_desc = "Mưa to diện rộng, trần mây thấp, bề mặt đường băng trơn ướt"
    origin_weather_code = 65
elif weather_mode == "⛈️ Dông Bão Nhiệt Đới / Gió Giật Mạnh (Mưa 60mm, Gió giật 50km/h - Cực đoan)":
    origin_temp = 22.5
    origin_rain = 60.0
    origin_wind = 35.0
    origin_gusts = 50.0
    origin_weather_desc = "Dông lốc nhiệt đới, mây đối lưu tích điện (CB), sấm sét nguy hiểm, gió giật trên 50 km/h"
    origin_weather_code = 95
elif weather_mode == "🌫️ Sương mù dày đặc / Trần mây thấp (Mùa đông miền Bắc)":
    origin_temp = 17.0
    origin_rain = 1.5
    origin_wind = 6.0
    origin_gusts = 8.0
    origin_weather_desc = "Sương mù bức xạ dày đặc, trần mây cực thấp, tầm nhìn đường băng (RVR) dưới 600m"
    origin_weather_code = 45
else: # Live API
    w_raw = fetch_live_weather(origin_code)
    if w_raw and 'current' in w_raw:
        cur_o = w_raw['current']
        origin_temp = float(cur_o.get('temperature_2m', 29.5))
        origin_rain = float(cur_o.get('precipitation', 0.0))
        origin_wind = float(cur_o.get('wind_speed_10m', 14.0))
        origin_gusts = float(cur_o.get('wind_gusts_10m', 18.0))
        origin_weather_code = int(cur_o.get('weather_code', 0))
        if origin_weather_code >= 95:
            origin_weather_desc = "Live API: Dông bão / sấm sét nguy hiểm"
        elif origin_weather_code >= 61:
            origin_weather_desc = "Live API: Mưa rào diện rộng"
        elif origin_weather_code >= 45:
            origin_weather_desc = "Live API: Sương mù / Tầm nhìn thấp"
        elif origin_weather_code >= 1:
            origin_weather_desc = "Live API: Có mây rải rác"
        else:
            origin_weather_desc = "Live API: Trời quang đãng"
    else:
        origin_temp, origin_rain, origin_wind, origin_gusts = 29.0, 0.0, 12.0, 15.0
        origin_weather_desc = "Live API (Dự phòng): Trời quang đãng"
        origin_weather_code = 0

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
    
    inp['hour'] = hour
    inp['dayofweek'] = dow
    inp['month'] = month
    inp['is_weekend'] = 1 if dow in [4, 6] else 0
    inp['is_peak_hours'] = 1 if 16 <= hour <= 20 else (1 if 7 <= hour <= 9 else 0)
    inp['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    inp['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    inp['day_sin'] = np.sin(2 * np.pi * dow / 7.0)
    inp['day_cos'] = np.cos(2 * np.pi * dow / 7.0)
    
    base_net_delay = 12.0 if 6 <= hour <= 9 else (28.0 if 16 <= hour <= 20 else 18.0)
    if dow in [4, 6]:
        base_net_delay += 6.0
    inp['network_delay_lag_1h'] = base_net_delay
    inp['network_delay_rolling_3h'] = base_net_delay - 2.0
    inp['passenger_count'] = 180.0
    inp['scheduled_duration'] = 115.0 if 'SGN' in [origin, dest] and 'HAN' in [origin, dest] else 80.0
    
    inp['origin_temp_c'] = temp_c
    inp['origin_rain_mm'] = rain_mm
    inp['origin_wind_kmh'] = wind_kmh
    inp['origin_gusts_kmh'] = gusts_kmh
    inp['origin_visibility_km'] = 10.0 if rain_mm == 0 else (2.0 if rain_mm > 20 else 5.0)
    inp['is_rain'] = 1 if rain_mm > 0.1 else 0
    inp['is_high_wind'] = 1 if gusts_kmh > 25.0 else 0
    inp['is_low_visibility'] = 1 if (rain_mm > 15.0 or origin_weather_code in [45, 48]) else 0
    inp['flight_load_cluster'] = 1 if 16 <= hour <= 20 else 0
    
    for key in [f"airline_{airline}", f"origin_{origin}", f"dest_{dest}", f"aircraft_{aircraft}"]:
        if key in inp:
            inp[key] = 1
            
    df_sample = pd.DataFrame([inp])[feature_cols]
    if "XGBoost" in selected_model_name:
        pred_delay = artifacts['xgb'].predict(df_sample)[0]
    elif "Random Forest" in selected_model_name:
        pred_delay = artifacts['rf'].predict(df_sample)[0]
    else:
        scaled = artifacts['scaler'].transform(df_sample)
        pred_delay = artifacts['ridge'].predict(scaled)[0]
        
    # Phụ phí trễ do điều kiện thời tiết dông bão / gió giật theo tiêu chuẩn Cục Hàng không (CAAV)
    if rain_mm > 30.0 or gusts_kmh > 40.0 or origin_weather_code >= 95:
        pred_delay += max(15.0, rain_mm * 0.35 + (gusts_kmh - 25.0) * 0.4)
        
    return float(max(2.0, pred_delay))

dow_val = target_date.weekday()
pred_minutes = predict_flight_delay(
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

delay_prob = 1.0 / (1.0 + np.exp(-(pred_minutes - 26.0) / 10.0))
delay_prob_pct = int(np.clip(delay_prob * 100, 5, 96))

# ----------------- 7. GIAO DIỆN CHÍNH 3 TAB -----------------
tab_delay, tab_pricing, tab_analytics = st.tabs([
    "✈️ DỰ BÁO NGUY CƠ TRỄ CHUYẾN BAY",
    "💰 SĂN VÉ MÁY BAY GIÁ RẺ",
    "📊 PHÂN TÍCH CHUYÊN SÂU & ĐÁNH GIÁ MÔ HÌNH"
])

# ================= TAB 1: DỰ BÁO TRỄ CHUYẾN BAY =================
with tab_delay:
    st.title("✈️ Dự báo Nguy cơ Trễ chuyến bay Nội địa Việt Nam")
    
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
            
        # Yếu tố Ngày trong tuần vs Cuối tuần (Khảo sát từ 42,000 chuyến bay)
        day_names_vn = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
        current_day_name = day_names_vn[dow_val]
        if dow_val in [4, 6]: # Thứ Sáu & Chủ Nhật
            reasons.append((f"Cuối tuần cao điểm ({current_day_name})", 30, f"Khảo sát từ 42,000 chuyến bay: {current_day_name} có mức trễ bình quân ~38.5p (tỷ lệ trễ ≥30p vọt lên 66.8%), do lượng khách đi du lịch/về quê tăng đột biến."))
        elif dow_val == 5:
            reasons.append(("Thứ Bảy di chuyển", 5, "Khảo sát thực nghiệm: Thứ Bảy có mức trễ bình quân ~30.3p, lưu lượng khách ổn định."))
        else:
            reasons.append((f"Ngày thường trong tuần ({current_day_name})", -15, f"Khảo sát thực nghiệm: {current_day_name} có mức trễ bình quân ~30.6p (thấp hơn ~8 phút so với Thứ Sáu và Chủ Nhật), quầy thủ tục thông thoáng."))

        if dest_code == 'SGN':
            reasons.append(("Điểm nghẽn Tân Sơn Nhất (SGN)", 30, "Sân bay Tân Sơn Nhất quá tải đường băng, máy bay đến thường phải bay vòng chờ hạ cánh."))
        elif origin_code == 'HAN':
            reasons.append(("Mật độ điều phối Nội Bài (HAN)", 15, "Mật độ cất cánh dày đặc vào các khung giờ vàng."))
            
        if selected_airline == 'Vietjet Air':
            reasons.append(("Đặc tính xoay vòng Vietjet Air", 25, "Thời gian quay đầu máy bay ngắn (30-40 phút) dễ làm lan truyền trễ."))
        elif selected_airline == 'Bamboo Airways':
            reasons.append(("Chỉ số đúng giờ Bamboo Airways", -15, "Bamboo Airways duy trì tỷ lệ đúng giờ bình quân cao trên 90%."))
        elif selected_airline == 'Vietnam Airlines':
            reasons.append(("Ưu tiên điều hành Vietnam Airlines", -10, "Đội tàu bay dự phòng lớn và vị trí slot cất cánh ưu tiên."))
            
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
        predict_flight_delay(selected_airline, origin_code, dest_code, selected_aircraft, h, dow_val, target_date.month, origin_temp, origin_rain, origin_wind, origin_gusts)
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

    # KHẢO SÁT THỰC NGHIỆM TẬP DỮ LIỆU MẪU: NGÀY TRONG TUẦN VS CUỐI TUẦN
    st.markdown("---")
    st.markdown(f"#### 📊 Khảo sát Thực nghiệm Dữ liệu Mẫu: Mức độ Trễ Ngày thường vs Cuối tuần")
    st.caption("Thống kê chi tiết từ tập dữ liệu 42,000 chuyến bay nội địa Việt Nam (Khám phá Dữ liệu Chuyên sâu - EDA):")
    
    col_d1, col_d2, col_d3, col_d4, col_d5, col_d6, col_d7 = st.columns(7)
    with col_d1:
        st.metric("Thứ Hai", "30.6p", "Trễ ≥30p: 50.8%")
    with col_d2:
        st.metric("Thứ Ba", "30.6p", "Trễ ≥30p: 50.7%")
    with col_d3:
        st.metric("Thứ Tư", "30.7p", "Trễ ≥30p: 51.3%")
    with col_d4:
        st.metric("Thứ Năm", "30.8p", "Trễ ≥30p: 50.8%")
    with col_d5:
        st.metric("🔥 Thứ Sáu", "38.7p", "Trễ ≥30p: 66.8% 🔴")
    with col_d6:
        st.metric("Thứ Bảy", "30.3p", "Trễ ≥30p: 49.6%")
    with col_d7:
        st.metric("🔥 Chủ Nhật", "38.4p", "Trễ ≥30p: 65.9% 🔴")

    st.info(f"""
    📌 **Khám phá Bản chất Dữ liệu (EDA Insight từ 42,000 chuyến bay mẫu):**  
    - **Ngày thường (Thứ 2 đến Thứ 5, Thứ 7):** Mức trễ bình quân ổn định ở ngưỡng **30.3 – 30.8 phút**, tỷ lệ trễ $\ge 30$p khoảng **50.5%**.  
    - **Cuối tuần cao điểm (Thứ 6 & Chủ Nhật):** Mức trễ bình quân tăng vọt lên **38.5 phút** (tăng thêm **+7.8 phút**, tương đương **+25.4%**), và tỷ lệ trễ $\ge 30$p nhảy vọt lên **66.4%** (cứ 3 chuyến bay thì có 2 chuyến bị trễ nặng).  
    - **Giải thích cơ chế vận hành:** Chiều tối Thứ Sáu là làn sóng hành khách từ Hà Nội/TP.HCM ồ ạt bay đi du lịch hoặc về quê, trong khi chiều Chủ Nhật là làn sóng quay trở lại thành phố, khiến slot cất/hạ cánh tại Nội Bài và Tân Sơn Nhất chạm ngưỡng quá tải cực đại!
    """)


# ================= TAB 2: SĂN VÉ MÁY BAY GIÁ RẺ =================
with tab_pricing:
    st.title("💰 Săn Vé Máy Bay Giá Rẻ & Dự Báo Đáy Giá Vé (Airfare Tracker)")
    st.markdown("""
    Phân hệ mô phỏng thuật toán **Định giá động (Dynamic Pricing)** của các hãng hàng không nội địa theo số ngày đặt trước, 
    giúp bạn xác định chính xác **Thời điểm vàng nên mua vé** để tiết kiệm chi phí tối đa.
    """)

    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        price_route = st.selectbox("Chọn chặng bay muốn săn vé:", ROUTES_LIST, format_func=lambda x: x[2], key='price_route_sel')
    with col_p2:
        intended_flight_date = st.date_input("Ngày bạn dự định bay:", datetime.now().date() + timedelta(days=25))

    p_orig, p_dest, p_label = price_route
    days_until_flight = max(0, (intended_flight_date - datetime.now().date()).days)

    st.info(f"📅 Từ hôm nay đến ngày bay còn: **{days_until_flight} ngày**. Dưới đây là phân tích đường cong giá vé:")

    if artifacts['pricing'] is not None:
        df_pr = artifacts['pricing']
        df_sub = df_pr[(df_pr['origin'] == p_orig) & (df_pr['destination'] == p_dest)]
        
        fig_p, ax_p = plt.subplots(figsize=(13, 5))
        for airline_p, grp in df_sub.groupby('airline'):
            grp_sort = grp.sort_values(by='days_before_departure', ascending=False)
            color_p = AIRLINES_INFO.get(airline_p, {}).get('color', '#333333')
            ax_p.plot(grp_sort['days_before_departure'], grp_sort['fare_vnd'] / 1000, marker='o', markersize=3, label=airline_p, color=color_p, linewidth=2.2)

        ax_p.axvspan(14, 28, color='#22c55e', alpha=0.18, label="VÙNG ĐÁY GIÁ VÉ (Nên mua trước 14 - 28 ngày)")
        ax_p.axvspan(0, 7, color='#ef4444', alpha=0.15, label="VÙNG GIÁ ĐỈNH GIỜ CHÓT (< 7 ngày)")

        if 0 <= days_until_flight <= 60:
            ax_p.axvline(days_until_flight, color='#dc2626', linestyle='--', linewidth=2.5, label=f"Hôm nay ({days_until_flight} ngày trước bay)")

        ax_p.set_title(f"Biến thiên Giá vé Máy bay {p_label} theo Ngày đặt trước", fontsize=13, fontweight='bold', pad=12)
        ax_p.set_xlabel("Số ngày đặt trước chuyến bay (Ngày)", fontsize=11)
        ax_p.set_ylabel("Giá vé cơ bản (Nghìn VNĐ)", fontsize=11)
        ax_p.invert_xaxis()
        ax_p.legend(loc='upper right')
        ax_p.grid(True, linestyle=':', alpha=0.6)
        fig_p.tight_layout()
        st.pyplot(fig_p)

    st.markdown("### 🎯 Khuyến Nghị Săn Vé Từ Hệ Thống:")
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.success("""
        **🟢 THỜI ĐIỂM MUA VÉ TỐT NHẤT:**  
        Nên mua vé trước từ **14 đến 25 ngày**. Đây là giai đoạn các hãng tung vé hạng phổ thông tiết kiệm (Promo) để đảm bảo hệ số lấp đầy ghế.
        """)
    with rec_col2:
        st.warning("""
        **🟡 CHỌN NGÀY TRONG TUẦN ĐỂ MUA:**  
        Mua vé vào **Thứ Ba hoặc Thứ Tư** thường rẻ hơn $8 - 15\%$ so với đặt vào tối Thứ Sáu hoặc Chủ Nhật (thời điểm nhiều người vào web săn vé làm tăng giá động).
        """)
    with rec_col3:
        st.error("""
        **🔴 TRÁNH MUA SÁT NGÀY (< 7 NGÀY):**  
        Giá vé cận ngày tăng phi mã từ **$50\% - 120\%$** do hãng nhận diện khách hàng bay gấp là khách công vụ có khả năng chi trả cao.
        """)


# ================= TAB 3: PHÂN TÍCH CHUYÊN SÂU & ĐÁNH GIÁ MÔ HÌNH =================
with tab_analytics:
    st.title("📊 Phân tích Chuyên sâu Dữ liệu & Đánh giá Mô hình Học máy")
    st.markdown("""
    Phân hệ trình bày toàn diện các kết quả thực nghiệm khoa học dữ liệu: Thống kê chuỗi thời gian, 
    ma trận tương quan khí tượng, bảng so sánh hiệu năng các mô hình và khảo sát hiện tượng Overfitting theo độ sâu cây.
    """)

    sub1, sub2, sub3, sub4 = st.tabs([
        "1. Chuỗi thời gian & Tự tương quan",
        "2. Tương tác Khí tượng & Giờ cao điểm",
        "3. So sánh Hiệu năng Mô hình",
        "4. Thực nghiệm Phân tích Overfitting"
    ])

    with sub1:
        st.subheader("Chuỗi Thời Gian & Tính Tự Tương Quan (Serial Correlation Analysis)")
        c_ts1, c_ts2 = st.columns(2)
        with c_ts1:
            img_c1 = os.path.join(base_dir, "docs", "figures", "task_c1_run_sequence.png")
            if os.path.exists(img_c1):
                st.image(img_c1, caption="Run Sequence Plot - Chuỗi thời gian trễ của các chuyến bay liên tiếp", use_container_width=True)
                st.caption("Khám phá: Mức độ trễ có tính chu kỳ dao động mạnh, các chuyến bay buổi tối thường xuyên chạm ngưỡng trễ trên 30 phút.")
        with c_ts2:
            img_c2 = os.path.join(base_dir, "docs", "figures", "task_c2_lag_plot.png")
            if os.path.exists(img_c2):
                st.image(img_c2, caption="Lag Plot (Mức trễ chuyến bay vs Mức trễ mạng bay giờ trước)", use_container_width=True)
                st.caption("Khám phá: Hệ số tự tương quan Pearson r đạt mức cao, chứng minh hiện tượng tắc nghẽn dây chuyền của mạng lưới hàng không.")

    with sub2:
        st.subheader("Tương Tác Khí Tượng & Khung Giờ Cao Điểm")
        c_hm1, c_hm2 = st.columns(2)
        with c_hm1:
            img_c3 = os.path.join(base_dir, "docs", "figures", "task_c3_heatmap_24x7.png")
            if os.path.exists(img_c3):
                st.image(img_c3, caption="Heatmap Ma Trận Mức Độ Trễ (24 Giờ x 7 Ngày)", use_container_width=True)
        with c_hm2:
            img_c4 = os.path.join(base_dir, "docs", "figures", "task_c4_correlation_heatmap.png")
            if os.path.exists(img_c4):
                st.image(img_c4, caption="Ma Trận Hệ Số Tương Quan Pearson", use_container_width=True)

    with sub3:
        st.subheader("Bảng Tổng Hợp So Sánh Hiệu Năng Các Mô Hình")
        if artifacts['metrics'] is not None:
            st.dataframe(artifacts['metrics'], use_container_width=True, hide_index=True)
        
        img_feat = os.path.join(base_dir, "docs", "figures", "task_d6_feature_importance.png")
        if os.path.exists(img_feat):
            st.image(img_feat, caption="Top đặc trưng quan trọng nhất trong mô hình XGBoost", use_container_width=True)
            st.markdown("""
            **Đánh giá:** Mô hình **XGBoost Regressor** đạt hiệu năng vượt trội với MAE chỉ **~9.98 phút** (giảm gần 35% sai số so với Baseline). 
            Độ chính xác phân loại các chuyến bay trễ $\\ge 30$ phút đạt trên **$80.4\%$** với điểm F1-Score đạt **$0.825$**.
            """)

    with sub4:
        st.subheader("Thực nghiệm Khảo sát Hiện tượng Quá khớp (Overfitting Analysis)")
        img_overfit = os.path.join(base_dir, "docs", "figures", "task_d5_overfitting_analysis.png")
        if os.path.exists(img_overfit):
            col_ov_img, col_ov_txt = st.columns([3, 2])
            with col_ov_img:
                st.image(img_overfit, caption="Đường cong sai số Training vs Test Error theo độ sâu mô hình cây", use_container_width=True)
            with col_ov_txt:
                st.markdown("""
                ##### 🔍 Ba vùng đặc trưng trong thực nghiệm:
                1. **Vùng Thiếu khớp (Underfitting, max_depth < 4):** Cả sai số Train và Test đều cao do mô hình quá đơn giản.
                2. **Điểm Dung lượng Tối ưu (Optimal, max_depth = 5 - 8):** Sai số Test đạt cực tiểu (MAE thấp nhất), đạt sự cân bằng Bias - Variance hoàn hảo.
                3. **Vùng Quá khớp (Overfitting, max_depth > 12):** Sai số Train tiếp tục giảm tiệm cận 0 trong khi sai số Test phân kỳ tăng trở lại.
                """)
