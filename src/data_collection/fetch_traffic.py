"""
Module: fetch_traffic.py
Task A1: Thu thập và quản lý luồng dữ liệu cảm biến giao thông (Traffic Sensor Stream).
Nguồn dữ liệu: Interstate I-94 Westbound (MN DoT / UCI Repository).
"""

import os
import pandas as pd

def extract_traffic_stream(raw_csv_path: str = "data/raw/Metro_Interstate_Traffic_Volume.csv",
                           output_path: str = "data/raw/traffic_sensor_stream.csv") -> pd.DataFrame:
    """
    Trích xuất nguồn dữ liệu cảm biến giao thông riêng biệt từ kho dữ liệu thô.
    Các trường cục bộ (Local Schema):
    - timestamp_local: Thời điểm ghi nhận từ cảm biến
    - traffic_volume: Lưu lượng xe qua trạm (xe/giờ)
    - holiday: Thông tin ghi nhận ngày lễ
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc tại {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)
    
    # Giữ các trường thuộc về hệ thống cảm biến giao thông
    traffic_stream = df[['date_time', 'traffic_volume', 'holiday']].copy()
    traffic_stream.rename(columns={'date_time': 'sensor_timestamp'}, inplace=True)
    traffic_stream['sensor_id'] = 'I94_WB_Station_101'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traffic_stream.to_csv(output_path, index=False)
    print(f"[Task A1] Đã trích xuất {len(traffic_stream)} bản ghi giao thông vào: {output_path}")
    return traffic_stream

if __name__ == "__main__":
    extract_traffic_stream()
