"""
Module: cleaner.py
Giai đoạn B: Làm sạch dữ liệu & Phân tích đơn biến.
- Task B1: Xử lý giá trị khuyết (Missing values) với quy tắc Median & Group-mean.
- Task B2: Xử lý ngoại lai (Outliers) bằng dải phân vị 1.5 * IQR.
- Task B3: Chuẩn hóa dữ liệu (Min-Max Scaling, Z-Score Standardization).
- Task B4: Thống kê mô tả đơn biến (Mean, Median, Mode, Variance, Std, Skewness, Kurtosis).
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple, Union

class DataCleaner:
    def __init__(self, data_path: str = "data/processed/integrated_flight_weather.csv"):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df['dep_time'] = pd.to_datetime(self.df['dep_time'])

    def check_missing_values(self) -> pd.DataFrame:
        """Thống kê tỷ lệ khuyết dữ liệu (Task B1)."""
        null_counts = self.df.isnull().sum()
        null_ratios = (null_counts / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': null_counts,
            'Missing_Ratio_%': null_ratios.round(2)
        })
        return missing_df

    def handle_missing_values(self) -> pd.DataFrame:
        """
        Xử lý giá trị khuyết theo các chiến lược phù hợp từng thuộc tính:
        - Nhiệt độ, gió, mưa: Điền median theo sân bay.
        - Số lượng hành khách: Điền trung vị.
        """
        df = self.df.copy()
        
        # Nếu có giá trị khuyết ở các biến thời tiết, điền median theo từng sân bay xuất phát
        if 'origin_temp_c' in df.columns:
            df['origin_temp_c'] = df.groupby('origin')['origin_temp_c'].transform(lambda x: x.fillna(x.median()))
            df['origin_rain_mm'] = df['origin_rain_mm'].fillna(0.0)
            df['origin_wind_kmh'] = df.groupby('origin')['origin_wind_kmh'].transform(lambda x: x.fillna(x.median()))
            df['origin_gusts_kmh'] = df.groupby('origin')['origin_gusts_kmh'].transform(lambda x: x.fillna(x.median()))
            df['origin_visibility_km'] = df['origin_visibility_km'].fillna(10.0)
            df['origin_weather'] = df['origin_weather'].fillna('Clear')
            
        if 'passenger_count' in df.columns:
            df['passenger_count'] = df['passenger_count'].fillna(df['passenger_count'].median())
            
        self.df = df
        return df

    def handle_outliers_iqr(self, col: str = 'delay_minutes', clip: bool = True) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Task B2: Phát hiện và xử lý ngoại lai bằng quy tắc 1.5 * IQR.
        - Đối với số phút trễ: Các chuyến bay trễ cực đoan (> 300 phút = 5 tiếng do sự cố kỹ thuật hiếm gặp)
          được cắt tỉa hoặc nén về biên trên (Winsorization/Clipping) để không làm lệch mô hình hồi quy.
        """
        df = self.df.copy()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        
        if clip:
            # Nén ngoại lai về ngưỡng biên Tukey (Winsorization) nhằm bảo toàn số lượng mẫu
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"[Task B2] Đã xử lý ngoại lai theo 1.5xIQR cho '{col}': Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
            print(f"Biên giới hạn: [{lower_bound:.2f}, {upper_bound:.2f}], đã nén {outliers_count} điểm ngoại lai.")
        else:
            # Lọc bỏ hoàn toàn các bản ghi ngoại lai
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            print(f"[Task B2] Đã loại bỏ {outliers_count} bản ghi ngoại lai ngoài biên [{lower_bound:.2f}, {upper_bound:.2f}].")
            
        self.df = df.reset_index(drop=True)
        return self.df, {
            'q1': q1, 'q3': q3, 'iqr': iqr,
            'lower_bound': lower_bound, 'upper_bound': upper_bound,
            'outliers_count': outliers_count
        }

    def compute_univariate_statistics(self, column: Union[str, pd.Series] = 'delay_minutes') -> Dict[str, Any]:
        """
        Task B4: Thống kê đơn biến:
        - Độ đo vị trí: Mean, Median, Mode
        - Độ đo quy mô: Variance, Standard Deviation
        - Hình dạng phân bố: Skewness (độ lệch) & Kurtosis (độ nhọn)
        Hỗ trợ cả tham số dạng chuỗi (tên cột) hoặc đối tượng pd.Series.
        """
        if isinstance(column, pd.Series):
            data = column.dropna()
            col_name = column.name or 'Target_Variable'
        else:
            col_name = str(column)
            data = self.df[col_name].dropna()
        
        mean_val = float(data.mean())
        median_val = float(data.median())
        mode_series = data.mode()
        mode_val = float(mode_series.iloc[0]) if not mode_series.empty else mean_val
        
        var_val = float(data.var())
        std_val = float(data.std())
        skew_val = float(stats.skew(data))
        kurt_val = float(stats.kurtosis(data))
        
        results = {
            'Variable': col_name,
            'Count': len(data),
            'Mean': round(mean_val, 2),
            'Median': round(median_val, 2),
            'Mode': round(mode_val, 2),
            'Variance': round(var_val, 2),
            'Std_Dev': round(std_val, 2),
            'Skewness': round(skew_val, 4),
            'Kurtosis': round(kurt_val, 4)
        }
        return results

    def save_cleaned_data(self, output_path: str = "data/processed/cleaned_flight_data.csv") -> pd.DataFrame:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"[Cleaner] Đã lưu dữ liệu làm sạch vào: {output_path} ({len(self.df)} dòng)")
        return self.df

    def clean_and_save(self, output_path: str = "data/processed/cleaned_flight_data.csv") -> pd.DataFrame:
        """Hàm tích hợp thực hiện tuần tự làm sạch và lưu dữ liệu (Dành cho Notebook)."""
        self.handle_missing_values()
        self.handle_outliers_iqr('delay_minutes', clip=True)
        return self.save_cleaned_data(output_path)

if __name__ == "__main__":
    cleaner = DataCleaner()
    print("Kiểm tra khuyết dữ liệu:\n", cleaner.check_missing_values())
    cleaner.handle_missing_values()
    cleaner.handle_outliers_iqr('delay_minutes')
    stats_delay = cleaner.compute_univariate_statistics('delay_minutes')
    print("Thống kê đơn biến thời gian trễ chuyến:\n", stats_delay)
    cleaner.save_cleaned_data()
