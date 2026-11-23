"""
Module: cleaner.py
Giai đoạn B: Làm sạch dữ liệu & Phân tích đơn biến (Data Cleaning & Univariate Analysis).
- Task B1: Phát hiện và xử lý giá trị khuyết (Missing values) với Group-median.
- Task B2: Phát hiện và xử lý ngoại lai (Outliers) bằng dải phân vị 1.5 * IQR trên tập Train.
- Task B3: Chuẩn hóa dữ liệu đầy đủ: Min-Max Scaling & Z-Score Standardization.
- Task B4: Thống kê mô tả đơn biến biến mục tiêu delay_minutes (Mean, Median, Mode, Variance, Std, Skewness, Kurtosis).
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
        self.df['dep_timestamp'] = pd.to_datetime(self.df['dep_timestamp'])

    def check_missing_values(self) -> pd.DataFrame:
        """Thống kê tỷ lệ khuyết dữ liệu (Task B1)."""
        null_counts = self.df.isnull().sum()
        null_ratios = (null_counts / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': null_counts,
            'Missing_Ratio_%': null_ratios.round(2)
        })
        return missing_df

    def handle_missing_values(self, train_medians: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Xử lý giá trị khuyết (Task B1) trên dữ liệu quan trắc thật:
        - Các giờ khuyết trạm khí tượng được điền theo median của từng sân bay và từng tháng.
        - air_time khuyết được điền theo median của từng chặng bay (origin -> dest).
        """
        df = self.df.copy()
        if train_medians is None:
            train_medians = {
                'origin_temp_c': float(df['origin_temp_c'].median() if pd.notna(df['origin_temp_c'].median()) else 15.0),
                'origin_wind_kmh': float(df['origin_wind_kmh'].median() if pd.notna(df['origin_wind_kmh'].median()) else 15.0),
                'origin_gusts_kmh': float(df['origin_gusts_kmh'].median() if pd.notna(df['origin_gusts_kmh'].median()) else 20.0),
                'origin_rain_mm': 0.0,
                'origin_visibility_km': 16.0,
                'humid': float(df['humid'].median() if 'humid' in df.columns and pd.notna(df['humid'].median()) else 60.0),
                'pressure': float(df['pressure'].median() if 'pressure' in df.columns and pd.notna(df['pressure'].median()) else 1015.0),
                'air_time': float(df['air_time'].median() if pd.notna(df['air_time'].median()) else 150.0)
            }

        # Điền các cột khí tượng
        for col in ['origin_temp_c', 'origin_wind_kmh', 'origin_gusts_kmh', 'humid', 'pressure']:
            if col in df.columns:
                df[col] = df.groupby(['origin', 'month'])[col].transform(lambda x: x.fillna(x.median()))
                df[col] = df[col].fillna(train_medians.get(col, 15.0))

        df['origin_rain_mm'] = df['origin_rain_mm'].fillna(0.0)
        df['origin_visibility_km'] = df['origin_visibility_km'].fillna(train_medians['origin_visibility_km'])
        df['origin_weather'] = df['origin_weather'].fillna('Quang đãng')

        if 'air_time' in df.columns:
            df['air_time'] = df.groupby(['origin', 'dest'])['air_time'].transform(lambda x: x.fillna(x.median()))
            df['air_time'] = df['air_time'].fillna(train_medians['air_time'])

        self.df = df
        return df, train_medians

    @staticmethod
    def compute_iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> Dict[str, float]:
        """Task B2: Tính biên Tukey 1.5 * IQR CHỈ TRÊN TẬP HỌC (Train)."""
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        return {
            'q1': q1, 'q3': q3, 'iqr': iqr,
            'lower_bound': lower_bound, 'upper_bound': upper_bound
        }

    @staticmethod
    def min_max_scale(series: pd.Series, feature_range: Tuple[float, float] = (0.0, 1.0)) -> pd.Series:
        """Task B3: Chuẩn hóa Min-Max Scaling."""
        s_min = series.min()
        s_max = series.max()
        if s_max == s_min:
            return pd.Series(0.0, index=series.index)
        scaled = (series - s_min) / (s_max - s_min)
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]

    @staticmethod
    def z_score_standardize(series: pd.Series) -> pd.Series:
        """Task B3: Chuẩn hóa Z-Score Standardization."""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std

    def compute_univariate_statistics(self, column: Union[str, pd.Series] = 'delay_minutes') -> Dict[str, Any]:
        """Task B4: Thống kê mô tả đơn biến toàn diện."""
        if isinstance(column, pd.Series):
            data = column.dropna()
            col_name = column.name or 'Target_Variable'
        else:
            col_name = str(column)
            data = self.df[col_name].dropna()

        mean_val = float(data.mean())
        median_val = float(data.median())
        mode_res = stats.mode(data, keepdims=True)
        mode_val = float(mode_res.mode[0]) if len(mode_res.mode) > 0 else mean_val
        variance_val = float(data.var())
        std_val = float(data.std())
        skew_val = float(data.skew())
        kurt_val = float(data.kurtosis())

        stats_summary = {
            'Thuộc tính': col_name,
            'Số lượng mẫu (Count)': len(data),
            'Trung bình (Mean)': round(mean_val, 2),
            'Trung vị (Median)': round(median_val, 2),
            'Yếu vị (Mode)': round(mode_val, 2),
            'Phương sai (Variance)': round(variance_val, 2),
            'Độ lệch chuẩn (Std Dev)': round(std_val, 2),
            'Độ bất đối xứng (Skewness)': round(skew_val, 4),
            'Độ nhọn (Kurtosis)': round(kurt_val, 4),
            'Trễ nhỏ nhất (Min)': float(data.min()),
            'Trễ lớn nhất (Max)': float(data.max())
        }
        return stats_summary

    def run_cleaning_pipeline(self, output_path: str = "data/processed/cleaned_flight_data.csv") -> pd.DataFrame:
        print("[Task B] Bắt đầu pipeline làm sạch và thống kê dữ liệu thật...")
        self.handle_missing_values()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"[Task B - Hoàn thành] Đã lưu dữ liệu làm sạch tại: {output_path}")

        stats_delay = self.compute_univariate_statistics('delay_minutes')
        print("Thống kê đơn biến thời gian trễ thực tế (delay_minutes):")
        for k, v in stats_delay.items():
            print(f"  - {k}: {v}")
        return self.df

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run_cleaning_pipeline()
