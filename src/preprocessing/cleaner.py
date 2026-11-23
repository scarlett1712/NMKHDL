"""
Module: cleaner.py
Giai đoạn B: Làm sạch dữ liệu & Phân tích đơn biến (Data Cleaning & Univariate Analysis).
- Task B1: Phát hiện và xử lý giá trị khuyết (Missing values) với Group-median.
- Task B2: Phát hiện và xử lý ngoại lai (Outliers) bằng dải phân vị 1.5 * IQR trên tập Train.
- Task B3: Chuẩn hóa dữ liệu đầy đủ: Min-Max Scaling & Z-Score Standardization.
- Task B4: Thống kê mô tả đơn biến biến mục tiêu delay_minutes (Mean, Median, Mode, Variance, Std, Skewness, Kurtosis).
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.splits import sort_chronologically, train_mask

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

    WEATHER_NUMERIC_COLS = ['origin_temp_c', 'origin_wind_kmh', 'origin_gusts_kmh',
                            'origin_rain_mm', 'origin_visibility_km', 'humid', 'pressure']

    def handle_missing_values(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Xử lý giá trị khuyết (Task B1) - CHỐNG RÒ RỈ: mọi median chỉ tính trên tập Train.
        - Cột khí tượng: điền theo median (sân bay, tháng) của Train; nếu tháng đó không có trong Train
          (Test rơi vào tháng 11-12), dùng median theo sân bay của Train; cuối cùng là median toàn Train.
        - Lượng mưa khuyết = 0 (trạm không ghi nhận mưa).
        - air_time (thời gian bay THỰC TẾ, chỉ biết sau khi hạ cánh) bị loại khỏi dữ liệu vì gây rò rỉ
          thông tin tương lai khi dự báo trễ cất cánh.
        """
        df = sort_chronologically(self.df)
        is_train = train_mask(df)
        train = df[is_train]

        fill_stats = {'by_origin_month': {}, 'by_origin': {}, 'global': {}}
        for col in self.WEATHER_NUMERIC_COLS:
            if col not in df.columns:
                continue
            if col == 'origin_rain_mm':
                df[col] = df[col].fillna(0.0)
                continue
            by_om = train.groupby(['origin', 'month'])[col].median()
            by_o = train.groupby('origin')[col].median()
            glob = float(train[col].median())
            fill_stats['by_origin_month'][col] = by_om.to_dict()
            fill_stats['by_origin'][col] = by_o.to_dict()
            fill_stats['global'][col] = glob

            na = df[col].isna()
            if na.any():
                keys = list(zip(df.loc[na, 'origin'], df.loc[na, 'month']))
                filled = pd.Series([by_om.get(k, np.nan) for k in keys], index=df.index[na])
                filled = filled.fillna(df.loc[na, 'origin'].map(by_o)).fillna(glob)
                df.loc[na, col] = filled

        if 'origin_weather' in df.columns:
            df['origin_weather'] = df['origin_weather'].fillna('Quang đãng')

        df = df.drop(columns=['air_time'], errors='ignore')

        self.df = df
        self.fill_stats = fill_stats
        return df, fill_stats

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
