"""
Module: splits.py
Quy ước phân chia dữ liệu chuỗi thời gian dùng chung cho toàn bộ pipeline.

Mọi bước "học" tham số từ dữ liệu (median điền khuyết, K-Means, top-k one-hot, biên IQR, StandardScaler)
chỉ được tính trên tập Train xác định tại đây, rồi mới áp dụng sang Validation và Test.
Như vậy cleaner.py, feature_engineering.py và train_evaluate.py dùng chung một ranh giới Train/Val/Test.
"""

import pandas as pd

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # Test = 1 - TRAIN_FRAC - VAL_FRAC


def sort_chronologically(df: pd.DataFrame, time_col: str = 'dep_timestamp') -> pd.DataFrame:
    """Sắp xếp ổn định theo thời gian cất cánh (flight_id là khóa phụ để thứ tự không đổi giữa các bước)."""
    keys = [time_col, 'flight_id'] if 'flight_id' in df.columns else [time_col]
    return df.sort_values(by=keys, kind='mergesort').reset_index(drop=True)


def split_bounds(n: int):
    """Trả về (train_end, val_end) theo chỉ số dòng của DataFrame đã sắp xếp theo thời gian."""
    return int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))


def split_labels(n: int) -> pd.Series:
    """Nhãn 'train' / 'val' / 'test' cho từng dòng của DataFrame đã sắp xếp theo thời gian."""
    train_end, val_end = split_bounds(n)
    labels = ['train'] * train_end + ['val'] * (val_end - train_end) + ['test'] * (n - val_end)
    return pd.Series(labels)


def train_mask(df: pd.DataFrame) -> pd.Series:
    """Mask boolean các dòng thuộc tập Train (df phải đã được sort_chronologically)."""
    train_end, _ = split_bounds(len(df))
    mask = pd.Series(False, index=df.index)
    mask.iloc[:train_end] = True
    return mask
