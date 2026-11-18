"""
Module: train_evaluate.py
Giai đoạn D: Huấn luyện, Regularization & So sánh mô hình.
- Task D1: Phân chia tập dữ liệu Time-series Train/Val/Test (Tránh Data Leakage).
- Task D2: Xây dựng Baseline Model (Naïve Persistence / Network Lag-1).
- Task D3: Huấn luyện Mô hình tuyến tính: Linear Regression & Ridge Regression (L2 Regularization, 24).
- Task D4: Huấn luyện Mô hình phi tuyến: Random Forest & XGBoost.
- Task D5: Phân tích thực nghiệm Overfitting (Train Error vs Test Error theo Tree Depth).
- Task D6: Đo lường & So sánh hiệu năng toàn diện (MAE, RMSE, R2, Accuracy, F1-Score); Trích xuất Feature Importance.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FIGURE_DIR = "docs/figures"
MODELS_DIR = "models_saved"

def prepare_data(data_path: str = "data/processed/featured_flight_dataset.csv"):
    df = pd.read_csv(data_path)
    df['dep_time'] = pd.to_datetime(df['dep_time'])
    df = df.sort_values(by='dep_time').reset_index(drop=True)

    # Chọn lọc các đặc trưng đầu vào (Features)
    base_feature_cols = [
        'hour', 'dayofweek', 'month', 'is_weekend', 'is_peak_hours',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'network_delay_lag_1h', 'network_delay_rolling_3h',
        'passenger_count', 'scheduled_duration',
        'origin_temp_c', 'origin_rain_mm', 'origin_wind_kmh', 'origin_gusts_kmh', 'origin_visibility_km',
        'is_rain', 'is_high_wind', 'is_low_visibility', 'flight_load_cluster'
    ]
    # Bổ sung các cột dummy One-Hot cho Hãng, Tàu bay, Sân bay
    dummy_cols = [c for c in df.columns if c.startswith(('airline_', 'origin_', 'dest_', 'aircraft_')) and c not in [
        'origin', 'destination', 'origin_temp_c', 'origin_rain_mm', 'origin_wind_kmh', 'origin_gusts_kmh', 'origin_visibility_km', 'origin_weather'
    ]]
    feature_cols = base_feature_cols + dummy_cols

    target_col = 'delay_minutes'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Task D1: Phân chia tuần tự theo thời gian (Time-series split, không shuffle)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # Chuẩn hóa StandardScaler: CHỈ fit trên Train, rồi transform Val và Test (Tránh Data Leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            X_train_scaled, X_val_scaled, X_test_scaled, scaler, feature_cols, df)

def evaluate_predictions(y_true, y_pred, model_name: str) -> Dict[str, Any]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Đánh giá tác vụ Phân loại nguy cơ trễ chuyến (Ngưỡng >= 30 phút)
    y_true_cls = (y_true >= 30).astype(int)
    y_pred_cls = (y_pred >= 30).astype(int)
    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    
    return {
        'Mô hình': model_name,
        'MAE (phút)': round(mae, 2),
        'RMSE (phút)': round(rmse, 2),
        'R2 Score': round(r2, 4),
        'Accuracy (≥30p)': round(acc * 100, 1),
        'F1-Score': round(f1, 4)
    }

def experiment_overfitting(X_train, y_train, X_test, y_test, save_dir: str = FIGURE_DIR):
    """
    Task D5: Phân tích thực nghiệm Overfitting theo độ sâu cây (Tree Depth).
    Minh họa trực quan vùng Underfitting, Điểm cân bằng tối ưu và Vùng Overfitting.
    """
    os.makedirs(save_dir, exist_ok=True)
    depths = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
    train_errors = []
    test_errors = []

    print("[Task D5] Đang chạy thực nghiệm Overfitting theo chiều sâu mô hình cây...")
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        
        pred_train = dt.predict(X_train)
        pred_test = dt.predict(X_test)
        
        train_errors.append(mean_absolute_error(y_train, pred_train))
        test_errors.append(mean_absolute_error(y_test, pred_test))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depths, train_errors, 'o-', color='#2ca02c', linewidth=2, label='Training Error (Sai số tập học)')
    ax.plot(depths, test_errors, 's-', color='#d62728', linewidth=2, label='Test Error (Sai số tập kiểm tra)')

    best_idx = int(np.argmin(test_errors))
    best_depth = depths[best_idx]
    best_test_err = test_errors[best_idx]

    ax.axvline(x=best_depth, color='blue', linestyle='--', alpha=0.7, label=f'Điểm cân bằng tối ưu (max_depth = {best_depth})')
    ax.scatter([best_depth], [best_test_err], color='blue', s=120, zorder=5)

    # Đánh dấu các vùng lý thuyết
    ax.annotate('Vùng Underfitting\n(Mô hình quá đơn giản)', xy=(2, test_errors[1]), xytext=(2.2, test_errors[1] + 3.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    ax.annotate('Vùng Overfitting\n(Học vẹt / Quá khớp)', xy=(20, test_errors[-2]), xytext=(15, test_errors[-2] + 3.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    ax.set_title("Task D5: Thực nghiệm Overfitting (Underfitting vs. Overfitting)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Độ phức tạp mô hình (Độ sâu tối đa cây max_depth)", fontsize=11)
    ax.set_ylabel("Sai số tuyệt đối trung bình MAE (phút)", fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_d5_overfitting_analysis.png"), dpi=300)
    plt.close(fig)
    print(f"[Task D5] Biểu đồ Overfitting đã được lưu tại: {os.path.join(save_dir, 'task_d5_overfitting_analysis.png')}")

def train_and_compare_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("[Task D] Bắt đầu quá trình huấn luyện và so sánh các mô hình học máy...")
    (X_train, y_train, X_val, y_val, X_test, y_test,
     X_train_scaled, X_val_scaled, X_test_scaled, scaler, feature_cols, df) = prepare_data()

    results = []

    # 1. Task D2: Baseline Model (Naïve Persistence / Network Lag-1)
    y_pred_baseline = X_test['network_delay_lag_1h'].values
    results.append(evaluate_predictions(y_test, y_pred_baseline, "Baseline (Naïve Persistence)"))

    # 2. Task D3: Hồi quy tuyến tính chuẩn (Linear Regression)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results.append(evaluate_predictions(y_test, y_pred_lr, "Linear Regression"))

    # 3. Task D3: Tuyến tính có Regularization (Ridge L2 Penalty, 24)
    ridge = Ridge(alpha=50.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    results.append(evaluate_predictions(y_test, y_pred_ridge, "Ridge Regression (L2 Regularization)"))

    # 4. Task D4: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate_predictions(y_test, y_pred_rf, "Random Forest Regressor"))

    # 5. Task D4: XGBoost Regressor (Gradient Boosting)
    xgb = XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results.append(evaluate_predictions(y_test, y_pred_xgb, "XGBoost Regressor (Tối ưu)"))

    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("BẢNG TỔNG HỢP HIỆU NĂNG SO SÁNH CÁC MÔ HÌNH (TASK D6):")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70 + "\n")
    results_df.to_csv(os.path.join(FIGURE_DIR, "model_comparison_metrics.csv"), index=False)

    # 6. Task D5: Chạy thực nghiệm phân tích Overfitting
    experiment_overfitting(X_train, y_train, X_test, y_test)

    # 7. Trích xuất Feature Importance từ XGBoost
    importances = xgb.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(12)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.sort_values().plot(kind='barh', color='#1f77b4', ax=ax)
    ax.set_title("Task D6: Top Đặc Trưng Quyết Định Trễ Chuyến (Feature Importance - XGBoost)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mức độ quan trọng tương đối", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "task_d6_feature_importance.png"), dpi=300)
    plt.close(fig)

    # 8. Đóng gói lưu lại mô hình tốt nhất và Scaler
    joblib.dump(xgb, os.path.join(MODELS_DIR, "best_model_xgb.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "model_rf.pkl"))
    joblib.dump(ridge, os.path.join(MODELS_DIR, "model_ridge.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    print(f"[Task D - Hoàn thành] Đã lưu các mô hình và scaler vào thư mục: {MODELS_DIR}")
    return results_df

if __name__ == "__main__":
    train_and_compare_models()
