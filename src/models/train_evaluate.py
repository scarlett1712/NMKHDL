"""
Module: train_evaluate.py
Giai đoạn D: Huấn luyện, Regularization, Phân tích Overfitting & So sánh mô hình trên DỮ LIỆU THẬT 100%.
- Task D1: Phân chia tập dữ liệu Time-series Train (70%) / Val (15%) / Test (15%) - Tuyệt đối không rò rỉ dữ liệu.
  + StandardScaler chỉ fit trên Train.
  + IQR clipping chỉ thực hiện trên y_train; y_val và y_test giữ nguyên vẹn 100%.
- Task D2: Bổ sung 2 Baseline chuẩn mực:
  + Baseline 1: Majority Class Classifier (Zero-Rule).
  + Baseline 2: Naïve Persistence Model (Airport Lag-1h).
- Task D3: Mô hình tuyến tính & Regularization:
  + Linear Regression.
  + RidgeCV (L2) tự động chọn alpha tối ưu.
  + LassoCV (L1) tự động chọn lambda tối ưu.
- Task D4: Mô hình phi tuyến: Random Forest & XGBoost Regressor.
- Task D5: Phân tích thực nghiệm Overfitting: Đánh giá Train Error vs. VALIDATION Error.
- Task D6: Đo lường & So sánh hiệu năng toàn diện trên tập Test; Trích xuất Feature Importance thực tế.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

FIGURE_DIR = "docs/figures"
MODELS_DIR = "models_saved"

def prepare_data(data_path: str = "data/processed/featured_flight_dataset.csv"):
    df = pd.read_csv(data_path)
    df['dep_timestamp'] = pd.to_datetime(df['dep_timestamp'])
    df = df.sort_values(by='dep_timestamp').reset_index(drop=True)

    base_feature_cols = [
        'month', 'dayofweek', 'is_peak_day', 'is_peak_hours',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'network_delay_lag_1h', 'network_delay_rolling_3h',
        'distance', 'air_time',
        'origin_temp_c', 'origin_rain_mm', 'origin_wind_kmh', 'origin_gusts_kmh', 'origin_visibility_km', 'humid', 'pressure',
        'is_rain', 'is_high_wind', 'is_low_visibility', 'flight_load_cluster'
    ]
    dummy_cols = [c for c in df.columns if c.startswith(('carrier_', 'origin_', 'dest_')) and c not in [
        'origin', 'dest', 'carrier_clean', 'dest_clean', 'origin_temp_c', 'origin_rain_mm', 'origin_wind_kmh', 'origin_gusts_kmh', 'origin_visibility_km', 'origin_weather'
    ]]
    feature_cols = base_feature_cols + dummy_cols
    target_col = 'delay_minutes'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Task D1: Phân chia chuỗi thời gian tuần tự: Train (70%), Val (15%), Test (15%)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end].copy(), y.iloc[:train_end].copy()
    X_val, y_val = X.iloc[train_end:val_end].copy(), y.iloc[train_end:val_end].copy()
    X_test, y_test = X.iloc[val_end:].copy(), y.iloc[val_end:].copy()

    # Nén ngoại lai: CHỈ thực hiện trên y_train để ổn định gradient huấn luyện
    # y_val và y_test giữ nguyên vẹn 100% để đánh giá trung thực
    q1 = y_train.quantile(0.25)
    q3 = y_train.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    y_train_clipped = y_train.clip(lower=lower_bound, upper=upper_bound)

    # Chuẩn hóa StandardScaler: CHỈ fit trên Train, rồi transform cho Val và Test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (X_train, y_train_clipped, y_train, X_val, y_val, X_test, y_test,
            X_train_scaled, X_val_scaled, X_test_scaled, scaler, feature_cols, df)

def evaluate_predictions(y_true, y_pred, model_name: str) -> Dict[str, Any]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Đánh giá phân loại trễ chuyến (Ngưỡng >= 30 phút theo chuẩn FAA)
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

def experiment_overfitting(X_train, y_train, X_val, y_val, save_dir: str = FIGURE_DIR):
    """
    Task D5: Phân tích thực nghiệm Overfitting theo độ sâu cây trên tập VALIDATION.
    """
    os.makedirs(save_dir, exist_ok=True)
    depths = [1, 2, 4, 6, 8, 10, 12, 16, 20]
    train_errors = []
    val_errors = []

    print("[Task D5] Đang phân tích Overfitting theo độ sâu cây trên tập Validation...")
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        
        pred_train = dt.predict(X_train)
        pred_val = dt.predict(X_val)
        
        train_errors.append(mean_absolute_error(y_train, pred_train))
        val_errors.append(mean_absolute_error(y_val, pred_val))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depths, train_errors, 'o-', color='#2ca02c', linewidth=2, label='Training Error (Sai số tập học)')
    ax.plot(depths, val_errors, 's-', color='#d62728', linewidth=2, label='Validation Error (Sai số tập kiểm định)')

    best_idx = int(np.argmin(val_errors))
    best_depth = depths[best_idx]
    best_val_err = val_errors[best_idx]

    ax.axvline(x=best_depth, color='blue', linestyle='--', alpha=0.7, label=f'Điểm tối ưu (max_depth = {best_depth})')
    ax.scatter([best_depth], [best_val_err], color='blue', s=120, zorder=5)

    ax.annotate('Vùng Underfitting\n(Mô hình quá đơn giản)', xy=(2, val_errors[1]), xytext=(2.2, val_errors[1] + 2.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    ax.annotate('Vùng Overfitting\n(Quá khớp dữ liệu)', xy=(16, val_errors[-2]), xytext=(12, val_errors[-2] + 2.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    ax.set_title("Task D5: Thực nghiệm Overfitting trên Dữ liệu Thật (Train vs. Validation)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Độ phức tạp mô hình (Độ sâu tối đa max_depth)", fontsize=11)
    ax.set_ylabel("Sai số tuyệt đối trung bình MAE (phút)", fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_d5_overfitting_analysis.png"), dpi=300)
    plt.close(fig)
    print(f"[Task D5] Biểu đồ Overfitting đã được lưu tại: {os.path.join(save_dir, 'task_d5_overfitting_analysis.png')}")
    return best_depth

def train_and_compare_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("[Task D] Bắt đầu huấn luyện và đánh giá trên DỮ LIỆU THẬT 100%...")
    (X_train, y_train_clipped, y_train_raw, X_val, y_val, X_test, y_test,
     X_train_scaled, X_val_scaled, X_test_scaled, scaler, feature_cols, df) = prepare_data()

    results = []

    # 1. Baseline 1: Majority Class / Zero-Rule
    train_delay_ratio = float((y_train_raw >= 30).mean())
    majority_pred_val = 35.0 if train_delay_ratio >= 0.5 else 0.0
    y_pred_majority = np.full(len(y_test), majority_pred_val)
    results.append(evaluate_predictions(y_test, y_pred_majority, "Baseline 1 (Majority Class / Zero-Rule)"))

    # 2. Baseline 2: Naïve Persistence (Airport Lag-1h)
    y_pred_persistence = X_test['network_delay_lag_1h'].values
    results.append(evaluate_predictions(y_test, y_pred_persistence, "Baseline 2 (Naïve Persistence)"))

    # 3. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_clipped)
    y_pred_lr = lr.predict(X_test_scaled)
    results.append(evaluate_predictions(y_test, y_pred_lr, "Linear Regression"))

    # 4. Ridge Regression (L2) qua RidgeCV
    alphas_ridge = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
    ridge_cv = RidgeCV(alphas=alphas_ridge)
    ridge_cv.fit(X_train_scaled, y_train_clipped)
    best_alpha_ridge = float(ridge_cv.alpha_)
    y_pred_ridge = ridge_cv.predict(X_test_scaled)
    print(f"[Task D3] RidgeCV tối ưu alpha = {best_alpha_ridge}")
    results.append(evaluate_predictions(y_test, y_pred_ridge, f"Ridge Regression (L2, alpha={best_alpha_ridge})"))

    # 5. Lasso Regression (L1) qua LassoCV
    lasso_cv = LassoCV(alphas=[0.01, 0.05, 0.1, 0.5], max_iter=2000, random_state=42)
    lasso_cv.fit(X_train_scaled, y_train_clipped)
    best_alpha_lasso = float(lasso_cv.alpha_)
    y_pred_lasso = lasso_cv.predict(X_test_scaled)
    print(f"[Task D3] LassoCV tối ưu alpha = {best_alpha_lasso}")
    results.append(evaluate_predictions(y_test, y_pred_lasso, f"Lasso Regression (L1, alpha={best_alpha_lasso})"))

    # 6. Task D5: Khảo sát Overfitting trên tập Validation
    best_depth = experiment_overfitting(X_train, y_train_clipped, X_val, y_val)

    # 7. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=best_depth, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_clipped)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate_predictions(y_test, y_pred_rf, "Random Forest Regressor"))

    # 8. XGBoost Regressor (Tối ưu)
    xgb = XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, subsample=0.85, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train_clipped)
    y_pred_xgb = xgb.predict(X_test)
    results.append(evaluate_predictions(y_test, y_pred_xgb, "XGBoost Regressor (Tối ưu)"))

    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("BẢNG TỔNG HỢP HIỆU NĂNG THỰC TẾ SO SÁNH CÁC MÔ HÌNH TRÊN TẬP TEST (TASK D6):")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")
    results_df.to_csv(os.path.join(FIGURE_DIR, "model_comparison_metrics.csv"), index=False)

    # 9. Feature Importance thực tế từ XGBoost
    importances = xgb.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(12)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.sort_values().plot(kind='barh', color='#1f77b4', ax=ax)
    ax.set_title("Task D6: Top Đặc Trưng Quyết Định Trễ Chuyến Thực Tế (XGBoost)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mức độ quan trọng tương đối", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "task_d6_feature_importance.png"), dpi=300)
    plt.close(fig)

    # 10. Huấn luyện bộ phân loại nhị phân thực tế phục vụ Web App
    y_train_cls = (y_train_raw >= 30).astype(int)
    clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train_cls)

    # 11. Đóng gói mô hình và bộ chuẩn hóa
    joblib.dump(xgb, os.path.join(MODELS_DIR, "best_model_xgb.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "model_rf.pkl"))
    joblib.dump(ridge_cv, os.path.join(MODELS_DIR, "model_ridge.pkl"))
    joblib.dump(lasso_cv, os.path.join(MODELS_DIR, "model_lasso.pkl"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "delay_classifier.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    print(f"[Task D - Hoàn thành] Đã lưu các mô hình và scaler vào: {MODELS_DIR}")
    return results_df

if __name__ == "__main__":
    train_and_compare_models()
