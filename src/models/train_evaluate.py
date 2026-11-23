"""
Module: train_evaluate.py
Giai đoạn D: Huấn luyện, Regularization, Phân tích Overfitting & So sánh mô hình trên DỮ LIỆU THẬT.
- Task D1: Phân chia Time-series Train (70%) / Val (15%) / Test (15%) theo quy ước chung src/preprocessing/splits.py.
  + StandardScaler chỉ fit trên Train.
  + Nén ngoại lai IQR (nếu được chọn) chỉ áp dụng cho y_train; y_val và y_test giữ nguyên.
- Task D2: Baseline:
  + Baseline 1: Hằng số = median của Train (cho hồi quy) / luôn đoán lớp đa số (cho phân loại).
  + Baseline 2: Naïve Persistence (trễ trung bình sân bay giờ trước).
- Task D3: Linear Regression, Ridge (L2), Lasso (L1) - hệ số phạt alpha chọn trên tập VALIDATION.
- Task D4: Random Forest & XGBoost - siêu tham số chọn trên tập VALIDATION (XGBoost dùng early stopping trên Val).
- Task D5: Phân tích Overfitting: Train Error vs. Validation Error theo độ sâu cây.
- Task D6: Đánh giá DUY NHẤT MỘT LẦN trên tập Test; Feature Importance.
- Task D7: Bài toán phân loại trễ >= 30 phút: XGBClassifier có cân bằng lớp, ngưỡng xác suất chọn trên Val,
  so sánh với baseline lớp đa số / persistence bằng Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.

Không dùng air_time làm đặc trưng: đó là thời gian bay thực tế, chỉ biết sau khi chuyến bay kết thúc.
"""

import os
import sys
import itertools
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, average_precision_score, brier_score_loss)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.splits import sort_chronologically, split_bounds

FIGURE_DIR = "docs/figures"
MODELS_DIR = "models_saved"
DELAY_THRESHOLD = 30  # phút - ngưỡng trễ nghiêm trọng

BASE_FEATURE_COLS = [
    'month', 'dayofweek', 'is_peak_day', 'is_peak_hours',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'network_delay_lag_1h', 'network_delay_rolling_3h',
    'distance', 'dest_hub_tier', 'dest_enplanements_m',
    'origin_temp_c', 'origin_rain_mm', 'origin_wind_kmh', 'origin_gusts_kmh', 'origin_visibility_km', 'humid', 'pressure',
    'is_rain', 'is_high_wind', 'is_low_visibility', 'flight_load_cluster'
]
NON_DUMMY_PREFIXED = {
    'origin', 'dest', 'carrier_clean', 'dest_clean', 'origin_temp_c', 'origin_rain_mm', 'origin_wind_kmh',
    'origin_gusts_kmh', 'origin_visibility_km', 'origin_weather', 'dest_hub_tier', 'dest_enplanements_m'
}


def prepare_data(data_path: str = "data/processed/featured_flight_dataset.csv"):
    df = pd.read_csv(data_path)
    df['dep_timestamp'] = pd.to_datetime(df['dep_timestamp'])
    df = sort_chronologically(df)

    dummy_cols = [c for c in df.columns if c.startswith(('carrier_', 'origin_', 'dest_')) and c not in NON_DUMMY_PREFIXED]
    feature_cols = [c for c in BASE_FEATURE_COLS if c in df.columns] + dummy_cols
    assert 'air_time' not in feature_cols, "air_time gây rò rỉ thông tin tương lai - không được dùng làm đặc trưng"

    X = df[feature_cols].astype(float)
    y = df['delay_minutes'].astype(float)

    train_end, val_end = split_bounds(len(df))
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # Biên IQR tính trên y_train; chỉ dùng để nén y_train (tùy chọn được chọn trên Val)
    q1, q3 = y_train.quantile(0.25), y_train.quantile(0.75)
    iqr = q3 - q1
    y_train_clipped = y_train.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    scaler = StandardScaler().fit(X_train)
    print(f"[Task D1] Train {len(X_train):,} ({df.dep_timestamp.iloc[0]:%d/%m} - {df.dep_timestamp.iloc[train_end - 1]:%d/%m}), "
          f"Val {len(X_val):,}, Test {len(X_test):,} (từ {df.dep_timestamp.iloc[val_end]:%d/%m}); "
          f"{len(feature_cols)} đặc trưng; biên IQR y_train = [{q1 - 1.5 * iqr:.1f}, {q3 + 1.5 * iqr:.1f}]")

    return {
        'X_train': X_train, 'y_train': y_train, 'y_train_clipped': y_train_clipped,
        'X_val': X_val, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test,
        'X_train_s': scaler.transform(X_train), 'X_val_s': scaler.transform(X_val), 'X_test_s': scaler.transform(X_test),
        'scaler': scaler, 'feature_cols': feature_cols, 'df': df,
    }


def evaluate_predictions(y_true, y_pred, model_name: str, config: str = "") -> Dict[str, Any]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_cls = (y_true >= DELAY_THRESHOLD).astype(int)
    y_pred_cls = (y_pred >= DELAY_THRESHOLD).astype(int)
    return {
        'Mô hình': model_name,
        'Cấu hình (chọn trên Val)': config,
        'MAE (phút)': round(mean_absolute_error(y_true, y_pred), 2),
        'RMSE (phút)': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        'R2 Score': round(r2_score(y_true, y_pred), 4),
        'Accuracy (≥30p)': round(accuracy_score(y_true_cls, y_pred_cls) * 100, 1),
        'F1-Score': round(f1_score(y_true_cls, y_pred_cls, zero_division=0), 4)
    }


def evaluate_classifier(y_true_cls, y_pred_cls, y_score, model_name: str, config: str = "") -> Dict[str, Any]:
    has_score = y_score is not None
    return {
        'Mô hình': model_name,
        'Cấu hình (chọn trên Val)': config,
        'Accuracy (%)': round(accuracy_score(y_true_cls, y_pred_cls) * 100, 1),
        'Precision': round(precision_score(y_true_cls, y_pred_cls, zero_division=0), 4),
        'Recall': round(recall_score(y_true_cls, y_pred_cls, zero_division=0), 4),
        'F1-Score': round(f1_score(y_true_cls, y_pred_cls, zero_division=0), 4),
        'ROC-AUC': round(roc_auc_score(y_true_cls, y_score), 4) if has_score else np.nan,
        'PR-AUC': round(average_precision_score(y_true_cls, y_score), 4) if has_score else np.nan,
    }


def experiment_overfitting(X_train, y_train, X_val, y_val, save_dir: str = FIGURE_DIR):
    """Task D5: Train Error vs. Validation Error theo độ sâu cây."""
    os.makedirs(save_dir, exist_ok=True)
    depths = [1, 2, 4, 6, 8, 10, 12, 16, 20]
    train_errors, val_errors = [], []

    print("[Task D5] Đang phân tích Overfitting theo độ sâu cây trên tập Validation...")
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
        train_errors.append(mean_absolute_error(y_train, dt.predict(X_train)))
        val_errors.append(mean_absolute_error(y_val, dt.predict(X_val)))

    best_idx = int(np.argmin(val_errors))
    best_depth, best_val_err = depths[best_idx], val_errors[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depths, train_errors, 'o-', color='#2ca02c', linewidth=2, label='Training Error (Sai số tập học)')
    ax.plot(depths, val_errors, 's-', color='#d62728', linewidth=2, label='Validation Error (Sai số tập kiểm định)')
    ax.axvline(x=best_depth, color='blue', linestyle='--', alpha=0.7, label=f'Điểm tối ưu (max_depth = {best_depth})')
    ax.scatter([best_depth], [best_val_err], color='blue', s=120, zorder=5)
    ax.set_title("Task D5: Thực nghiệm Overfitting trên Dữ liệu Thật (Train vs. Validation)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Độ phức tạp mô hình (Độ sâu tối đa max_depth)", fontsize=11)
    ax.set_ylabel("Sai số tuyệt đối trung bình MAE (phút)", fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_d5_overfitting_analysis.png"), dpi=300)
    plt.close(fig)
    print(f"[Task D5] max_depth tối ưu trên Val = {best_depth} (Val MAE = {best_val_err:.2f})")
    return best_depth, pd.DataFrame({'max_depth': depths, 'train_mae': train_errors, 'val_mae': val_errors})


def _target(d, clip: bool):
    return d['y_train_clipped'] if clip else d['y_train']


def tune_linear(d, model_cls, alphas, name, tuning_log):
    """Chọn alpha và tùy chọn nén y_train theo MAE trên Validation."""
    best = None
    for alpha, clip in itertools.product(alphas, [True, False]):
        kwargs = {'alpha': alpha} if alpha is not None else {}
        if model_cls is Lasso:
            kwargs['max_iter'] = 5000
        m = model_cls(**kwargs).fit(d['X_train_s'], _target(d, clip))
        mae = mean_absolute_error(d['y_val'], m.predict(d['X_val_s']))
        tuning_log.append({'model': name, 'params': f"alpha={alpha}, clip_y={clip}", 'val_mae': round(mae, 3)})
        if best is None or mae < best[0]:
            best = (mae, m, alpha, clip)
    mae, m, alpha, clip = best
    cfg = (f"alpha={alpha}, " if alpha is not None else "") + f"nén y_train={'có' if clip else 'không'}"
    print(f"[Task D3] {name}: {cfg} (Val MAE = {mae:.2f})")
    return m, cfg


def tune_random_forest(d, tuning_log):
    best = None
    for depth, leaf, clip in itertools.product([8, 12, 16], [5, 20], [True, False]):
        m = RandomForestRegressor(n_estimators=150, max_depth=depth, min_samples_leaf=leaf,
                                  max_features=0.5, random_state=42, n_jobs=-1).fit(d['X_train'], _target(d, clip))
        mae = mean_absolute_error(d['y_val'], m.predict(d['X_val']))
        tuning_log.append({'model': 'Random Forest', 'params': f"max_depth={depth}, min_samples_leaf={leaf}, clip_y={clip}",
                           'val_mae': round(mae, 3)})
        if best is None or mae < best[0]:
            best = (mae, m, depth, leaf, clip)
    mae, m, depth, leaf, clip = best
    cfg = f"max_depth={depth}, min_samples_leaf={leaf}, nén y_train={'có' if clip else 'không'}"
    print(f"[Task D4] Random Forest: {cfg} (Val MAE = {mae:.2f})")
    return m, cfg


def tune_xgboost(d, objective, tuning_log):
    """Grid nhỏ + early stopping trên Val (số cây tối ưu chọn tự động)."""
    best = None
    for depth, lr, clip in itertools.product([4, 6, 8], [0.03, 0.1], [True, False]):
        m = XGBRegressor(n_estimators=2000, max_depth=depth, learning_rate=lr, subsample=0.85, colsample_bytree=0.8,
                         objective=objective, eval_metric='mae', early_stopping_rounds=50, random_state=42, n_jobs=-1)
        m.fit(d['X_train'], _target(d, clip), eval_set=[(d['X_val'], d['y_val'])], verbose=False)
        mae = mean_absolute_error(d['y_val'], m.predict(d['X_val']))
        tuning_log.append({'model': f'XGBoost ({objective})', 'params': f"max_depth={depth}, lr={lr}, clip_y={clip}, n_trees={m.best_iteration + 1}",
                           'val_mae': round(mae, 3)})
        if best is None or mae < best[0]:
            best = (mae, m, depth, lr, clip)
    mae, m, depth, lr, clip = best
    cfg = f"max_depth={depth}, lr={lr}, số cây={m.best_iteration + 1}, nén y_train={'có' if clip else 'không'}"
    print(f"[Task D4] XGBoost ({objective}): {cfg} (Val MAE = {mae:.2f})")
    return m, cfg, mae


def train_classifier(d, tuning_log):
    """Task D7: XGBClassifier cân bằng lớp; chọn cấu hình theo PR-AUC trên Val, ngưỡng theo F1 trên Val."""
    y_tr = (d['y_train'] >= DELAY_THRESHOLD).astype(int)
    y_va = (d['y_val'] >= DELAY_THRESHOLD).astype(int)
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    best = None
    for depth, lr in itertools.product([4, 6], [0.03, 0.1]):
        clf = XGBClassifier(n_estimators=2000, max_depth=depth, learning_rate=lr, subsample=0.85, colsample_bytree=0.8,
                            scale_pos_weight=spw, eval_metric='aucpr', early_stopping_rounds=50, random_state=42, n_jobs=-1)
        clf.fit(d['X_train'], y_tr, eval_set=[(d['X_val'], y_va)], verbose=False)
        ap = average_precision_score(y_va, clf.predict_proba(d['X_val'])[:, 1])
        tuning_log.append({'model': 'XGBClassifier', 'params': f"max_depth={depth}, lr={lr}, n_trees={clf.best_iteration + 1}",
                           'val_pr_auc': round(ap, 4)})
        if best is None or ap > best[0]:
            best = (ap, clf, depth, lr)
    ap, clf, depth, lr = best

    # Ngưỡng xác suất tối ưu F1 trên Val
    p_val = clf.predict_proba(d['X_val'])[:, 1]
    thresholds = np.round(np.arange(0.10, 0.91, 0.02), 2)
    f1s = [f1_score(y_va, (p_val >= t).astype(int), zero_division=0) for t in thresholds]
    thr = float(thresholds[int(np.argmax(f1s))])
    cfg = f"max_depth={depth}, lr={lr}, số cây={clf.best_iteration + 1}, scale_pos_weight={spw:.2f}, ngưỡng={thr}"
    print(f"[Task D7] XGBClassifier: {cfg} (Val PR-AUC = {ap:.3f}, Val F1 = {max(f1s):.3f})")

    # scale_pos_weight làm điểm số bị đẩy lên (không còn là xác suất thật) -> hiệu chỉnh Isotonic trên Val
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(p_val, y_va)
    return clf, thr, cfg, calibrator


def train_and_compare_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("[Task D] Bắt đầu huấn luyện và đánh giá trên DỮ LIỆU THẬT...")
    d = prepare_data()
    X_test, y_test, feature_cols = d['X_test'], d['y_test'], d['feature_cols']
    tuning_log = []
    results = []

    # Baseline 1: hằng số = median Train (dự báo MAE-tối ưu khi không có thông tin)
    median_train = float(d['y_train'].median())
    results.append(evaluate_predictions(y_test, np.full(len(y_test), median_train),
                                        "Baseline 1 (Hằng số = Median Train)", f"dự báo {median_train:.0f} phút"))
    # Baseline 2: Naïve Persistence
    results.append(evaluate_predictions(y_test, X_test['network_delay_lag_1h'].values, "Baseline 2 (Naïve Persistence)",
                                        "trễ TB sân bay giờ trước"))

    # Tuyến tính & Regularization (alpha chọn trên Val)
    lr_model, cfg = tune_linear(d, LinearRegression, [None], "Linear Regression", tuning_log)
    results.append(evaluate_predictions(y_test, lr_model.predict(d['X_test_s']), "Linear Regression", cfg))
    ridge, cfg = tune_linear(d, Ridge, [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0], "Ridge (L2)", tuning_log)
    results.append(evaluate_predictions(y_test, ridge.predict(d['X_test_s']), "Ridge Regression (L2)", cfg))
    lasso, cfg = tune_linear(d, Lasso, [0.001, 0.01, 0.05, 0.1, 0.5, 1.0], "Lasso (L1)", tuning_log)
    n_zero = int((np.abs(lasso.coef_) < 1e-9).sum())
    results.append(evaluate_predictions(y_test, lasso.predict(d['X_test_s']), "Lasso Regression (L1)",
                                        f"{cfg}, {n_zero}/{len(feature_cols)} hệ số = 0"))

    # Overfitting (Train vs Val)
    best_depth, overfit_df = experiment_overfitting(d['X_train'], d['y_train_clipped'], d['X_val'], d['y_val'])

    # Random Forest & XGBoost (siêu tham số chọn trên Val)
    rf, cfg = tune_random_forest(d, tuning_log)
    results.append(evaluate_predictions(y_test, rf.predict(X_test), "Random Forest Regressor", cfg))

    xgb_candidates = []
    for objective, label in [('reg:squarederror', 'XGBoost (squared error)'), ('reg:absoluteerror', 'XGBoost (absolute error)')]:
        m, cfg, val_mae = tune_xgboost(d, objective, tuning_log)
        results.append(evaluate_predictions(y_test, m.predict(X_test), label, cfg))
        xgb_candidates.append((val_mae, m, label))
    _, xgb, xgb_label = min(xgb_candidates, key=lambda t: t[0])
    print(f"[Task D4] Mô hình XGBoost được chọn (Val MAE thấp nhất): {xgb_label}")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("HIỆU NĂNG HỒI QUY TRÊN TẬP TEST (đánh giá 1 lần, cấu hình đã chọn trên Val):")
    print("=" * 80)
    print(results_df.drop(columns=['Cấu hình (chọn trên Val)']).to_string(index=False))
    results_df.to_csv(os.path.join(FIGURE_DIR, "model_comparison_metrics.csv"), index=False)

    # Task D7: Phân loại trễ >= 30 phút
    y_test_cls = (y_test >= DELAY_THRESHOLD).astype(int)
    clf, thr, clf_cfg, calibrator = train_classifier(d, tuning_log)
    p_test = clf.predict_proba(X_test)[:, 1]
    p_test_cal = calibrator.predict(p_test)
    calib_df = pd.DataFrame([
        {'Xác suất': 'Điểm thô XGBClassifier (scale_pos_weight)', 'Trung bình dự báo': round(float(p_test.mean()), 4),
         'Tỷ lệ trễ thực tế': round(float(y_test_cls.mean()), 4), 'Brier score': round(brier_score_loss(y_test_cls, p_test), 4)},
        {'Xác suất': 'Sau hiệu chỉnh Isotonic (fit trên Val)', 'Trung bình dự báo': round(float(p_test_cal.mean()), 4),
         'Tỷ lệ trễ thực tế': round(float(y_test_cls.mean()), 4), 'Brier score': round(brier_score_loss(y_test_cls, p_test_cal), 4)},
    ])
    print("[Task D7] Hiệu chỉnh xác suất trên Test:\n" + calib_df.to_string(index=False))
    calib_df.to_csv(os.path.join(FIGURE_DIR, "probability_calibration.csv"), index=False)
    persistence = X_test['network_delay_lag_1h'].values
    cls_results = [
        evaluate_classifier(y_test_cls, np.zeros(len(y_test_cls), dtype=int), None, "Baseline 1 (Luôn đoán lớp đa số: không trễ)"),
        evaluate_classifier(y_test_cls, (persistence >= DELAY_THRESHOLD).astype(int), persistence, "Baseline 2 (Persistence ≥ 30p)"),
        evaluate_classifier(y_test_cls, (xgb.predict(X_test) >= DELAY_THRESHOLD).astype(int), xgb.predict(X_test),
                            f"{xgb_label} - ngưỡng 30p", "hồi quy rồi cắt ngưỡng"),
        evaluate_classifier(y_test_cls, (p_test >= 0.5).astype(int), p_test, "XGBClassifier (ngưỡng 0.5)", clf_cfg.rsplit(', ngưỡng', 1)[0]),
        evaluate_classifier(y_test_cls, (p_test >= thr).astype(int), p_test, f"XGBClassifier (ngưỡng {thr} tối ưu F1 trên Val)", clf_cfg),
    ]
    cls_df = pd.DataFrame(cls_results)
    print("\n" + "=" * 80)
    print(f"HIỆU NĂNG PHÂN LOẠI TRỄ ≥ 30 PHÚT TRÊN TẬP TEST (tỷ lệ lớp dương = {y_test_cls.mean() * 100:.1f}%):")
    print("=" * 80)
    print(cls_df.drop(columns=['Cấu hình (chọn trên Val)']).to_string(index=False))
    print("=" * 80 + "\n")
    cls_df.to_csv(os.path.join(FIGURE_DIR, "classification_metrics.csv"), index=False)
    pd.DataFrame(tuning_log).to_csv(os.path.join(FIGURE_DIR, "hyperparameter_tuning_val.csv"), index=False)

    # Feature Importance từ XGBoost đã chọn
    feat_imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.sort_values().plot(kind='barh', color='#1f77b4', ax=ax)
    ax.set_title(f"Task D6: Top Đặc Trưng Quan Trọng ({xgb_label})", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mức độ quan trọng tương đối (gain)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "task_d6_feature_importance.png"), dpi=300)
    plt.close(fig)
    feat_imp.round(4).to_csv(os.path.join(FIGURE_DIR, "feature_importance_top12.csv"), header=['importance'])

    # Đóng gói mô hình
    joblib.dump(xgb, os.path.join(MODELS_DIR, "best_model_xgb.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "model_rf.pkl"))
    joblib.dump(ridge, os.path.join(MODELS_DIR, "model_ridge.pkl"))
    joblib.dump(lasso, os.path.join(MODELS_DIR, "model_lasso.pkl"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "delay_classifier.pkl"))
    joblib.dump(thr, os.path.join(MODELS_DIR, "classifier_threshold.pkl"))
    joblib.dump(calibrator, os.path.join(MODELS_DIR, "classifier_calibrator.pkl"))
    joblib.dump({'objective_label': xgb_label, 'best_tree_depth': best_depth}, os.path.join(MODELS_DIR, "model_info.pkl"))
    overfit_df.round(3).to_csv(os.path.join(FIGURE_DIR, "overfitting_curve.csv"), index=False)
    joblib.dump(d['scaler'], os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    print(f"[Task D - Hoàn thành] Đã lưu các mô hình và scaler vào: {MODELS_DIR}")
    return results_df, cls_df

if __name__ == "__main__":
    train_and_compare_models()
