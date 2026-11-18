"""
Module: eda_plots.py
Giai đoạn C: Trực quan hóa dữ liệu đa biến & Phân tích thăm dò EDA.
- Task C1: Run Sequence Plot
- Task C2: Lag Plot
- Task C3: Heatmap ma trận 24 giờ x 7 ngày
- Task C4: Correlation Heatmap Pearson
- Task C5: Parallel Coordinates Plot
- Task C6: Biểu đồ đường cong biến thiên giá vé máy bay theo ngày đặt trước (Dynamic Pricing)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

FIGURE_DIR = "docs/figures"

def plot_run_sequence(df: pd.DataFrame, n_flights: int = 300, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C1: Run Sequence Plot - Chuỗi thời gian trễ của các chuyến bay liên tiếp."""
    os.makedirs(save_dir, exist_ok=True)
    sample_df = df.tail(n_flights).copy().reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample_df.index, sample_df['delay_minutes'], color='#1f77b4', linewidth=1.5, alpha=0.85, label='Số phút trễ (phút)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(30, color='red', linestyle=':', linewidth=1.8, label='Ngưỡng trễ nghiêm trọng (≥ 30 phút)')
    
    ax.set_title(f"Task C1: Run Sequence Plot - Chuỗi thời gian mức độ trễ của {n_flights} chuyến bay liên tiếp", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Chỉ số thứ tự chuyến bay theo dòng thời gian", fontsize=11)
    ax.set_ylabel("Số phút trễ (phút)", fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c1_run_sequence.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_lag_plot(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C2: Lag Plot - Kiểm chứng tính tự tương quan trễ dây chuyền của mạng bay."""
    os.makedirs(save_dir, exist_ok=True)
    y_t = df['delay_minutes']
    y_lag = df['network_delay_lag_1h']
    
    valid_idx = (~y_t.isna()) & (~y_lag.isna())
    corr = np.corrcoef(y_lag[valid_idx], y_t[valid_idx])[0, 1]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_lag, y_t, alpha=0.2, color='#2ca02c', s=14, edgecolors='none')
    
    # Đường hồi quy xu hướng
    m, b = np.polyfit(y_lag[valid_idx], y_t[valid_idx], 1)
    x_vals = np.linspace(y_lag.min(), y_lag.max(), 100)
    ax.plot(x_vals, m * x_vals + b, color='red', linestyle='--', linewidth=2, label=f'Đường xu hướng (r = {corr:.3f})')
    
    ax.set_title("Task C2: Lag Plot (Tự tương quan trễ dây chuyền)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mức trễ bình quân mạng bay giờ trước (phút)", fontsize=11)
    ax.set_ylabel("Số phút trễ chuyến bay hiện tại (phút)", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c2_lag_plot.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_hourly_weekly_heatmap(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C3: Heatmap 2D ma trận 24 giờ x 7 ngày."""
    os.makedirs(save_dir, exist_ok=True)
    pivot_table = df.pivot_table(index='dayofweek', columns='hour', values='delay_minutes', aggfunc='mean')
    day_labels = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ Nhật']
    pivot_table.index = day_labels
    
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt=".1f", cbar_kws={'label': 'Thời gian trễ trung bình (phút)'}, ax=ax)
    
    ax.set_title("Task C3: Heatmap Ma trận Mức độ Trễ chuyến (24 Giờ x 7 Ngày)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Khung giờ khởi hành trong ngày (0 - 23h)", fontsize=11)
    ax.set_ylabel("Ngày trong tuần", fontsize=11)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c3_heatmap_24x7.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C4: Correlation Heatmap Pearson."""
    os.makedirs(save_dir, exist_ok=True)
    cols = ['delay_minutes', 'hour', 'is_weekend', 'passenger_count', 'origin_rain_mm', 'origin_wind_kmh', 'origin_gusts_kmh', 'origin_visibility_km']
    corr_matrix = df[cols].corr(method='pearson')
    
    col_names = {
        'delay_minutes': 'Số phút trễ',
        'hour': 'Khung giờ',
        'is_weekend': 'Cuối tuần',
        'passenger_count': 'Tải khách',
        'origin_rain_mm': 'Lượng mưa',
        'origin_wind_kmh': 'Tốc độ gió',
        'origin_gusts_kmh': 'Gió giật',
        'origin_visibility_km': 'Tầm nhìn'
    }
    corr_matrix.rename(index=col_names, columns=col_names, inplace=True)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Task C4: Ma trận Hệ số Tương quan Pearson (Thời tiết, Giờ & Trễ chuyến)", fontsize=13, fontweight='bold', pad=12)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c4_correlation_heatmap.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_parallel_coordinates(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C5: Parallel Coordinates Plot."""
    os.makedirs(save_dir, exist_ok=True)
    sample_df = df.sample(n=min(300, len(df)), random_state=42).copy()
    
    norm_cols = ['hour', 'passenger_count', 'origin_rain_mm', 'origin_wind_kmh', 'delay_minutes']
    for col in norm_cols:
        sample_df[col + '_norm'] = (sample_df[col] - sample_df[col].min()) / (sample_df[col].max() - sample_df[col].min() + 1e-6)
        
    plot_df = sample_df[['hour_norm', 'passenger_count_norm', 'origin_rain_mm_norm', 'origin_wind_kmh_norm', 'delay_minutes_norm', 'delay_status']].copy()
    plot_df.columns = ['Khung giờ', 'Lượng khách', 'Mưa', 'Gió', 'Mức trễ', 'Trạng thái trễ']
    
    fig, ax = plt.subplots(figsize=(11, 5))
    parallel_coordinates(plot_df, 'Trạng thái trễ', color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.45, ax=ax)
    
    ax.set_title("Task C5: Parallel Coordinates Plot - Tọa độ song song đa chiều", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Giá trị chuẩn hóa (0 - 1)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Phân loại trễ", loc='upper right')
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c5_parallel_coordinates.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_pricing_curve(pricing_csv: str = "data/processed/airfare_pricing_dataset.csv", save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Vẽ đường cong biến thiên giá vé máy bay theo số ngày đặt trước."""
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(pricing_csv):
        return None
    df_price = pd.read_csv(pricing_csv)
    df_han_sgn = df_price[(df_price['origin'] == 'HAN') & (df_price['destination'] == 'SGN')].copy()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for airline, group in df_han_sgn.groupby('airline'):
        group_sorted = group.sort_values(by='days_before_departure', ascending=False)
        ax.plot(group_sorted['days_before_departure'], group_sorted['fare_vnd'] / 1000, marker='o', markersize=3, label=airline, linewidth=2)
        
    # Vùng đáy giá vé
    ax.axvspan(14, 28, color='#22c55e', alpha=0.18, label="VÙNG ĐÁY GIÁ VÉ (Nên mua trước 14 - 28 ngày)")
    ax.axvspan(0, 7, color='#ef4444', alpha=0.15, label="VÙNG GIÁ ĐỈNH (Sát ngày bay < 7 ngày)")
    
    ax.set_title("Đường cong Biến thiên Giá vé Máy bay HAN - SGN theo Số ngày Đặt trước", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Số ngày đặt trước chuyến bay (Ngày)", fontsize=11)
    ax.set_ylabel("Giá vé trung bình (Nghìn VNĐ)", fontsize=11)
    ax.invert_xaxis() # Đảo ngược trục X: Từ 60 ngày về 0 ngày bay
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c6_pricing_curve.png"), dpi=300)
    plt.close(fig)
    return fig

def generate_all_eda_plots(featured_csv: str = "data/processed/featured_flight_dataset.csv"):
    print("[Task C] Đang sinh toàn bộ đồ thị trực quan hóa dữ liệu EDA...")
    df = pd.read_csv(featured_csv)
    plot_run_sequence(df)
    plot_lag_plot(df)
    plot_hourly_weekly_heatmap(df)
    plot_correlation_heatmap(df)
    plot_parallel_coordinates(df)
    plot_pricing_curve()
    print(f"[Task C - Hoàn thành] Toàn bộ biểu đồ EDA đã được lưu trong: {FIGURE_DIR}")

if __name__ == "__main__":
    generate_all_eda_plots()
