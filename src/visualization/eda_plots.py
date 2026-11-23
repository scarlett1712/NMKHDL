"""
Module: eda_plots.py
Giai đoạn C: Trực quan hóa dữ liệu đa biến & Phân tích thăm dò EDA toàn diện.
Bao gồm đầy đủ các đồ thị chuẩn mực của môn Khoa học Dữ liệu:
1. Run Sequence Plot: Chuỗi thời gian trễ của các chuyến bay liên tiếp
2. Lag Plot: Đồ thị tự tương quan trễ mạng bay (r)
3. Heatmap 24 giờ x 7 ngày: Ma trận nhiệt trễ chuyến
4. Correlation Heatmap Pearson: Ma trận tương quan các thuộc tính khí tượng & vận hành
5. Parallel Coordinates Plot: Tọa độ song song đa chiều
6. Monthly & Hub Delay: Mức trễ theo tháng và theo quy mô sân bay đến (dữ liệu Web Crawler)
7. Histogram & KDE: Phân bố tần suất thời gian trễ (delay_minutes)
8. Boxplot: Phân bố trễ theo Hãng hàng không & Sân bay xuất phát
9. Violin Plot: Mật độ xác suất trễ theo Hãng hàng không
10. PCA 2D Projection: Phân tích thành phần chính giảm chiều dữ liệu
11. K-Means Elbow & Silhouette: Cơ sở thực nghiệm lựa chọn k=3
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.cleaner import DataCleaner

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

FIGURE_DIR = "docs/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def plot_run_sequence(df: pd.DataFrame, n_flights: int = 300, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C1: Run Sequence Plot - Chuỗi thời gian mức độ trễ của 300 chuyến bay liên tiếp."""
    sample_df = df.tail(n_flights).copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample_df.index, sample_df['delay_minutes'], color='#1f77b4', linewidth=1.5, alpha=0.85, label='Số phút trễ (phút)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(30, color='red', linestyle=':', linewidth=1.8, label='Ngưỡng trễ nghiêm trọng (≥ 30 phút)')
    ax.set_title(f"Task C1: Run Sequence Plot - Chuỗi thời gian trễ của {n_flights} chuyến bay liên tiếp", fontsize=13, fontweight='bold', pad=12)
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
    y_t = df['delay_minutes']
    y_lag = df['network_delay_lag_1h']
    valid_idx = (~y_t.isna()) & (~y_lag.isna())
    corr = np.corrcoef(y_lag[valid_idx], y_t[valid_idx])[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_lag, y_t, alpha=0.15, color='#2ca02c', s=14, edgecolors='none')
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
    """Task C4: Correlation Heatmap Pearson các thuộc tính quan trọng."""
    cols = ['delay_minutes', 'network_delay_lag_1h', 'distance', 'origin_rain_mm', 'origin_wind_kmh', 'origin_gusts_kmh', 'origin_visibility_km']
    corr_matrix = df[cols].corr(method='pearson')
    col_names = {
        'delay_minutes': 'Số phút trễ',
        'network_delay_lag_1h': 'Trễ sân bay lag-1h',
        'distance': 'Cự ly bay (dặm)',
        'origin_rain_mm': 'Lượng mưa (mm)',
        'origin_wind_kmh': 'Tốc độ gió',
        'origin_gusts_kmh': 'Gió giật (km/h)',
        'origin_visibility_km': 'Tầm nhìn (km)'
    }
    corr_matrix = corr_matrix.rename(index=col_names, columns=col_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Task C4: Ma trận Tương quan Tuyến tính Pearson (Dữ liệu Thật)", fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c4_correlation_heatmap.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_parallel_coordinates(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C5: Parallel Coordinates Plot - Tọa độ song song."""
    cols_parallel = ['distance', 'origin_rain_mm', 'origin_gusts_kmh', 'delay_minutes', 'delay_status']
    sample_df = df[cols_parallel].sample(n=min(400, len(df)), random_state=42).copy()
    norm_cols = ['distance', 'origin_rain_mm', 'origin_gusts_kmh', 'delay_minutes']
    for c in norm_cols:
        sample_df[c] = DataCleaner.min_max_scale(sample_df[c])  # Task B3: Min-Max Scaling về [0, 1]
    fig, ax = plt.subplots(figsize=(11, 5))
    parallel_coordinates(sample_df, 'delay_status', color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.4, ax=ax)
    ax.set_title("Task C5: Biểu đồ Tọa độ Song song (Parallel Coordinates)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Giá trị chuẩn hóa [0, 1]", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c5_parallel_coordinates.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_monthly_and_hub_delay(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C6: Mức trễ theo tháng (mùa vụ) và theo quy mô sân bay đến (dữ liệu Web Crawler)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    by_month = df.groupby('month').agg(mean_delay=('delay_minutes', 'mean'),
                                       severe=('is_delayed_30', 'mean')).reset_index()
    ax = axes[0]
    ax.bar(by_month['month'], by_month['mean_delay'], color='#1f77b4', alpha=0.85, label='Trễ trung bình (phút)')
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Tháng (năm 2013)", fontsize=10)
    ax.set_ylabel("Trễ trung bình (phút)", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(by_month['month'], by_month['severe'] * 100, 'o-', color='#d62728', linewidth=2, label='% trễ ≥ 30 phút')
    ax2.set_ylabel("% chuyến trễ ≥ 30 phút", fontsize=10, color='#d62728')
    ax2.grid(False)
    ax.set_title("Mức trễ theo tháng", fontsize=12, fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left')

    if 'dest_hub_tier' in df.columns:
        tier_names = {0: 'Không phải hub\nlớn/vừa', 1: 'Medium hub', 2: 'Large hub'}
        by_hub = df.groupby('dest_hub_tier').agg(mean_delay=('delay_minutes', 'mean'), n=('delay_minutes', 'size')).reset_index()
        labels = [f"{tier_names.get(t, t)}\n(n={n:,})" for t, n in zip(by_hub['dest_hub_tier'], by_hub['n'])]
        axes[1].bar(labels, by_hub['mean_delay'], color=['#9ca3af', '#f59e0b', '#16a34a'][:len(by_hub)])
        axes[1].set_title("Trễ trung bình theo quy mô sân bay đến (FAA, Wikipedia)", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Trễ trung bình (phút)", fontsize=10)

    fig.suptitle("Task C6: Mùa vụ và Quy mô Sân bay đến", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c6_monthly_hub_delay.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_delay_distribution(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C7: Histogram & KDE phân bố thời gian trễ thực tế."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['delay_minutes'], kde=True, bins=40, color='#1f77b4', edgecolor='white', ax=ax)
    ax.axvline(df['delay_minutes'].mean(), color='red', linestyle='--', linewidth=1.8, label=f"Mean: {df['delay_minutes'].mean():.1f}p")
    ax.axvline(df['delay_minutes'].median(), color='green', linestyle=':', linewidth=2, label=f"Median: {df['delay_minutes'].median():.1f}p")
    ax.set_title("Task C7: Phân bố Tần suất Thời gian Trễ chuyến (Histogram & KDE)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Số phút trễ (phút)", fontsize=11)
    ax.set_ylabel("Số lượng chuyến bay", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c7_delay_histogram.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_boxplot_by_airline_and_airport(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C8: Boxplot so sánh mức trễ theo Hãng bay và Sân bay xuất phát."""
    top_airlines = df['airline'].value_counts().head(8).index
    df_top = df[df['airline'].isin(top_airlines)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x='airline', y='delay_minutes', data=df_top, palette='Set2', ax=axes[0], showfliers=False)
    axes[0].set_title("Phân bố Trễ theo Top Hãng hàng không", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Hãng hàng không", fontsize=10)
    axes[0].set_ylabel("Số phút trễ (phút)", fontsize=10)
    axes[0].tick_params(axis='x', rotation=15)

    sns.boxplot(x='origin', y='delay_minutes', data=df, palette='Pastel1', ax=axes[1], showfliers=False)
    axes[1].set_title("Phân bố Trễ theo Sân bay xuất phát (JFK, EWR, LGA)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Sân bay xuất phát", fontsize=10)
    axes[1].set_ylabel("Số phút trễ (phút)", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c8_boxplot_airline_airport.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_violin_delay(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C9: Violin Plot biểu diễn mật độ xác suất trễ theo Hãng hàng không."""
    top_airlines = df['airline'].value_counts().head(6).index
    df_top = df[df['airline'].isin(top_airlines)].copy()
    # Giới hạn hiển thị dưới 120 phút để nhìn rõ thân đàn violin
    df_top['delay_capped'] = df_top['delay_minutes'].clip(lower=-20, upper=120)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(x='airline', y='delay_capped', data=df_top, palette='muted', inner='quartile', ax=ax)
    ax.set_title("Task C9: Biểu đồ Violin Phân bố Mật độ Trễ theo Top Hãng bay", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Hãng hàng không", fontsize=11)
    ax.set_ylabel("Số phút trễ (phút)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c9_violin_plot.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_pca_analysis(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C10: Phân tích thành phần chính PCA (2D Projection) giảm chiều dữ liệu."""
    features = ['hour', 'dayofweek', 'distance', 'origin_rain_mm', 'origin_wind_kmh', 'origin_gusts_kmh', 'network_delay_lag_1h']
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    sample_idx = np.random.default_rng(42).choice(len(components), size=min(1500, len(components)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(components[sample_idx, 0], components[sample_idx, 1],
                         c=df.loc[sample_idx, 'delay_minutes'].clip(lower=-15, upper=120), cmap='viridis', alpha=0.6, s=18)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Số phút trễ (phút)', fontsize=10)
    ax.set_title(f"Task C10: PCA 2D Không gian Đặc trưng Thực tế (Var Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c10_pca_analysis.png"), dpi=300)
    plt.close(fig)
    return fig

def plot_kmeans_elbow(df: pd.DataFrame, save_dir: str = FIGURE_DIR) -> plt.Figure:
    """Task C11: Biểu đồ Elbow lựa chọn tối ưu số cụm k cho K-Means."""
    X = StandardScaler().fit_transform(df[['hour', 'distance']].dropna())
    sample_X = X[np.random.default_rng(42).choice(len(X), size=min(4000, len(X)), replace=False)]
    inertias = []
    k_range = list(range(1, 8))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(sample_X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, 'o-', color='#e377c2', linewidth=2, markersize=7)
    ax.axvline(3, color='blue', linestyle='--', label='Điểm gập khuỷu tối ưu (Elbow k=3)')
    ax.set_title("Task C11: Phương pháp Elbow chọn số cụm K-Means tối ưu (Dữ liệu Thật)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Số cụm k", fontsize=10)
    ax.set_ylabel("Tổng bình phương khoảng cách (Inertia)", fontsize=10)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "task_c11_kmeans_elbow.png"), dpi=300)
    plt.close(fig)
    return fig

def generate_all_eda_plots(featured_path: str = "data/processed/featured_flight_dataset.csv"):
    print("[Task C] Bắt đầu sinh toàn bộ các biểu đồ EDA...")
    df = pd.read_csv(featured_path)
    plot_run_sequence(df)
    plot_lag_plot(df)
    plot_hourly_weekly_heatmap(df)
    plot_correlation_heatmap(df)
    plot_parallel_coordinates(df)
    plot_monthly_and_hub_delay(df)
    plot_delay_distribution(df)
    plot_boxplot_by_airline_and_airport(df)
    plot_violin_delay(df)
    plot_pca_analysis(df)
    plot_kmeans_elbow(df)
    print(f"[Task C - Hoàn tất] Toàn bộ các đồ thị EDA đã được lưu tại: {FIGURE_DIR}")

if __name__ == "__main__":
    generate_all_eda_plots()
