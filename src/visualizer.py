import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

class CryptoVisualizer:
    """Lớp trực quan hóa với các hàm tái sử dụng"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.fig_count = 0
        self.stats_cache = {}
    
    
    def _create_figure(self, title, nrows=2, ncols=3, figsize=(18, 12)):
        """Tạo figure với title và layout"""
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        return fig, axes
    
    
    def plot_distribution(self, ax, data, title, xlabel, color='skyblue', 
                         show_stats=True, bins=50):
        """Vẽ histogram với thống kê"""
        ax.hist(data.dropna(), bins=bins, alpha=0.7, edgecolor='black', color=color)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        if show_stats and len(data.dropna()) > 0:
            stats_text = self._get_stats_text(data)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    
    def plot_data_distribution(self):
        """Basic data exploration using reusable functions"""
        print("="*50)
        
        # Lấy các cột số quan trọng
        important_numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades_count', 'taker_buy_base_volume', 'taker_buy_quote_volume']
        available_cols = [col for col in important_numeric_cols if col in self.df.columns]
        
        if len(available_cols) > 0:
            print("Đang tạo biểu đồ phân phối cho các cột số...")
            
            # Tạo figure lớn
            n_cols = min(3, len(available_cols))
            n_rows = (len(available_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
            fig.suptitle('Phân Phối Các Biến Số Quan Trọng', fontsize=16, fontweight='bold')
            
            # Flatten axes nếu cần
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows > 1 and n_cols > 1:
                axes = axes.flatten()
            
            # Vẽ từng biểu đồ phân phối
            for idx, col in enumerate(available_cols):
                if idx < len(axes):
                    self.plot_distribution(
                        axes[idx], 
                        self.df[col], 
                        f'Phân phối của {col}', 
                        col, 
                        color='skyblue', 
                        show_stats=True, 
                        bins=50
                    )
                    
                    # Thêm log scale cho volume
                    if col == 'volume' or col == 'quote_volume':
                        axes[idx].set_xscale('log')
            
            # Ẩn các axes không sử dụng
            for idx in range(len(available_cols), len(axes)):
                axes[idx].set_visible(False)

        else:
            print("Không có đủ dữ liệu số để phân tích")
    
    def _get_stats_text(self, data):
        """Tạo text thống kê"""
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return "No data"
        
        return f"""N: {len(clean_data):,}
                Mean: {clean_data.mean():.4f}
                Std: {clean_data.std():.4f}
                Min: {clean_data.min():.4f}
                25%: {clean_data.quantile(0.25):.4f}
                50%: {clean_data.median():.4f}
                75%: {clean_data.quantile(0.75):.4f}
                Max: {clean_data.max():.4f}
                Skew: {clean_data.skew():.3f}"""