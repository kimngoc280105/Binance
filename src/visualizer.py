# ============================================================
# VISUALIZER.PY - OPTIMIZED WITH REUSABLE FUNCTIONS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, pearsonr, shapiro, f_oneway, spearmanr
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
    
    # ============================================================
    # REUSABLE VISUALIZATION FUNCTIONS
    # ============================================================
    
    def _create_figure(self, title, nrows=2, ncols=3, figsize=(18, 12)):
        """Tạo figure với title và layout"""
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        return fig, axes
    
    def _save_figure(self, filename):
        """Lưu figure"""
        plt.savefig(f'figure_{self.fig_count}_{filename}.png', 
                   dpi=300, bbox_inches='tight')
        self.fig_count += 1
        plt.show()
    
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
    
    def plot_scatter_with_correlation(self, ax, x, y, title, xlabel, ylabel,
                                     log_scale=False, alpha=0.3, s=10):
        """Vẽ scatter plot với correlation"""
        if log_scale:
            ax.set_xscale('log')
        
        ax.scatter(x, y, alpha=alpha, s=s, color='blue')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Tính correlation
        valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(valid_data) > 2:
            corr = valid_data['x'].corr(valid_data['y'])
            ax.annotate(f'Correlation: {corr:.3f}', 
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Thêm regression line
            if len(valid_data) > 10:
                z = np.polyfit(valid_data['x'], valid_data['y'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(valid_data['x'].min(), valid_data['x'].max(), 100)
                ax.plot(x_range, p(x_range), 'r-', linewidth=2)
    
    def plot_line_chart(self, ax, x, y, title, xlabel, ylabel, 
                       marker='o', color='blue', label=None, grid=True):
        """Vẽ line chart"""
        ax.plot(x, y, marker=marker, color=color, linewidth=2, label=label)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if grid:
            ax.grid(True, alpha=0.3)
        
        if label:
            ax.legend()
    
    def plot_bar_chart(self, ax, categories, values, title, xlabel, ylabel,
                      colors=None, horizontal=False, add_values=True):
        """Vẽ bar chart"""
        if colors is None:
            colors = 'skyblue'
        
        if horizontal:
            bars = ax.barh(categories, values, color=colors)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            bars = ax.bar(categories, values, color=colors)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')
        
        if add_values:
            for bar in bars:
                if horizontal:
                    width = bar.get_width()
                    ax.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.3f}', va='center')
                else:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.01,
                           f'{height:.3f}', ha='center', va='bottom')
    
    def plot_pie_chart(self, ax, sizes, labels, title, colors=None):
        """Vẽ pie chart"""
        if colors is None:
            colors = ['lightcoral', 'lightgreen', 'gold', 'lightskyblue']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(title, fontsize=14)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def plot_box_plot(self, ax, data_list, labels, title, xlabel, ylabel):
        """Vẽ box plot"""
        bp = ax.boxplot(data_list, labels=labels)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
    
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