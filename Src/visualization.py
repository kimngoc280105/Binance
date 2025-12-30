import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        
        # Định nghĩa các khoảng chia cho từng loại cột
        self.custom_bins = {
            # Các cột giá (open, high, low, close) - phạm vi từ 0.000004 đến 116000
            'price': [0, 1, 10, 100, 1000, 10000, 50000, 120000],
            # Volume - phạm vi từ 22 đến 6.3 nghìn tỷ
            'volume': [0, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 10000000000000],
            # Quote volume - phạm vi từ 1142 đến 889 triệu
            'quote_volume': [0, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 500000000, 1000000000],
            # Trades count - phạm vi từ 7 đến 984003
            'trades_count': [0, 100, 500, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000],
            # Taker buy base volume
            'taker_buy_base': [0, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 100000000000, 10000000000000],
            # Taker buy quote volume
            'taker_buy_quote': [0, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 500000000]
        }
        
        # Map cột với loại bins
        self.column_bin_type = {
            'open': 'price',
            'high': 'price', 
            'low': 'price',
            'close': 'price',
            'volume': 'volume',
            'quote_volume': 'quote_volume',
            'trades_count': 'trades_count',
            'taker_buy_base_volume': 'taker_buy_base',
            'taker_buy_quote_volume': 'taker_buy_quote'
        }
    
    
    def _create_figure(self, title, nrows=2, ncols=3, figsize=(18, 12)):
        """Tạo figure với title và layout"""
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        return fig, axes
    
    def _get_custom_bins_for_column(self, column, data):
        """
        Lấy custom bins cho một cột dựa trên phân phối dữ liệu thực tế
        """
        bin_type = self.column_bin_type.get(column)
        
        if bin_type and bin_type in self.custom_bins:
            bins = self.custom_bins[bin_type].copy()
            data_max = data.max()
            data_min = data.min()
            
            # Lọc bins nằm trong phạm vi dữ liệu
            filtered_bins = [b for b in bins if b <= data_max * 1.1]
            
            # Đảm bảo có bin cuối cùng lớn hơn max
            if filtered_bins[-1] < data_max:
                filtered_bins.append(data_max * 1.1)
            
            # Đảm bảo bin đầu tiên nhỏ hơn hoặc bằng min
            if filtered_bins[0] > data_min:
                filtered_bins.insert(0, data_min * 0.9 if data_min > 0 else 0)
            
            return filtered_bins
        
        # Fallback: tạo bins logarithmic nếu không có custom
        return self._create_log_bins(data)
    
    def _create_log_bins(self, data, n_bins=15):
        """Tạo bins theo thang logarithm cho dữ liệu có phân phối lệch"""
        clean_data = data.dropna()
        data_min = clean_data.min()
        data_max = clean_data.max()
        
        if data_min <= 0:
            data_min = clean_data[clean_data > 0].min() if (clean_data > 0).any() else 1
        
        if data_max <= data_min:
            return np.linspace(data_min, data_max + 1, n_bins + 1)
        
        # Tạo bins logarithmic
        log_bins = np.logspace(np.log10(data_min), np.log10(data_max), n_bins + 1)
        return log_bins
    
    def _format_bin_label(self, low, high, is_last=False):
        """Format nhãn cho bin với số đẹp, không hiển thị số 0 dư thừa"""
        def format_num(n):
            # Xử lý số 0
            if n == 0:
                return "0"
            if n >= 1e12:
                return f"{n/1e12:.0f}T" if n/1e12 == int(n/1e12) else f"{n/1e12:.1f}T"
            elif n >= 1e9:
                return f"{n/1e9:.0f}B" if n/1e9 == int(n/1e9) else f"{n/1e9:.1f}B"
            elif n >= 1e6:
                return f"{n/1e6:.0f}M" if n/1e6 == int(n/1e6) else f"{n/1e6:.1f}M"
            elif n >= 1e3:
                return f"{n/1e3:.0f}K" if n/1e3 == int(n/1e3) else f"{n/1e3:.1f}K"
            elif n >= 1:
                return f"{int(n)}" if n == int(n) else f"{n:.1f}"
            elif n >= 0.01:
                return f"{n:.2f}"
            elif n >= 0.001:
                return f"{n:.3f}"
            elif n >= 0.0001:
                return f"{n:.4f}"
            else:
                return f"{n:.1e}"
        
        if is_last:
            return f">{format_num(low)}"
        return f"{format_num(low)}-{format_num(high)}"
    
    def _merge_small_bins(self, hist_counts, bin_edges, bin_labels, min_count_threshold=None):
        """
        Gộp các bins liền kề có số lượng nhỏ lại với nhau
        
        Parameters:
        -----------
        hist_counts : array
            Số lượng trong mỗi bin
        bin_edges : array  
            Cạnh của các bins
        bin_labels : list
            Labels của các bins
        min_count_threshold : int
            Ngưỡng tối thiểu để gộp bins (mặc định 1% tổng số)
        """
        total = hist_counts.sum()
        if min_count_threshold is None:
            min_count_threshold = max(total * 0.01, 10)  # 1% hoặc tối thiểu 10
        
        merged_counts = []
        merged_labels = []
        merged_edges = [bin_edges[0]]
        
        i = 0
        while i < len(hist_counts):
            current_count = hist_counts[i]
            current_start = bin_edges[i]
            current_end = bin_edges[i + 1]
            
            # Gộp các bins liền kề nếu tổng vẫn nhỏ
            while i + 1 < len(hist_counts) and current_count < min_count_threshold:
                i += 1
                current_count += hist_counts[i]
                current_end = bin_edges[i + 1]
            
            merged_counts.append(current_count)
            merged_edges.append(current_end)
            
            # Tạo label mới cho bin đã gộp
            is_last = (i == len(hist_counts) - 1)
            label = self._format_bin_label(current_start, current_end, is_last)
            merged_labels.append(label)
            
            i += 1
        
        return np.array(merged_counts), np.array(merged_edges), merged_labels
    
    def plot_distribution(self, ax, data, title, xlabel, color='skyblue', 
                         show_stats=True, bins=50):
        """Vẽ histogram với thống kê (matplotlib version - legacy)"""
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
        """
        Vẽ biểu đồ phân phối cho các cột số sử dụng Plotly
        Mỗi cột vẽ riêng một biểu đồ to, bins không đều theo phân phối dữ liệu
        Kèm bảng thống kê phần trăm sau mỗi biểu đồ
        """
        print("="*50)
        print("BIỂU ĐỒ PHÂN PHỐI DỮ LIỆU (PLOTLY)")
        print("="*50)
        
        # Lấy các cột số quan trọng
        important_numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                                  'quote_volume', 'trades_count', 
                                  'taker_buy_base_volume', 'taker_buy_quote_volume']
        available_cols = [col for col in important_numeric_cols if col in self.df.columns]
        
        if len(available_cols) == 0:
            print("Không có đủ dữ liệu số để phân tích")
            return
        
        print(f"Đang tạo biểu đồ phân phối cho {len(available_cols)} cột số...")
        
        # Màu sắc cho các biểu đồ
        colors = px.colors.qualitative.Set2
        
        for idx, col in enumerate(available_cols):
            color = colors[idx % len(colors)]
            
            # Lấy dữ liệu
            data = self.df[col].dropna()
            
            if len(data) == 0:
                print(f"   • {col}: Không có dữ liệu")
                continue
            
            # Lấy custom bins cho cột này
            bins = self._get_custom_bins_for_column(col, data)
            
            # Tính histogram với custom bins
            hist_counts, bin_edges = np.histogram(data, bins=bins)
            
            # Tạo labels cho các bins
            bin_labels = []
            for i in range(len(bin_edges) - 1):
                is_last = (i == len(bin_edges) - 2)
                label = self._format_bin_label(bin_edges[i], bin_edges[i+1], is_last)
                bin_labels.append(label)
            
            # Gộp các bins nhỏ lại với nhau
            merged_counts, merged_edges, merged_labels = self._merge_small_bins(
                hist_counts, bin_edges, bin_labels
            )
            
            # Chỉ giữ lại các bins có dữ liệu (count > 0)
            non_zero_mask = merged_counts > 0
            filtered_labels = [merged_labels[i] for i in range(len(merged_labels)) if non_zero_mask[i]]
            filtered_counts = merged_counts[non_zero_mask]
            
            # Tính phần trăm
            total = filtered_counts.sum()
            percentages = (filtered_counts / total * 100)
            
            # Tạo biểu đồ riêng cho mỗi cột
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=filtered_labels,
                y=filtered_counts,
                marker_color=color,
                marker_line_color='darkgray',
                marker_line_width=1,
                opacity=0.85,
                text=[f'<b>{c:,}</b><br>({p:.1f}%)' for c, p in zip(filtered_counts, percentages)],
                textposition='auto',
                textfont=dict(size=12),
                hovertemplate=(
                    f'<b>{col}</b><br>'
                    'Khoảng: %{x}<br>'
                    'Số lượng: %{y:,}<br>'
                    '<extra></extra>'
                )
            ))
            
            # Tính thống kê
            stats_text = (
                f"<b>Thống kê:</b><br>"
                f"N = {len(data):,}<br>"
                f"Mean = {data.mean():,.4f}<br>"
                f"Median = {data.median():,.4f}<br>"
                f"Min = {data.min():,.6g}<br>"
                f"Max = {data.max():,.2f}"
            )
            
            # Cập nhật layout
            fig.update_layout(
                title=dict(
                    text=f'<b>Phân Phối của {col.upper()}</b>',
                    x=0.5,
                    font=dict(size=20)
                ),
                xaxis_title=dict(text=f'Khoảng giá trị {col}', font=dict(size=14)),
                yaxis_title=dict(text='Số lượng bản ghi', font=dict(size=14)),
                height=650,
                width=1200,
                template='plotly_white',
                font=dict(family="Arial", size=12),
                showlegend=False,
                # Thêm annotation thống kê
                annotations=[
                    dict(
                        x=0.98,
                        y=0.95,
                        xref='paper',
                        yref='paper',
                        text=stats_text,
                        showarrow=False,
                        font=dict(size=11),
                        align='left',
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='gray',
                        borderwidth=1,
                        borderpad=5
                    )
                ],
                bargap=0.15,
                margin=dict(t=80, b=80)
            )
            
            # Xoay labels nếu có nhiều bins
            if len(filtered_labels) > 8:
                fig.update_xaxes(tickangle=45)
            
            # Hiển thị biểu đồ
            fig.show()
            
            # In bảng thống kê phần trăm cho cột này
            self._print_percentage_table(col, filtered_labels, filtered_counts, percentages)
        
        # In bảng thống kê tổng hợp cuối cùng
        self._print_distribution_stats(available_cols)
    
    def _print_percentage_table(self, col_name, labels, counts, percentages):
        """In bảng phần trăm cho từng khoảng giá trị"""
        print(f"\n📊 BẢNG PHÂN BỐ: {col_name.upper()}")
        print("-" * 60)
        print(f"{'Khoảng giá trị':<25} {'Số lượng':>12} {'Phần trăm':>12}")
        print("-" * 60)
        
        for label, count, pct in zip(labels, counts, percentages):
            print(f"{label:<25} {count:>12,} {pct:>11.2f}%")
        
        print("-" * 60)
        print(f"{'TỔNG':<25} {sum(counts):>12,} {100.00:>11.2f}%")
        print("=" * 60)
    
    def _print_distribution_stats(self, columns):
        """In bảng thống kê phân phối chi tiết"""
        print("\n" + "="*80)
        print("THỐNG KÊ PHÂN PHỐI CHI TIẾT")
        print("="*80)
        
        stats_data = []
        for col in columns:
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
                
            stats_data.append({
                'Cột': col,
                'Số lượng': f"{len(data):,}",
                'Mean': f"{data.mean():.4f}",
                'Std': f"{data.std():.4f}",
                'Min': f"{data.min():.6f}",
                'Q1 (25%)': f"{data.quantile(0.25):.4f}",
                'Median': f"{data.median():.4f}",
                'Q3 (75%)': f"{data.quantile(0.75):.4f}",
                'Max': f"{data.max():.2f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        print(stats_df.to_string(index=False))
    
    def plot_single_distribution(self, column):
        """
        Vẽ biểu đồ phân phối chi tiết cho một cột cụ thể
        
        Parameters:
        -----------
        column : str
            Tên cột cần vẽ
        """
        if column not in self.df.columns:
            print(f"Cột '{column}' không tồn tại trong dataframe")
            return
        
        data = self.df[column].dropna()
        bins = self._get_custom_bins_for_column(column, data)
        
        # Tính histogram
        hist_counts, bin_edges = np.histogram(data, bins=bins)
        
        # Tạo labels
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            is_last = (i == len(bin_edges) - 2)
            label = self._format_bin_label(bin_edges[i], bin_edges[i+1], is_last)
            bin_labels.append(label)
        
        # Lọc bins có dữ liệu
        non_zero_mask = hist_counts > 0
        filtered_labels = [bin_labels[i] for i in range(len(bin_labels)) if non_zero_mask[i]]
        filtered_counts = hist_counts[non_zero_mask]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=filtered_labels,
            y=filtered_counts,
            marker_color='steelblue',
            marker_line_color='darkblue',
            marker_line_width=1,
            text=[f'{c:,}' if c > 0 else '' for c in filtered_counts],
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f'<b>Phân Phối của {column.upper()}</b>',
            xaxis_title=column,
            yaxis_title='Số lượng',
            height=700,
            width=1400,
            template='plotly_white'
        )
        
        fig.show()
    
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