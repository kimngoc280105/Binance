import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.stats import zscore, norm, ttest_ind, f_oneway
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class CryptoDataProcessor:
    """Lớp xử lý dữ liệu crypto với các kỹ thuật xử lý nâng cao"""
    
    def __init__(self, file_path):
        """
        Khởi tạo processor với đường dẫn file
        
        Parameters:
        -----------
        file_path : str
            Đường dẫn đến file CSV chứa dữ liệu
        """
        self.file_path = file_path
        self.df = None
        self.scalers = {}
        self.pca = None
        
    def load_data(self):
        """Tải dữ liệu từ file CSV"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Đã tải {len(self.df)} bản ghi, {self.df.shape[1]} cột")
            
            # Kiểm tra tính hợp lệ của dữ liệu
            self.validate_data_integrity()
            
            return self.df
        except Exception as e:
            print(f"✗ Lỗi khi tải dữ liệu: {str(e)}")
            return None
    
    def validate_data_integrity(self):
        """Kiểm tra tính hợp lệ của dữ liệu"""
        print("\n" + "="*60)
        print("KIỂM TRA TÍNH HỢP LỆ CỦA DỮ LIỆU")
        print("="*60)
        
        # Kiểm tra giá trị âm cho các cột không được âm
        non_negative_cols = ['open', 'high', 'low', 'close', 'volume', 
                           'quote_volume', 'taker_buy_base_volume']
        
        for col in non_negative_cols:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    print(f"⚠ Cảnh báo: {negative_count} giá trị âm trong cột {col}")
                    
        required_cols = ['open', 'high', 'low', 'close']

        if all(col in self.df.columns for col in required_cols):

            # Max và Min giữa open-close
            max_oc = self.df[['open', 'close']].max(axis=1)
            min_oc = self.df[['open', 'close']].min(axis=1)

            # Điều kiện sai
            invalid_high = (self.df['high'] < max_oc).sum()
            invalid_low  = (self.df['low']  > min_oc).sum()

            # In cảnh báo
            if invalid_high > 0:
                print(f"Cảnh báo: {invalid_high} bản ghi có high < max(open, close)")

            if invalid_low > 0:
                print(f"Cảnh báo: {invalid_low} bản ghi có low > min(open, close)")
    
    def explore_data(self):
        """Khám phá thông tin cơ bản của dữ liệu"""
        if self.df is None:
            self.load_data()
            
        print("=" * 60)
        print("PHÂN TÍCH DỮ LIỆU CHI TIẾT")
        print("=" * 60)

         # 1. Ý nghĩa của từng cột
        print("\n0. Ý NGHĨA CÁC CỘT DỮ LIỆU:")
        print("-" * 50)
        
        column_descriptions = {
            'open': 'Giá mở cửa tại đầu mỗi giờ',
            'high': 'Giá cao nhất trong giờ',
            'low': 'Giá thấp nhất trong giờ',
            'close': 'Giá đóng cửa cuối giờ',
            'volume': 'Khối lượng giao dịch (base currency)',
            'quote_volume': 'Khối lượng giao dịch (quote currency)',
            'trades_count': 'Số lượng giao dịch trong giờ',
            'taker_buy_base_volume': 'Khối lượng mua từ taker (base)',
            'taker_buy_quote_volume': 'Khối lượng mua từ taker (quote)',
            'timestamp': 'Thời điểm bắt đầu của khoảng thời gian',
            'date': 'Ngày',
            'time': 'Giờ',
            'symbol': 'Cặp giao dịch (ví dụ: BTCUSDT)',
            'interval': 'Khoảng thời gian (1h, 4h, 1d, v.v.)'
        }
        
        for col in self.df.columns:
            desc = column_descriptions.get(col, 'Không có mô tả')
            print(f"• {col:<25}: {desc}")

        # 2. Thông tin cơ bản
        print(f"\n1. KÍCH THƯỚC DỮ LIỆU:")
        print(f"   • Số hàng: {self.df.shape[0]:,}")
        print(f"   • Số cột: {self.df.shape[1]}")
        print(f"   • Kích thước bộ nhớ: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 3. Kiểu dữ liệu và thông tin cột
        print("\n2. THÔNG TIN CÁC CỘT:")
        column_info = []
        for col in self.df.columns:
            col_type = self.df[col].dtype
            missing = self.df[col].isnull().sum()
            missing_pct = (missing / len(self.df)) * 100
            unique = self.df[col].nunique()
            
            # Thêm thông tin thống kê cho cột số
            if pd.api.types.is_numeric_dtype(self.df[col]):
                stats_info = f"Min: {self.df[col].min():.4f}, Max: {self.df[col].max():.4f}"
            else:
                stats_info = f"Unique: {unique}"
                
            column_info.append({
                'Cột': col,
                'Kiểu': col_type,
                'Missing': f"{missing} ({missing_pct:.2f}%)",
                'Thống kê': stats_info
            })
        
        col_df = pd.DataFrame(column_info)
        print(col_df.to_string(index=False))
        
        # 4. Thống kê mô tả chi tiết
        print("\n3. THỐNG KÊ MÔ TẢ CHI TIẾT:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            stats_df = self.df[numeric_cols].describe().T
            stats_df['skewness'] = self.df[numeric_cols].skew()
            stats_df['kurtosis'] = self.df[numeric_cols].kurtosis()
            stats_df['cv'] = (stats_df['std'] / stats_df['mean']) * 100  # Coefficient of variation
            
            print(stats_df.round(4))
            
            # Phân tích phân phối
            print("\n4. PHÂN TÍCH PHÂN PHỐI:")
            for col in numeric_cols[:9]:
                skew_val = stats_df.loc[col, 'skewness']
                kurt_val = stats_df.loc[col, 'kurtosis']
                
                distribution_type = "Normal" if abs(skew_val) < 0.5 and abs(kurt_val) < 1 else "Non-normal"
                print(f"   • {col}: Skewness={skew_val:.3f}, Kurtosis={kurt_val:.3f} → {distribution_type}")
        
        return self.df
    
    def handle_missing_values(self, strategy='knn', n_neighbors=5):
        """
        Xử lý giá trị missing bằng nhiều phương pháp
        
        Parameters:
        -----------
        strategy : str
            'mean', 'median', 'knn', 'interpolate'
        n_neighbors : int
            Số neighbors cho KNN imputation
        """
        print(f"\nXỬ LÝ MISSING VALUES (Strategy: {strategy})")
        
        initial_missing = self.df.isnull().sum().sum()
        if initial_missing == 0:
            print("✓ Không có giá trị missing")
            return self.df
        
        print(f"• Tổng giá trị missing ban đầu: {initial_missing}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if strategy == 'mean':
            # Fill với giá trị mean của từng cột
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    
        elif strategy == 'median':
            # Fill với giá trị median (robust với outliers)
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    
        elif strategy == 'knn':
            # Sử dụng KNN Imputation
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed_data = imputer.fit_transform(self.df[numeric_cols])
            self.df[numeric_cols] = imputed_data
            
        elif strategy == 'interpolate':
            # Nội suy theo thời gian (nếu có cột timestamp)
            if 'timestamp' in self.df.columns:
                self.df = self.df.sort_values('timestamp')
                for col in numeric_cols:
                    if self.df[col].isnull().sum() > 0:
                        self.df[col] = self.df[col].interpolate(method='time')
        
        # Forward/backward fill cho các giá trị còn lại
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        final_missing = self.df.isnull().sum().sum()
        print(f"• Tổng giá trị missing sau xử lý: {final_missing}")
        print(f"✓ Đã xử lý {initial_missing - final_missing} giá trị missing")
        
        return self.df
    
    def detect_and_handle_outliers(self, method='iqr', threshold=3, by_symbol=True):
        """
        Phát hiện và xử lý outliers
        
        Parameters:
        -----------
        method : str
            'iqr', 'zscore', 'isolation'
        threshold : float
            Ngưỡng cho phương pháp zscore
        by_symbol : bool
            True: Xử lý outliers riêng cho từng symbol
            False: Xử lý chung toàn bộ dữ liệu
        """
        print(f"\nXỬ LÝ OUTLIERS (Method: {method}, By Symbol: {by_symbol})")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        important_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        important_cols = [col for col in important_cols if col in numeric_cols]
        
        outlier_counts = {}
        processed_symbols = 0
        
        if by_symbol and 'symbol' in self.df.columns and len(important_cols) > 0:
            # Xử lý outliers riêng cho từng symbol
            symbols = self.df['symbol'].unique()
            
            for symbol in symbols:
                symbol_mask = self.df['symbol'] == symbol
                symbol_df = self.df[symbol_mask].copy()
                
                if len(symbol_df) < 10:  # Bỏ qua symbols có ít dữ liệu
                    continue
                    
                symbol_outliers = 0
                
                for col in important_cols:
                    if method == 'iqr':
                        Q1 = symbol_df[col].quantile(0.25)
                        Q3 = symbol_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        if IQR == 0:  # Tránh chia cho 0
                            continue
                            
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Capping outliers
                        self.df.loc[symbol_mask & (self.df[col] < lower_bound), col] = lower_bound
                        self.df.loc[symbol_mask & (self.df[col] > upper_bound), col] = upper_bound
                        
                        # Đếm outliers
                        original_outliers = ((symbol_df[col] < lower_bound) | (symbol_df[col] > upper_bound)).sum()
                        symbol_outliers += original_outliers
                        
                        if col not in outlier_counts:
                            outlier_counts[col] = 0
                        outlier_counts[col] += original_outliers
                        
                    elif method == 'zscore':
                        mean_val = symbol_df[col].mean()
                        std_val = symbol_df[col].std()
                        
                        if std_val == 0:  # Tránh chia cho 0
                            continue
                            
                        # Capping outliers
                        z_scores = np.abs((self.df.loc[symbol_mask, col] - mean_val) / std_val)
                        outliers_mask = z_scores > threshold
                        
                        capped_values = np.where(
                            self.df.loc[symbol_mask, col] > mean_val,
                            mean_val + threshold * std_val,
                            mean_val - threshold * std_val
                        )
                        
                        self.df.loc[symbol_mask & outliers_mask, col] = capped_values[outliers_mask]
                        
                        # Đếm outliers
                        original_outliers = outliers_mask.sum()
                        symbol_outliers += original_outliers
                        
                        if col not in outlier_counts:
                            outlier_counts[col] = 0
                        outlier_counts[col] += original_outliers
                
                processed_symbols += 1
                
                # Hiển thị tiến độ
                if processed_symbols % 5 == 0:
                    print(f"  Đã xử lý {processed_symbols}/{len(symbols)} symbols...")
            
            print(f"\n• Đã xử lý outliers cho {processed_symbols} symbols")
            
        else:
            # Xử lý outliers chung toàn bộ dữ liệu
            for col in important_cols:
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        outlier_counts[col] = outlier_count
                        
                        # Capping outliers
                        self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                        self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                        
                elif method == 'zscore':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    outliers = self.df[np.abs((self.df[col] - mean_val) / std_val) > threshold]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        outlier_counts[col] = outlier_count
                        
                        # Capping outliers
                        self.df[col] = np.where(
                            np.abs((self.df[col] - mean_val) / std_val) > threshold,
                            mean_val + threshold * std_val * np.sign(self.df[col] - mean_val),
                            self.df[col]
                        )
        
        if outlier_counts:
            print("• Outliers được xử lý (capped):")
            total_outliers = sum(outlier_counts.values())
            total_records = len(self.df) * len(important_cols)
            
            for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(self.df)) * 100
                print(f"  - {col}: {count:,} outliers ({pct:.2f}%)")
            
            print(f"  - Tổng outliers: {total_outliers:,}")
            print(f"  - Tỷ lệ outliers: {(total_outliers/total_records*100):.2f}%")
        else:
            print("✓ Không phát hiện outliers đáng kể")
        
        return self.df
    

    
    def get_processed_data(self):
        """Trả về dữ liệu đã được xử lý đầy đủ"""
        if self.df is None:
            self.load_data()
            
        # Pipeline xử lý dữ liệu đầy đủ
        print("\n" + "="*60)
        print("BẮT ĐẦU XỬ LÝ DỮ LIỆU HOÀN CHỈNH")
        print("="*60)
        
        # 1. Tải dữ liệu
        self.load_data()
        
        # 2. Xử lý missing values
        self.handle_missing_values(strategy='knn', n_neighbors=5)
        
        # 3. Xử lý outliers
        self.detect_and_handle_outliers(method='iqr')
        
        print("\n" + "="*60)
        print("XỬ LÝ DỮ LIỆU HOÀN TẤT")
        print("="*60)
        print(f"✓ Kích thước dữ liệu cuối: {self.df.shape}")
        print(f"✓ Số features: {len(self.df.columns)}")
        print(f"✓ Số bản ghi: {len(self.df)}")
        
        return self.df

