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
            print(f"Đã tải {len(self.df)} bản ghi, {self.df.shape[1]} cột")
            
            # Kiểm tra tính hợp lệ của dữ liệu
            self.validate_data_integrity()
            
            return self.df
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {str(e)}")
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
                    print(f"Cảnh báo: {negative_count} giá trị âm trong cột {col}")
                    
        # Kiểm tra tính nhất quán của OHLC:
        # Điều kiện:
        #   high >= max(open, close)
        #   low  <= min(open, close)

        required_cols = ['open', 'high', 'low', 'close']

        if all(col in self.df.columns for col in required_cols):

            # Max và Min giữa open-close
            max_oc = self.df[['open', 'close']].max(axis=1)
            min_oc = self.df[['open', 'close']].min(axis=1)

            # Điều kiện sai
            invalid_high = (self.df['high'] < max_oc).sum()
            invalid_low  = (self.df['low']  > min_oc).sum()

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
        
        # 1. Thông tin cơ bản
        print(f"\n1. KÍCH THƯỚC DỮ LIỆU:")
        print(f"   • Số hàng: {self.df.shape[0]:,}")
        print(f"   • Số cột: {self.df.shape[1]}")
        print(f"   • Kích thước bộ nhớ: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 2. Kiểu dữ liệu và thông tin cột
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
        
        # 3. Thống kê mô tả chi tiết
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
            for col in numeric_cols[:]:  # Chỉ hiển thị 5 cột đầu
                skew_val = stats_df.loc[col, 'skewness']
                kurt_val = stats_df.loc[col, 'kurtosis']
                
                distribution_type = "Normal" if abs(skew_val) < 0.5 and abs(kurt_val) < 1 else "Non-normal"
                print(f"   • {col}: Skewness={skew_val:.3f}, Kurtosis={kurt_val:.3f} → {distribution_type}")
        
        # 5. Kiểm tra outliers bằng IQR
        print("\n5. PHÂN TÍCH OUTLIERS (IQR Method):")
        for col in numeric_cols[:]:  # Chỉ hiển thị 5 cột đầu
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            outlier_pct = (len(outliers) / len(self.df)) * 100
            
            if len(outliers) > 0:
                print(f"   • {col}: {len(outliers)} outliers ({outlier_pct:.2f}%) "
                      f"[{lower_bound:.4f}, {upper_bound:.4f}]")
        
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
    
    def detect_and_handle_outliers(self, method='iqr', threshold=3):
        """
        Phát hiện và xử lý outliers
        
        Parameters:
        -----------
        method : str
            'iqr', 'zscore', 'isolation'
        threshold : float
            Ngưỡng cho phương pháp zscore
        """
        print(f"\nXỬ LÝ OUTLIERS (Method: {method})")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = self.df[np.abs(z_scores) > threshold]
            
            outlier_count = len(outliers)
            if outlier_count > 0:
                outlier_counts[col] = outlier_count
                
                # Capping outliers (thay vì loại bỏ để giữ nguyên số lượng mẫu)
                if method == 'iqr':
                    self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                    self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                elif method == 'zscore':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    self.df[col] = np.where(
                        np.abs((self.df[col] - mean_val) / std_val) > threshold,
                        mean_val + threshold * std_val * np.sign(self.df[col] - mean_val),
                        self.df[col]
                    )
        
        if outlier_counts:
            print("• Outliers được xử lý (capped):")
            for col, count in outlier_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"  - {col}: {count} outliers ({pct:.2f}%)")
        else:
            print("✓ Không phát hiện outliers đáng kể")
        
        return self.df
    
    def apply_normalization(self, method='standard', columns=None):
        """
        Áp dụng normalization/standardization cho dữ liệu
        
        Parameters:
        -----------
        method : str
            'standard', 'minmax', 'robust', 'log', 'decimal'
        columns : list
            Danh sách các cột cần normalize
        """
        print(f"\nNORMALIZATION (Method: {method})")
        
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            # Loại bỏ các cột không cần normalize (như timestamp-based)
            exclude_cols = ['hour', 'day', 'month', 'year', 'day_of_week', 'is_weekend']
            columns = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            original_data = self.df[col].copy()
            
            if method == 'standard':
                # Z-score standardization (mean=0, std=1)
                scaler = StandardScaler()
                self.df[f"{col}_standardized"] = scaler.fit_transform(self.df[[col]])
                self.scalers[f"{col}_standardized"] = scaler
                
            elif method == 'minmax':
                # Min-Max scaling [0, 1]
                scaler = MinMaxScaler()
                self.df[f"{col}_minmax"] = scaler.fit_transform(self.df[[col]])
                self.scalers[f"{col}_minmax"] = scaler
                
            elif method == 'robust':
                # Robust scaling (sử dụng IQR)
                scaler = RobustScaler()
                self.df[f"{col}_robust"] = scaler.fit_transform(self.df[[col]])
                self.scalers[f"{col}_robust"] = scaler
                
            elif method == 'log':
                # Log transformation cho data skewed
                if (self.df[col] > 0).all():
                    self.df[f"{col}_log"] = np.log1p(self.df[col])
                else:
                    # Shift để tránh log(0) hoặc log(âm)
                    min_val = self.df[col].min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                        self.df[f"{col}_log"] = np.log1p(self.df[col] + shift)
                        
            elif method == 'decimal':
                # Decimal scaling
                max_abs = np.max(np.abs(self.df[col]))
                if max_abs > 0:
                    scale = 10 ** np.ceil(np.log10(max_abs))
                    self.df[f"{col}_decimal"] = self.df[col] / scale
            
            # So sánh trước/sau
            print(f"  • {col}:")
            print(f"    Original - Mean: {original_data.mean():.4f}, Std: {original_data.std():.4f}")
            if f"{col}_{method}" in self.df.columns:
                normalized = self.df[f"{col}_{method}"]
                print(f"    Normalized - Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
        
        return self.df
    
    def feature_engineering(self):
        """Tạo thêm các đặc trưng mới từ dữ liệu hiện có"""
        print("\nFEATURE ENGINEERING")
        
        if 'timestamp' not in self.df.columns:
            print("✗ Không có cột timestamp để tạo features")
            return self.df
        
        # 1. Chuyển đổi timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # 2. Tạo features thời gian
        print("• Tạo features thời gian:")
        time_features = {
            'hour': self.df['timestamp'].dt.hour,
            'day': self.df['timestamp'].dt.day,
            'month': self.df['timestamp'].dt.month,
            'year': self.df['timestamp'].dt.year,
            'day_of_week': self.df['timestamp'].dt.dayofweek,
            'quarter': self.df['timestamp'].dt.quarter,
            'week_of_year': self.df['timestamp'].dt.isocalendar().week,
            'is_weekend': self.df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int),
            'is_month_start': self.df['timestamp'].dt.is_month_start.astype(int),
            'is_month_end': self.df['timestamp'].dt.is_month_end.astype(int)
        }
        
        for name, feature in time_features.items():
            self.df[name] = feature
            print(f"  - {name}")
        
        # 3. Features giá
        print("• Tạo features giá:")
        if all(col in self.df.columns for col in ['open', 'high', 'low', 'close']):
            # Price movements
            self.df['price_change'] = self.df['close'] - self.df['open']
            self.df['price_change_pct'] = (self.df['price_change'] / self.df['open']) * 100
            
            # Price ranges
            self.df['daily_range'] = self.df['high'] - self.df['low']
            self.df['daily_range_pct'] = (self.df['daily_range'] / self.df['open']) * 100
            
            # Price position
            self.df['price_position'] = (self.df['close'] - self.df['low']) / self.df['daily_range'].replace(0, np.nan)
            
            # Typical price
            self.df['typical_price'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            
            print("  - price_change, price_change_pct, daily_range, daily_range_pct")
            print("  - price_position, typical_price")
        
        # 4. Features volume
        print("• Tạo features volume:")
        if 'volume' in self.df.columns:
            # Volume indicators
            self.df['volume_ma_5'] = self.df['volume'].rolling(window=5).mean()
            self.df['volume_ma_20'] = self.df['volume'].rolling(window=20).mean()
            self.df['volume_ratio'] = self.df['volume'] / self.df['volume_ma_20'].replace(0, np.nan)
            
            # Volume-price relationship
            if 'close' in self.df.columns:
                self.df['volume_price_trend'] = self.df['volume'] * self.df['price_change_pct']
            
            # Taker buy ratio
            if all(col in self.df.columns for col in ['taker_buy_base_volume', 'volume']):
                self.df['taker_buy_ratio'] = self.df['taker_buy_base_volume'] / self.df['volume'].replace(0, np.nan)
            
            print("  - volume_ma_5, volume_ma_20, volume_ratio")
            if 'volume_price_trend' in self.df.columns:
                print("  - volume_price_trend")
            if 'taker_buy_ratio' in self.df.columns:
                print("  - taker_buy_ratio")
        
        # 5. Technical indicators
        print("• Tạo technical indicators:")
        self._calculate_technical_indicators()
        
        # 6. Statistical features
        print("• Tạo statistical features:")
        if 'close' in self.df.columns:
            # Rolling statistics
            for window in [5, 10, 20]:
                self.df[f'returns_{window}'] = self.df['close'].pct_change(window)
                self.df[f'volatility_{window}'] = self.df[f'returns_{window}'].rolling(window).std()
                self.df[f'skewness_{window}'] = self.df['close'].rolling(window).skew()
                self.df[f'kurtosis_{window}'] = self.df['close'].rolling(window).kurt()
            
            print("  - returns_5/10/20, volatility_5/10/20")
            print("  - skewness_5/10/20, kurtosis_5/10/20")
        
        # 7. Interaction features
        print("• Tạo interaction features:")
        if all(col in self.df.columns for col in ['volume', 'daily_range_pct']):
            self.df['volume_range_interaction'] = self.df['volume'] * self.df['daily_range_pct']
            print("  - volume_range_interaction")
        
        if all(col in self.df.columns for col in ['taker_buy_ratio', 'price_change_pct']):
            self.df['taker_momentum'] = self.df['taker_buy_ratio'] * self.df['price_change_pct']
            print("  - taker_momentum")
        
        # 8. Lag features
        print("• Tạo lag features:")
        if 'close' in self.df.columns:
            for lag in [1, 2, 3, 5, 10]:
                self.df[f'close_lag_{lag}'] = self.df['close'].shift(lag)
            
            print("  - close_lag_1/2/3/5/10")
        
        # Xử lý NaN từ các features mới
        self.df = self.df.dropna()
        
        print(f"✓ Đã tạo thêm {len([col for col in self.df.columns if col not in ['timestamp', 'date', 'time', 'symbol', 'interval']])} features")
        
        return self.df
    
    def _calculate_technical_indicators(self):
        """Tính toán các chỉ số kỹ thuật"""
        if 'close' not in self.df.columns:
            return
        
        # RSI
        self._calculate_rsi()
        
        # MACD
        self._calculate_macd()
        
        # Bollinger Bands
        self._calculate_bollinger_bands()
        
        # Moving Averages
        for window in [5, 10, 20, 50]:
            self.df[f'sma_{window}'] = self.df['close'].rolling(window=window).mean()
            self.df[f'ema_{window}'] = self.df['close'].ewm(span=window, adjust=False).mean()
        
        print("  - RSI, MACD, Bollinger Bands")
        print("  - SMA/EMA 5/10/20/50")
    
    def _calculate_rsi(self, window=14):
        """Tính RSI"""
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
    
    def _calculate_macd(self):
        """Tính MACD"""
        exp1 = self.df['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = exp1 - exp2
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
    
    def _calculate_bollinger_bands(self, window=20):
        """Tính Bollinger Bands"""
        self.df['bb_middle'] = self.df['close'].rolling(window=window).mean()
        bb_std = self.df['close'].rolling(window=window).std()
        self.df['bb_upper'] = self.df['bb_middle'] + (bb_std * 2)
        self.df['bb_lower'] = self.df['bb_middle'] - (bb_std * 2)
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
    
    def perform_statistical_tests(self):
        """Thực hiện các kiểm định thống kê"""
        print("\n" + "="*60)
        print("KIỂM ĐỊNH THỐNG KÊ")
        print("="*60)
        
        if 'close' not in self.df.columns:
            print("✗ Không có dữ liệu giá để thực hiện kiểm định")
            return
        
        # 1. Kiểm định tính chuẩn của phân phối returns
        print("\n1. KIỂM ĐỊNH TÍNH CHUẨN (Normality Test):")
        
        # Tính returns
        self.df['returns'] = self.df['close'].pct_change().dropna()
        returns = self.df['returns'].dropna()
        
        if len(returns) > 0:
            # Shapiro-Wilk test
            from scipy.stats import shapiro
            
            # Chỉ lấy mẫu 5000 điểm để tránh quá tải
            sample_size = min(5000, len(returns))
            sample_returns = returns.sample(sample_size, random_state=42)
            
            stat, p_value = shapiro(sample_returns)
            
            # Giả thuyết
            print("   • Giả thuyết H0: Returns có phân phối chuẩn")
            print("   • Giả thuyết H1: Returns không có phân phối chuẩn")
            print(f"   • Shapiro-Wilk test: W={stat:.4f}, p-value={p_value:.6f}")
            
            alpha = 0.05
            if p_value > alpha:
                print(f"   • Kết luận: Không đủ bằng chứng để bác bỏ H0 (p>{alpha})")
                print("     → Returns có thể có phân phối chuẩn")
            else:
                print(f"   • Kết luận: Bác bỏ H0 (p≤{alpha})")
                print("     → Returns không có phân phối chuẩn")
        
        # 2. Kiểm định sự khác biệt giữa weekend và weekday
        print("\n2. KIỂM ĐỊNH WEEKEND VS WEEKDAY:")
        
        if 'is_weekend' in self.df.columns and 'returns' in self.df.columns:
            weekend_returns = self.df[self.df['is_weekend'] == 1]['returns'].dropna()
            weekday_returns = self.df[self.df['is_weekend'] == 0]['returns'].dropna()
            
            if len(weekend_returns) > 30 and len(weekday_returns) > 30:
                # T-test độc lập
                t_stat, p_value = ttest_ind(weekend_returns, weekday_returns, equal_var=False)
                
                # Giả thuyết
                print("   • Giả thuyết H0: Không có sự khác biệt về returns giữa weekend và weekday")
                print("   • Giả thuyết H1: Có sự khác biệt về returns giữa weekend và weekday")
                print(f"   • Welch's t-test: t={t_stat:.4f}, p-value={p_value:.6f}")
                
                alpha = 0.05
                if p_value > alpha:
                    print(f"   • Kết luận: Không đủ bằng chứng để bác bỏ H0 (p>{alpha})")
                    print("     → Không có sự khác biệt đáng kể")
                else:
                    print(f"   • Kết luận: Bác bỏ H0 (p≤{alpha})")
                    print("     → Có sự khác biệt đáng kể giữa weekend và weekday")
                    
                # Tính mean difference
                mean_diff = weekend_returns.mean() - weekday_returns.mean()
                print(f"   • Chênh lệch trung bình: {mean_diff:.6f}")
        
        # 3. Kiểm định phương sai (Volume theo giờ)
        print("\n3. KIỂM ĐỊNH PHƯƠNG SAI THEO GIỜ:")
        
        if 'hour' in self.df.columns and 'volume' in self.df.columns:
            # Nhóm theo giờ
            hourly_groups = [self.df[self.df['hour'] == h]['volume'] for h in range(24)]
            
            # ANOVA test
            f_stat, p_value = f_oneway(*hourly_groups)
            
            # Giả thuyết
            print("   • Giả thuyết H0: Không có sự khác biệt về volume trung bình giữa các giờ")
            print("   • Giả thuyết H1: Có ít nhất một giờ có volume trung bình khác biệt")
            print(f"   • One-way ANOVA: F={f_stat:.4f}, p-value={p_value:.6f}")
            
            alpha = 0.05
            if p_value > alpha:
                print(f"   • Kết luận: Không đủ bằng chứng để bác bỏ H0 (p>{alpha})")
            else:
                print(f"   • Kết luận: Bác bỏ H0 (p≤{alpha})")
                print("     → Có sự khác biệt về volume giữa các giờ")
        
        # 4. Kiểm định tương quan
        print("\n4. KIỂM ĐỊNH TƯƠNG QUAN:")
        
        if all(col in self.df.columns for col in ['volume', 'price_change_pct']):
            from scipy.stats import pearsonr, spearmanr
            
            # Pearson correlation (linear)
            pearson_corr, pearson_p = pearsonr(
                self.df['volume'].dropna(),
                self.df['price_change_pct'].dropna()
            )
            
            # Spearman correlation (monotonic)
            spearman_corr, spearman_p = spearmanr(
                self.df['volume'].dropna(),
                self.df['price_change_pct'].dropna()
            )
            
            print("   • Pearson correlation (Volume vs Price Change %):")
            print(f"     Correlation: {pearson_corr:.4f}, p-value: {pearson_p:.6f}")
            
            print("   • Spearman correlation (Volume vs Price Change %):")
            print(f"     Correlation: {spearman_corr:.4f}, p-value: {spearman_p:.6f}")
            
            # Đánh giá ý nghĩa
            alpha = 0.05
            if pearson_p < alpha:
                print(f"   • Kết luận: Có tương quan tuyến tính đáng kể (p<{alpha})")
            else:
                print(f"   • Kết luận: Không có tương quan tuyến tính đáng kể (p≥{alpha})")
    
    def apply_dimensionality_reduction(self, n_components=10, method='pca'):
        """
        Áp dụng dimensionality reduction
        
        Parameters:
        -----------
        n_components : int
            Số components cần giữ lại
        method : str
            'pca', 'tsne', 'umap'
        """
        print(f"\nDIMENSIONALITY REDUCTION (Method: {method})")
        
        # Chọn các features số để giảm chiều
        numeric_features = self.df.select_dtypes(include=[np.number]).columns
        
        # Loại bỏ các features không phù hợp
        exclude_features = ['hour', 'day', 'month', 'year', 'day_of_week', 
                          'is_weekend', 'is_month_start', 'is_month_end']
        numeric_features = [col for col in numeric_features if col not in exclude_features]
        
        if len(numeric_features) < n_components:
            print(f"✗ Số features ({len(numeric_features)}) ít hơn n_components ({n_components})")
            return self.df
        
        # Chuẩn hóa dữ liệu trước khi giảm chiều
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[numeric_features].fillna(0))
        
        if method == 'pca':
            from sklearn.decomposition import PCA
            
            self.pca = PCA(n_components=n_components, random_state=42)
            pca_result = self.pca.fit_transform(scaled_data)
            
            # Tạo tên cho các components
            for i in range(n_components):
                self.df[f'pca_component_{i+1}'] = pca_result[:, i]
            
            # Phân tích explained variance
            explained_variance = self.pca.explained_variance_ratio_
            cumulative_variance = explained_variance.cumsum()
            
            print(f"• PCA Results:")
            print(f"  - Total features: {len(numeric_features)}")
            print(f"  - Reduced to: {n_components} components")
            print(f"  - Explained variance: {cumulative_variance[-1]:.2%}")
            
            print("\n  Component Analysis:")
            for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
                print(f"    Component {i+1}: {var:.2%} (Cumulative: {cum_var:.2%})")
            
            # Phân tích feature importance trong components
            print("\n  Top Features in First 3 Components:")
            for i in range(min(3, n_components)):
                component = self.pca.components_[i]
                top_indices = np.argsort(np.abs(component))[-5:][::-1]
                print(f"    Component {i+1}:")
                for idx in top_indices:
                    feature_name = numeric_features[idx]
                    weight = component[idx]
                    print(f"      - {feature_name}: {weight:.4f}")
        
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            
            tsne = TSNE(n_components=min(3, n_components), 
                       random_state=42,
                       perplexity=30,
                       n_iter=1000)
            tsne_result = tsne.fit_transform(scaled_data)
            
            for i in range(tsne_result.shape[1]):
                self.df[f'tsne_component_{i+1}'] = tsne_result[:, i]
            
            print(f"• t-SNE Results:")
            print(f"  - Reduced to: {tsne_result.shape[1]} components")
        
        return self.df
    
    def optimize_numerical_computations(self):
        """Tối ưu hóa tính toán số học và giảm sai số"""
        print("\nTỐI ƯU HÓA TÍNH TOÁN SỐ HỌC")
        
        # 1. Giảm độ chính xác để tăng tốc độ tính toán
        print("1. Giảm độ chính xác số học:")
        float_cols = self.df.select_dtypes(include=['float64']).columns
        
        for col in float_cols:
            # Chuyển từ float64 sang float32 nếu phù hợp
            if self.df[col].dtype == 'float64':
                max_val = self.df[col].abs().max()
                min_val = self.df[col].abs().min()
                
                # Kiểm tra xem float32 có đủ không
                if max_val < 3.4e38 and min_val > 1.2e-38:
                    self.df[col] = self.df[col].astype('float32')
                    print(f"  • {col}: float64 → float32")
        
        # 2. Tính toán với độ ổn định số học cao hơn
        print("\n2. Cải thiện độ ổn định số học:")
        
        if all(col in self.df.columns for col in ['close', 'open']):
            # Sử dụng log difference thay vì percentage change cho tính ổn định
            self.df['log_returns'] = np.log(self.df['close'] / self.df['open'])
            
            # Sử dụng Kahan summation để giảm sai số tích lũy
            def kahan_sum(values):
                """Kahan summation algorithm giảm sai số làm tròn"""
                total = 0.0
                compensation = 0.0
                for val in values:
                    y = val - compensation
                    t = total + y
                    compensation = (t - total) - y
                    total = t
                return total
            
            # Tính tổng với Kahan summation
            sample_values = self.df['log_returns'].dropna().values[:1000]
            regular_sum = np.sum(sample_values)
            kahan_sum_result = kahan_sum(sample_values)
            
            print(f"  • Regular sum: {regular_sum:.12f}")
            print(f"  • Kahan sum:   {kahan_sum_result:.12f}")
            print(f"  • Difference:  {abs(regular_sum - kahan_sum_result):.2e}")
        
        # 3. Sử dụng compensated algorithms
        print("\n3. Sử dụng compensated algorithms:")
        
        if 'volume' in self.df.columns:
            # Tính variance với compensated algorithm
            def online_variance(data):
                """Welford's online algorithm cho variance"""
                n = 0
                mean = 0.0
                M2 = 0.0
                
                for x in data:
                    n += 1
                    delta = x - mean
                    mean += delta / n
                    delta2 = x - mean
                    M2 += delta * delta2
                
                if n < 2:
                    return float('nan')
                else:
                    return M2 / (n - 1)
            
            volume_data = self.df['volume'].dropna().values[:10000]
            numpy_var = np.var(volume_data, ddof=1)
            online_var = online_variance(volume_data)
            
            print(f"  • NumPy variance: {numpy_var:.4f}")
            print(f"  • Online variance: {online_var:.4f}")
            print(f"  • Relative error: {abs(numpy_var - online_var)/numpy_var:.2e}")
        
        # 4. Tính toán với extended precision khi cần
        print("\n4. Extended precision tính toán:")
        
        if all(col in self.df.columns for col in ['taker_buy_base_volume', 'volume']):
            # Tính toán ratio với precision cao hơn
            self.df['taker_buy_ratio_precise'] = (
                self.df['taker_buy_base_volume'].astype('float64') / 
                self.df['volume'].astype('float64').replace(0, np.nan)
            )
            
            print(f"  • Taker buy ratio calculated with higher precision")
    
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
        
        # 2. Feature engineering
        self.feature_engineering()
        
        # 3. Xử lý missing values
        self.handle_missing_values(strategy='knn', n_neighbors=5)
        
        # 4. Xử lý outliers
        self.detect_and_handle_outliers(method='iqr')
        
        # 5. Normalization
        self.apply_normalization(method='standard')
        self.apply_normalization(method='log')
        
        # 6. Tối ưu tính toán số học
        self.optimize_numerical_computations()
        
        # 7. Kiểm định thống kê
        self.perform_statistical_tests()
        
        # 8. Dimensionality reduction (tùy chọn)
        # self.apply_dimensionality_reduction(n_components=10, method='pca')
        
        print("\n" + "="*60)
        print("XỬ LÝ DỮ LIỆU HOÀN TẤT")
        print("="*60)
        print(f"✓ Kích thước dữ liệu cuối: {self.df.shape}")
        print(f"✓ Số features: {len(self.df.columns)}")
        print(f"✓ Số bản ghi: {len(self.df)}")
        
        return self.df

