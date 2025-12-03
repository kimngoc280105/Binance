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
                    
        # Kiểm tra tính nhất quán: high >= low, high >= close, low <= close
        if all(col in self.df.columns for col in ['high', 'low', 'close']):
            invalid_high_low = (self.df['high'] < self.df['low']).sum()
            invalid_high_close = (self.df['high'] < self.df['close']).sum()
            invalid_low_close = (self.df['low'] > self.df['close']).sum()
            
            if invalid_high_low > 0:
                print(f"⚠ Cảnh báo: {invalid_high_low} bản ghi có high < low")
            if invalid_high_close > 0:
                print(f"⚠ Cảnh báo: {invalid_high_close} bản ghi có high < close")
            if invalid_low_close > 0:
                print(f"⚠ Cảnh báo: {invalid_low_close} bản ghi có low > close")
    
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
            for col in numeric_cols[:5]:  # Chỉ hiển thị 5 cột đầu
                skew_val = stats_df.loc[col, 'skewness']
                kurt_val = stats_df.loc[col, 'kurtosis']
                
                distribution_type = "Normal" if abs(skew_val) < 0.5 and abs(kurt_val) < 1 else "Non-normal"
                print(f"   • {col}: Skewness={skew_val:.3f}, Kurtosis={kurt_val:.3f} → {distribution_type}")
        
        # 5. Kiểm tra outliers bằng IQR
        print("\n5. PHÂN TÍCH OUTLIERS (IQR Method):")
        for col in numeric_cols[:5]:  # Chỉ hiển thị 5 cột đầu
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
    