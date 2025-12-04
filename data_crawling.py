# ============================================================
# PHASE 1: THU THẬP DỮ LIỆU TỪ BINANCE API
# Mục tiêu: Lấy >10,000 records với >10 attributes
# ============================================================

import requests
import pandas as pd
import time
from datetime import datetime
import os

class CryptoDataCollector:
    """Thu thập dữ liệu crypto từ Binance API"""
    
    def __init__(self):
        self.base_url = "https://data-api.binance.vision"
        
    def get_top_symbols(self, top_n=50):
        """Lấy top N trading pairs theo volume"""
        print("🔍 Đang tìm top trading pairs...")
        
        url = f"{self.base_url}/api/v3/ticker/24hr"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Lọc chỉ lấy USDT pairs
            df = df[df['symbol'].str.endswith('USDT')]
            df['quoteVolume'] = pd.to_numeric(df['quoteVolume'])
            
            # Lấy top theo volume
            top_symbols = df.nlargest(top_n, 'quoteVolume')['symbol'].tolist()
            
            print(f"Lấy được top {len(top_symbols)} pairs")
            print(f"Top 10: {top_symbols[:10]}")
            
            return top_symbols
        else:
            print(f"Lỗi: {response.status_code}")
            return []
    
    def get_historical_data(self, symbol, interval='1h', limit=1000):
        """
        Lấy dữ liệu lịch sử của 1 symbol
        interval: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
        limit: max 1000 per request
        """
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            klines = response.json()
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            # Chuyển đổi kiểu dữ liệu
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.time
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'trades_count',
                          'taker_buy_base_volume', 'taker_buy_quote_volume']
            
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['symbol'] = symbol
            df['interval'] = interval
            
            # Bỏ cột không cần thiết
            df = df.drop(['open_time', 'close_time', 'ignore'], axis=1)
            
            return df
        else:
            return None
    
    def collect_multiple_symbols(self, symbols, interval='1h', limit=1000):
        """Thu thập dữ liệu nhiều symbols"""
        print("\n" + "="*70)
        print("BẮT ĐẦU THU THẬP DỮ LIỆU")
        print("="*70)
        print(f"Số symbols: {len(symbols)}")
        print(f"Interval: {interval}")
        print(f"Records mỗi symbol: {limit}")
        print(f"Tổng dự kiến: ~{len(symbols) * limit:,} records")
        print()
        
        all_data = []
        success = 0
        failed = 0
        
        for idx, symbol in enumerate(symbols, 1):
            print(f"[{idx}/{len(symbols)}] {symbol}...", end=' ')
            
            df = self.get_historical_data(symbol, interval, limit)
            
            if df is not None and len(df) > 0:
                all_data.append(df)
                success += 1
                print(f"{len(df)} records")
            else:
                failed += 1
                print("Lỗi khi lấy dữ liệu")
            
            time.sleep(0.15)  # Rate limit protection
            
            # Progress update mỗi 10 symbols
            if idx % 10 == 0:
                total_records = sum(len(df) for df in all_data)
                print(f"   📊 Tiến độ: {success} thành công | {total_records:,} records\n")
        
        print("\n" + "="*70)
        print(f"Hoàn thành: {success}/{len(symbols)} symbols")
        print(f"Thất bại: {failed}")
        print("="*70)
        
        if all_data:
            df_final = pd.concat(all_data, ignore_index=True)
            return df_final
        return None
    
    def save_data(self, df, filename='crypto_raw_data.csv'):
        """Lưu dữ liệu vào CSV"""
        if df is None or len(df) == 0:
            print("Không có dữ liệu để lưu!")
            return
        
        # Tạo thư mục nếu chưa có
        os.makedirs('data', exist_ok=True)
        
        filepath = f'data/{filename}'
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*70}")
        print(f"ĐÃ LƯU DỮ LIỆU THÀNH CÔNG!")
        print(f"{'='*70}")
        print(f"File: {filepath}")
        print(f"Tổng records: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"Số symbols: {df['symbol'].nunique()}")
        print(f"Từ {df['timestamp'].min()} đến {df['timestamp'].max()}")
        print(f"{'='*70}")
        
        # Thống kê cơ bản
        print("\nTHỐNG KÊ CƠ BẢN:")
        print(f"   • Kích thước file: ~{os.path.getsize(filepath) / (1024*1024):.2f} MB")
        print(f"   • Giá trị null: {df.isnull().sum().sum()}")
        print(f"   • Records/symbol: {len(df) // df['symbol'].nunique():.0f}")
        
        return filepath
# ============================================================
# SỬ DỤNG
# ============================================================

if __name__ == "__main__":
    
    collector = CryptoDataCollector()
    
    # Bước 1: Lấy top symbols
    top_symbols = collector.get_top_symbols(top_n=30)
    
    if not top_symbols:
        print("Không lấy được danh sách symbols!")
        exit()
    
    # Bước 2: Thu thập dữ liệu
    # OPTION A: Hourly data (nhiều records hơn)
    # 30 symbols × 1000 hours = 30,000 records
    df_hourly = collector.collect_multiple_symbols(
        symbols=top_symbols,
        interval='1h',
        limit=1000
    )
    
    if df_hourly is not None:
        collector.save_data(df_hourly, 'crypto_hourly_data.csv')
        
        print("\nPREVIEW DỮ LIỆU:")
        print(df_hourly[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']].head())
        
        print("\nMÔ TẢ THỐNG KÊ:")
        print(df_hourly[['open', 'high', 'low', 'close', 'volume']].describe())
    