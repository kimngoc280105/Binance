# Binance Crypto Data Analysis & Prediction

Đồ án phân tích dữ liệu thị trường Crypto (Binance) và dự đoán giá sử dụng các mô hình Machine Learning & Time Series. Dự án tập trung vào việc khai thác dữ liệu giao dịch theo giờ (hourly data) của 30 loại tiền mã hóa khác nhau để tìm ra các quy luật biến động và xây dựng mô hình dự báo hiệu quả.

## Thông Tin Nhóm
*(Vui lòng điền tên thành viên nhóm vào đây)*
*   Thành viên 1: 23120047 - Nguyễn Gia Huy
*   Thành viên 2: 23120062 - Trần Kim Ngọc
*   Thành viên 3: 23120063 - Nguyễn Thành Nguyên
*   Thành viên 4: 23120084 - Nguyễn Mạnh Thắng

## Dataset: Binance Hourly Data

Dữ liệu được thu thập từ sàn giao dịch Binance với các thông số chi tiết sau:

*   **Nguồn dữ liệu:** Binance API (Spot Market)
*   **Khoảng thời gian:** 2025-10-23 đến 2025-12-04 (42 ngày)
*   **Số lượng Coins:** 30 cặp giao dịch (VD: BTCUSDT, ETHUSDT, PEPEUSDT, SAPIENUSDT...)
*   **Tổng số bản ghi:** ~28,806 dòng
*   **Đặc trưng (Columns):** 14 trường thông tin bao gồm OHLC (Open, High, Low, Close), Volume, Quote Asset Volume, Number of Trades, Taker Buy Base/Quote Asset Volume.

## Câu Hỏi Nghiên Cứu (Research Questions)

Dự án tập trung trả lời 6 câu hỏi chính để hiểu sâu hơn về hành vi thị trường:

1.  **Volatility Analysis:** Trong 30 coins được quan sát, coin nào có độ biến động giá cao nhất (rủi ro/lợi nhuận cao) và thấp nhất (ổn định) trong giai đoạn 42 ngày?
2.  **Volume & Amplitude:** Có mối tương quan như thế nào giữa volume giao dịch và biên độ dao động giá (High - Low) trong mỗi giờ?
3.  **Active Hours:** Khung giờ nào trong ngày (0-23h) có volume giao dịch cao nhất? Có pattern đặc biệt nào không?
4.  **Market Pressure:** Tỷ lệ người mua chủ động vs bán chủ động (Taker Buy Ratio) ảnh hưởng thế nào đến chiều hướng tăng/giảm giá?
5.  **Bitcoin Correlation:** Trong các altcoins, đồng coin nào biến động giá giống Bitcoin nhất (tương quan dương mạnh) và khác Bitcoin nhất?
6.  **Liquidity:** Trong 30 coins, coins nào có số lượng giao dịch (trades count) cao nhất? Có mối liên hệ gì giữa số lượng giao dịch và tính thanh khoản?

## Tóm Tắt Kết Quả (Key Findings)

Một số điểm nổi bật rút ra từ quá trình phân tích:

*   **Biến động giá (Volatility):**
    *   **Cao nhất:** `SAPIENUSDT` (CV ~35%) - Phù hợp cho đầu tư mạo hiểm ngắn hạn.
    *   **Thấp nhất:** `BFUSDUSDT` & `USDCUSDT` (CV ~0.03%) - Các Stablecoins, phù hợp để trú ẩn an toàn.
*   **Mối tương quan (Correlation):** Dữ liệu cho thấy sự phân hóa rõ rệt giữa nhóm stablecoins và các coins biến động mạnh (meme coins/new coins).
*   *(Các kết quả chi tiết khác về khung giờ giao dịch và ảnh hưởng của Taker Buy Volume được trình bày chi tiết trong notebook số 02)*

## Cấu Trúc Dự Án

```
Binance/
├── Data/                   # Thư mục chứa dữ liệu
│   ├── raw/                # Dữ liệu thô (crypto_hourly_data.csv)
│   └── processed/          # Dữ liệu đã qua xử lý
├── Notebooks/              # Jupyter Notebooks
│   ├── 01_exploration.ipynb    # Data Loading, Cleaning, EDA cơ bản
│   ├── 02_question.ipynb       # Trả lời 6 câu hỏi nghiên cứu chi tiết
│   └── 03_modeling.ipynb       # Feature Engineering & Model Training (LR, XGB, ARIMA)
├── Src/                    # Source code Python
│   ├── data_exploration.py     # Class Helper cho EDA
│   ├── visualization.py        # Class Helper cho trực quan hóa (Plotly/Matplotlib)
│   └── models.py               # Model Pipeline
├── data_crawling.py        # Script thu thập dữ liệu
├── Requirements.txt        # Các thư viện phụ thuộc
├── Slides.pdf              # Slide báo cáo
└── README.md               # Tài liệu này
```

## Hướng Dẫn Cài Đặt & Chạy

1.  **Cài đặt môi trường:**
    ```bash
    pip install -r Requirements.txt
    ```

2.  **Thu thập dữ liệu (Optional):**
    Nếu muốn cập nhật dữ liệu mới nhất:
    ```bash
    python data_crawling.py
    ```

3.  **Chạy phân tích:**
    Mở và chạy lần lượt các Notebooks theo thứ tự 01 -> 03 để tái hiện toàn bộ quá trình phân tích và dự báo.

## Dependencies

Dự án sử dụng các thư viện chính: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `statsmodels`, `matplotlib`, `seaborn`, `plotly`.
Chi tiết xem tại `Requirements.txt`.
