project/
│
├── data/
│ ├── raw/
│ │ └── crypto_hourly_data.csv
│ │
│ ├── processed/
│ │ └── crypto_clean.csv
│ │
│ └── external/ # File tham khảo nếu có (optional)
│
├── src/
│ ├── data_collection.py # lấy dữ liệu từ API (requests/selenium)
│ ├── preprocessing.py # xử lý missing, outlier, formatting
│ ├── feature_engineering.py # tạo feature mới
│ ├── modeling.py # chạy model ML + evaluate
│ ├── utils.py # hàm phụ trợ
│ └── **init**.py
│
├── notebooks/
│ ├── 01_data_collection.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_data_analysis.ipynb # phần câu hỏi meaningful questions
│ ├── 04_feature_engineering.ipynb
│ ├── 05_modeling.ipynb
│ └── 06_final_summary.ipynb
│
├── reports/
│ ├── presentation/ # slide để thuyết trình
│ │ └── final_presentation.pdf
│ │
│ └── final_report/ # phần reflection + kết luận
│ └── project_report.pdf
│
├── models/
│ ├── trained_models.pkl # các model đã train
│ └── model_results.csv
│
├── requirements.txt # danh sách thư viện dùng để chạy project
├── README.md # mô tả tổng quan project cho GitHub
└── .gitignore # bỏ bớt file không commit
