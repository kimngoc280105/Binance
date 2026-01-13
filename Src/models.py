"""
Module chứa các functions để train và evaluate machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def train_and_evaluate_models(df_input, dataset_name="Dataset", include_arima=False, arima_order=(5, 1, 0)):
    """
    Train và đánh giá các models trên một dataset
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        DataFrame đã được xử lý features với các cột cần thiết
    dataset_name : str, optional
        Tên của dataset để hiển thị (default: "Dataset")
    include_arima : bool, optional
        Có bao gồm ARIMA model hay không (default: False)
        Lưu ý: ARIMA với rolling forecast sẽ chạy lâu hơn
    arima_order : tuple, optional
        Tham số (p, d, q) cho ARIMA model (default: (5, 1, 0))
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame chứa kết quả đánh giá (RMSE, R2) cho từng model
    results : dict
        Dictionary chứa chi tiết kết quả (RMSE, R2, Predictions) cho từng model
    X_test : pd.DataFrame
        Features của tập test
    y_test : pd.Series
        Target của tập test
        
    Example:
    --------
    >>> results_df, results, X_test, y_test = train_and_evaluate_models(
    ...     df, 
    ...     dataset_name="Full Data",
    ...     include_arima=True
    ... )
    """
    print(f"\n{'='*60}")
    print(f"Training Models on: {dataset_name}")
    print(f"{'='*60}")
    
    # Chuẩn bị features và target
    features = ['close_lag1', 'close_lag2', 'MA_24', 'volatility', 
                'taker_buy_ratio', 'volume', 'trades_count']
    target = 'Target'
    
    X = df_input[features]
    y = df_input[target]
    
    # Chia train/test (80/20)
    split_point = int(len(df_input) * 0.8)
    
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")
    print(f"Time range: {df_input.index.min()} to {df_input.index.max()}")
    
    # Định nghĩa ML models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    
    # Train và đánh giá ML models
    results = {}
    print(f"\n{'Model':<20} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 45)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"RMSE": rmse, "R2": r2, "Prediction": y_pred}
        print(f"{name:<20} | {rmse:<10.4f} | {r2:<10.4f}")
    
    # Train ARIMA nếu được yêu cầu
    if include_arima:
        print(f"\n--- Training ARIMA{arima_order} (Rolling Forecast) ---")
        
        # Lấy chuỗi close để dự báo
        close_series = df_input['close'].copy()
        train_ts = close_series.iloc[:split_point]
        test_ts = close_series.iloc[split_point:]
        
        # Rolling Forecast
        history = list(train_ts.values)
        predictions_list = []
        
        print(f"Đang thực hiện Rolling Forecast cho {len(test_ts)} bước...")
        
        for t in range(len(test_ts)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit()
            yhat = model_fit.forecast(steps=1)[0]
            predictions_list.append(yhat)
            history.append(test_ts.iloc[t])
            
            if (t + 1) % 50 == 0:
                print(f"  Đã dự báo {t + 1}/{len(test_ts)} bước...")
        
        arima_pred = np.array(predictions_list)
        
        # ARIMA dự báo giá Close hiện tại, không phải Target (giá tiếp theo)
        # Nên ta shift lại để so sánh đúng với y_test
        arima_rmse = np.sqrt(mean_squared_error(test_ts.values, arima_pred))
        arima_r2 = r2_score(test_ts.values, arima_pred)
        
        arima_name = f'ARIMA{arima_order}'
        results[arima_name] = {"RMSE": arima_rmse, "R2": arima_r2, "Prediction": arima_pred}
        print(f"{arima_name:<20} | {arima_rmse:<10.4f} | {arima_r2:<10.4f}")
    
    results_df = pd.DataFrame(results).T[['RMSE', 'R2']].sort_values(by='RMSE')
    
    return results_df, results, X_test, y_test


def train_arima_model(df_input, order=(5, 1, 0), dataset_name="Dataset", use_rolling=True):
    """
    Train và đánh giá ARIMA model cho time series forecasting
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        DataFrame đã được xử lý với cột 'close' và 'Target'
    order : tuple, optional
        Tham số (p, d, q) cho ARIMA model (default: (5, 1, 0))
        - p: Số lag của AR (AutoRegressive)
        - d: Bậc sai phân (Differencing)
        - q: Số lag của MA (Moving Average)
    dataset_name : str, optional
        Tên của dataset để hiển thị (default: "Dataset")
    use_rolling : bool, optional
        Sử dụng rolling forecast thay vì multi-step forecast (default: True)
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame chứa kết quả đánh giá (RMSE, R2, MAE, MAPE)
    arima_fitted : ARIMAResults
        Fitted ARIMA model
    test_ts : pd.Series
        Chuỗi giá thực tế (test set)
    predictions : pd.Series
        Chuỗi giá dự báo
        
    Example:
    --------
    >>> results_df, model, test_ts, predictions = train_arima_model(
    ...     df, 
    ...     order=(5, 1, 0),
    ...     dataset_name="ETHUSDT"
    ... )
    """
    print(f"\n{'='*60}")
    print(f"Training ARIMA Model on: {dataset_name}")
    print(f"{'='*60}")
    
    # Lấy chuỗi giá Close để dự báo
    close_series = df_input['close'].copy()
    
    # Chia train/test theo thời gian (80/20)
    split_idx = int(len(close_series) * 0.8)
    train_ts = close_series.iloc[:split_idx]
    test_ts = close_series.iloc[split_idx:]
    
    print(f"Train size: {len(train_ts)}")
    print(f"Test size: {len(test_ts)}")
    print(f"Time range: {df_input.index.min()} to {df_input.index.max()}")
    print(f"ARIMA order: (p={order[0]}, d={order[1]}, q={order[2]})")
    print(f"Rolling Forecast: {use_rolling}")
    
    if use_rolling:
        # Rolling Forecast - dự báo từng bước một
        history = list(train_ts.values)
        predictions_list = []
        
        print(f"\nĐang thực hiện Rolling Forecast cho {len(test_ts)} bước...")
        
        for t in range(len(test_ts)):
            # Fit model trên history hiện tại
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            
            # Dự báo 1 bước tiếp theo
            yhat = model_fit.forecast(steps=1)[0]
            predictions_list.append(yhat)
            
            # Thêm giá trị thực tế vào history để dự báo bước tiếp
            history.append(test_ts.iloc[t])
            
            # Progress
            if (t + 1) % 50 == 0:
                print(f"  Đã dự báo {t + 1}/{len(test_ts)} bước...")
        
        predictions = pd.Series(predictions_list, index=test_ts.index)
        arima_fitted = model_fit  # Model cuối cùng
        
    else:
        # Multi-step Forecast (có thể ra đường thẳng)
        arima_model = ARIMA(train_ts, order=order)
        arima_fitted = arima_model.fit()
        predictions = arima_fitted.forecast(steps=len(test_ts))
        predictions.index = test_ts.index
    
    # Tính các metrics
    rmse = np.sqrt(mean_squared_error(test_ts, predictions))
    r2 = r2_score(test_ts, predictions)
    mae = np.mean(np.abs(test_ts.values - predictions.values))
    mape = np.mean(np.abs((test_ts.values - predictions.values) / test_ts.values)) * 100
    
    print(f"\n{'Model':<20} | {'RMSE':<10} | {'R2 Score':<10} | {'MAE':<10} | {'MAPE (%)':<10}")
    print("-" * 70)
    print(f"{'ARIMA' + str(order):<20} | {rmse:<10.4f} | {r2:<10.4f} | {mae:<10.4f} | {mape:<10.4f}")
    
    # Tạo results DataFrame
    results = {
        'Model': [f'ARIMA{order}'],
        'RMSE': [rmse],
        'R2': [r2],
        'MAE': [mae],
        'MAPE (%)': [mape]
    }
    results_df = pd.DataFrame(results).set_index('Model')
    
    # In summary từ ARIMA
    print(f"\n--- ARIMA Model Summary ---")
    print(f"AIC: {arima_fitted.aic:.4f}")
    print(f"BIC: {arima_fitted.bic:.4f}")
    
    return results_df, arima_fitted, test_ts, predictions
