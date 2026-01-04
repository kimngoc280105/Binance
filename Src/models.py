"""
Module chứa các functions để train và evaluate machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def train_and_evaluate_models(df_input, dataset_name="Dataset"):
    """
    Train và đánh giá các models trên một dataset
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        DataFrame đã được xử lý features với các cột cần thiết
    dataset_name : str, optional
        Tên của dataset để hiển thị (default: "Dataset")
    
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
    ...     dataset_name="Full Data"
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
    
    # Định nghĩa models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    
    # Train và đánh giá
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
    
    results_df = pd.DataFrame(results).T[['RMSE', 'R2']].sort_values(by='RMSE')
    
    return results_df, results, X_test, y_test
