# src/data_processing.py
"""
Data processing utilities for time-series / tabular projects.

Contains:
- Loading / type coercion / basic validation
- Missing value handling (simple + KNN + model-based skeleton)
- Outlier detection & removal (z-score, IQR, robust clipping)
- Transformations: log1p, decimal scaling, min-max, standardization (z-score)
- Numerically-stable financial computations (log returns)
- Feature engineering: time features, lag features, rolling stats, interactions
- Dimensionality reduction (PCA wrapper)
- Basic statistical tests: t-test (1-sample), Shapiro normality, Wilcoxon, ADF (if available)
- Persistence helpers (joblib)
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import pandas as pd
import math
import joblib
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# optional stats imports
try:
    from scipy import stats
except Exception:
    stats = None

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:
    adfuller = None


# ---------------------------
# Loading & basic validation
# ---------------------------
def load_csv(path: str, parse_dates: Optional[List[str]] = None, **pd_kwargs) -> pd.DataFrame:
    """
    Load CSV into DataFrame, parse dates if provided, sort by first parse_date if any.
    """
    df = pd.read_csv(path, parse_dates=parse_dates, **pd_kwargs)
    if parse_dates:
        df = df.sort_values(by=parse_dates[0]).reset_index(drop=True)
    return df


def ensure_numeric(df: pd.DataFrame, cols: List[str], inplace: bool = True) -> pd.DataFrame:
    """
    Coerce given columns to numeric (float). Non-coercible entries become NaN.
    """
    df_out = df if inplace else df.copy()
    for c in cols:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
    return df_out


def summary_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarizing missing counts and percentages per column.
    """
    s = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_pct': (df.isna().sum() / len(df)).round(4)
    })
    return s.sort_values('missing_pct', ascending=False)


# ---------------------------
# Numeric-stable computations
# ---------------------------
def safe_pct_change(series: pd.Series) -> pd.Series:
    """
    Compute percentage change safely: (x_t / x_{t-1}) - 1, with clipping to avoid inf/NaN.
    """
    s = series.astype(float)
    prev = s.shift(1)
    res = np.where(prev == 0, np.nan, s / prev - 1)
    return pd.Series(res, index=series.index)


def log_return(series: pd.Series) -> pd.Series:
    """
    Compute log returns: ln(x_t) - ln(x_{t-1}); use log1p if values may be near zero.
    Preferable for numerical stability when chaining returns.
    """
    s = series.astype(float)
    # require positive values; if zeros or negatives present, use log1p on clipped positive.
    if (s <= 0).any():
        # shift by small epsilon to keep positive (document this decision)
        eps = 1e-9
        s = s.clip(lower=eps)
    return np.log(s).diff()


# ---------------------------
# Outlier detection & handling
# ---------------------------
def detect_outliers_zscore(series: pd.Series, thresh: float = 3.0) -> pd.Series:
    """
    Return boolean mask where absolute z-score > thresh.
    Uses population std (ddof=0).
    """
    s = series.dropna().astype(float)
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=series.index)
    z = (series - mean) / std
    return z.abs() > thresh


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Return boolean mask where value is outside [Q1 - k*IQR, Q3 + k*IQR].
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return (series < low) | (series > high)


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', thresh: float = 3.0,
                    inplace: bool = False) -> pd.DataFrame:
    """
    Remove rows flagged as outliers for `column`.
    method: 'iqr' or 'zscore'
    """
    df_out = df if inplace else df.copy()
    if column not in df_out.columns:
        return df_out
    col = df_out[column]
    if method == 'zscore':
        mask = detect_outliers_zscore(col, thresh=thresh)
    elif method == 'iqr':
        mask = detect_outliers_iqr(col, k=thresh)
    else:
        raise ValueError("method must be 'zscore' or 'iqr'")
    # keep rows that are NOT outliers
    return df_out.loc[~mask].reset_index(drop=True)


def clip_extremes(df: pd.DataFrame, column: str, lower: Optional[float] = None, upper: Optional[float] = None,
                  inplace: bool = False) -> pd.DataFrame:
    """
    Clip values in column to [lower, upper] if provided.
    """
    df_out = df if inplace else df.copy()
    if column in df_out.columns:
        df_out[column] = df_out[column].clip(lower=lower, upper=upper)
    return df_out


# ---------------------------
# Missing value imputation
# ---------------------------
def impute_simple(df: pd.DataFrame, strategy: str = 'median', columns: Optional[List[str]] = None,
                  inplace: bool = False) -> pd.DataFrame:
    """
    Impute numeric columns using simple strategies: 'mean', 'median', 'constant', 'ffill', 'bfill'.
    Returns DataFrame with imputed values.
    """
    df_out = df if inplace else df.copy()
    cols = columns or df_out.select_dtypes(include=[np.number]).columns.tolist()
    if strategy in ('mean', 'median', 'constant'):
        imp = SimpleImputer(strategy=strategy if strategy != 'constant' else 'constant', fill_value=0)
        df_out[cols] = imp.fit_transform(df_out[cols])
    elif strategy == 'ffill':
        df_out[cols] = df_out[cols].fillna(method='ffill').fillna(method='bfill')
    else:
        raise ValueError("Unsupported strategy: choose 'mean','median','constant','ffill'")
    return df_out


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5, columns: Optional[List[str]] = None,
               inplace: bool = False) -> pd.DataFrame:
    """
    KNN imputation for numeric columns. Beware: this uses Euclidean distance on scaled features by default.
    """
    df_out = df if inplace else df.copy()
    cols = columns or df_out.select_dtypes(include=[np.number]).columns.tolist()
    if not cols:
        return df_out
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_out[cols] = imputer.fit_transform(df_out[cols])
    return df_out


def impute_model_predictor(df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]] = None,
                           model: Optional[Any] = None, test_size: float = 0.2, random_state: int = 42,
                           inplace: bool = False) -> pd.DataFrame:
    """
    A model-based imputer: train a regressor to predict missing values of target_col using feature_cols.
    Steps:
      - Use rows where target_col is present to train the model.
      - Predict on rows where it's missing and fill them.
    If no model provided, use RandomForestRegressor (robust).
    Warning: only use if missingness is predictable and not MCAR.
    """
    df_out = df if inplace else df.copy()
    if feature_cols is None:
        # use all numeric except the target
        feature_cols = [c for c in df_out.select_dtypes(include=[np.number]).columns if c != target_col]
    # rows with known target
    train_idx = df_out[df_out[target_col].notna()].index
    predict_idx = df_out[df_out[target_col].isna()].index
    if len(predict_idx) == 0:
        return df_out  # nothing to do
    X_train = df_out.loc[train_idx, feature_cols].copy()
    y_train = df_out.loc[train_idx, target_col].copy()
    X_pred = df_out.loc[predict_idx, feature_cols].copy()
    # drop rows with NaN in features (could impute them first)
    X_train = X_train.fillna(X_train.median())
    X_pred = X_pred.fillna(X_train.median())
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_pred)
    df_out.loc[predict_idx, target_col] = preds
    return df_out


# ---------------------------
# Transformations: Normalization & Scaling
# ---------------------------
def transform_log1p(df: pd.DataFrame, columns: List[str], suffix: str = '_log1p', inplace: bool = False) -> pd.DataFrame:
    """
    Create log1p transformed columns to handle skewed (positive) distributions.
    Uses np.log1p to support zero values.
    """
    df_out = df if inplace else df.copy()
    for c in columns:
        if c in df_out.columns:
            df_out[c + suffix] = np.log1p(df_out[c].clip(lower=0))
    return df_out


def transform_decimal_scaling(df: pd.DataFrame, columns: List[str], inplace: bool = False) -> pd.DataFrame:
    """
    Decimal scaling: scale by 10^j where j = ceil(log10(max(abs(x))))
    New column name: col + '_decscaled'
    """
    df_out = df if inplace else df.copy()
    for c in columns:
        if c in df_out.columns:
            max_abs = df_out[c].abs().max(skipna=True)
            if pd.isna(max_abs) or max_abs == 0:
                df_out[c + '_decscaled'] = df_out[c]
            else:
                j = math.ceil(math.log10(max_abs + 1e-12))
                df_out[c + '_decscaled'] = df_out[c] / (10 ** j)
    return df_out


def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Fit scaler on X_train and transform X_train & X_test. Return (X_train_scaled, X_test_scaled, scaler_object).
    method: 'standard' (z-score) or 'minmax'.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    scaler.fit(X_train)
    Xtr = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    Xte = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return Xtr, Xte, scaler


# ---------------------------
# Feature engineering helpers
# ---------------------------
def build_time_features(df: pd.DataFrame, time_col: str = 'timestamp', inplace: bool = False) -> pd.DataFrame:
    """
    Add hour, dayofweek, day, month, is_weekend features.
    """
    df_out = df if inplace else df.copy()
    t = pd.to_datetime(df_out[time_col])
    df_out['hour'] = t.dt.hour
    df_out['dayofweek'] = t.dt.dayofweek
    df_out['day'] = t.dt.day
    df_out['month'] = t.dt.month
    df_out['is_weekend'] = (df_out['dayofweek'] >= 5).astype(int)
    return df_out


def add_lag_features(df: pd.DataFrame, column: str = 'close', lags: List[int] = [1, 2, 3, 6, 12, 24],
                     inplace: bool = False) -> pd.DataFrame:
    """
    Add lag features for a numeric column.
    """
    df_out = df if inplace else df.copy()
    for lag in lags:
        df_out[f'{column}_lag_{lag}'] = df_out[column].shift(lag)
    return df_out


def add_rolling_features(df: pd.DataFrame, column: str = 'close', windows: List[int] = [12, 24, 72],
                         stats: List[str] = ['mean', 'std', 'min', 'max'], inplace: bool = False) -> pd.DataFrame:
    """
    Add rolling statistics. windows in number of rows (e.g. hours).
    """
    df_out = df if inplace else df.copy()
    for w in windows:
        ro = df_out[column].rolling(window=w, min_periods=1)
        if 'mean' in stats:
            df_out[f'{column}_roll_mean_{w}'] = ro.mean()
        if 'std' in stats:
            df_out[f'{column}_roll_std_{w}'] = ro.std()
        if 'min' in stats:
            df_out[f'{column}_roll_min_{w}'] = ro.min()
        if 'max' in stats:
            df_out[f'{column}_roll_max_{w}'] = ro.max()
    return df_out


def add_interaction_features(df: pd.DataFrame, pairs: List[Tuple[str, str]], op: str = 'mul', inplace: bool = False) -> pd.DataFrame:
    """
    Add interaction features for given pairs of columns.
    op: 'mul' (product) or 'div' (safe division: a / (b + eps)).
    """
    df_out = df if inplace else df.copy()
    eps = 1e-9
    for a, b in pairs:
        if a in df_out.columns and b in df_out.columns:
            if op == 'mul':
                df_out[f'{a}_x_{b}'] = df_out[a] * df_out[b]
            elif op == 'div':
                df_out[f'{a}_div_{b}'] = df_out[a] / (df_out[b] + eps)
            else:
                raise ValueError("op must be 'mul' or 'div'")
    return df_out


# ---------------------------
# Dimensionality reduction
# ---------------------------
def apply_pca(X: pd.DataFrame, n_components: Union[int, float] = 0.95, whiten: bool = False) -> Tuple[pd.DataFrame, PCA]:
    """
    Fit PCA on X and return transformed DataFrame and PCA object.
    n_components: int (num comps) or float in (0,1] to preserve variance ratio.
    """
    pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
    Xt = pca.fit_transform(X.fillna(0).values)
    cols = [f'PC{i+1}' for i in range(Xt.shape[1])]
    return pd.DataFrame(Xt, index=X.index, columns=cols), pca


# ---------------------------
# Statistical tests / hypothesis testing
# ---------------------------
def ttest_onesample(series: pd.Series, popmean: float = 0.0) -> Dict[str, float]:
    """
    One-sample t-test H0: mean == popmean. Returns t-stat, pvalue, mean.
    Requires scipy.stats, otherwise raises.
    """
    if stats is None:
        raise ImportError("scipy is required for t-test (scipy.stats).")
    series_clean = series.dropna().astype(float)
    res = stats.ttest_1samp(series_clean, popmean)
    return {'t_stat': float(res.statistic), 'pvalue': float(res.pvalue), 'sample_mean': float(series_clean.mean())}


def shapiro_test(series: pd.Series) -> Dict[str, float]:
    """
    Shapiro-Wilk normality test. H0: data is from a normal distribution.
    """
    if stats is None:
        raise ImportError("scipy is required for Shapiro test (scipy.stats).")
    series_clean = series.dropna().astype(float)
    # Shapiro may fail for n > 5000 in scipy versions; warn caller to sample if necessary.
    if len(series_clean) > 5000:
        series_clean = series_clean.sample(5000, random_state=42)
    res = stats.shapiro(series_clean)
    return {'W': float(res[0]), 'pvalue': float(res[1])}


def wilcoxon_test(x: pd.Series, y: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test (paired) if y provided, otherwise one-sample Wilcoxon against zero.
    """
    if stats is None:
        raise ImportError("scipy is required for Wilcoxon test (scipy.stats).")
    if y is not None:
        x_clean = x.dropna()
        y_clean = y.dropna()
        n = min(len(x_clean), len(y_clean))
        res = stats.wilcoxon(x_clean.iloc[:n], y_clean.iloc[:n])
    else:
        res = stats.wilcoxon(x.dropna())
    return {'statistic': float(res.statistic), 'pvalue': float(res.pvalue)}


def adf_test(series: pd.Series) -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller test for stationarity. Returns dict of results.
    Requires statsmodels.
    """
    if adfuller is None:
        raise ImportError("statsmodels is required for ADF test (statsmodels.tsa.stattools.adfuller).")
    series_clean = series.dropna().astype(float)
    res = adfuller(series_clean)
    return {
        'adf_stat': float(res[0]),
        'pvalue': float(res[1]),
        'usedlag': int(res[2]),
        'nobs': int(res[3]),
        'critical_values': res[4],
        'icbest': float(res[5])
    }


# ---------------------------
# Persistence helpers
# ---------------------------
def save_obj(obj: Any, path: str) -> None:
    """Save object (scaler, model, pipeline) using joblib."""
    joblib.dump(obj, path)


def load_obj(path: str) -> Any:
    """Load object saved with joblib."""
    return joblib.load(path)


# ---------------------------
# Utility pipeline example
# ---------------------------
def full_preprocessing_pipeline(df: pd.DataFrame,
                                time_col: str = 'timestamp',
                                numeric_cols: Optional[List[str]] = None,
                                remove_outlier_cols: Optional[List[str]] = None,
                                outlier_method: str = 'iqr',
                                impute_strategy: str = 'median',
                                log_transform_cols: Optional[List[str]] = None,
                                lag_cols: Optional[List[str]] = None,
                                lag_sizes: Optional[List[int]] = None,
                                rolling_cols: Optional[List[str]] = None,
                                rolling_windows: Optional[List[int]] = None,
                                scale_method: Optional[str] = 'standard') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Example full pipeline that applies common steps in order and returns (processed_df, artifacts)
    artifacts contains fitted scaler/pca/others for reproducibility.
    This function is a convenience wrapper; for real projects, prefer composing smaller functions above.
    """
    artifacts: Dict[str, Any] = {}
    df_proc = df.copy()

    # 1. parse time features
    if time_col in df_proc.columns:
        df_proc = build_time_features(df_proc, time_col=time_col, inplace=True)

    # 2. ensure numeric
    if numeric_cols is None:
        numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    df_proc = ensure_numeric(df_proc, numeric_cols, inplace=True)

    # 3. outlier removal (optional)
    if remove_outlier_cols:
        for col in remove_outlier_cols:
            if col in df_proc.columns:
                df_proc = remove_outliers(df_proc, column=col, method=outlier_method, thresh=3.0, inplace=False)

    # 4. impute missing
    df_proc = impute_simple(df_proc, strategy=impute_strategy, columns=numeric_cols, inplace=True)

    # 5. log transform
    if log_transform_cols:
        df_proc = transform_log1p(df_proc, log_transform_cols, inplace=True)

    # 6. lags
    if lag_cols is None:
        lag_cols = ['close'] if 'close' in df_proc.columns else []
    lag_sizes = lag_sizes or [1, 2, 3, 6, 12, 24]
    for col in lag_cols:
        df_proc = add_lag_features(df_proc, column=col, lags=lag_sizes, inplace=True)

    # 7. rolling features
    if rolling_cols is None:
        rolling_cols = ['close'] if 'close' in df_proc.columns else []
    windows = rolling_windows or [12, 24, 72]
    for col in rolling_cols:
        df_proc = add_rolling_features(df_proc, column=col, windows=windows, inplace=True)

    # 8. drop rows with NaNs produced by lags/rolling
    df_proc = df_proc.dropna().reset_index(drop=True)

    # 9. scale numeric features (fit scaler later when train/test decided)
    artifacts['scale_method'] = scale_method

    return df_proc, artifacts
