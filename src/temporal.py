from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .schema import find_datetime_column


def ensure_datetime(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    formats: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = df.copy()
    if date_col is None:
        date_col = find_datetime_column(df)
    if date_col is None or date_col not in df.columns:
        return df

    if not is_datetime64_any_dtype(df[date_col]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def aggregate_time(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    freq: str = "ME",
    values: Optional[Sequence[str]] = None,
    aggfunc: str = "sum",
) -> pd.DataFrame:
    date_col = date_col or find_datetime_column(df)
    if date_col is None or date_col not in df.columns:
        return pd.DataFrame()

    df = ensure_datetime(df, date_col)
    if df[date_col].isna().all():
        return pd.DataFrame()

    index = pd.DatetimeIndex(df[date_col]).to_period(freq)
    if values is None:
        values = list(df.select_dtypes(include=[np.number]).columns)
    values = [col for col in values if col in df.columns]
    if not values:
        return pd.DataFrame()

    grouped = df.groupby(index)[values].agg(aggfunc)
    grouped.index = grouped.index.to_timestamp()
    grouped.index.name = date_col
    result = grouped.reset_index()
    return result


def rolling_average(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    window: int = 7,
    min_periods: int = 1,
) -> pd.DataFrame:
    time_df = aggregate_time(df, date_col=date_col, values=value_cols, aggfunc="sum")
    if time_df.empty:
        return pd.DataFrame()

    date_col = date_col or find_datetime_column(df)
    if date_col is None or date_col not in time_df.columns:
        return pd.DataFrame()

    if value_cols is None:
        value_cols = [col for col in time_df.select_dtypes(include=[np.number]).columns if col != date_col]
    value_cols = [col for col in (value_cols or []) if col in time_df.columns]
    if not value_cols:
        return pd.DataFrame()

    time_df = time_df.set_index(pd.to_datetime(time_df[date_col], errors="coerce"))
    smoothed = time_df[value_cols].rolling(window=window, min_periods=min_periods).mean()
    smoothed = smoothed.reset_index().rename(columns={"index": date_col})
    return smoothed


def detect_trend(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    window: int = 7,
) -> Dict[str, Any]:
    date_col = date_col or find_datetime_column(df)
    if date_col is None or date_col not in df.columns:
        return {"trend": "no_date_column"}

    df = ensure_datetime(df, date_col)
    timeline = df.dropna(subset=[date_col]).sort_values(date_col)
    numeric = timeline.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"trend": "no_numeric_values"}

    if value_col is None or value_col not in numeric.columns:
        value_col = numeric.columns[0]

    series = timeline[[date_col, value_col]].dropna()
    if len(series) < 3:
        return {"trend": "insufficient_points"}

    x = np.arange(len(series))
    y = series[value_col].astype(float).values
    slope = float(np.polyfit(x, y, 1)[0])
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    seasonal_strength = np.nan
    if len(series) >= 12:
        seasonal_strength = float(
            series[value_col].diff(1).abs().rolling(7, min_periods=1).mean().mean()
            / (series[value_col].mean() or 1)
        )

    return {
        "date_column": date_col,
        "value_column": value_col,
        "slope": slope,
        "trend": direction,
        "seasonality_index": seasonal_strength,
    }


def detect_temporal_anomalies(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    window: int = 7,
    z_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Detect temporal anomalies using rolling mean deviation and z-score."""
    date_col = date_col or find_datetime_column(df)
    if date_col is None or date_col not in df.columns:
        return {"count": 0, "anomalies": []}

    df = ensure_datetime(df, date_col)
    timeline = df.dropna(subset=[date_col]).sort_values(date_col)
    numeric = timeline.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"count": 0, "anomalies": []}

    if value_col is None or value_col not in numeric.columns:
        value_col = numeric.columns[0]

    series = timeline[[date_col, value_col]].dropna().copy()
    if len(series) < window:
        return {"count": 0, "anomalies": []}

    # Z-score based detection
    z_scores = np.abs((series[value_col].astype(float) - series[value_col].astype(float).mean()) / (series[value_col].astype(float).std() + 1e-8))
    anomaly_mask = z_scores > z_threshold

    anomalies = series[anomaly_mask].to_dict(orient="records")
    return {
        "count": int(anomaly_mask.sum()),
        "anomalies": anomalies[:5],
        "z_threshold": z_threshold,
    }
