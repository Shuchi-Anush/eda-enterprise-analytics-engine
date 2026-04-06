from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence


def describe_numerical(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if numeric_cols is None:
        numeric = df.select_dtypes(include=[np.number])
    else:
        numeric = df.loc[:, [col for col in numeric_cols if col in df.columns]]

    if numeric.empty:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "median",
                "std",
                "variance",
                "skewness",
                "kurtosis",
                "q1",
                "q3",
                "iqr",
                "outlier_count",
                "outlier_ratio",
            ]
        )

    rows = []
    for col in numeric.columns:
        series = numeric[col].dropna().astype(float)
        if series.empty:
            rows.append(
                {
                    "column": col,
                    "count": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "variance": np.nan,
                    "skewness": np.nan,
                    "kurtosis": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "iqr": np.nan,
                    "outlier_count": 0,
                    "outlier_ratio": 0.0,
                }
            )
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        rows.append(
            {
                "column": col,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std(ddof=0)),
                "variance": float(series.var(ddof=0)),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "outlier_count": int(outliers.count()),
                "outlier_ratio": float(outliers.count() / series.count()),
            }
        )

    return pd.DataFrame(rows).sort_values("outlier_ratio", ascending=False).reset_index(drop=True)


def describe_numerical_dict(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, float]]]:
    summary = describe_numerical(df, numeric_cols)
    return {
        "summary": summary.to_dict(orient="records"),
        "outliers": [
            {
                "column": row["column"],
                "outlier_count": int(row["outlier_count"]),
                "outlier_ratio": float(row["outlier_ratio"]),
            }
            for row in summary.to_dict(orient="records")
        ],
    }


def iqr_outlier_summary(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    summary = describe_numerical(df, numeric_cols)
    return {
        row["column"]: {
            "outlier_count": int(row["outlier_count"]),
            "outlier_ratio": float(row["outlier_ratio"]),
        }
        for row in summary.to_dict(orient="records")
    }


def compute_correlation(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    if numeric_cols is None:
        numeric = df.select_dtypes(include=[np.number])
    else:
        numeric = df.loc[:, [col for col in numeric_cols if col in df.columns]]
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr(method=method)


def identify_strong_relationships(
    corr: pd.DataFrame,
    threshold: float = 0.7,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    if corr.empty:
        return result

    for col in corr.columns:
        strong = [other for other in corr.index if other != col and abs(corr.at[other, col]) >= threshold]
        if strong:
            result[col] = strong
    return result


def calculate_feature_importance(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    top_k: int = 3,
) -> List[tuple[str, float]]:
    """Rank numeric columns by importance: variance * non_null_ratio."""
    if numeric_cols is None:
        numeric = df.select_dtypes(include=[np.number])
    else:
        numeric = df.loc[:, [col for col in numeric_cols if col in df.columns]]
    
    if numeric.empty:
        return []
    
    importance_scores = []
    for col in numeric.columns:
        variance = float(numeric[col].var(ddof=0))
        non_null_ratio = float(numeric[col].notna().mean())
        importance = variance * non_null_ratio
        importance_scores.append((col, importance))
    
    return sorted(importance_scores, key=lambda x: x[1], reverse=True)[:top_k]


def calculate_advanced_feature_importance(
    df: pd.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    exclude_cols: Optional[Sequence[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Advanced feature importance: normalized variance, correlation penalty, cardinality.
    
    Excludes identifier columns from analysis. Applies correlation penalty to avoid
    multicollinearity and normalizes by max variance to remove scale bias.
    """
    exclude_cols = set(exclude_cols or [])
    
    if numeric_cols is None:
        numeric = df.select_dtypes(include=[np.number])
    else:
        numeric = df.loc[:, [col for col in numeric_cols if col in df.columns and col not in exclude_cols]]
    
    if numeric.empty:
        return {"numeric": [], "categorical": [], "excluded_identifiers": list(exclude_cols), "penalty_applied": True}
    
    variances = {col: float(numeric[col].var(ddof=0)) for col in numeric.columns}
    max_var = max(variances.values()) if variances else 1.0
    normalized_var = {col: v / (max_var + 1e-8) for col, v in variances.items()}
    
    corr_matrix = numeric.corr().abs()
    correlation_penalty = {}
    for col in numeric.columns:
        high_corr = (corr_matrix[col] > 0.7).sum() - 1
        correlation_penalty[col] = max(0, 1.0 - (high_corr * 0.15))
    
    numeric_scores = []
    for col in numeric.columns:
        non_null_ratio = float(numeric[col].notna().mean())
        score = normalized_var[col] * non_null_ratio * correlation_penalty[col]
        numeric_scores.append((col, score))
    
    numeric_scores = sorted(numeric_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    categorical_scores = []
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns and col not in exclude_cols:
                cardinality = float(df[col].nunique() / len(df))
                completeness = float(df[col].notna().mean())
                score = cardinality * completeness
                categorical_scores.append((col, score))
    
    categorical_scores = sorted(categorical_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    return {"numeric": numeric_scores, "categorical": categorical_scores, "excluded_identifiers": list(exclude_cols), "penalty_applied": True}
