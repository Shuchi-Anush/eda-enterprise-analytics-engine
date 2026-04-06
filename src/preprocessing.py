from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", "%b %d %Y", "%d %b %Y"]


def normalize_column_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or str(name)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def safe_load_dataframe(source: Union[str, Path, pd.DataFrame, Any], **kwargs) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, dtype="object", low_memory=False, **kwargs)
        if suffix in {".parquet", ".parq"}:
            return pd.read_parquet(path, **kwargs)
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path, dtype="object", **kwargs)
        return pd.read_csv(path, dtype="object", low_memory=False, **kwargs)

    if hasattr(source, "read"):
        return pd.read_csv(source, dtype="object", low_memory=False, **kwargs)

    raise ValueError("Unsupported data source type for loading.")


load_dataframe = safe_load_dataframe


def _sample_values(series: pd.Series, n: int = 500) -> pd.Series:
    return series.dropna().head(n)


def _guess_series_type(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return "numeric"
    if is_object_dtype(series):
        sample = _sample_values(series)
        if sample.empty:
            return "categorical"

        numeric_coercion = pd.to_numeric(sample, errors="coerce").notna().mean()
        if numeric_coercion >= 0.75:
            return "numeric"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            datetime_parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            datetime_coercion = datetime_parsed.notna().mean()
        if datetime_coercion >= 0.5:
            return "datetime"

        return "categorical"
    return "other"


def detect_column_types(df: pd.DataFrame, datetime_formats: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    if df.empty:
        return {"numeric": [], "categorical": [], "datetime": [], "boolean": [], "other": []}

    numeric, categorical, datetime_cols, boolean, other = [], [], [], [], []
    for col in df.columns:
        series = df[col]
        if is_bool_dtype(series):
            boolean.append(col)
            continue
        inferred = _guess_series_type(series)
        if inferred == "numeric":
            numeric.append(col)
        elif inferred == "datetime":
            datetime_cols.append(col)
        elif inferred == "categorical":
            categorical.append(col)
        else:
            other.append(col)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "boolean": boolean,
        "other": other,
    }


def summarize_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "dtype", "non_null", "null_ratio", "unique"])

    return (
        pd.DataFrame(
            [
                {
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].notna().sum()),
                    "null_ratio": float(1 - df[col].notna().mean()),
                    "unique": int(df[col].nunique(dropna=True)),
                }
                for col in df.columns
            ]
        )
        .sort_values(["null_ratio", "unique"], ascending=[False, True])
        .reset_index(drop=True)
    )


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_ratio"])

    missing = df.isna().sum()
    return (
        pd.DataFrame(
            {
                "column": missing.index,
                "missing_count": missing.values,
                "missing_ratio": missing.values / len(df),
            }
        )
        .sort_values("missing_ratio", ascending=False)
        .reset_index(drop=True)
    )


def detect_duplicates(df: pd.DataFrame, subset: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[df.duplicated(subset=subset, keep=False)].copy()


def detect_mixed_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    mixed = {}
    for col in df.select_dtypes(include=["object"]).columns:
        distinct_types = {type(v).__name__ for v in df[col].dropna().head(500).tolist()}
        if len(distinct_types) > 1:
            mixed[col] = sorted(distinct_types)
    return mixed


def find_datetime_column(df: pd.DataFrame, hint: Optional[Sequence[str]] = None) -> Optional[str]:
    """Find datetime column with primary keyword matching, then fallback heuristics."""
    hint = hint or ["date", "timestamp", "time", "reported", "created", "updated"]
    
    for col in df.columns:
        for pattern in hint:
            if pattern in col.lower():
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() >= 0.5:
                    return col
    
    schema = detect_column_types(df)
    if schema["datetime"]:
        return schema["datetime"][0]
    
    for col in df.select_dtypes(include=[np.number]).columns:
        sample = df[col].dropna().head(100)
        if sample.empty:
            continue
        if (sample.min() >= 1900 and sample.max() <= 2100) or \
           (sample.min() >= 1000000000 and sample.max() <= 9999999999):
            try:
                parsed = pd.to_datetime(sample, errors="coerce", unit="s")
                if parsed.notna().mean() >= 0.5:
                    return col
            except (ValueError, TypeError):
                pass
    
    return None


def infer_column_semantics(df: pd.DataFrame, schema: Dict[str, List[str]]) -> Dict[str, Any]:
    """Infer semantic roles of columns with refined rules: identifiers, metrics, dimensions, temporal.

    Rules:
      - Identifiers: Contains 'id'/'postal'/'code' OR cardinality > 90%
      - Metrics: Numeric, not identifier
      - Dimensions: Categorical, cardinality < 50
      - Temporal: Schema-detected datetime columns
    """
    roles = {
        "target_like": [],
        "identifiers": [],
        "dimensions": [],
        "metrics": [],
        "temporal": list(schema.get("datetime", [])),
        "identifier_reasons": {},
    }

    metric_candidates: List[str] = []
    for col in schema.get("numeric", []):
        col_lower = col.lower()
        cardinality = df[col].nunique() / len(df)

        is_id_by_name = any(pattern in col_lower for pattern in ["id", "postal", "code"])
        is_id_by_cardinality = cardinality > 0.9

        if is_id_by_name or is_id_by_cardinality:
            reason = []
            if is_id_by_name:
                reason.append("name_contains_id")
            if is_id_by_cardinality:
                reason.append(f"high_cardinality_{cardinality:.1%}")
            roles["identifiers"].append(col)
            roles["identifier_reasons"][col] = reason
        else:
            metric_candidates.append(col)

    for col in schema.get("categorical", []):
        unique_count = df[col].nunique()
        cardinality = unique_count / len(df)

        if cardinality > 0.9:
            roles["identifiers"].append(col)
            roles["identifier_reasons"][col] = ["high_cardinality_categorical", f"{cardinality:.1%}"]
        elif unique_count < 50:
            roles["dimensions"].append(col)

    roles["metrics"] = list(metric_candidates)

    if roles["metrics"]:
        metric_vars = df[roles["metrics"]].var()
        median_var = metric_vars.median() if len(metric_vars) > 0 else 0
        for col in roles["metrics"]:
            variance = df[col].var()
            completeness = df[col].notna().mean()
            if variance > median_var and completeness > 0.8:
                roles["target_like"].append(col)
                break

    return roles


def validate_and_convert_types(
    df: pd.DataFrame,
    datetime_formats: Optional[Sequence[str]] = None,
    numeric_threshold: float = 0.75,
) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if is_bool_dtype(series):
            continue
        if is_numeric_dtype(series):
            df[col] = pd.to_numeric(series, errors="coerce")
            continue
        if is_datetime64_any_dtype(series):
            df[col] = pd.to_datetime(series, errors="coerce")
            continue
        if is_object_dtype(series):
            sample = _sample_values(series)
            if not sample.empty:
                numeric_coercion = pd.to_numeric(sample, errors="coerce").notna().mean()
                if numeric_coercion >= numeric_threshold:
                    df[col] = pd.to_numeric(series, errors="coerce")
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    datetime_parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                    datetime_coercion = datetime_parsed.notna().mean()
                if datetime_coercion >= 0.5:
                    df[col] = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
                    continue
        df[col] = series
    return df


def handle_missing_values(
    df: pd.DataFrame,
    time_column: Optional[str] = None,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if df.empty:
        return df.copy(), {"warning": "Empty dataset; no missing-value handling applied."}

    df = df.copy()
    if time_column and time_column in df.columns:
        df = df.sort_values(time_column)

    cleaned_df, justification = impute_missing_values(
        df,
        time_column=time_column,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
    )
    return cleaned_df, justification


def impute_missing_values(
    df: pd.DataFrame,
    time_column: Optional[str] = None,
    numeric_strategy: str = "auto",
    categorical_strategy: str = "mode",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    schema = detect_column_types(df)
    justification: Dict[str, str] = {}

    if time_column and time_column in df.columns:
        df = df.sort_values(time_column)
        df = df.ffill().bfill()
        justification["time_series"] = (
            "Forward/backward fill applied along the time index to preserve trends and avoid artificial jumps."
            if df[time_column].notna().all()
            else "No time-series fill applied because time column is incomplete."
        )

    for col in schema["numeric"]:
        if df[col].isna().all():
            justification[col] = "All values missing; numeric imputation skipped to avoid misleading constants."
            continue
        skew = float(df[col].skew(skipna=True)) if df[col].count() > 0 else 0.0
        if numeric_strategy == "median" or (numeric_strategy == "auto" and abs(skew) > 0.5):
            fill_value = float(df[col].median(skipna=True))
            rationale = "median because the distribution is skewed." if numeric_strategy == "auto" else "median by explicit strategy."
        else:
            fill_value = float(df[col].mean(skipna=True))
            rationale = "mean because the distribution is roughly symmetric." if numeric_strategy == "auto" else "mean by explicit strategy."
        df[col] = df[col].fillna(fill_value)
        justification[col] = f"Filled missing numeric values with {fill_value:.4g}; {rationale}"

    for col in schema["categorical"] + schema["boolean"]:
        if df[col].isna().all():
            df[col] = df[col].fillna("Unknown")
            justification[col] = "All values missing; replaced with Unknown to preserve categorical structure."
            continue
        mode = df[col].mode(dropna=True)
        if categorical_strategy == "unknown" or mode.empty or df[col].nunique(dropna=True) > len(df) * 0.5:
            fill_value = "Unknown"
            rationale = "Unknown because categories are high-cardinality or no dominant mode exists."
        else:
            fill_value = mode.iloc[0]
            rationale = "mode because the category has the strongest support."
        df[col] = df[col].fillna(fill_value)
        justification[col] = f"Filled missing categorical values with {fill_value}; {rationale}"

    return df, justification


def auto_extract_date_features(df: pd.DataFrame, date_column: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    if date_column is None:
        date_column = find_datetime_column(df)
    if date_column is None or date_column not in df.columns:
        return df

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce", infer_datetime_format=True)
    if df[date_column].isna().all():
        return df

    df[f"{date_column}_year"] = df[date_column].dt.year
    df[f"{date_column}_month"] = df[date_column].dt.month
    df[f"{date_column}_day"] = df[date_column].dt.day
    df[f"{date_column}_weekday"] = df[date_column].dt.day_name()
    df[f"{date_column}_quarter"] = df[date_column].dt.quarter
    return df


def safe_merge_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[Union[str, Sequence[str]]] = None,
    how: str = "inner",
    suffixes: Tuple[str, str] = ("_left", "_right"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if on is None:
        common = list(set(left.columns).intersection(right.columns))
        if not common:
            raise ValueError("No common columns found for merge.")
        on = common

    merged = left.merge(right, on=on, how=how, suffixes=suffixes, validate=None)
    stats = {
        "left_rows": len(left),
        "right_rows": len(right),
        "merged_rows": len(merged),
        "nulls_after_merge": int(merged.isna().sum().sum()),
        "join_type": how,
        "keys": on if isinstance(on, list) else [on],
    }
    return merged, stats
