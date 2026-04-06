from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype

from .preprocessing import detect_column_types, normalize_column_name
from .schema_config import EXPECTED_TYPES


def _match_any(value: str, patterns: Sequence[str]) -> bool:
    token = normalize_column_name(value)
    return any(pattern in token for pattern in patterns)


def detect_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    if df is None or df.empty:
        return {"numeric": [], "categorical": [], "datetime": [], "boolean": [], "other": []}

    assigned = set()
    numeric: List[str] = []
    categorical: List[str] = []
    datetime_cols: List[str] = []
    boolean: List[str] = []
    other: List[str] = []

    for col in df.columns:
        if _match_any(col, EXPECTED_TYPES.get("date", [])):
            datetime_cols.append(col)
            assigned.add(col)
            continue
        if _match_any(col, EXPECTED_TYPES.get("amount", [])):
            numeric.append(col)
            assigned.add(col)
            continue
        if _match_any(col, EXPECTED_TYPES.get("category", [])):
            categorical.append(col)
            assigned.add(col)
            continue
        if _match_any(col, EXPECTED_TYPES.get("region", [])) or _match_any(col, EXPECTED_TYPES.get("customer", [])):
            categorical.append(col)
            assigned.add(col)
            continue
        if col.lower().endswith("_id") or col.lower().startswith("id_"):
            categorical.append(col)
            assigned.add(col)
            continue
        if is_bool_dtype(df[col]):
            boolean.append(col)
            assigned.add(col)
            continue

    fallback = detect_column_types(df)
    for col in fallback["boolean"]:
        if col not in assigned:
            boolean.append(col)
            assigned.add(col)
    for col in fallback["datetime"]:
        if col not in assigned:
            datetime_cols.append(col)
            assigned.add(col)
    for col in fallback["numeric"]:
        if col not in assigned:
            numeric.append(col)
            assigned.add(col)
    for col in fallback["categorical"]:
        if col not in assigned:
            categorical.append(col)
            assigned.add(col)
    for col in df.columns:
        if col not in assigned:
            other.append(col)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "boolean": boolean,
        "other": other,
    }


def find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    schema = detect_schema(df)
    return schema["datetime"][0] if schema["datetime"] else None


def find_columns_by_keywords(df: pd.DataFrame, keywords: Sequence[str]) -> List[str]:
    if df is None or df.empty:
        return []
    return [col for col in df.columns if any(keyword in normalize_column_name(col) for keyword in keywords)]
