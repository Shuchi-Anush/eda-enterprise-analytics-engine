from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd


def _safe_columns(df: pd.DataFrame, cols: Optional[Sequence[str]]) -> List[str]:
    if not cols:
        return []
    return [col for col in cols if col in df.columns]


def rollup(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
    aggfunc: Union[str, Dict[str, str]] = "sum",
) -> pd.DataFrame:
    keys = _safe_columns(df, list(group_cols))
    metrics = _safe_columns(df, list(metrics))
    if not keys or not metrics:
        raise ValueError("Roll-up requires at least one valid group column and one valid metric.")

    grouped = df.groupby(keys, dropna=False)[metrics].agg(aggfunc)
    return grouped.reset_index()


def drilldown(
    df: pd.DataFrame,
    hierarchy: Sequence[str],
    metrics: Sequence[str],
    aggfunc: Union[str, Dict[str, str]] = "sum",
) -> pd.DataFrame:
    return rollup(df, group_cols=hierarchy, metrics=metrics, aggfunc=aggfunc)


def slice_olap(
    df: pd.DataFrame,
    filters: Dict[str, Any],
    metrics: Sequence[str],
    aggfunc: Union[str, Dict[str, str]] = "sum",
) -> pd.DataFrame:
    filtered = df.copy()
    for column, value in filters.items():
        if column in filtered.columns:
            filtered = filtered[filtered[column] == value]
    return rollup(filtered, list(filters.keys()), metrics, aggfunc)


def dice(
    df: pd.DataFrame,
    conditions: Dict[str, Any],
    metrics: Sequence[str],
    aggfunc: Union[str, Dict[str, str]] = "sum",
) -> pd.DataFrame:
    filtered = df.copy()
    for column, value in conditions.items():
        if column in filtered.columns:
            filtered = filtered[filtered[column] == value]
    return rollup(filtered, list(conditions.keys()), metrics, aggfunc)


def pivot_analysis(
    df: pd.DataFrame,
    index: Sequence[str],
    columns: Optional[Sequence[str]] = None,
    values: Optional[Union[str, Sequence[str]]] = None,
    aggfunc: str = "sum",
    fill_value: Any = 0,
) -> pd.DataFrame:
    index_cols = _safe_columns(df, list(index))
    column_cols = _safe_columns(df, list(columns)) if columns else []
    if values is None:
        values = [col for col in df.select_dtypes(include=["number"]).columns]
    if isinstance(values, str):
        values = [values]
    value_cols = _safe_columns(df, list(values))

    if not index_cols or not value_cols:
        raise ValueError("Pivot requires valid index and value columns.")

    pivot = pd.pivot_table(
        df,
        index=index_cols,
        columns=column_cols or None,
        values=value_cols,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=False,
        dropna=False,
    )
    if isinstance(pivot.columns, pd.MultiIndex):
        pivot.columns = ["_".join(map(str, col)).strip("_") for col in pivot.columns.values]
    return pivot.reset_index()


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
