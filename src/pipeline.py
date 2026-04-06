from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .olap import dice, pivot_analysis, rollup, slice_olap
from .preprocessing import (
    detect_duplicates,
    detect_missing_values,
    handle_missing_values,
    normalize_columns,
    summarize_schema,
    validate_and_convert_types,
)
from .schema import detect_schema, find_datetime_column
from .stats import (
    compute_correlation,
    describe_numerical_dict,
    iqr_outlier_summary,
    calculate_feature_importance,
    calculate_advanced_feature_importance,
)
from .temporal import aggregate_time, detect_trend, rolling_average, detect_temporal_anomalies
from .preprocessing import infer_column_semantics


def generate_insights(report: Dict[str, Any]) -> List[str]:
    """Generate context-aware insights with multi-column reasoning (v4).
    
    Includes semantic awareness of identifiers, metrics, and dimensions to provide
    decision-support recommendations.
    """
    insights: List[str] = []
    if not report:
        return ["No report generated."]

    stats_summary = report.get("stats", {}).get("summary", [])
    duplicates = report.get("duplicates", 0)
    semantics = report.get("column_semantics", {})

    # Signal 0: Identifier columns detected and excluded from analysis
    identifiers = semantics.get("identifier", [])
    if identifiers:
        id_reasons = semantics.get("identifier_reasons", {})
        reason_strs = []
        for col in identifiers:
            reasons = id_reasons.get(col, [])
            reason_strs.append(f"{col} ({', '.join(reasons)})")
        insights.append(
            f"Identifier columns detected and excluded from analysis: {', '.join(reason_strs)}. "
            "These columns represent row identifiers, not predictive features."
        )

    # Signal 1: Extreme skewness with transformation guidance
    heavy_tail_cols = []
    for row in stats_summary:
        skew = abs(row.get("skewness", 0.0))
        outlier_ratio = row.get("outlier_ratio", 0.0)
        if skew > 1.0 and outlier_ratio > 0.15:
            heavy_tail_cols.append((row["column"], skew, outlier_ratio))
    if heavy_tail_cols:
        top_col = heavy_tail_cols[0]
        insights.append(
            f"Extreme skewness in '{top_col[0]}' (skew={top_col[1]:.2f}, outliers={top_col[2]:.0%}). "
            "Distribution highly asymmetric—consider log transformation or robust scaling."
        )

    # Signal 2: Negative correlations (margin erosion pattern)
    corr_df = report.get("correlation")
    neg_pairs = []
    if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
        for i, row in corr_df.iterrows():
            for j, value in row.items():
                if i < j and value < -0.6:
                    neg_pairs.append((i, j, value))
    if neg_pairs:
        top_neg = min(neg_pairs, key=lambda x: x[2])
        insights.append(
            f"Margin erosion detected: {top_neg[0]} inversely driven by {top_neg[1]} (r={top_neg[2]:.2f}). "
            "Review pricing or cost dynamics."
        )

    # Signal 3: Business concentration risk (dominant categories)
    dominant_cat = semantics.get("dominant_categories", {})
    if dominant_cat:
        top_cat_col = max(dominant_cat.items(), key=lambda x: x[1][1])
        if top_cat_col[1][1] > 0.5:
            insights.append(
                f"Business concentration risk: {top_cat_col[0]} dominated by '{top_cat_col[1][0]}' "
                f"({top_cat_col[1][1]:.0%}). Diversification recommended."
            )

    # Signal 4: Data quality issues
    quality_issues = []
    if duplicates > 0:
        quality_issues.append(f"{duplicates} duplicate rows")
    missing = report.get("missing", [])
    for row in missing:
        if row.get("missing_ratio", 0.0) >= 0.3:
            quality_issues.append(f"{row['column']}: {row['missing_ratio']:.0%} missing")
    if quality_issues:
        insights.append(f"Data quality issues: {', '.join(quality_issues[:1])}. Cleaning required before modeling.")

    if not insights:
        insights.append("Dataset is clean, well-distributed, and ready for advanced analytics.")

    return insights[:3]


def run_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {"warnings": []}
    if df is None or df.empty:
        report["warnings"].append("Input dataset is empty or invalid.")
        return pd.DataFrame(), report

    df = normalize_columns(df)
    df = validate_and_convert_types(df)

    schema = detect_schema(df)
    report["schema"] = schema
    report["summary"] = summarize_schema(df).to_dict(orient="records")
    report["missing"] = detect_missing_values(df).to_dict(orient="records")
    report["duplicates"] = int(detect_duplicates(df).shape[0])

    datetime_col = find_datetime_column(df)
    df_clean, justification = handle_missing_values(
        df,
        time_column=datetime_col,
        numeric_strategy="median",
        categorical_strategy="mode",
    )
    report["cleaning"] = {"missing_imputation": justification}

    report["stats"] = describe_numerical_dict(df_clean)
    report["correlation"] = compute_correlation(df_clean)
    report["outlier_summary"] = iqr_outlier_summary(df_clean)
    
    # Column semantics inference (target, identifier, dimension, metric)
    column_semantics = infer_column_semantics(df_clean, schema)
    report["column_semantics"] = column_semantics
    report["semantics"] = column_semantics
    
    # Advanced adaptive importance (normalized + correlation penalty + cardinality)
    # Exclude identifier columns from importance calculation
    numeric_cols = schema.get("numeric", [])
    identifiers = column_semantics.get("identifier", [])
    advanced_importance = calculate_advanced_feature_importance(
        df_clean,
        numeric_cols=numeric_cols,
        categorical_cols=schema.get("categorical", []),
        exclude_cols=identifiers,
        top_k=5,
    )
    report["advanced_importance"] = advanced_importance
    
    # Correlation pattern analysis for context
    if isinstance(report["correlation"], pd.DataFrame) and not report["correlation"].empty:
        positive_corr = []
        negative_corr = []
        for i, row in report["correlation"].iterrows():
            for j, value in row.items():
                if i < j:
                    if abs(value) >= 0.6:
                        if value > 0:
                            positive_corr.append((i, j, value))
                        else:
                            negative_corr.append((i, j, value))
        report["correlation_analysis"] = {"positive": positive_corr, "negative": negative_corr}
    else:
        report["correlation_analysis"] = {"positive": [], "negative": []}
    
    # Detect dominant categories for business insights
    dominant_categories = {}
    for col in schema.get("categorical", []):
        if col in df_clean.columns:
            top_cat = df_clean[col].value_counts().head(1)
            if not top_cat.empty:
                top_value = top_cat.index[0]
                ratio = top_cat.values[0] / len(df_clean)
                dominant_categories[col] = (top_value, ratio)
    report["column_semantics"]["dominant_categories"] = dominant_categories
    
    # Temporal anomaly detection
    if datetime_col:
        anomalies = detect_temporal_anomalies(df_clean, date_col=datetime_col)
        report["anomalies"] = anomalies
    else:
        report["anomalies"] = {"count": 0, "anomalies": []}

    if datetime_col:
        report["temporal"] = {}
        aggregated = aggregate_time(df_clean, date_col=datetime_col)
        report["temporal"]["series"] = aggregated.to_dict(orient="records") if not aggregated.empty else []
        rolled = rolling_average(df_clean, date_col=datetime_col)
        report["temporal"]["rolling"] = rolled.to_dict(orient="records") if not rolled.empty else []
        report["temporal"]["trend"] = detect_trend(df_clean, date_col=datetime_col)
    else:
        report["warnings"].append("No datetime column detected; temporal analysis skipped.")
        report["temporal"] = {}

    report["olap"] = {}
    category_cols = schema.get("categorical", [])
    numeric_cols = schema.get("numeric", [])
    if category_cols and numeric_cols:
        try:
            report["olap"]["pivot"] = pivot_analysis(
                df_clean,
                index=[category_cols[0]],
                columns=[category_cols[1]] if len(category_cols) > 1 else None,
                values=numeric_cols[0],
            )
        except ValueError:
            report["olap"]["pivot"] = pd.DataFrame()

        report["olap"]["rollup"] = rollup(
            df_clean,
            group_cols=[category_cols[0]],
            metrics=[numeric_cols[0]],
        ).to_dict(orient="records")

        if df_clean[category_cols[0]].notna().any():
            first_value = df_clean[category_cols[0]].dropna().iloc[0]
            report["olap"]["slice"] = slice_olap(
                df_clean,
                {category_cols[0]: first_value},
                metrics=[numeric_cols[0]],
            ).to_dict(orient="records")
        if len(category_cols) > 1 and df_clean[category_cols[1]].notna().any():
            second_value = df_clean[category_cols[1]].dropna().iloc[0]
            report["olap"]["dice"] = dice(
                df_clean,
                {category_cols[0]: first_value, category_cols[1]: second_value},
                metrics=[numeric_cols[0]],
            ).to_dict(orient="records")
    else:
        report["warnings"].append(
            "No suitable categorical and numeric columns available for OLAP analysis."
        )

    report["insights"] = generate_insights(report)
    return df_clean, report