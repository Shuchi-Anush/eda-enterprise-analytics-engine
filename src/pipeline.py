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


def classify_dataset(df: pd.DataFrame, schema: dict, semantics: dict) -> str:
    numeric = semantics.get("metrics", [])
    categorical = semantics.get("dimensions", [])
    datetime_cols = semantics.get("temporal", [])

    effective_categorical = [
        col for col in categorical
        if col in df.columns and df[col].nunique() < 50
    ]

    valid_temporal = []
    for col in datetime_cols:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.5:
            valid_temporal.append(col)

    if len(numeric) > 0 and len(effective_categorical) == 0 and len(valid_temporal) == 0:
        return "feature_dataset"

    if len(valid_temporal) > 0 and len(numeric) > 0:
        return "temporal_dataset"

    if len(effective_categorical) > 0 and len(numeric) > 0:
        return "analytical_dataset"

    return "unknown"


def generate_insights(report: Dict[str, Any]) -> List[str]:
    """Generate scored insights with impact × confidence prioritization."""
    insights: List[Dict[str, Any]] = []
    if not report:
        return ["No report generated."]

    stats_summary = report.get("stats", {}).get("summary", [])
    duplicates = report.get("duplicates", 0)
    semantics = report.get("column_semantics", {})
    corr_df = report.get("correlation")
    dataset_type = report.get("dataset_type", "unknown")
    total_rows = len(report.get("summary", []))

    # Helper to calculate confidence based on data size and statistical strength
    def calculate_confidence(statistical_strength: float, data_size_factor: float = 1.0) -> float:
        size_confidence = min(1.0, total_rows / 1000)  # Max confidence at 1000+ rows
        return min(1.0, statistical_strength * size_confidence * data_size_factor)

    # Priority 1: Data Quality Issues
    quality_issues = []
    if duplicates > 0:
        quality_issues.append(f"{duplicates} duplicate rows detected")
    missing = report.get("missing", [])
    for row in missing:
        if row.get("missing_ratio", 0.0) >= 0.3:
            quality_issues.append(f"{row['column']}: {row['missing_ratio']:.0%} missing values")
    if quality_issues:
        impact = 0.6  # Medium-high impact
        confidence = calculate_confidence(0.9, 1.0)  # High statistical confidence
        score = impact * confidence
        insights.append({
            "text": f"Data quality issues: {', '.join(quality_issues)}. Cleaning required before modeling.",
            "score": score,
            "observation": f"Found {len(quality_issues)} data quality problems",
            "why_matters": "Poor data quality can lead to unreliable analysis and biased models",
            "action": "Remove duplicates and impute or remove missing values"
        })

    # Priority 2: Outlier Analysis
    outlier_cols = []
    for row in stats_summary:
        outlier_ratio = row.get("outlier_ratio", 0.0)
        if outlier_ratio > 0.05:  # More than 5% outliers
            outlier_cols.append((row["column"], outlier_ratio))
    if outlier_cols:
        top_outlier = max(outlier_cols, key=lambda x: x[1])
        impact = 0.8  # High impact
        confidence = calculate_confidence(0.8, top_outlier[1])  # Confidence based on outlier ratio
        score = impact * confidence
        insights.append({
            "text": f"Outlier detection: '{top_outlier[0]}' has {top_outlier[1]:.1%} outliers (IQR method). Consider robust statistics or outlier treatment.",
            "score": score,
            "observation": f"Column '{top_outlier[0]}' shows significant outlier presence",
            "why_matters": "Outliers can distort statistical measures and model performance",
            "action": "Use robust statistics, remove outliers, or apply transformations"
        })

    # Priority 3: Skewness Analysis
    skew_cols = []
    for row in stats_summary:
        skew = abs(row.get("skewness", 0.0))
        if skew > 1.0:
            skew_cols.append((row["column"], skew))
    if skew_cols:
        top_skew = max(skew_cols, key=lambda x: x[1])
        impact = 0.8  # High impact
        confidence = calculate_confidence(0.9, min(1.0, top_skew[1] / 5))  # Confidence based on skew magnitude
        score = impact * confidence
        direction = "right-skewed" if top_skew[1] > 0 else "left-skewed"
        insights.append({
            "text": f"Skewness alert: '{top_skew[0]}' is highly {direction} (skewness={top_skew[1]:.2f}). Consider log transformation or non-parametric methods.",
            "score": score,
            "observation": f"Column '{top_skew[0]}' shows asymmetric distribution",
            "why_matters": "Skewed data can violate statistical assumptions and affect model accuracy",
            "action": "Apply log transformation, use non-parametric methods, or normalize the data"
        })

    # Priority 4: Correlation Analysis
    neg_pairs = []
    pos_pairs = []
    if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
        for i, row in corr_df.iterrows():
            for j, value in row.items():
                if i < j and abs(value) >= 0.6:
                    if value > 0:
                        pos_pairs.append((i, j, value))
                    else:
                        neg_pairs.append((i, j, value))
    if neg_pairs or pos_pairs:
        if neg_pairs:
            top_corr = min(neg_pairs, key=lambda x: x[2])
            corr_type = "negative"
            corr_desc = "inverse relationship"
        else:
            top_corr = max(pos_pairs, key=lambda x: x[2])
            corr_type = "positive"
            corr_desc = "direct relationship"
        
        impact = 0.8  # High impact
        confidence = calculate_confidence(0.85, abs(top_corr[2]))  # Confidence based on correlation strength
        score = impact * confidence
        insights.append({
            "text": f"Strong {corr_type} correlation: {top_corr[0]} and {top_corr[1]} show {corr_desc} (r={top_corr[2]:.2f}).",
            "score": score,
            "observation": f"Columns '{top_corr[0]}' and '{top_corr[1]}' are strongly correlated",
            "why_matters": "Correlated features can cause multicollinearity in models",
            "action": "Consider feature selection, PCA, or domain knowledge to handle correlation"
        })

    # Additional insights if we have space
    if len(insights) < 3:
        # Identifier columns detected and excluded
        identifiers = semantics.get("identifier", [])
        if identifiers:
            impact = 0.4  # Lower impact
            confidence = calculate_confidence(0.95, 1.0)  # High confidence in detection
            score = impact * confidence
            id_reasons = semantics.get("identifier_reasons", {})
            reason_strs = []
            for col in identifiers:
                reasons = id_reasons.get(col, [])
                reason_strs.append(f"{col} ({', '.join(reasons)})")
            insights.append({
                "text": f"Identifier columns detected and excluded: {', '.join(reason_strs)}. These represent row identifiers, not predictive features.",
                "score": score,
                "observation": f"Found {len(identifiers)} identifier columns",
                "why_matters": "Including identifiers can lead to data leakage and poor generalization",
                "action": "Exclude these columns from modeling and analysis"
            })

    if len(insights) < 3:
        # Business concentration risk
        dominant_cat = semantics.get("dominant_categories", {})
        if dominant_cat:
            top_cat_col = max(dominant_cat.items(), key=lambda x: x[1][1])
            if top_cat_col[1][1] > 0.5:
                impact = 0.5  # Medium impact
                confidence = calculate_confidence(0.8, top_cat_col[1][1])  # Confidence based on dominance ratio
                score = impact * confidence
                insights.append({
                    "text": f"Business concentration risk: {top_cat_col[0]} dominated by '{top_cat_col[1][0]}' ({top_cat_col[1][1]:.0%}). Diversification recommended.",
                    "score": score,
                    "observation": f"Column '{top_cat_col[0]}' shows high concentration in one category",
                    "why_matters": "High concentration can indicate business risk or data imbalance",
                    "action": "Monitor concentration trends and consider diversification strategies"
                })

    if dataset_type == "feature_dataset":
        insights.append({
            "text": "Dataset is feature-centric: focus on distributions and relationships rather than grouping.",
            "score": 0.5,
            "observation": "No categorical grouping or time dimension available",
            "why_matters": "Feature datasets are best analyzed through distributions, correlations, and feature importance",
            "action": "Focus on feature relationships and distribution diagnostics"
        })

    if not insights:
        insights.append({
            "text": "Dataset is clean, well-distributed, and ready for advanced analytics.",
            "score": 0.1,
            "observation": "No significant issues detected",
            "why_matters": "Clean data enables reliable analysis",
            "action": "Proceed with modeling and analysis"
        })

    # Sort by score and return top 3
    insights.sort(key=lambda x: x["score"], reverse=True)
    return [insight["text"] for insight in insights[:3]]


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
    
    dataset_type = classify_dataset(df_clean, schema, column_semantics)
    report["dataset_type"] = dataset_type

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