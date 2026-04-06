import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import generate_insights, run_pipeline
from src.preprocessing import load_dataframe, normalize_columns
from src.schema import find_columns_by_keywords


@st.cache_data
def load_and_analyze(uploaded_file):
    """Cached preprocessing and pipeline execution."""
    df = load_dataframe(uploaded_file)
    df = normalize_columns(df)
    df_clean, report = run_pipeline(df)
    return df_clean, report


@st.cache_data
def compute_stats_cache(df: pd.DataFrame, schema: dict):
    """Cache-friendly stats computation."""
    numeric_cols = schema.get("numeric", [])
    return numeric_cols


def human_readable_label(col_name: str) -> str:
    """Convert snake_case column names to human-readable labels."""
    return col_name.replace("_", " ").title()


def build_filter_panel(df: pd.DataFrame, schema: dict, semantics: dict, feature_importance: dict, dataset_type: str) -> tuple[dict, pd.DataFrame]:
    """Intelligent filter system with prioritization based on feature importance."""
    st.sidebar.header("🔍 Data Filters")

    if dataset_type == "feature_dataset":
        st.sidebar.info(
            "Filtering is not applicable for feature-only datasets. "
            "Focus is on distributions and relationships."
        )
        return {}, df

    df_filtered = df.copy()
    selections = {}
    filter_count = 0

    # Get semantic columns - NEVER show identifiers as filters
    dimensions = semantics.get("dimensions", [])
    metrics = semantics.get("metrics", [])
    datetime_cols = semantics.get("temporal", [])

    # Get importance scores for prioritization
    categorical_importance = dict(feature_importance.get("categorical", []))
    
    # Sort dimensions by importance (highest first)
    prioritized_dimensions = sorted(dimensions, key=lambda x: categorical_importance.get(x, 0), reverse=True)
    
    # Top 5-7 dimensions for primary filters
    top_dimensions = prioritized_dimensions[:7]
    advanced_dimensions = prioritized_dimensions[7:]

    # 📦 Primary Categorical Filters (Top importance)
    if top_dimensions:
        st.sidebar.subheader("📦 Primary Filters")
        for col in top_dimensions:
            if col not in df.columns:
                continue

            unique_vals = sorted(df[col].dropna().unique())
            n_unique = len(unique_vals)

            if n_unique <= 1 or n_unique > 50:
                continue

            filter_count += 1
            label = human_readable_label(col)
            importance_score = categorical_importance.get(col, 0)
            selected = st.sidebar.multiselect(
                f"{label} (imp: {importance_score:.3f})",
                unique_vals,
                default=[],
                help=f"Filter by {label.lower()} - Importance: {importance_score:.3f}"
            )
            if selected:
                df_filtered = df_filtered[df_filtered[col].isin(selected)]
                selections[col] = selected

    # 📊 Numeric Filters Section
    if metrics:
        st.sidebar.subheader("📊 Numeric Filters")
        # Get numeric importance
        numeric_importance = dict(feature_importance.get("numeric", []))
        
        for col in metrics:
            if col not in df.columns:
                continue

            filter_count += 1
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            label = human_readable_label(col)
            importance_score = numeric_importance.get(col, 0)

            # Use sliders for numeric ranges
            range_vals = st.sidebar.slider(
                f"{label} (imp: {importance_score:.3f})",
                min_val, max_val,
                (min_val, max_val),
                help=f"Filter {label.lower()} between selected range - Importance: {importance_score:.3f}"
            )
            if range_vals != (min_val, max_val):
                df_filtered = df_filtered[(df_filtered[col] >= range_vals[0]) & (df_filtered[col] <= range_vals[1])]
                selections[col] = range_vals

    # ⏱ Time Filters Section
    if datetime_cols:
        st.sidebar.subheader("⏱ Time Filters")
        date_col = datetime_cols[0]  # Use first datetime column
        if date_col in df.columns:
            df_dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if not df_dates.empty:
                filter_count += 1
                min_date = df_dates.min().date()
                max_date = df_dates.max().date()
                label = human_readable_label(date_col)

                date_range = st.sidebar.date_input(
                    f"{label} Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date,
                    help=f"Filter by {label.lower()} date range"
                )
                if len(date_range) == 2 and date_range[0] != min_date or date_range[1] != max_date:
                    start_date, end_date = date_range
                    df_filtered = df_filtered[
                        (pd.to_datetime(df[date_col], errors='coerce').dt.date >= start_date) &
                        (pd.to_datetime(df[date_col], errors='coerce').dt.date <= end_date)
                    ]
                    selections[date_col] = date_range

    # Advanced Filters (collapsible)
    if advanced_dimensions:
        with st.sidebar.expander("🔧 Advanced Filters", expanded=False):
            st.write("Additional dimensions (lower importance):")
            for col in advanced_dimensions:
                if col not in df.columns:
                    continue

                unique_vals = sorted(df[col].dropna().unique())
                n_unique = len(unique_vals)

                if n_unique <= 1 or n_unique > 50:
                    continue

                filter_count += 1
                label = human_readable_label(col)
                importance_score = categorical_importance.get(col, 0)
                selected = st.multiselect(
                    f"{label} (imp: {importance_score:.3f})",
                    unique_vals,
                    default=[],
                    key=f"adv_{col}"
                )
                if selected:
                    df_filtered = df_filtered[df_filtered[col].isin(selected)]
                    selections[col] = selected

    # Reset and Summary
    if filter_count == 0:
        st.sidebar.info("No applicable filters for this dataset.")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🔄 Reset Filters", help="Clear all filters"):
            st.rerun()
    with col2:
        st.sidebar.metric("📊 Filtered Rows", f"{len(df_filtered):,}")

    # Handle empty results
    if df_filtered.empty:
        st.sidebar.warning("⚠️ No data matches current filters. Try adjusting filters.")
        df_filtered = df.copy()  # Reset to show all data

    return selections, df_filtered


def render_kpis(df: pd.DataFrame, semantics: dict, feature_importance: dict) -> None:
    """Render intelligence-driven KPIs using highest importance metric."""
    total_records = len(df)

    # Select primary KPI as highest importance numeric feature
    numeric_importance = dict(feature_importance.get("numeric", []))
    if numeric_importance:
        primary_metric = max(numeric_importance.keys(), key=lambda x: numeric_importance[x])
        metric_name = human_readable_label(primary_metric)
    else:
        # Fallback to semantic metrics
        metrics = semantics.get("metrics", [])
        if metrics:
            primary_metric = metrics[0]
            metric_name = human_readable_label(primary_metric)
        else:
            # Final fallback to any numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            primary_metric = numeric_cols[0] if numeric_cols else None
            metric_name = "Total Metric" if primary_metric else "Records"

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Records", f"{total_records:,}")

    if primary_metric and primary_metric in df.columns:
        total_value = df[primary_metric].sum()
        average_value = df[primary_metric].mean()
        col2.metric(f"🔢 Total {metric_name}", f"{total_value:,.2f}")
        col3.metric(f"📈 Avg {metric_name}", f"{average_value:,.2f}")
    else:
        col2.metric("Total Value", "N/A")
        col3.metric("Average Value", "N/A")


def plot_time_trend(df: pd.DataFrame, date_col: str | None, measure: str | None) -> None:
    if not date_col or not measure or date_col not in df.columns or measure not in df.columns:
        st.info("⏰ No datetime column detected → temporal analysis skipped")
        return

    period = pd.to_datetime(df[date_col], errors="coerce")
    time_df = df.copy()
    time_df[date_col] = period
    time_df = time_df.dropna(subset=[date_col])
    if time_df.empty:
        st.info("⏰ Datetime column contains no valid timestamps → temporal analysis skipped")
        return

    aggregated = time_df.resample("ME", on=date_col)[measure].sum().reset_index()
    fig = px.line(aggregated, x=date_col, y=measure, markers=True, title=f"{human_readable_label(measure)} over time")
    fig.update_layout(yaxis_title=human_readable_label(measure), xaxis_title=human_readable_label(date_col))
    st.plotly_chart(fig, width="stretch")


def plot_rolling_average(df: pd.DataFrame, date_col: str, measure: str) -> None:
    """Plot rolling average trend for temporal analysis."""
    if date_col not in df.columns or measure not in df.columns:
        return

    period = pd.to_datetime(df[date_col], errors="coerce")
    time_df = df.copy()
    time_df[date_col] = period
    time_df = time_df.dropna(subset=[date_col])
    if time_df.empty:
        return

    aggregated = time_df.resample("ME", on=date_col)[measure].sum().reset_index()
    if len(aggregated) < 3:
        return

    # Calculate rolling average
    aggregated = aggregated.sort_values(date_col)
    aggregated['rolling_avg'] = aggregated[measure].rolling(window=3, min_periods=1).mean()

    fig = px.line(aggregated, x=date_col, y=['rolling_avg', measure],
                  title=f"{human_readable_label(measure)} with 3-month rolling average",
                  labels={'value': human_readable_label(measure), 'variable': 'Type'})
    fig.update_layout(yaxis_title=human_readable_label(measure), xaxis_title=human_readable_label(date_col))
    st.plotly_chart(fig, width="stretch")


def plot_category_bar(df: pd.DataFrame, semantics: dict, feature_importance: dict) -> None:
    """Intelligent category chart using top importance dimension and metric."""
    dimensions = semantics.get("dimensions", [])
    metrics = semantics.get("metrics", [])
    
    if not dimensions or not metrics:
        st.info("📊 No suitable categorical and numeric columns for category chart")
        return

    # Select top importance dimension and metric
    categorical_importance = dict(feature_importance.get("categorical", []))
    numeric_importance = dict(feature_importance.get("numeric", []))
    
    top_dimension = max(dimensions, key=lambda x: categorical_importance.get(x, 0)) if categorical_importance else dimensions[0]
    top_metric = max(metrics, key=lambda x: numeric_importance.get(x, 0)) if numeric_importance else metrics[0]

    if top_dimension not in df.columns or top_metric not in df.columns:
        st.info("📊 Selected columns not available in filtered data")
        return

    bar_data = (
        df.groupby(top_dimension, dropna=False)[top_metric]
        .sum()
        .reset_index()
        .sort_values(top_metric, ascending=False)
        .head(20)
    )
    if bar_data.empty:
        st.info("📊 No data available for category aggregation")
        return

    fig = px.bar(
        bar_data,
        x=top_dimension,
        y=top_metric,
        title=f"Top {human_readable_label(top_dimension)} by {human_readable_label(top_metric)}"
    )
    fig.update_layout(
        xaxis_title=human_readable_label(top_dimension),
        yaxis_title=human_readable_label(top_metric)
    )
    st.plotly_chart(fig, width="stretch")


def plot_pivot_heatmap(pivot_df: pd.DataFrame) -> None:
    if pivot_df.empty:
        st.info("Pivot view unavailable for this dataset.")
        return

    matrix = pivot_df.set_index(pivot_df.columns[0]).select_dtypes(include=["number"]).fillna(0)
    if matrix.empty:
        st.info("Pivot heatmap requires numeric pivot values.")
        return

    fig = px.imshow(
        matrix,
        labels={"x": "Columns", "y": pivot_df.columns[0], "color": "Value"},
        x=matrix.columns,
        y=matrix.index,
        text_auto=True,
        aspect="auto",
        title="Pivot heatmap",
    )
    st.plotly_chart(fig, width="stretch")


def plot_histogram_and_box(df: pd.DataFrame, metric: str, stats_summary: dict, feature_importance: dict) -> None:
    """Intelligent distribution analysis with skewness-aware visualization."""
    if not metric or metric not in df.columns:
        st.info("📈 No numeric column available for distribution analysis")
        return

    # Check for skewness to suggest log-scale
    skewness_info = stats_summary.get("summary", [])
    skewness = 0
    for row in skewness_info:
        if row.get("column") == metric:
            skewness = abs(row.get("skewness", 0))
            break
    
    if skewness > 1.0:
        st.info(f"⚠️ '{human_readable_label(metric)}' is highly skewed (skewness: {skewness:.2f}). Consider log transformation for better visualization.")
        # Offer log-scale option
        use_log = st.checkbox(f"Use log scale for {human_readable_label(metric)}", key="log_scale")
        plot_data = np.log1p(df[metric]) if use_log else df[metric]
        title_suffix = " (log scale)" if use_log else ""
    else:
        plot_data = df[metric]
        title_suffix = ""

    hist = px.histogram(df, x=metric, nbins=30, title=f"Distribution of {human_readable_label(metric)}{title_suffix}")
    box = px.box(df, y=metric, title=f"Boxplot of {human_readable_label(metric)}{title_suffix}")
    st.plotly_chart(hist, width="stretch")
    st.plotly_chart(box, width="stretch")


def build_interactive_olap(df: pd.DataFrame, schema: dict, semantics: dict, feature_importance: dict, report: dict) -> None:
    """Intelligent OLAP UI with importance-based dimension and metric selection."""
    st.subheader("⚙️ Interactive OLAP Analysis")
    
    dimensions = semantics.get("dimensions", [])
    metrics = semantics.get("metrics", [])
    
    if not dimensions or not metrics:
        st.info("Interactive OLAP requires semantic dimensions and metrics.")
        return
    
    # Get importance scores
    categorical_importance = dict(feature_importance.get("categorical", []))
    numeric_importance = dict(feature_importance.get("numeric", []))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sort dimensions by importance for default selection
        sorted_dims = sorted(dimensions, key=lambda x: categorical_importance.get(x, 0), reverse=True)
        row_dim = st.selectbox("Row Dimension", sorted_dims, key="row_dim")
    with col2:
        col_options = [None] + sorted_dims
        col_dim = st.selectbox("Column Dimension", col_options, key="col_dim")
    with col3:
        # Sort metrics by importance for default selection
        sorted_metrics = sorted(metrics, key=lambda x: numeric_importance.get(x, 0), reverse=True)
        measure = st.selectbox("Metric", sorted_metrics, key="measure")

    # Drill-down level selector
    drill_level = st.radio("Drill-down Level", ["Summary", "Top 10", "Detailed"], horizontal=True)

    # Build pivot table based on selections
    if col_dim:
        pivot_data = df.pivot_table(
            index=row_dim,
            columns=col_dim,
            values=measure,
            aggfunc="sum",
            fill_value=0,
        )
    else:
        pivot_data = df.groupby(row_dim)[measure].agg(["sum", "mean", "count"]).reset_index()
        pivot_data.columns = [row_dim, "Total", "Average", "Count"]

    # Apply drill-down filtering
    if drill_level == "Summary" and not isinstance(pivot_data, pd.DataFrame):
        display_data = pivot_data.head(5)
        st.write("**Summary (Top 5)**")
    elif drill_level == "Top 10":
        display_data = pivot_data.head(10)
        st.write("**Top 10**")
    else:
        display_data = pivot_data
        st.write("**Detailed View**")

    st.dataframe(display_data, width="stretch")

    # Visualization
    if not isinstance(pivot_data, pd.DataFrame):
        if col_dim:
            st.write("**Pivot Heatmap**")
            fig = px.imshow(pivot_data, text_auto=True, aspect="auto", color_continuous_scale="Blues", title="Dimension Analysis")
            st.plotly_chart(fig, width="stretch")
    else:
        st.write("**Aggregation Chart**")
        fig = px.bar(display_data, x=row_dim if row_dim in display_data.columns else display_data.index, y="Total", title=f"{human_readable_label(measure)} by {human_readable_label(row_dim)}")
        st.plotly_chart(fig, width="stretch")


def format_warning_list(warnings: list[str]) -> None:
    if warnings:
        with st.expander("System warnings"):
            for warning in warnings:
                st.warning(warning)


def show_table(df: pd.DataFrame, title: str) -> None:
    st.write(f"### {title}")
    st.dataframe(df)


def main() -> None:
    st.set_page_config(page_title="Enterprise EDA Platform", layout="wide")
    st.title("🔬 Enterprise EDA Platform")
    st.markdown(
        "📊 Upload any dataset and explore schema, quality, trends, OLAP analysis, and intelligent insights."
    )
    st.markdown("---")

    uploaded = st.file_uploader("📁 Upload dataset", type=["csv", "xlsx", "parquet"])
    if uploaded is None:
        st.info("👆 Upload a file to begin analysis.")
        return

    with st.spinner("⏳ Analyzing dataset..."):
        try:
            df_clean, report = load_and_analyze(uploaded)
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return
    
    if df_clean.empty:
        st.error("The uploaded dataset is empty or could not be loaded.")
        return

    insights = report.get("insights", [])
    schema = report.get("schema", {})
    semantics = report.get("semantics", {})
    numeric_cols = schema.get("numeric", [])
    cat_cols = schema.get("categorical", [])
    date_cols = schema.get("datetime") or []
    date_col = date_cols[0] if date_cols else None
    metrics = semantics.get("metrics", [])
    dimensions = semantics.get("dimensions", [])
    main_metric = metrics[0] if metrics else None
    dataset_type = report.get("dataset_type", "unknown")
    
    # Extract feature importance for intelligent UI
    feature_importance = report.get("advanced_importance", {"categorical": [], "numeric": []})

    if dataset_type == "feature_dataset":
        st.info(
            "This dataset appears to be a feature/attribute dataset. "
            "Analysis is focused on distributions, correlations, and feature relationships. "
            "Grouping and time-based analysis are not applicable."
        )
    elif dataset_type == "analytical_dataset":
        st.success(
            "Analytical dataset detected: supports grouping (OLAP), filtering, and aggregations."
        )
    elif dataset_type == "temporal_dataset":
        st.success(
            "Temporal dataset detected: supports time-series trends, rolling averages, and seasonality."
        )

    df_original = df_clean.copy()
    filters, filtered_df = build_filter_panel(df_original, schema, semantics, feature_importance, dataset_type)

    format_warning_list(report.get("warnings", []))

    # 📊 1. DATA OVERVIEW - Start with the big picture
    st.subheader("📊 Data Overview")
    render_kpis(filtered_df, semantics, feature_importance)
    
    with st.expander("👀 View Sample Data"):
        st.dataframe(filtered_df.head(10), width="stretch")

    st.markdown("---")
    
    # 🔍 2. DATA QUALITY - Address data issues first
    st.subheader("🔍 Data Quality Summary")
    col_quality1, col_quality2 = st.columns(2)
    with col_quality1:
        show_table(pd.DataFrame(report.get("missing", [])), "Missing Values")
    with col_quality2:
        show_table(pd.DataFrame(report.get("stats", {}).get("outliers", [])), "Outlier Summary")

    st.markdown("---")
    
    # 📈 3. DISTRIBUTION ANALYSIS - Understand data spread
    st.subheader("📈 Distribution Analysis")
    
    if main_metric:
        plot_histogram_and_box(filtered_df, main_metric, report.get("stats", {}), feature_importance)
    else:
        st.info("📊 No suitable numeric columns for distribution analysis")

    st.markdown("---")
    
    # 🔗 4. FEATURE RELATIONSHIPS - Explore correlations
    st.subheader("🔗 Feature Relationships")
    if isinstance(report.get("correlation"), pd.DataFrame) and not report["correlation"].empty:
        fig = px.imshow(
            report["correlation"],
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Correlation Matrix",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("🔗 No correlation data available")

    st.markdown("---")
    
    # ⏰ 5. TEMPORAL TRENDS - Time-based patterns
    st.subheader("⏰ Temporal Trends")
    
    if date_col and main_metric:
        # Time trend and rolling average
        col1, col2 = st.columns(2)
        with col1:
            plot_time_trend(filtered_df, date_col, main_metric)
        with col2:
            plot_rolling_average(filtered_df, date_col, main_metric)
    else:
        st.info("⏰ No datetime column detected → temporal analysis skipped")

    st.markdown("---")
    
    # 💡 6. INTELLIGENT INSIGHTS - Key findings
    st.subheader("💡 Intelligent Insights")
    for i, insight in enumerate(insights, 1):
        st.write(f"**{i}.** {insight}")

    st.markdown("---")
    
    # ⚙️ 7. INTERACTIVE ANALYSIS - Deep dive exploration
    st.subheader("⚙️ Interactive Analysis")
    
    # Category chart
    if dataset_type == "feature_dataset":
        st.info(
            "No categorical dimensions detected. OLAP and category-based analysis are not applicable for this dataset."
        )
    elif dataset_type == "analytical_dataset":
        plot_category_bar(filtered_df, semantics, feature_importance)
    else:
        st.info("📊 No categorical dimensions available for category-based chart")

    # Interactive OLAP UI
    if dataset_type == "analytical_dataset":
        build_interactive_olap(filtered_df, schema, semantics, feature_importance, report)

    st.markdown("---")
    
    # 🏷️ COLUMN SEMANTICS - Technical details
    st.subheader("🏷️ Column Semantics & Types")
    col_semantics = report.get("column_semantics", {})
    
    sem_col1, sem_col2, sem_col3 = st.columns(3)
    
    with sem_col1:
        st.write("**📊 Metrics** (for analysis)")
        metrics = col_semantics.get("metrics", [])
        if metrics:
            st.success(", ".join(metrics))
        else:
            st.info("None")
        
        st.write("**🎯 Target-like** (high variance)")
        target_like = col_semantics.get("target_like", [])
        if target_like:
            st.info(", ".join(target_like))
        else:
            st.info("None")
    
    with sem_col2:
        st.write("**📈 Dimensions** (for grouping)")
        dimensions = col_semantics.get("dimensions", [])
        if dimensions:
            st.success(", ".join(dimensions))
        else:
            st.info("None")
    
    with sem_col3:
        st.write("**⚠️ Identifiers** (EXCLUDED)")
        identifiers = col_semantics.get("identifier", [])
        if identifiers:
            id_reasons = col_semantics.get("identifier_reasons", {})
            id_details = []
            for col in identifiers:
                reasons = id_reasons.get(col, [])
                id_details.append(f"{col} ({', '.join(reasons)})")
            st.warning("\n".join(id_details))
            st.caption("Identifiers excluded from feature importance and analysis")
        else:
            st.info("None detected")


if __name__ == "__main__":
    main()
