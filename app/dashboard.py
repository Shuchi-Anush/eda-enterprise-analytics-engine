import sys
from pathlib import Path

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

st.set_page_config(page_title="Enterprise EDA Platform", layout="wide")

FILTER_KEYWORDS = {
    "Category": ["category", "product", "department", "class", "type", "sub_category", "subcategory", "item", "brand", "group"],
    "Region": ["region", "state", "country", "city", "area", "zone", "territory", "location"],
    "Segment": ["segment", "customer", "client", "account", "user", "buyer"],
}


def apply_filters(df: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """Apply selected filters to dataframe."""
    filtered = df.copy()
    for column, values in selections.items():
        if values and column in filtered.columns:
            filtered = filtered[filtered[column].isin(values)]
    return filtered


def build_filter_panel(df: pd.DataFrame, schema: dict) -> tuple[dict, pd.DataFrame]:
    st.sidebar.header("Filters")

    cat_cols = schema.get("categorical", [])
    fallback_cols = df.select_dtypes(include=["object"]).columns.tolist()
    filter_cols = cat_cols if cat_cols else fallback_cols

    st.write("Filter Columns:", filter_cols)
    st.write("Schema:", schema)

    selections: dict = {}
    df_filtered = df.copy()

    if not filter_cols:
        st.sidebar.info("No columns available for filtering.")
        st.sidebar.markdown("### 📊 Data Summary")
        st.sidebar.metric("Rows", len(df_filtered))
        return selections, df_filtered

    for col in filter_cols:
        if col not in df.columns:
            continue

        unique_vals = sorted(df[col].dropna().unique())
        n_unique = len(unique_vals)
        if n_unique <= 1:
            continue

        if n_unique <= 50:
            selected = st.sidebar.multiselect(col, unique_vals, default=[])
            selections[col] = selected
            if selected:
                df_filtered = df_filtered[df_filtered[col].isin(selected)]

        elif n_unique <= 200:
            selected = st.sidebar.selectbox(col, [None] + unique_vals)
            selections[col] = [selected] if selected else []
            if selected:
                df_filtered = df_filtered[df_filtered[col] == selected]

        else:
            query = st.sidebar.text_input(f"Search {col}")
            selections[col] = [query] if query else []
            if query:
                df_filtered = df_filtered[df_filtered[col].astype(str).str.contains(query, case=False, na=False)]

    if st.sidebar.button("Reset Filters"):
        st.experimental_rerun()

    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.metric("Rows", len(df_filtered))

    if df_filtered.empty:
        st.warning("No data after applying filters. Reset filters.")
        df_filtered = df.copy()

    return selections, df_filtered


def render_kpis(df: pd.DataFrame, numeric_col: str | None) -> None:
    """Render dynamic KPIs with intelligent metric naming."""
    total_records = len(df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Records", f"{total_records:,}")
    
    if numeric_col and numeric_col in df.columns:
        total_value = df[numeric_col].sum()
        average_value = df[numeric_col].mean()
        metric_name = numeric_col.replace("_", " ").title()
        col2.metric(f"🔢 Total {metric_name}", f"{total_value:,.2f}")
        col3.metric(f"📈 Avg {metric_name}", f"{average_value:,.2f}")
    else:
        col2.metric("Total Value", "N/A")
        col3.metric("Average Value", "N/A")


def plot_time_trend(df: pd.DataFrame, date_col: str, measure: str) -> None:
    if date_col not in df.columns or measure not in df.columns:
        st.info("Time trend unavailable for this dataset.")
        return

    period = pd.to_datetime(df[date_col], errors="coerce")
    time_df = df.copy()
    time_df[date_col] = period
    time_df = time_df.dropna(subset=[date_col])
    if time_df.empty:
        st.info("Datetime column contains no valid timestamps.")
        return

    aggregated = time_df.resample("ME", on=date_col)[measure].sum().reset_index()
    fig = px.line(aggregated, x=date_col, y=measure, markers=True, title=f"{measure} over time")
    fig.update_layout(yaxis_title=measure, xaxis_title=date_col)
    st.plotly_chart(fig, width="stretch")


def plot_category_bar(df: pd.DataFrame, category_col: str, measure: str) -> None:
    if category_col not in df.columns or measure not in df.columns:
        st.info("Category chart unavailable for this dataset.")
        return

    bar_data = (
        df.groupby(category_col, dropna=False)[measure]
        .sum()
        .reset_index()
        .sort_values(measure, ascending=False)
        .head(20)
    )
    if bar_data.empty:
        st.info("No data available for category aggregation.")
        return

    fig = px.bar(bar_data, x=category_col, y=measure, title=f"Top {category_col} by {measure}")
    fig.update_layout(xaxis_title=category_col, yaxis_title=measure)
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


def plot_histogram_and_box(df: pd.DataFrame, measure: str) -> None:
    if measure not in df.columns:
        st.info("Histogram and boxplot unavailable for this dataset.")
        return

    hist = px.histogram(df, x=measure, nbins=30, title=f"Distribution of {measure}")
    box = px.box(df, y=measure, title=f"Boxplot of {measure}")
    st.plotly_chart(hist, width="stretch")
    st.plotly_chart(box, width="stretch")


def build_interactive_olap(df: pd.DataFrame, schema: dict, report: dict) -> None:
    """Interactive OLAP UI with user-selected dimensions and drill-down."""
    st.subheader("⚙️ Interactive OLAP Analysis")
    
    cat_cols = schema.get("categorical", [])
    numeric_cols = schema.get("numeric", [])
    
    if not cat_cols or not numeric_cols:
        st.info("Interactive OLAP requires categorical and numeric columns.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        row_dim = st.selectbox("Row Dimension", cat_cols, key="row_dim")
    with col2:
        col_dim = st.selectbox("Column Dimension", [None] + cat_cols, key="col_dim")
    with col3:
        measure = st.selectbox("Metric", numeric_cols, key="measure")
    
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
        fig = px.bar(display_data, x=row_dim if row_dim in display_data.columns else display_data.index, y="Total", title=f"{measure} by {row_dim}")
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
    main_metric = semantics.get("metrics", [])[0] if semantics.get("metrics") else None

    df_original = df_clean.copy()
    filters, filtered_df = build_filter_panel(df_original, schema)

    st.write("Schema:", schema)
    st.write("Semantics:", semantics)

    format_warning_list(report.get("warnings", []))

    st.subheader("📊 Key Performance Indicators")
    render_kpis(filtered_df, main_metric)

    with st.container(border=True):
        st.subheader("💡 Top Insights")
        for i, insight in enumerate(insights, 1):
            st.write(f"**{i}.** {insight}")

    st.markdown("---")
    st.subheader("📈 Visualizations")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if date_col and main_metric:
            plot_time_trend(filtered_df, date_col, main_metric)
        else:
            st.info("⏰ Time trend chart not available.")
    with chart_col2:
        if cat_cols and main_metric:
            plot_category_bar(filtered_df, cat_cols[0], main_metric)
        else:
            st.info("📊 Category bar chart not available.")

    st.markdown("---")
    st.subheader("🔥 Feature Analysis")

    if report.get("olap", {}).get("pivot") is not None:
        st.write("**Pivot Heatmap**")
        plot_pivot_heatmap(report["olap"]["pivot"])

    if main_metric:
        st.write("**Distribution Analysis**")
        plot_histogram_and_box(filtered_df, main_metric)

    # Interactive OLAP UI
    st.markdown("---")
    build_interactive_olap(filtered_df, schema, report)

    st.markdown("---")
    st.subheader("🏷️ Column Semantics & Types")
    col_semantics = report.get("column_semantics", {})
    
    sem_col1, sem_col2, sem_col3 = st.columns(3)
    
    with sem_col1:
        st.write("**📊 Metrics** (for analysis)")
        metrics = col_semantics.get("metric", [])
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
        dimensions = col_semantics.get("dimension", [])
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

    st.markdown("---")
    st.subheader("🔍 Data Quality Summary")
    col_quality1, col_quality2 = st.columns(2)
    with col_quality1:
        show_table(pd.DataFrame(report.get("missing", [])), "Missing Values")
    with col_quality2:
        show_table(pd.DataFrame(report.get("stats", {}).get("outliers", [])), "Outlier Summary")

    st.markdown("---")
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

    st.markdown("---")
    with st.expander("👀 View Sample Data"):
        st.dataframe(filtered_df.head(10), width="stretch")


if __name__ == "__main__":
    main()
