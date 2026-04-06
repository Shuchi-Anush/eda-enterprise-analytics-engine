# EDA Enterprise Dashboard

A modular exploratory data analysis system built for unknown datasets. The design separates ingestion, preprocessing, statistical summarization, OLAP operations, and temporal insights into reusable components.

## Structure

- `src/preprocessing.py`: schema detection, safe loading, type conversion, missing-value handling, date feature engineering, and merge validation.
- `src/stats.py`: descriptive statistics, skewness, kurtosis, IQR outlier detection, and correlation analysis.
- `src/olap.py`: roll-up, drill-down, slice, dice, pivot analysis, and safe table merge logic.
- `src/temporal.py`: datetime parsing, time aggregation, rolling averages, and trend detection.
- `app/app.py`: CLI runner for a single dataset.
- `app/dashboard.py`: Streamlit front-end for interactive EDA.

## Usage

CLI:

```bash
python app/app.py data/superstore.csv
```

Dashboard:

```bash
streamlit run app/dashboard.py
```

## Design principles

- Dynamic schema detection for unknown column names and types.
- No hardcoded columns; all functions infer numeric, categorical, and datetime types.
- Robust handling of missing data, duplicates, mixed types, and empty datasets.
- Separate logic from UI so analysis can be reused in production or tests.

## Exam-ready patterns

- `pd.to_numeric(..., errors='coerce')` for safe numeric conversion
- `pd.to_datetime(..., errors='coerce', infer_datetime_format=True)` for date parsing
- `df.groupby(...).agg(...)` and `pd.pivot_table(...)` for OLAP-style analytics
- `rolling().mean()` for temporal smoothing and trend detection

## Notes

- If a dataset has no datetime column, temporal steps are skipped safely.
- Correlation only runs on numeric data to avoid invalid relationships.
- The modular design supports future expansion for multi-table joins and feature engineering.
