from .preprocessing import (
    auto_extract_date_features,
    detect_column_types,
    detect_duplicates,
    detect_mixed_types,
    detect_missing_values,
    find_datetime_column,
    impute_missing_values,
    load_dataframe,
    normalize_columns,
    safe_merge_tables,
    safe_load_dataframe,
    summarize_schema,
    validate_and_convert_types,
)
from .stats import (
    compute_correlation,
    describe_numerical,
    identify_strong_relationships,
)
from .olap import (
    dice,
    pivot_analysis,
    rollup,
    slice_olap,
    drilldown,
    safe_merge_tables as merge_tables,
)
from .temporal import (
    aggregate_time,
    detect_trend,
    ensure_datetime,
    find_datetime_column,
    rolling_average,
)
