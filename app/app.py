import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.preprocessing import load_dataframe
from src.pipeline import generate_insights, run_pipeline


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modular exploratory data analysis.")
    parser.add_argument("path", help="Path to a dataset file (CSV, Excel, Parquet).")
    args = parser.parse_args()

    df = load_dataframe(Path(args.path))
    if df.empty:
        print("The dataset is empty or could not be loaded.")
        return

    df_clean, report = run_pipeline(df)

    if report.get("warnings"):
        print_section("Warnings")
        for warning in report["warnings"]:
            print(f"- {warning}")

    print_section("Schema summary")
    print(pd.DataFrame(report.get("summary", [])).to_string(index=False))

    print_section("Missing values")
    print(pd.DataFrame(report.get("missing", [])).to_string(index=False))
    print(f"Duplicate rows: {report.get('duplicates', 0)}")

    print_section("Descriptive statistics")
    stats_df = pd.DataFrame(report.get("stats", {}).get("summary", []))
    print(stats_df.to_string(index=False) if not stats_df.empty else "No numeric columns available for descriptive statistics.")

    print_section("Correlation")
    corr = report.get("correlation")
    if isinstance(corr, pd.DataFrame) and not corr.empty:
        print(corr.to_string())
    else:
        print("No numeric columns available for correlation.")

    temporal = report.get("temporal", {})
    if temporal.get("trend"):
        print_section("Temporal insights")
        print(f"Trend: {temporal['trend']}")
        if temporal.get("series"):
            print(pd.DataFrame(temporal["series"]).head().to_string(index=False))
        if temporal.get("rolling"):
            print(pd.DataFrame(temporal["rolling"]).head().to_string(index=False))
    else:
        print_section("Temporal insights")
        print("No datetime column detected; temporal analysis skipped.")

    print_section("Column Semantics")
    semantics = report.get("column_semantics", {})
    if semantics.get("identifier"):
        id_reasons = semantics.get("identifier_reasons", {})
        print("Identifiers (EXCLUDED from analysis):")
        for col in semantics["identifier"]:
            reasons = id_reasons.get(col, [])
            print(f"  - {col}: {', '.join(reasons)}")
    else:
        print("No identifiers detected")
    metrics = semantics.get("metric", [])
    print(f"Metrics: {', '.join(metrics) if metrics else 'None'}")
    dimensions = semantics.get("dimension", [])
    print(f"Dimensions: {', '.join(dimensions) if dimensions else 'None'}")
    target_like = semantics.get("target_like", [])
    print(f"Target-like: {', '.join(target_like) if target_like else 'None'}")

    print_section("Important Features (excluding identifiers)")
    adv_imp = report.get("advanced_importance", {})
    if adv_imp.get("numeric"):
        for i, (col, score) in enumerate(adv_imp["numeric"], 1):
            print(f"{i}. {col:20s} (importance: {score:.6f})")
    if adv_imp.get("categorical"):
        for i, (col, score) in enumerate(adv_imp["categorical"], 1):
            print(f"{i}. {col:20s} (cardinality importance: {score:.6f})")
    if adv_imp.get("excluded_identifiers"):
        print(f"\nExcluded from importance rankings: {', '.join(adv_imp['excluded_identifiers'])}")

    print_section("Insights")
    for insight in report.get("insights", []):
        print(f"- {insight}")


if __name__ == "__main__":
    main()
