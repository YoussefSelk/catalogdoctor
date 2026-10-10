"""Evidence-preserving dataset and column summaries."""

from collections import Counter
from functools import partial
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from catalogdoctor.models import parse_date, scalar


def profile(
    df: pd.DataFrame, rare_fraction: float = 0.01, date_formats: dict[str, str] | None = None
) -> dict[str, Any]:
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    columns: dict[str, Any] = {}
    for name in df.columns:
        series = df[name]
        nonnull = series.dropna()
        kind = "categorical"
        if is_datetime64_any_dtype(series) or name in (date_formats or {}):
            kind = "date"
        elif is_numeric_dtype(series) and not is_bool_dtype(series):
            kind = "numeric"
        counts = nonnull.value_counts()
        column: dict[str, Any] = {
            "dtype": str(series.dtype),
            "kind": kind,
            "non_null_count": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "missing_pct": 100 * series.isna().sum() / rows if rows else None,
            "unique_count": int(series.nunique()),
            "unique_pct": 100 * series.nunique() / len(nonnull) if len(nonnull) else None,
            "top_values": [
                {"value": scalar(value), "count": int(count)}
                for value, count in counts.head(10).items()
            ],
        }
        if kind == "numeric":
            finite = (
                pd.to_numeric(nonnull).replace([float("inf"), float("-inf")], float("nan")).dropna()
            )
            column["nonfinite_count"] = len(nonnull) - len(finite)
            if len(finite):
                q1, q3 = finite.quantile(0.25), finite.quantile(0.75)
                column.update(
                    mean=finite.mean(),
                    median=finite.median(),
                    std=finite.std(ddof=0),
                    min=finite.min(),
                    max=finite.max(),
                    q1=q1,
                    q3=q3,
                    iqr=q3 - q1,
                )
        elif kind == "date":
            dates = series
            if date_formats and name in date_formats:
                dates = series.map(partial(parse_date, format=date_formats[name]))
            valid_dates = dates.dropna()
            column["invalid_date_count"] = int((dates.isna() & series.notna()).sum())
            if len(valid_dates):
                column.update(min=valid_dates.min(), max=valid_dates.max())
        else:
            column["cardinality"] = int(series.nunique())
            column["rare_categories"] = [
                scalar(value)
                for value, count in counts.items()
                if count / max(1, len(nonnull)) < rare_fraction
            ][:100]
        columns[name] = scalar(column)
    return {
        "rows": rows,
        "column_count": cols,
        "missing_cells": missing,
        "missing_pct": 100 * missing / (rows * cols) if rows * cols else None,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": 100 * duplicate_rows / rows if rows else None,
        "type_distribution": dict(Counter(column["kind"] for column in columns.values())),
        "columns": columns,
    }
