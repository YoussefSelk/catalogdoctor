"""Shared serializable issue records and scalar conversion."""

import math
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

Severity = Literal["critical", "high", "medium", "low"]


def parse_date(value: Any, format: str) -> datetime | None:
    try:
        parsed = datetime.strptime(str(value), format)
        return parsed if parsed.strftime(format) == str(value) else None
    except (ValueError, TypeError):
        return None


def scalar(value: Any) -> Any:
    """Convert data scalars without leaking NaN/Infinity into JSON."""
    if isinstance(value, dict):
        return {str(k): scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [scalar(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return scalar(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str
    fields: list[str]
    row_ref: int
    source_index: Any = None
    severity: Severity = "medium"
    explanation: str
    actual: Any
    expected: str
    method: str


def issue(df: pd.DataFrame, position: int, fields: list[str], **kwargs: Any) -> Issue:
    return Issue(
        fields=fields,
        row_ref=position,
        source_index=scalar(df.index[position]),
        actual=scalar(
            {
                field: df.attrs["raw_numeric"][field][position]
                if field in df.attrs.get("raw_numeric", {})
                else df.iloc[position][field]
                for field in fields
            }
        ),
        **kwargs,
    )
