import json

import pandas as pd
import pytest

from catalogdoctor.profiling import profile


def test_dataset_and_column_statistics():
    df = pd.DataFrame({"price": [1.0, 1.0, 3.0, None], "category": ["A", "A", "B", None]})
    result = profile(df)
    assert result["rows"] == 4
    assert result["missing_cells"] == 2
    assert result["missing_pct"] == 25
    assert result["duplicate_rows"] == 1
    price = result["columns"]["price"]
    assert price["mean"] == pytest.approx(5 / 3)
    assert price["median"] == 1
    assert price["iqr"] == 1
    assert price["unique_count"] == 2
    assert result["columns"]["category"]["cardinality"] == 2
    json.dumps(result, allow_nan=False)


def test_empty_constant_mixed_and_dates():
    assert profile(pd.DataFrame())["missing_pct"] is None
    df = pd.DataFrame(
        {
            "mixed": [1, "two", None],
            "constant": [5, 5, 5],
            "date": pd.to_datetime(["2025-01-01", None, "2025-02-01"]),
        }
    )
    result = profile(df)
    assert result["columns"]["constant"]["std"] == 0
    assert result["columns"]["mixed"]["kind"] == "categorical"
    assert result["columns"]["date"]["min"] == "2025-01-01T00:00:00"
    assert df["mixed"].tolist() == [1, "two", None]
    json.dumps(result, allow_nan=False)
