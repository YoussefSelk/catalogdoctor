import numpy as np
import pandas as pd
import pytest

from catalogdoctor.detection import isolation_forest, statistical


@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_statistical_outlier_boundaries(method):
    df = pd.DataFrame({"price": [10.0] * 20 + [1000.0, None], "constant": [1.0] * 22})
    result = statistical(df, ["price", "constant"], method=method)
    assert [i.row_ref for i in result["issues"]] == [20]
    assert result["issues"][0].method == f"statistical:{method}"
    assert "threshold" in result["issues"][0].explanation
    assert result["boundaries"]["constant"]["status"] == "constant"


def test_small_and_invalid_detection():
    assert not statistical(pd.DataFrame({"x": [1, None]}), ["x"])["issues"]
    assert isolation_forest(pd.DataFrame({"x": [1, 2]}))["status"] == "unavailable"
    with pytest.raises(ValueError):
        statistical(pd.DataFrame({"x": [1]}), ["x"], threshold=-1)
    with pytest.raises(ValueError):
        isolation_forest(pd.DataFrame({"x": [1]}), contamination=0.8)


def test_nullable_numeric_outliers_preserve_missing_evidence():
    df = pd.DataFrame({"x": pd.Series([1, 2, 3, 4, 100, None], dtype="Int64")})
    result = statistical(df, ["x"])
    assert [i.row_ref for i in result["issues"]] == [4]
    assert pd.isna(df.iloc[5]["x"])


def test_isolation_reproducible_imputed_and_excludes_identifiers():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "product_id": np.arange(100),
            "price": rng.normal(10, 1, 100),
            "stock": rng.normal(20, 3, 100),
            "empty": [np.nan] * 100,
        }
    )
    df.loc[0, "price"] = 1000
    df.loc[1, "stock"] = np.nan
    a, b = isolation_forest(df), isolation_forest(df)
    assert a["scores"] == b["scores"]
    assert "product_id" not in a["features"]
    assert "empty" not in a["features"]
    assert 0 in [i.row_ref for i in a["issues"]]
    assert "not causal" in a["context_note"]
