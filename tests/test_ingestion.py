from io import BytesIO

import pandas as pd
import pytest
from sqlalchemy import create_engine

from catalogdoctor.ingestion import load_file, load_sqlite, sqlite_tables


def test_csv_encoding_separator_and_row_refs():
    df = load_file(
        BytesIO("name;price\ncafé;12\n;5\n".encode("cp1252")),
        "x.csv",
        encoding="cp1252",
        separator=";",
    )
    assert df.iloc[0]["name"] == "café"
    assert df.index.tolist() == [0, 1]
    assert pd.isna(df.iloc[1]["name"])


def test_parquet_and_empty(tmp_path):
    path = tmp_path / "x.parquet"
    pd.DataFrame({"a": [1, None]}).to_parquet(path)
    assert load_file(path).shape == (2, 1)
    assert load_file(BytesIO(b"a,b\n"), "empty.csv").empty
    with pytest.raises(ValueError, match="empty|invalid"):
        load_file(BytesIO(b""), "empty.csv")


def test_limits_and_invalid_files():
    with pytest.raises(ValueError, match="row limit"):
        load_file(BytesIO(b"a\n1\n2\n"), "x.csv", max_rows=1)
    with pytest.raises(ValueError, match="size limit"):
        load_file(BytesIO(b"a\n1\n"), "x.csv", max_bytes=2)
    with pytest.raises(ValueError, match="CSV or Parquet"):
        load_file(BytesIO(b"abc"), "x.exe")
    with pytest.raises(ValueError, match="memory limit"):
        load_file(BytesIO(b"a\nhello\n"), "x.csv", max_memory=1)


def test_sqlite_real_aggregates_and_safe_inspection(tmp_path):
    path = tmp_path / "example.db"
    engine = create_engine(f"sqlite:///{path}")
    pd.DataFrame({"a": [1, None, 1], 'weird " name': ["x", "y", None]}).to_sql(
        "products", engine, index=False
    )
    engine.dispose()
    assert sqlite_tables(path) == ["products"]
    df, stats = load_sqlite(path, "products")
    assert df.shape == (3, 2)
    assert stats == {"rows": 3, "null_counts": {"a": 1, 'weird " name': 1}}
    with pytest.raises(ValueError, match="Unknown table"):
        load_sqlite(path, "products; DROP TABLE products")
    with pytest.raises(ValueError, match="row limit"):
        load_sqlite(path, "products", max_rows=2)
    assert sqlite_tables(path) == ["products"]


def test_missing_sqlite_does_not_create_file(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(ValueError, match="does not exist"):
        sqlite_tables(path)
    assert not path.exists()
