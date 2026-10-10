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


def test_sqlite_aggregates_and_rows_share_snapshot(tmp_path):
    from sqlalchemy import event

    from catalogdoctor.ingestion import sqlite_engine

    path = tmp_path / "snapshot.db"
    writer = create_engine(f"sqlite:///{path}")
    with writer.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
    pd.DataFrame({"a": [1, 2]}).to_sql("products", writer, index=False)
    reader = sqlite_engine(path)
    inserted = False

    def insert_after_count(connection, cursor, statement, parameters, context, many):
        nonlocal inserted
        if "SELECT count(*)" in statement and not inserted:
            inserted = True
            with writer.begin() as write:
                write.exec_driver_sql("INSERT INTO products (a) VALUES (3)")

    event.listen(reader, "after_cursor_execute", insert_after_count)
    try:
        df, stats = load_sqlite(path, "products", shared_engine=reader)
        assert inserted
        assert len(df) == stats["rows"] == 2
        assert stats["null_counts"]["a"] == 0
    finally:
        reader.dispose()
        writer.dispose()


def test_nested_parquet_has_a_helpful_error(tmp_path):
    path = tmp_path / "nested.parquet"
    pd.DataFrame({"x": [[1, 2], [3]]}).to_parquet(path)
    with pytest.raises(ValueError, match="scalar"):
        load_file(path)


def test_csv_preserves_identifiers_and_literal_na():
    df = load_file(BytesIO(b"sku,category,price\n001,NA,1.20\n002,Other,2.50\n"), "x.csv")
    assert df["sku"].tolist() == ["001", "002"]
    assert df["category"].tolist() == ["NA", "Other"]
    assert df["price"].tolist() == [1.2, 2.5]
    assert df.attrs["raw_numeric"]["price"] == ["1.20", "2.50"]


def test_parquet_retains_explicit_index(tmp_path):
    path = tmp_path / "indexed.parquet"
    pd.DataFrame({"x": [1, 2]}, index=["row-a", "row-b"]).to_parquet(path)
    assert load_file(path).index.tolist() == ["row-a", "row-b"]


def test_invalid_sqlite_is_reported_as_input_error(tmp_path):
    path = tmp_path / "bad.db"
    path.write_text("this is not SQLite")
    with pytest.raises(ValueError, match="SQLite"):
        sqlite_tables(path)
