"""Bounded file ingestion and reflected, read-only SQLite queries."""

from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import MetaData, Table, create_engine, func, inspect, select
from sqlalchemy.engine import Engine

MAX_BYTES = 50 * 1024 * 1024
MAX_ROWS = 100_000
MAX_MEMORY = 256 * 1024 * 1024


def check_frame(df: pd.DataFrame, max_rows: int, max_memory: int) -> pd.DataFrame:
    if len(df) > max_rows:
        raise ValueError(f"Dataset exceeds row limit ({max_rows:,}).")
    if int(df.memory_usage(deep=True).sum()) > max_memory:
        raise ValueError("Dataset exceeds memory limit.")
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError("Column names must be unique strings.")
    return df


def load_file(
    source: str | Path | BinaryIO,
    filename: str | None = None,
    *,
    encoding: str = "utf-8-sig",
    separator: str = ",",
    max_bytes: int = MAX_BYTES,
    max_rows: int = MAX_ROWS,
    max_memory: int = MAX_MEMORY,
) -> pd.DataFrame:
    """Read CSV or Parquet; reject limits rather than silently truncating."""
    name = filename or (str(source) if isinstance(source, (str, Path)) else "")
    suffix = Path(name).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError("Select a CSV or Parquet file.")
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > max_bytes:
            raise ValueError("File exceeds size limit.")
        data = path.read_bytes()
    else:
        source.seek(0)
        data = source.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("File exceeds size limit.")
    try:
        if suffix == ".csv":
            if len(separator) != 1:
                raise ValueError("CSV separator must be one character.")
            df = pd.read_csv(BytesIO(data), encoding=encoding, sep=separator, nrows=max_rows + 1)
        else:
            parquet = pq.ParquetFile(BytesIO(data))
            if parquet.metadata.num_rows > max_rows:
                raise ValueError("Dataset exceeds row limit.")
            # Batch decoding bounds peak allocation; compressed file size is not enough.
            frames: list[pd.DataFrame] = []
            memory = 0
            for batch in parquet.iter_batches(batch_size=2048):
                frame = batch.to_pandas()
                memory += int(frame.memory_usage(deep=True).sum())
                if memory > max_memory:
                    raise ValueError("Dataset exceeds memory limit.")
                frames.append(frame)
            df = pd.concat(frames, ignore_index=True) if frames else parquet.read().to_pandas()
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, OSError) as exc:
        raise ValueError(f"File is empty or invalid: {exc}") from exc
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Invalid {suffix[1:]} file: {exc}") from exc
    return check_frame(df, max_rows, max_memory)


def sqlite_engine(path: str | Path) -> Engine:
    file = Path(path).resolve()
    if not file.is_file():
        raise ValueError("SQLite database does not exist.")
    # SQLite URI read-only mode prevents mutations, including database creation.
    return create_engine(f"sqlite:///file:{file.as_posix()}?mode=ro&uri=true")


def sqlite_tables(path: str | Path) -> list[str]:
    engine = sqlite_engine(path)
    try:
        return inspect(engine).get_table_names()
    finally:
        engine.dispose()


def load_sqlite(
    path: str | Path,
    table: str,
    *,
    max_rows: int = MAX_ROWS,
    max_memory: int = MAX_MEMORY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    engine = sqlite_engine(path)
    try:
        if table not in inspect(engine).get_table_names():
            raise ValueError("Unknown table; select an inspected table name.")
        relation = Table(table, MetaData(), autoload_with=engine)
        with engine.connect() as connection, connection.begin():
            # One read transaction keeps aggregates and loaded rows consistent.
            rows = int(connection.scalar(select(func.count()).select_from(relation)) or 0)
            if rows > max_rows:
                raise ValueError("Dataset exceeds row limit.")
            expressions = [func.count().label("total")]
            expressions += [
                func.count(column).label(f"nonnull_{i}") for i, column in enumerate(relation.c)
            ]
            counts = connection.execute(select(*expressions).select_from(relation)).one()
            nulls = {column.name: rows - int(counts[i + 1]) for i, column in enumerate(relation.c)}
            frames: list[pd.DataFrame] = []
            memory = 0
            query = select(relation)
            if relation.primary_key.columns:
                query = query.order_by(*relation.primary_key.columns)
            for frame in pd.read_sql(query, connection, chunksize=2048):
                memory += int(frame.memory_usage(deep=True).sum())
                if memory > max_memory:
                    raise ValueError("Dataset exceeds memory limit.")
                frames.append(frame)
            df = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=relation.c.keys())
            )
        return check_frame(df, max_rows, max_memory), {"rows": rows, "null_counts": nulls}
    finally:
        engine.dispose()
