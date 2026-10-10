"""Run from any directory after installing CatalogDoctor."""

from pathlib import Path

from catalogdoctor.demo import write_examples

if __name__ == "__main__":
    write_examples(Path(__file__).resolve().parent)
    print("Generated 1,500 synthetic records, CSV/Parquet/SQLite, rules and ground truth.")
