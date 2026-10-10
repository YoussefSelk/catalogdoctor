import json

import pandas as pd

from catalogdoctor.demo import generate_catalog, write_examples
from catalogdoctor.ingestion import load_sqlite
from catalogdoctor.reporting import analyze, dataset_identity, export_csv, export_html, export_json


def test_demo_reproducible_and_sql_roundtrip(tmp_path):
    a, manifest = generate_catalog()
    b, other = generate_catalog()
    pd.testing.assert_frame_equal(a, b)
    assert manifest == other
    assert len(a) == 1500
    assert len(manifest) > 50
    write_examples(tmp_path)
    df, stats = load_sqlite(tmp_path / "catalog.db", "products")
    assert stats["rows"] == 1500
    assert stats["null_counts"]["name"] > 0
    assert df.shape == a.shape
    assert json.loads((tmp_path / "ground_truth.json").read_text())["seed"] == 42


def test_exports_serialize_escape_and_report_real_results():
    df = pd.DataFrame({"name": ["<script>alert(1)</script>", None], "price": [-1, 3]})
    report = analyze(
        df, "rules: [{column: name, type: required}, {column: price, type: min_value, value: 0}]"
    )
    parsed = json.loads(export_json(report))
    assert parsed["profile"]["rows"] == 2
    assert len(parsed["issues"]) == 2
    assert "row_ref" in export_csv(report)
    html = export_html(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "Quality score" in html
    assert export_csv(analyze(pd.DataFrame(), "rules: []")).startswith("rule_id,")


def test_export_identity_includes_schema():
    df = pd.DataFrame({"price": [1, 2]})
    assert dataset_identity(df) != dataset_identity(df.rename(columns={"price": "weight"}))
    assert dataset_identity(df) != dataset_identity(df.astype(float))
