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


def test_configured_dates_are_profiled_without_changing_values():
    df = pd.DataFrame({"date": ["2025-01-01", "2025-02-30", None]})
    report = analyze(df, "rules: [{column: date, type: date_validity, format: '%Y-%m-%d'}]")
    column = report["profile"]["columns"]["date"]
    assert column["kind"] == "date"
    assert column["invalid_date_count"] == 1
    assert column["min"] == "2025-01-01T00:00:00"
    assert df.iloc[1]["date"] == "2025-02-30"


def test_original_numeric_lexemes_reach_issues():
    from io import BytesIO

    from catalogdoctor.ingestion import load_file

    df = load_file(BytesIO(b"price\n-1.20\n3.50\n"), "x.csv")
    report = analyze(df, "rules: [{column: price, type: min_value, value: 0}]")
    assert report["issues"][0]["actual"]["price"] == "-1.20"


def test_csv_formula_safety():
    report = analyze(
        pd.DataFrame({"x": ["=2+2"]}),
        'rules: [{column: x, type: allowed_values, values: [safe], id: "=BAD"}]',
    )
    assert "'=BAD" in export_csv(report)
