"""One reusable analysis pipeline and honest, escaped report exports."""

import csv
import html
import json
from collections import Counter
from datetime import UTC, datetime
from io import StringIO
from typing import Any

import pandas as pd

from catalogdoctor.detection import (
    classification_review,
    isolation_forest,
    numeric_features,
    statistical,
)
from catalogdoctor.models import Issue, scalar
from catalogdoctor.profiling import profile
from catalogdoctor.quality import evaluate_rules, load_rules, quality_score


def serializable(value: Any) -> Any:
    if isinstance(value, Issue):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return scalar(value)


def analyze(df: pd.DataFrame, rules_yaml: str) -> dict[str, Any]:
    config = load_rules(rules_yaml)
    deterministic = evaluate_rules(df, config)
    features = numeric_features(df)
    stats = statistical(df, features)
    ml = isolation_forest(df, features)
    texts = [c for c in ["name", "description"] if c in df]
    labels = (
        classification_review(df, texts, "category")
        if texts and "category" in df
        else {
            "status": "unavailable",
            "reason": "Choose text and category columns in Classification Review.",
            "issues": [],
            "metrics": None,
        }
    )
    all_issues = deterministic + stats["issues"] + ml["issues"] + labels["issues"]
    return serializable(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "profile": profile(df),
            "score": quality_score(df, config, deterministic),
            "rules": config.model_dump(),
            "issues": all_issues,
            "failed_rule_counts": dict(Counter(i.rule_id for i in deterministic)),
            "severity_distribution": dict(Counter(i.severity for i in deterministic)),
            "statistical": stats,
            "isolation_forest": ml,
            "classification": labels,
            "preview": df.head(10).to_dict(orient="records"),
            "row_reference": "zero-based position in original loaded dataset; "
            "source_index also preserved",
        }
    )


def export_json(report: dict[str, Any]) -> str:
    return json.dumps(serializable(report), indent=2, allow_nan=False, ensure_ascii=False)


def export_csv(report: dict[str, Any]) -> str:
    output = StringIO(newline="")
    fields = list(Issue.model_fields)
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in report["issues"]:
        row = {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            for k, v in record.items()
        }
        # Protect spreadsheet users from formula execution on untrusted text.
        row = {
            k: "'" + v
            if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r"))
            else v
            for k, v in row.items()
        }
        writer.writerow(row)
    return output.getvalue()


def export_html(report: dict[str, Any]) -> str:
    escaped = html.escape(export_json(report))
    score = report["score"]["score"]
    score_label = f"{score:.1f}/100" if score is not None else "Unavailable"
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>CatalogDoctor report</title>'
        + (
            "<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem;"
            "color:#142b42;background:#f4f7fb}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
            "background:white;padding:2rem;border-radius:12px}h1{color:#067e83}</style>"
            f"<h1>CatalogDoctor</h1><h2>Quality score: {score_label}</h2>"
            "<p>Deterministic quality measurements. "
            "Outliers and model disagreements require review.</p>"
            f"<pre>{escaped}</pre></html>"
        )
    )
