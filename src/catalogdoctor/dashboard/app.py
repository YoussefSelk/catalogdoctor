"""CatalogDoctor dashboard. Launch with `catalogdoctor dashboard`."""

import os
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy.engine import Engine

from catalogdoctor.demo import default_rules, generate_catalog
from catalogdoctor.detection import (
    classification_review,
    isolation_forest,
    numeric_features,
    statistical,
)
from catalogdoctor.ingestion import load_file, load_sqlite, sqlite_engine, sqlite_tables
from catalogdoctor.reporting import (
    analyze,
    dataset_identity,
    export_csv,
    export_html,
    export_json,
    serializable,
)

PAGES = [
    "Overview",
    "Data Profiling",
    "Quality Rules",
    "Anomaly Detection",
    "Classification Review",
    "Reports & Export",
]


@st.cache_resource(show_spinner=False)
def cached_sqlite_engine(path: str, modified_ns: int, size: int) -> Engine:
    """Reuse the read-only connection pool; never cache live database results."""
    return sqlite_engine(path)


@st.cache_data(show_spinner=False, max_entries=4, hash_funcs={pd.DataFrame: dataset_identity})
def cached_analysis(df: pd.DataFrame, yaml_text: str) -> dict[str, Any]:
    return analyze(df, yaml_text)


@st.cache_data(show_spinner=False, max_entries=8, hash_funcs={pd.DataFrame: dataset_identity})
def cached_numeric(
    df: pd.DataFrame, features: list[str], method: str, threshold: float, contamination: float
) -> dict[str, Any]:
    return serializable(
        {
            "statistical": statistical(df, features, method=method, threshold=threshold),  # type: ignore[arg-type]
            "isolation_forest": isolation_forest(df, features, contamination=contamination),
        }
    )


@st.cache_data(show_spinner=False, max_entries=4, hash_funcs={pd.DataFrame: dataset_identity})
def cached_classification(
    df: pd.DataFrame, text: list[str], target: str, confidence: float
) -> dict[str, Any]:
    return serializable(classification_review(df, text, target, confidence=confidence))


def issue_table(issues: list[dict[str, Any]], *, key: str) -> None:
    if not issues:
        st.success("No findings with the current settings.")
        return
    st.dataframe(
        pd.DataFrame(issues)[["row_ref", "severity", "fields", "explanation", "actual", "method"]],
        width="stretch",
        hide_index=True,
        key=key,
    )
    st.caption("Row references are zero-based positions in the original loaded dataset.")


def chart(figure: Any) -> None:
    figure.update_layout(
        template="plotly_white",
        margin=dict(l=15, r=15, t=35, b=15),
        colorway=["#087f8c", "#5e60ce", "#e09f3e", "#d1495b"],
    )
    st.plotly_chart(figure, width="stretch")


def get_dataset(source: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    if source == "Built-in demo":
        return generate_catalog()[0], None
    if source == "Upload file":
        upload = st.sidebar.file_uploader("CSV or Parquet", type=["csv", "parquet"])
        encoding = st.sidebar.selectbox("CSV encoding", ["utf-8-sig", "utf-8", "cp1252", "latin-1"])
        separator = st.sidebar.selectbox("CSV separator", [",", ";", "\t", "|"])
        if upload is None:
            st.info(
                "Upload a CSV or Parquet file to start. "
                "Maximum 50 MB, 100,000 rows, 256 MB decoded data."
            )
            return None, None
        return load_file(
            BytesIO(upload.getvalue()), upload.name, encoding=encoding, separator=separator
        ), None
    path = os.getenv("CATALOGDOCTOR_SQLITE", "examples/catalog.db")
    if not Path(path).is_file():
        st.info(
            "Configure CATALOGDOCTOR_SQLITE with an existing SQLite database, or generate examples."
        )
        return None, None
    metadata = Path(path).stat()
    engine = cached_sqlite_engine(path, metadata.st_mtime_ns, metadata.st_size)
    table = st.sidebar.selectbox("SQLite table", sqlite_tables(path, shared_engine=engine))
    if not table:
        st.info("This database has no tables.")
        return None, None
    return load_sqlite(path, table, shared_engine=engine)


def overview(report: dict[str, Any], df: pd.DataFrame, sql: dict[str, Any] | None) -> None:
    st.subheader("A clear picture of catalog health")
    st.write(
        "Measure completeness and validation failures, "
        "then review unusual records with their evidence."
    )
    profile, score = report["profile"], report["score"]
    violations = [i for i in report["issues"] if i["method"] == "rule"]
    cols = st.columns(5)
    for column, label, value in zip(
        cols,
        ["Quality score", "Records", "Missing cells", "Duplicate rows", "Rule violations"],
        [
            f"{score['score']:.1f}/100" if score["score"] is not None else "Unavailable",
            f"{len(df):,}",
            f"{profile['missing_cells']:,}",
            f"{profile['duplicate_rows']:,}",
            f"{len(violations):,}",
        ],
        strict=True,
    ):
        column.metric(label, value)
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Measured quality dimensions")
        components = pd.DataFrame(
            [
                {"Dimension": name.title(), **component}
                for name, component in score["components"].items()
            ]
        )
        chart(
            px.bar(
                components.dropna(subset=["score"]),
                x="score",
                y="Dimension",
                orientation="h",
                range_x=[0, 100],
                color_discrete_sequence=["#087f8c"],
            )
        )
        st.caption(
            "Unmeasured dimensions are omitted; weights are renormalized. "
            "ML flags do not affect this score."
        )
    with right:
        st.markdown("#### Deterministic issue severity")
        severity = report["severity_distribution"]
        if severity:
            chart(px.pie(names=list(severity), values=list(severity.values()), hole=0.6))
        else:
            st.info("No deterministic violations.")
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(8), width="stretch")
    st.caption(
        f"{profile['column_count']} columns · Original cell values retained. "
        "Built-in demo uses synthetic data with deliberately injected issues."
    )
    if sql is not None:
        st.markdown("#### SQL verification")
        st.json(sql)
        st.caption("Executed in SQLite using reflected tables, COUNT(*) and COUNT(column).")


def profiling_page(report: dict[str, Any], df: pd.DataFrame) -> None:
    st.subheader("Explore the shape of your data")
    columns = pd.DataFrame.from_dict(report["profile"]["columns"], orient="index")
    left, right = st.columns([3, 2])
    with left:
        chart(
            px.bar(
                columns.reset_index(),
                x="index",
                y="missing_pct",
                labels={"index": "Column", "missing_pct": "Missing (%)"},
            )
        )
    with right:
        types = report["profile"]["type_distribution"]
        chart(
            px.pie(names=list(types), values=list(types.values()), title="Inferred types", hole=0.5)
        )
    st.dataframe(
        columns[["dtype", "kind", "non_null_count", "null_count", "unique_count", "missing_pct"]],
        width="stretch",
    )
    selected = st.selectbox("Inspect column", list(df.columns))
    details = report["profile"]["columns"][selected]
    if details["kind"] == "numeric":
        chart(px.histogram(df, x=selected, nbins=40, title=f"Distribution · {selected}"))
    else:
        counts = pd.DataFrame(details["top_values"])
        if not counts.empty:
            counts["value"] = counts["value"].astype(str)
            chart(px.bar(counts, x="value", y="count", title="Most frequent values"))
    with st.expander("Detailed column statistics"):
        st.json(details)


def rules_page(report: dict[str, Any]) -> None:
    st.subheader("Trace every deterministic violation")
    st.caption(
        "Rules are configured in the sidebar. Optional rules skip nulls; "
        "use required rules for missing values."
    )
    with st.expander("Active rules", expanded=False):
        st.dataframe(pd.DataFrame(report["rules"]["rules"]), width="stretch", hide_index=True)
    issues = [i for i in report["issues"] if i["method"] == "rule"]
    severity = st.multiselect(
        "Severity",
        ["critical", "high", "medium", "low"],
        default=["critical", "high", "medium", "low"],
    )
    st.write(f"{len(issues):,} violations across {len(report['failed_rule_counts'])} failed rules")
    issue_table([i for i in issues if i["severity"] in severity], key="rules_issues")
    if report["failed_rule_counts"]:
        chart(
            px.bar(
                x=list(report["failed_rule_counts"]),
                y=list(report["failed_rule_counts"].values()),
                labels={"x": "Rule", "y": "Violations"},
            )
        )


def anomalies_page(df: pd.DataFrame, report: dict[str, Any], identity: str) -> None:
    st.subheader("Review unusual numerical records")
    st.info(
        "Outliers are review candidates. IsolationForest comparisons are descriptive, "
        "not causal explanations."
    )
    choices = list(df.select_dtypes(include="number").columns)
    features = st.multiselect("Numeric features", choices, default=numeric_features(df))
    left, middle, right = st.columns(3)
    method = left.selectbox("Statistical method", ["iqr", "zscore"])
    threshold = middle.number_input(
        "Threshold multiplier",
        min_value=0.1,
        value=1.5 if method == "iqr" else 3.0,
        step=0.1,
        key=f"threshold_{method}",
    )
    contamination = right.slider("IsolationForest contamination", 0.01, 0.5, 0.05, 0.01)
    result = cached_numeric(df, features, method, threshold, contamination)
    st.session_state["numeric_override"] = (identity, result)
    stats, ml = result["statistical"], result["isolation_forest"]
    st.metric("Statistical findings", len(stats["issues"]))
    st.dataframe(pd.DataFrame.from_dict(stats["boundaries"], orient="index"), width="stretch")
    if features:
        selected = st.selectbox("Plot numeric feature", features)
        figure = px.histogram(df, x=selected, nbins=40)
        boundary = stats["boundaries"].get(selected, {})
        if boundary.get("status") == "ok":
            figure.add_vline(x=boundary["lower"], line_dash="dash", line_color="#d1495b")
            figure.add_vline(x=boundary["upper"], line_dash="dash", line_color="#d1495b")
        chart(figure)
    with st.expander("Statistical finding details"):
        issue_table(stats["issues"], key="stats_issues")
    st.markdown("#### IsolationForest")
    if ml["status"] != "ok":
        st.info(ml["reason"])
    else:
        st.caption(
            f"Seed {ml['seed']} · {len(ml['issues'])} flagged rows · "
            "Higher scores are more unusual."
        )
        issue_table(ml["issues"], key="ml_issues")
        with st.expander("Descriptive feature comparisons"):
            st.json(ml["context"][:20])


def classification_page(df: pd.DataFrame, report: dict[str, Any], identity: str) -> None:
    st.subheader("Review category assignments with evidence")
    st.info(
        "Suggestions require human review. "
        "A model trained on noisy labels can reproduce their errors."
    )
    texts = st.multiselect(
        "Text columns", list(df.columns), default=[c for c in ["name", "description"] if c in df]
    )
    choices = [c for c in df.columns if c not in texts]
    if not choices:
        st.info("Leave one column available as the category target.")
        return
    target = st.selectbox(
        "Category column", choices, index=choices.index("category") if "category" in choices else 0
    )
    confidence = st.slider("Minimum model probability", 0.5, 1.0, 0.65, 0.05)
    if not texts:
        st.info("Select at least one text column.")
        return
    result = cached_classification(df, texts, target, confidence)
    st.session_state["classification_override"] = (identity, result)
    if result["status"] != "ok":
        st.info(result["reason"])
        return
    metrics = result["metrics"]
    cols = st.columns(4)
    for col, name, value in zip(
        cols,
        ["Held-out accuracy", "Balanced accuracy", "Macro F1", "Review candidates"],
        [
            f"{metrics['accuracy']:.1%}",
            f"{metrics['balanced_accuracy']:.1%}",
            f"{metrics['macro_f1']:.3f}",
            str(len(result["issues"])),
        ],
        strict=True,
    ):
        col.metric(name, value)
    st.caption(f"{metrics['evaluated_rows']:,} evaluated rows · {result['evaluation']}")
    predictions = pd.DataFrame(result["predictions"])
    flagged = predictions[predictions["flagged"]].copy()
    flagged["text_preview"] = [str(df.iloc[int(i)][texts[0]]) for i in flagged["row_ref"]]
    st.dataframe(
        flagged[["row_ref", "text_preview", "original", "suggested", "confidence"]],
        width="stretch",
        hide_index=True,
    )
    with st.expander("Finding explanations"):
        issue_table(result["issues"], key="classification_issues")
    st.caption(result["limitation"])


def main() -> None:
    st.set_page_config(page_title="CatalogDoctor", page_icon="🩺", layout="wide")
    st.sidebar.title("CatalogDoctor")
    st.sidebar.caption("EXPLAINABLE DATA QUALITY")
    page = st.sidebar.radio("Workspace", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    source = st.sidebar.selectbox("Data source", ["Built-in demo", "Upload file", "SQLite table"])
    st.title("CatalogDoctor")
    st.caption(f"{page.upper()}  /  {source}")
    try:
        df, sql = get_dataset(source)
        if df is None:
            return
        if df.empty or not len(df.columns):
            st.info("This dataset is empty. Quality scores and models are unavailable.")
            return
        if len(df) > 50_000:
            st.warning("Large dataset: model fitting and profiling may take longer.")
        rules_yaml = st.sidebar.text_area(
            "Quality rules · YAML",
            value=default_rules() if source == "Built-in demo" else "rules: []",
            height=170,
            key=f"rules_{source}",
        )
        st.sidebar.caption("Edit trusted rule configuration. Rules never modify the dataset.")
        with st.spinner("Analyzing observed data…"):
            report = cached_analysis(df, rules_yaml)
        identity = dataset_identity(df)
        if page == "Overview":
            overview(report, df, sql)
        elif page == "Data Profiling":
            profiling_page(report, df)
        elif page == "Quality Rules":
            rules_page(report)
        elif page == "Anomaly Detection":
            anomalies_page(df, report, identity)
        elif page == "Classification Review":
            classification_page(df, report, identity)
        else:
            st.subheader("Take your evidence with you")
            export = deepcopy(report)
            for state, names in [
                ("numeric_override", ["statistical", "isolation_forest"]),
                ("classification_override", ["classification"]),
            ]:
                override = st.session_state.get(state)
                if override and override[0] == identity:
                    for name in names:
                        export[name] = (
                            override[1][name] if state == "numeric_override" else override[1]
                        )
            export["issues"] = [i for i in report["issues"] if i["method"] == "rule"] + sum(
                [
                    export[name]["issues"]
                    for name in ["statistical", "isolation_forest", "classification"]
                ],
                [],
            )
            st.write(
                f"{len(export['issues']):,} findings, "
                "with original row references and detection methods."
            )
            st.caption(
                "Includes the most recently selected anomaly and classification settings "
                "for this dataset; defaults apply until those pages are visited."
            )
            left, middle, right = st.columns(3)
            left.download_button(
                "Download issues · CSV",
                export_csv(export),
                "catalogdoctor-issues.csv",
                "text/csv",
                width="stretch",
            )
            middle.download_button(
                "Download analysis · JSON",
                export_json(export),
                "catalogdoctor-analysis.json",
                "application/json",
                width="stretch",
            )
            right.download_button(
                "Download report · HTML",
                export_html(export),
                "catalogdoctor-report.html",
                "text/html",
                width="stretch",
            )
            st.markdown("#### Quality score calculation")
            st.json(report["score"])
            st.caption("Exports contain observed values. Share them only with intended recipients.")
    except (ValueError, OSError) as exc:
        st.error(f"Unable to analyze this input: {exc}")
        st.caption(
            "Check the file format, configured SQLite database, "
            "or YAML column names and parameters."
        )


if __name__ == "__main__":
    main()
