"""Explainable heuristics and reproducible, deliberately limited ML review."""

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from catalogdoctor.models import Issue, issue, scalar


def numeric_features(df: pd.DataFrame) -> list[str]:
    return [
        str(c)
        for c in df.select_dtypes(include="number").columns
        if str(c).lower() not in {"id", "sku"} and not str(c).lower().endswith(("_id", "_key"))
    ]


def statistical(
    df: pd.DataFrame,
    features: list[str],
    *,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float | None = None,
) -> dict[str, Any]:
    if method not in {"iqr", "zscore"}:
        raise ValueError("Method must be iqr or zscore")
    threshold = threshold if threshold is not None else (1.5 if method == "iqr" else 3.0)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Threshold must be finite and positive")
    boundaries: dict[str, Any] = {}
    issues: list[Issue] = []
    for column in features:
        if column not in df or not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Choose numeric features: {column}")
        values = pd.to_numeric(df[column]).replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if len(finite) < 4 or finite.nunique() < 2:
            boundaries[column] = {"status": "constant" if len(finite) >= 4 else "insufficient data"}
            continue
        if method == "iqr":
            q1, q3 = float(finite.quantile(0.25)), float(finite.quantile(0.75))
            lower, upper = q1 - threshold * (q3 - q1), q3 + threshold * (q3 - q1)
        else:
            mean, std = float(finite.mean()), float(finite.std(ddof=0))
            lower, upper = mean - threshold * std, mean + threshold * std
        boundaries[column] = {
            "status": "ok",
            "lower": float(lower),
            "upper": float(upper),
            "threshold": threshold,
        }
        mask = (values < lower) | (values > upper)
        for position in np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool)):
            value = values.iloc[position]
            issues.append(
                issue(
                    df,
                    int(position),
                    [column],
                    rule_id=f"{method}:{column}",
                    method=f"statistical:{method}",
                    expected=f"between {lower:.4g} and {upper:.4g}",
                    explanation=f"{column} {value:.4g} falls outside {method.upper()} thresholds "
                    f"[{lower:.4g}, {upper:.4g}]. An outlier is not necessarily an error.",
                )
            )
    return {"issues": issues, "boundaries": boundaries}


def isolation_forest(
    df: pd.DataFrame,
    features: list[str] | None = None,
    *,
    contamination: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    if not np.isfinite(contamination) or not 0 < contamination <= 0.5:
        raise ValueError("Contamination must be in (0, 0.5]")
    features = numeric_features(df) if features is None else features
    unavailable = {
        "status": "unavailable",
        "reason": "Need at least 20 rows and a varying numeric feature.",
        "issues": [],
        "scores": [],
        "features": [],
    }
    if len(df) < 20 or not features:
        return unavailable
    if set(features) - set(df.select_dtypes(include="number").columns):
        raise ValueError("Select existing numeric features")
    numeric = df[features].replace([np.inf, -np.inf], np.nan)
    usable = [c for c in features if numeric[c].notna().sum() >= 5 and numeric[c].nunique() >= 2]
    if not usable:
        return unavailable
    numeric = numeric[usable]
    complete_evidence = numeric.notna().any(axis=1).to_numpy()
    positions = np.flatnonzero(complete_evidence)
    if len(positions) < 20:
        return unavailable
    matrix = SimpleImputer(strategy="median").fit_transform(numeric.iloc[positions])
    model = IsolationForest(
        n_estimators=150, contamination=contamination, random_state=seed, n_jobs=1
    )
    predicted = model.fit_predict(matrix)
    scores = -model.decision_function(matrix)  # Positive = flagged, larger = more unusual.
    issues = []
    contexts = []
    medians = numeric.median()
    spreads = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, np.nan)
    for local in np.flatnonzero(predicted == -1):
        position = int(positions[local])
        context = {
            c: {
                "observed": scalar(numeric.iloc[position][c]),
                "median": scalar(medians[c]),
                "distance_in_iqr": scalar((numeric.iloc[position][c] - medians[c]) / spreads[c]),
            }
            for c in usable
        }
        contexts.append(
            {"row_ref": position, "score": float(scores[local]), "comparisons": context}
        )
        issues.append(
            issue(
                df,
                position,
                usable,
                rule_id="isolation_forest",
                method="ml:isolation_forest",
                expected="review unusually isolated numeric combinations",
                explanation=f"IsolationForest flagged this row (score {scores[local]:.4f}). "
                "Feature comparisons describe unusual values, not causal model explanations.",
            )
        )
    return {
        "status": "ok",
        "issues": issues,
        "features": usable,
        "scores": [
            {"row_ref": int(position), "score": float(score)}
            for position, score in zip(positions, scores, strict=True)
        ],
        "context": contexts,
        "context_note": "Descriptive comparisons only; not causal explanations.",
        "excluded_rows": len(df) - len(positions),
        "seed": seed,
        "contamination": contamination,
    }


def classification_review(
    df: pd.DataFrame,
    text_columns: list[str],
    target: str,
    *,
    confidence: float = 0.65,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate each row out of fold; keep normalized identical text groups together."""
    if not 0.5 <= confidence <= 1:
        raise ValueError("Confidence must be between 0.5 and 1")
    if not text_columns or target in text_columns or set(text_columns + [target]) - set(df.columns):
        raise ValueError("Choose existing text fields distinct from the target")
    unavailable: dict[str, Any] = {
        "status": "unavailable",
        "issues": [],
        "metrics": None,
        "predictions": [],
    }
    texts = df[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    labels = df[target].fillna("").astype(str)
    valid = texts.ne("") & labels.str.strip().ne("")
    positions = np.flatnonzero(valid.to_numpy())
    x = texts.iloc[positions].reset_index(drop=True)
    y = labels.iloc[positions].reset_index(drop=True)
    groups = x.str.lower().str.replace(r"\s+", " ", regex=True)
    if len(x) < 40 or y.nunique() < 2 or y.value_counts().min() < 5:
        return {
            **unavailable,
            "reason": "Need 40 labeled text rows, two classes, "
            "and at least five records per class.",
        }
    group_counts = pd.DataFrame({"y": y, "group": groups}).groupby("y")["group"].nunique()
    if group_counts.min() < 3:
        return {
            **unavailable,
            "reason": "Need at least three distinct text groups per class "
            "for leakage-safe evaluation.",
        }
    predictions: list[dict[str, Any]] = []
    observed: list[str] = []
    suggested: list[str] = []
    issues: list[Issue] = []
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    try:
        for train, test in folds.split(x, y, groups):
            if y.iloc[train].nunique() != y.nunique():
                return {
                    **unavailable,
                    "reason": "Class imbalance prevents a fold with all classes in training.",
                }
            pipeline = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), max_features=15_000, sublinear_tf=True),
                LogisticRegression(max_iter=500, random_state=seed, class_weight="balanced"),
            )
            pipeline.fit(x.iloc[train], y.iloc[train])
            probabilities = pipeline.predict_proba(x.iloc[test])
            classes = pipeline.classes_
            for local, test_index in enumerate(test):
                best = int(probabilities[local].argmax())
                prediction, probability = str(classes[best]), float(probabilities[local, best])
                actual = str(y.iloc[test_index])
                position = int(positions[test_index])
                observed.append(actual)
                suggested.append(prediction)
                predictions.append(
                    {
                        "row_ref": position,
                        "original": actual,
                        "suggested": prediction,
                        "confidence": probability,
                        "flagged": prediction != actual and probability >= confidence,
                    }
                )
                if prediction != actual and probability >= confidence:
                    issues.append(
                        issue(
                            df,
                            position,
                            [target, *text_columns],
                            rule_id="text_classifier",
                            method="classification_suspicion",
                            expected="human review of category assignment",
                            explanation=f"Text resembles records labeled {prediction} "
                            f"({probability:.1%} model probability), "
                            f"but the observed category is {actual}. "
                            "This is a suggestion, not a confirmed error.",
                        )
                    )
    except ValueError as exc:
        return {**unavailable, "reason": f"Text model unavailable: {exc}"}
    return {
        "status": "ok",
        "issues": sorted(issues, key=lambda i: i.row_ref),
        "predictions": sorted(predictions, key=lambda p: p["row_ref"]),
        "metrics": {
            "accuracy": float(accuracy_score(observed, suggested)),
            "balanced_accuracy": float(balanced_accuracy_score(observed, suggested)),
            "macro_f1": float(f1_score(observed, suggested, average="macro", zero_division=0)),
            "evaluated_rows": len(observed),
        },
        "evaluation": "stratified group out-of-fold; identical text groups held out",
        "limitation": "Metrics measure agreement with existing labels. "
        "Noisy labels can reproduce errors; "
        "probabilities are uncalibrated, and novel categories cannot be discovered.",
    }
