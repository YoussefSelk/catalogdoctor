"""Validated YAML rules, deterministic violations, and explicit quality denominators."""

import operator
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catalogdoctor.models import Issue, Severity, issue, parse_date

RuleType = Literal[
    "required",
    "unique",
    "min_value",
    "max_value",
    "allowed_values",
    "regex",
    "string_length",
    "date_validity",
    "cross_field",
    "duplicates",
]


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str | None = None
    type: RuleType
    column: str | None = None
    columns: list[str] = Field(default_factory=list)
    severity: Severity = "medium"
    value: float | None = None
    values: list[str | int | float | bool] | None = None
    pattern: str | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    format: str | None = None
    other_column: str | None = None
    operator: Literal["lt", "le", "eq", "ne", "ge", "gt"] = "le"

    @model_validator(mode="after")
    def validate_parameters(self) -> "Rule":
        if self.type == "duplicates":
            if not self.columns or len(set(self.columns)) != len(self.columns):
                raise ValueError("duplicates requires distinct columns")
        elif not self.column:
            raise ValueError("Rule requires a column")
        if self.type in {"min_value", "max_value"} and self.value is None:
            raise ValueError("Numeric rule requires value")
        if self.type == "allowed_values" and not self.values:
            raise ValueError("allowed_values requires values")
        if self.type == "regex":
            if not self.pattern or len(self.pattern) > 256:
                raise ValueError("Regex requires a pattern of at most 256 characters")
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regex: {exc}") from exc
        if self.type == "date_validity" and not self.format:
            raise ValueError("Date rule requires format")
        if self.type == "cross_field" and not self.other_column:
            raise ValueError("cross_field requires other_column")
        if self.type == "string_length":
            if self.min_length is None and self.max_length is None:
                raise ValueError("String length requires at least one bound")
            if (
                self.min_length is not None
                and self.max_length is not None
                and self.min_length > self.max_length
            ):
                raise ValueError("min_length exceeds max_length")
        return self

    def fields(self) -> list[str]:
        if self.type == "duplicates":
            return self.columns
        return [str(self.column)] + ([self.other_column] if self.other_column else [])


class RuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rules: list[Rule]
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "completeness": 0.35,
            "uniqueness": 0.25,
            "validity": 0.25,
            "consistency": 0.15,
        }
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> "RuleConfig":
        identifiers = []
        for i, rule in enumerate(self.rules):
            if rule.id is None:
                rule.id = f"{rule.type}:{i}:{','.join(rule.fields())}"
            identifiers.append(rule.id)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Rule identifiers must be unique")
        allowed = {"completeness", "uniqueness", "validity", "consistency"}
        if not self.weights or set(self.weights) - allowed:
            raise ValueError("Unknown scoring weight")
        if (
            any(not np.isfinite(v) or v < 0 for v in self.weights.values())
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError("Weights must be finite, nonnegative, and sum to more than zero")
        return self


def load_rules(text: str) -> RuleConfig:
    if len(text.encode("utf-8")) > 256_000:
        raise ValueError("Rules exceed configuration size limit")
    try:
        return RuleConfig.model_validate(yaml.safe_load(text))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc


def applicable(df: pd.DataFrame, rule: Rule) -> pd.Series:
    fields = rule.fields()
    missing = set(fields) - set(df.columns)
    if missing:
        raise ValueError(f"Rule {rule.id} has missing columns: {sorted(missing)}")
    if rule.type == "required":
        return pd.Series(True, index=df.index)
    return df[fields].notna().all(axis=1)


def violation_mask(df: pd.DataFrame, rule: Rule) -> tuple[pd.Series, str]:
    eligible = applicable(df, rule)
    series = df[rule.fields()[0]]
    if rule.type == "required":
        mask = series.isna() | series.map(lambda v: isinstance(v, str) and not v.strip())
        expected = "a non-null, nonblank value"
    elif rule.type in {"unique", "duplicates"}:
        mask = df.duplicated(subset=rule.fields(), keep=False)
        expected = f"unique values across {', '.join(rule.fields())}"
    elif rule.type in {"min_value", "max_value"}:
        numeric = pd.to_numeric(series, errors="coerce")
        comparison = numeric < rule.value if rule.type == "min_value" else numeric > rule.value
        mask = comparison | numeric.isna() | ~np.isfinite(numeric)
        expected = (
            f"a finite numeric value {'>=' if rule.type == 'min_value' else '<='} {rule.value}"
        )
    elif rule.type == "allowed_values":
        mask = ~series.isin(rule.values or [])
        expected = f"one of {rule.values}"
    elif rule.type == "regex":
        pattern = re.compile(str(rule.pattern))
        mask = series.map(lambda v: not isinstance(v, str) or pattern.fullmatch(v) is None)
        expected = f"a full match for {rule.pattern}"
    elif rule.type == "string_length":

        def invalid_length(value: Any) -> bool:
            return (
                not isinstance(value, str)
                or (rule.min_length is not None and len(value) < rule.min_length)
                or (rule.max_length is not None and len(value) > rule.max_length)
            )

        mask = series.map(invalid_length)
        expected = f"string length between {rule.min_length} and {rule.max_length}"
    elif rule.type == "date_validity":

        def invalid_date(value: Any) -> bool:
            return parse_date(value, str(rule.format)) is None

        mask = series.map(invalid_date)
        expected = f"a valid date in exact format {rule.format}"
    else:
        left = pd.to_numeric(series, errors="coerce")
        right = pd.to_numeric(df[str(rule.other_column)], errors="coerce")
        operation = getattr(operator, rule.operator)
        mask = ~operation(left, right) | ~np.isfinite(left) | ~np.isfinite(right)
        expected = f"{rule.column} {rule.operator} {rule.other_column} (finite numeric comparison)"
    return mask.fillna(False) & eligible, expected


def evaluate_rules(df: pd.DataFrame, config: RuleConfig) -> list[Issue]:
    issues = []
    for rule in config.rules:
        mask, expected = violation_mask(df, rule)
        for position in np.flatnonzero(mask.to_numpy(dtype=bool)):
            issues.append(
                issue(
                    df,
                    int(position),
                    rule.fields(),
                    rule_id=str(rule.id),
                    severity=rule.severity,
                    method="rule",
                    expected=expected,
                    explanation=(
                        f"{', '.join(rule.fields())} violates {rule.type}: expected {expected}."
                    ),
                )
            )
    return issues


def quality_score(df: pd.DataFrame, config: RuleConfig, issues: list[Issue]) -> dict[str, Any]:
    """Deduplicate checks by row/field; renormalize weights over measured dimensions."""
    # Validate every rule even when called independently from evaluate_rules.
    eligible: dict[str, set[tuple[int, tuple[str, ...]]]] = {
        "validity": set(),
        "consistency": set(),
    }
    by_id = {str(rule.id): rule for rule in config.rules}
    for rule in config.rules:
        mask = applicable(df, rule)
        dimension = "consistency" if rule.type == "cross_field" else "validity"
        if rule.type not in {"required", "unique", "duplicates"}:
            eligible[dimension].update(
                (int(i), tuple(rule.fields())) for i in np.flatnonzero(mask.to_numpy(dtype=bool))
            )
    missing = {(int(i), str(c)) for c in df for i in np.flatnonzero(df[c].isna().to_numpy())}
    duplicate_positions = set(map(int, np.flatnonzero(df.duplicated().to_numpy())))
    failures: dict[str, set[tuple[int, tuple[str, ...]]]] = {
        "validity": set(),
        "consistency": set(),
    }
    for finding in issues:
        if finding.method != "rule" or finding.rule_id not in by_id:
            continue
        rule = by_id[finding.rule_id]
        if rule.type == "required":
            missing.add((finding.row_ref, str(rule.column)))
        elif rule.type in {"unique", "duplicates"}:
            duplicate_positions.add(finding.row_ref)
        else:
            dimension = "consistency" if rule.type == "cross_field" else "validity"
            failures[dimension].add((finding.row_ref, tuple(rule.fields())))
    denominators = {
        "completeness": df.size,
        "uniqueness": len(df),
        **{key: len(value) for key, value in eligible.items()},
    }
    failed = {
        "completeness": len(missing),
        "uniqueness": len(duplicate_positions),
        **{key: len(value) for key, value in failures.items()},
    }
    components: dict[str, Any] = {
        name: {
            "score": 100 * (1 - failed[name] / total) if total else None,
            "failed": failed[name],
            "denominator": total,
            "weight": config.weights.get(name, 0),
        }
        for name, total in denominators.items()
    }
    measured = [c for c in components.values() if c["score"] is not None and c["weight"] > 0]
    total_weight = sum(c["weight"] for c in measured)
    score = sum(c["score"] * c["weight"] for c in measured) / total_weight if total_weight else None
    return {
        "score": score,
        "components": components,
        "formula": "sum(weight * component) / sum(measured weights)",
        "note": "Deterministic checks only; heuristic and ML findings are excluded.",
    }
