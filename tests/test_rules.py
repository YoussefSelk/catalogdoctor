import pandas as pd
import pytest
from pydantic import ValidationError

from catalogdoctor.quality import Rule, RuleConfig, evaluate_rules, load_rules


@pytest.mark.parametrize(
    ("rule", "values", "bad"),
    [
        ({"type": "required"}, [None, "", "ok"], [0, 1]),
        ({"type": "unique"}, ["a", "a", None], [0, 1]),
        ({"type": "min_value", "value": 0}, [-1, "bad", 2, None], [0, 1]),
        ({"type": "max_value", "value": 5}, [10, "bad", 2, None], [0, 1]),
        ({"type": "allowed_values", "values": ["A"]}, ["B", "A", None], [0]),
        (
            {"type": "regex", "pattern": r"^[A-Z]{3}-[0-9]{4}$"},
            ["ABC-1234", "ABC-1234\nx", None],
            [1],
        ),
        (
            {"type": "string_length", "min_length": 2, "max_length": 4},
            ["a", "abcde", "abc", None],
            [0, 1],
        ),
        ({"type": "date_validity", "format": "%Y-%m-%d"}, ["2025-02-30", "2025-02-01", None], [0]),
    ],
)
def test_rule_families(rule, values, bad):
    config = RuleConfig(rules=[Rule(column="x", severity="critical", **rule)])
    df = pd.DataFrame({"x": values}, index=["same"] * len(values))
    result = evaluate_rules(df, config)
    assert [i.row_ref for i in result] == bad
    assert all(i.severity == "critical" and i.method == "rule" for i in result)
    assert all(i.expected and i.explanation and i.source_index == "same" for i in result)


def test_cross_field_and_duplicate_subset():
    df = pd.DataFrame({"a": [1, 4, 1], "b": [2, 3, 2]})
    rules = RuleConfig(
        rules=[
            Rule(type="cross_field", column="a", other_column="b", operator="le"),
            Rule(type="duplicates", columns=["a", "b"]),
        ]
    )
    issues = evaluate_rules(df, rules)
    assert [i.row_ref for i in issues if len(i.fields) == 2 and "cross" in i.rule_id] == [1]
    assert len(issues) == 3


@pytest.mark.parametrize(
    "text",
    [
        "rules: [{column: x, type: unknown}]",
        'rules: [{column: x, type: regex, pattern: "["}]',
        "rules: [{column: x, type: min_value}]",
        "rules: [{column: x, type: required, severity: urgent}]",
        "rules: [{type: duplicates, columns: []}]",
        "rules: [{column: x, type: required, typo: true}]",
        "rules: [{id: a, column: x, type: required}, {id: a, column: x, type: unique}]",
    ],
)
def test_invalid_configuration(text):
    with pytest.raises((ValueError, ValidationError)):
        load_rules(text)


def test_missing_columns_fail_loudly():
    with pytest.raises(ValueError, match="missing columns"):
        evaluate_rules(pd.DataFrame({"a": [1]}), load_rules("rules: [{column: b, type: required}]"))


def test_text_rules_use_original_csv_lexemes():
    from io import BytesIO

    from catalogdoctor.ingestion import load_file

    df = load_file(BytesIO(b"code\n12345\n54321\n"), "x.csv")
    config = load_rules(
        "rules: [{column: code, type: regex, pattern: '^[0-9]{5}$'}, "
        "{column: code, type: string_length, min_length: 5, max_length: 5}, "
        "{column: code, type: allowed_values, values: ['12345', '54321']}]"
    )
    assert not evaluate_rules(df, config)
