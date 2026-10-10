import pandas as pd
import pytest

from catalogdoctor.quality import evaluate_rules, load_rules, quality_score


def test_score_components_and_no_double_counting():
    df = pd.DataFrame({"x": [None, -2.0, 2.0, 2.0]})
    rules = load_rules(
        "rules: [{column: x, type: required}, {column: x, type: min_value, value: 0}, "
        "{column: x, type: min_value, value: 1}]"
    )
    issues = evaluate_rules(df, rules)
    result = quality_score(df, rules, issues)
    assert result["components"]["completeness"]["score"] == 75
    assert result["components"]["uniqueness"]["score"] == 75
    assert result["components"]["validity"]["score"] == pytest.approx(200 / 3)
    assert result["components"]["validity"]["failed"] == 1
    assert result["components"]["validity"]["denominator"] == 3
    assert result["components"]["consistency"]["score"] is None
    assert 0 <= result["score"] <= 100


def test_empty_unavailable_and_configurable_weights():
    rules = load_rules("rules: []")
    assert quality_score(pd.DataFrame(), rules, [])["score"] is None
    rules.weights = {"completeness": 1, "uniqueness": 0, "validity": 0, "consistency": 0}
    assert quality_score(pd.DataFrame({"x": [1, None]}), rules, [])["score"] == 50
    with pytest.raises(ValueError):
        load_rules("rules: []\nweights: {completeness: -1}")
