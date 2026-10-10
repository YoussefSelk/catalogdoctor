import pandas as pd

from catalogdoctor.detection import classification_review


def test_insufficient_labels_and_empty_vocabulary():
    df = pd.DataFrame({"name": ["x", "y"], "category": ["A", "B"]})
    assert classification_review(df, ["name"], "category")["status"] == "unavailable"
    df = pd.DataFrame({"name": ["!"] * 50, "category": ["A", "B"] * 25})
    assert classification_review(df, ["name"], "category")["status"] == "unavailable"


def test_out_of_fold_disagreements_and_real_metrics():
    records = []
    for category, text in [
        ("Electronics", "wireless bluetooth headphones audio speaker"),
        ("Kitchen", "kitchen cooking saucepan utensils cookware"),
    ]:
        for i in range(60):
            records.append({"name": f"{text} model {i}", "category": category})
    records[0]["category"] = "Kitchen"
    df = pd.DataFrame(records)
    result = classification_review(df, ["name"], "category", confidence=0.65)
    assert result["status"] == "ok"
    assert result["metrics"]["accuracy"] > 0.9
    assert result["metrics"]["evaluated_rows"] == len(df)
    assert 0 in [i.row_ref for i in result["issues"]]
    assert all(i.method == "classification_suspicion" for i in result["issues"])
    assert result["predictions"][0]["suggested"] == "Electronics"
    assert result["evaluation"] == "stratified group out-of-fold; identical text groups held out"


def test_duplicate_text_cannot_supply_its_own_training_evidence():
    df = pd.DataFrame(
        {
            "name": ["electronics headphones"] * 30 + ["kitchen pot"] * 30,
            "category": ["A"] * 30 + ["B"] * 30,
        }
    )
    result = classification_review(df, ["name"], "category")
    assert result["status"] == "unavailable"
    assert "distinct text" in result["reason"]
