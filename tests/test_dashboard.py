from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src/catalogdoctor/dashboard/app.py"


def test_demo_dashboard_all_pages():
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    assert app.metric[1].value == "1,500"
    for page in [
        "Data Profiling",
        "Quality Rules",
        "Anomaly Detection",
        "Classification Review",
        "Reports & Export",
    ]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_invalid_rules_have_helpful_error():
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.sidebar.text_area[0].set_value("rules: [{column: absent, type: required}]").run()
    assert not app.exception
    assert any("missing columns" in item.value for item in app.error)
