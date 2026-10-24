from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import Settings


def test_offline_ui_runs_and_can_reject(settings, monkeypatch):
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    run_button = next(button for button in app.button if button.label == "开始分析")
    run_button.click().run()
    assert not app.exception
    assert app.session_state["result"]["status"] == "awaiting_review"
    next(button for button in app.button if button.label == "拒绝生成").click().run()
    assert not app.exception
    assert app.session_state["result"]["status"] == "rejected"


def test_repeated_ui_submission_reuses_id_and_new_task_rotates_it(settings, monkeypatch):
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
    next(box for box in app.checkbox if box.label == "生成报告前人工确认").uncheck().run()
    next(button for button in app.button if button.label == "开始分析").click().run()
    first = app.session_state["result"]["task_id"]
    next(button for button in app.button if button.label == "开始分析").click().run()
    assert not app.exception
    assert app.session_state["result"]["task_id"] == first
    next(button for button in app.button if button.label == "新建任务").click().run()
    next(button for button in app.button if button.label == "开始分析").click().run()
    assert not app.exception
    assert app.session_state["result"]["task_id"] != first
