"""The Streamlit page, driven headlessly with ``streamlit.testing``.

Runs against the synthetic sample with the Haar detector and the offline
template, so it needs no browser, model or network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("DERMA_FACE_DETECTOR", "haar")
    monkeypatch.setenv("DERMA_NARRATOR_ENABLED", "false")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from src.config import get_settings

    get_settings.cache_clear()
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    yield at
    get_settings.cache_clear()


def _text(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def test_the_page_boots_on_the_sample_without_an_exception(app):
    assert not app.exception
    assert "Photo quality" in _text(app)
    assert "Before and after" in _text(app)


def test_landmark_procedures_are_disabled_on_a_haar_face(app):
    sliders = {s.key: s for s in app.slider}
    assert sliders["slider_wrinkle"].disabled is False
    assert sliders["slider_lip"].disabled is True
    assert sliders["slider_under_eye"].disabled is True


def test_notes_written_for_one_edit_are_hidden_after_a_slider_moves(app):
    app.button(key="write_notes").click().run()
    assert "This preview applied" in _text(app)

    app.slider(key="slider_wrinkle").set_value(80).run()
    assert not app.exception
    assert "This preview applied" not in _text(app)


def test_setting_every_slider_to_zero_says_nothing_was_applied(app):
    for key in ("slider_wrinkle", "slider_tone", "slider_pigment"):
        app.slider(key=key).set_value(0)
    app.run()
    assert "Nothing applied yet" in _text(app)
