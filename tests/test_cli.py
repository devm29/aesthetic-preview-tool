"""Tests for the headless CLI.

The CLI exists to prove the pipeline runs without Streamlit, so these double as
an architecture guard.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import cli
from src.config import DISCLAIMER
from src.imaging import encode_png
from src.treatments import all_treatments


@pytest.fixture
def sample_path(tmp_path, sample_bytes):
    path = tmp_path / "face.jpg"
    path.write_bytes(sample_bytes)
    return path


@pytest.fixture(autouse=True)
def haar_only(monkeypatch):
    """Pin the detector so the CLI output is identical on every machine."""
    monkeypatch.setenv("DERMA_FACE_DETECTOR", "haar")
    monkeypatch.setenv("DERMA_NARRATOR_ENABLED", "false")
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_the_parser_exposes_one_flag_per_registered_treatment():
    """Regression guard for the registry seam: a new treatment must not need
    a CLI edit."""
    parser = cli.build_parser()
    actions = {a.dest for a in parser._actions}
    for treatment in all_treatments():
        assert treatment.key in actions


def test_missing_file_exits_with_a_usage_error(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope.jpg")]) == 2
    assert "No such image" in capsys.readouterr().err


def test_a_faceless_image_exits_nonzero_and_says_so(tmp_path, capsys):
    blank = tmp_path / "blank.png"
    blank.write_bytes(encode_png(np.zeros((300, 300, 3), np.uint8)))
    assert cli.main([str(blank)]) == 1
    out = capsys.readouterr().out
    assert "No face detected" in out
    assert "[FAIL] Face detected" in out


def test_a_successful_run_prints_the_checks_and_the_disclaimer(sample_path, capsys):
    assert cli.main([str(sample_path), "--wrinkle", "50"]) == 0
    out = capsys.readouterr().out
    assert "[PASS] Face detected" in out
    assert "[SKIP] Head tilt" in out, "the bbox-only detector must report a skip"
    assert "applied  Wrinkle smoothing @ 50%" in out
    assert "Identity   : SSIM" in out
    assert DISCLAIMER in out


def test_zero_intensities_report_that_nothing_was_applied(sample_path, capsys):
    argv = [str(sample_path)]
    for treatment in all_treatments():
        argv += [f"--{treatment.key.replace('_', '-')}", "0"]
    assert cli.main(argv) == 0
    assert "applied  nothing" in capsys.readouterr().out


def test_json_output_is_machine_readable(sample_path, capsys):
    assert cli.main([str(sample_path), "--wrinkle", "40", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["disclaimer"] == DISCLAIMER
    assert payload["detector"] == "haar"
    assert payload["landmarks"] is False
    assert payload["treatments"][0]["key"] == "wrinkle"
    assert payload["identity"]["within_threshold"] is True


def test_landmark_treatments_are_reported_as_notes_on_a_bbox_only_face(sample_path, capsys):
    assert cli.main([str(sample_path), "--lip", "60"]) == 0
    assert "note     Lip volume" in capsys.readouterr().out


def test_out_writes_a_before_after_png(sample_path, tmp_path, capsys):
    from src.imaging import decode_image

    destination = tmp_path / "nested" / "out.png"
    assert cli.main([str(sample_path), "--wrinkle", "60", "--out", str(destination)]) == 0
    assert destination.is_file()
    assert f"Wrote {destination}" in capsys.readouterr().out

    export = decode_image(destination.read_bytes())
    assert export.shape[1] > export.shape[0], "export should be a side-by-side pair"


def test_failing_quality_checks_block_rendering_until_forced(tmp_path, capsys):
    """A soft image must not be silently rendered."""
    import cv2

    from src.imaging import decode_image

    from .conftest import SAMPLE_PATH

    blurred = tmp_path / "soft.png"
    image = decode_image(SAMPLE_PATH.read_bytes())
    soft = cv2.GaussianBlur(image, (0, 0), sigmaX=6)
    blurred.write_bytes(encode_png(soft))

    assert cli.main([str(blurred)]) == 1
    assert "Re-run with --force" in capsys.readouterr().out

    assert cli.main([str(blurred), "--force", "--wrinkle", "30"]) == 0
    assert "applied  Wrinkle smoothing" in capsys.readouterr().out
