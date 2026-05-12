"""Tests for write_json_atomic: ensures the .tmp+rename pattern is bulletproof."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from utils import write_json_atomic


def test_atomic_write_creates_final_file_only(tmp_path: Path):
    path = tmp_path / "out.json"
    write_json_atomic({"a": 1}, path)

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["out.json"]
    assert not (tmp_path / "out.json.tmp").exists()


def test_atomic_write_round_trips_data(tmp_path: Path):
    path = tmp_path / "out.json"
    obj = {"x": [1, 2, 3], "nested": {"y": "hello"}, "n": 42}
    write_json_atomic(obj, path)

    assert json.loads(path.read_text()) == obj


def test_atomic_write_overwrites_existing_file(tmp_path: Path):
    path = tmp_path / "out.json"
    path.write_text('{"old": true}')

    write_json_atomic({"new": True}, path)

    assert json.loads(path.read_text()) == {"new": True}
    assert not (tmp_path / "out.json.tmp").exists()


def test_atomic_write_failure_does_not_corrupt_existing(tmp_path: Path):
    path = tmp_path / "out.json"
    path.write_text('{"original": "intact"}')

    with patch("utils.json.dump", side_effect=RuntimeError("disk full")):
        with pytest.raises(RuntimeError):
            write_json_atomic({"new": "data"}, path)

    assert json.loads(path.read_text()) == {"original": "intact"}


def test_atomic_write_with_unicode_and_default_str(tmp_path: Path):
    path = tmp_path / "out.json"
    obj = {
        "unicode": "café — naïve",
        "datetime": datetime(2026, 4, 27),
    }
    write_json_atomic(obj, path)

    raw = path.read_text()
    assert "café" in raw
    assert "naïve" in raw
    assert "2026-04-27" in raw


def test_atomic_write_returns_path(tmp_path: Path):
    path = tmp_path / "out.json"
    returned = write_json_atomic({"k": "v"}, path)
    assert returned == path


def test_atomic_write_accepts_str_path(tmp_path: Path):
    path = tmp_path / "out.json"
    write_json_atomic({"k": "v"}, str(path))
    assert path.exists()


def test_atomic_write_no_partial_visible_at_path(tmp_path: Path):
    """The tmp file must never be the final destination, even briefly.

    Patches `os.replace` to fail; verifies the .tmp is what existed up to that
    point — and the final path is still empty (no half-write).
    """
    path = tmp_path / "out.json"
    tmp = tmp_path / "out.json.tmp"

    with patch("utils.os.replace", side_effect=OSError("rename failed")):
        with pytest.raises(OSError):
            write_json_atomic({"a": 1}, path)

    assert not path.exists()
    assert tmp.exists()
