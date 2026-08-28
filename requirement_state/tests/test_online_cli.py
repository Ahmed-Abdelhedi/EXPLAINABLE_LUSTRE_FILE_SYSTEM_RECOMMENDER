import json
from pathlib import Path

from requirement_state.online_cli import (
    _json_value,
    _save_final_json,
)


def test_json_value_is_valid_json():
    raw = _json_value(
        {
            "capacity": 100,
            "ha": True,
            "missing": None,
        }
    )

    assert json.loads(raw) == {
        "capacity": 100,
        "ha": True,
        "missing": None,
    }


def test_save_final_json(tmp_path: Path):
    path = tmp_path / "result.json"

    saved = _save_final_json(
        '{"a": 1}',
        path,
    )

    assert saved == path.resolve()
    assert json.loads(
        path.read_text(encoding="utf-8")
    ) == {"a": 1}
