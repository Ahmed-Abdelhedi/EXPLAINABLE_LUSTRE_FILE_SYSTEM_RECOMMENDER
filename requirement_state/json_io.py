from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .models import FinalRequirementState


def canonical_json_string(
    state: FinalRequirementState,
    *,
    indent: int = 2,
) -> str:
    return json.dumps(
        state.to_canonical_json_dict(),
        indent=indent,
        ensure_ascii=False,
        sort_keys=False,
        allow_nan=False,
    )


def full_state_json_string(
    state: FinalRequirementState,
    *,
    indent: int = 2,
) -> str:
    return json.dumps(
        state.to_dict(
            include_traceability=True
        ),
        indent=indent,
        ensure_ascii=False,
        sort_keys=False,
        allow_nan=False,
    )


def write_canonical_json(
    state: FinalRequirementState,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.write_text(
        canonical_json_string(state) + "\n",
        encoding="utf-8",
    )
    return output
