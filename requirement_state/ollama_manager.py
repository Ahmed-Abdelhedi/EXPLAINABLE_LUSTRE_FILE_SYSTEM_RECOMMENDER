from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"


@dataclass(frozen=True)
class OllamaStatus:
    ready: bool
    host: str
    model: str
    server_reachable: bool
    model_available: bool
    auto_started: bool = False
    message: str = ""


def _tags(host: str, timeout: float = 1.5) -> Optional[dict]:
    url = host.rstrip("/") + "/api/tags"

    try:
        with urllib.request.urlopen(
            url,
            timeout=timeout,
        ) as response:
            return json.loads(
                response.read().decode("utf-8")
            )
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ):
        return None


def _model_names(payload: dict) -> set[str]:
    names = set()

    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue

        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str):
                names.add(value.strip())

    return names


def ensure_ollama_ready(
    *,
    host: Optional[str] = None,
    model: Optional[str] = None,
    auto_start: bool = True,
    start_timeout_seconds: float = 8.0,
) -> OllamaStatus:
    resolved_host = (
        host
        or os.getenv("OLLAMA_HOST")
        or DEFAULT_HOST
    )
    resolved_model = (
        model
        or os.getenv("OLLAMA_MODEL")
        or DEFAULT_MODEL
    )

    if not resolved_host.startswith(
        ("http://", "https://")
    ):
        resolved_host = (
            "http://" + resolved_host
        )

    payload = _tags(resolved_host)

    auto_started = False

    if payload is None and auto_start:
        executable = shutil.which("ollama")

        if executable is not None:
            creationflags = 0

            if os.name == "nt":
                creationflags = getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0,
                )

            try:
                subprocess.Popen(
                    [executable, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                auto_started = True
            except OSError:
                pass

            deadline = (
                time.monotonic()
                + start_timeout_seconds
            )

            while (
                payload is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.4)
                payload = _tags(
                    resolved_host
                )

    if payload is None:
        return OllamaStatus(
            ready=False,
            host=resolved_host,
            model=resolved_model,
            server_reachable=False,
            model_available=False,
            auto_started=auto_started,
            message=(
                "Ollama is not reachable. Start it with 'ollama serve'."
            ),
        )

    names = _model_names(payload)
    available = resolved_model in names

    if not available:
        return OllamaStatus(
            ready=False,
            host=resolved_host,
            model=resolved_model,
            server_reachable=True,
            model_available=False,
            auto_started=auto_started,
            message=(
                f"Model {resolved_model!r} is not installed. "
                f"Run: ollama pull {resolved_model}"
            ),
        )

    return OllamaStatus(
        ready=True,
        host=resolved_host,
        model=resolved_model,
        server_reachable=True,
        model_available=True,
        auto_started=auto_started,
        message="Ollama fallback is ready.",
    )
