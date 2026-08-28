from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from input_orchestrator import production_preflight
from input_orchestrator.production_wiring import (
    build_frozen_production_orchestrator,
)

from .production import finalize_orchestrator_session


DEFAULT_OUTPUT = Path("output") / "final_requirement.json"


def _json_value(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
    )


def _print_updated_fields(
    session,
    updated_fields: list[str],
) -> None:
    if not updated_fields:
        return

    print()
    print("[EXTRACTION / UPDATE]")

    for field_name in updated_fields:
        record = session.get(field_name)

        confidence = (
            ""
            if record.confidence is None
            else f" | confidence={record.confidence:.4f}"
        )

        print(
            f"  - {field_name} = {_json_value(record.value)} "
            f"| {record.state.value} "
            f"| source={record.source}"
            f"{confidence}"
        )


def _print_conflicts(response) -> None:
    if not response.conflicts:
        return

    print()
    print("[CONFLICT]")

    for conflict in response.conflicts:
        print(
            f"  - {conflict.field}: "
            f"previous={_json_value(conflict.previous_value)} "
            f"new={_json_value(conflict.new_value)}"
        )


def _print_state(response) -> None:
    print()
    print(
        f"[STATE] {response.conversation_state.value}"
    )


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  /state  show the current collected Requirement fields\n"
        "  /quit   stop the interactive session\n"
        "  /help   show these commands\n"
        "\nDuring clarification you can answer normally. For optional "
        "fields, 'skip' is accepted when you have no constraint.\n"
    )


def _print_session_state(session) -> None:
    print()
    print("[CURRENT REQUIREMENT STATE]")

    for field_name, record in session.fields.items():
        if field_name == "preference_weights":
            continue

        print(
            f"  - {field_name}: "
            f"{_json_value(record.value)} "
            f"[{record.state.value}]"
        )

    weights = session.get("preference_weights")
    print(
        "  - preference_weights: "
        f"{_json_value(weights.value)} "
        f"[{weights.state.value}]"
    )


def _save_final_json(
    canonical_json: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        canonical_json + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()


def run_interactive(
    *,
    device: Optional[str] = None,
    enable_llm_fallback: bool = False,
    output_path: Path = DEFAULT_OUTPUT,
) -> int:
    preflight = production_preflight()

    if not preflight["ready"]:
        print(
            "ERROR: production artifacts are missing."
        )
        print(
            json.dumps(
                preflight,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2

    print("=" * 72)
    print("LUSTRE REQUIREMENT ONLINE PIPELINE")
    print("=" * 72)
    print(
        "Enter your Lustre requirement in natural language.\n"
        "The system will extract fields, ask clarifications, run BWM "
        "if needed, validate the final state, and emit canonical JSON."
    )
    print(
        f"\nDevice: {device or 'auto'}"
        f"\nLLM fallback: {'ENABLED' if enable_llm_fallback else 'DISABLED'}"
        f"\nFinal JSON output: {output_path}"
    )
    _print_help()

    orchestrator = (
        build_frozen_production_orchestrator(
            device=device,
            enable_llm_fallback=enable_llm_fallback,
        )
    )
    session = orchestrator.new_session(
        "MANUAL_ONLINE_SESSION"
    )

    while True:
        try:
            user_text = input("USER> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession stopped.")
            return 130

        if not user_text:
            continue

        command = user_text.lower()

        if command == "/quit":
            print("Session stopped.")
            return 0

        if command == "/help":
            _print_help()
            continue

        if command == "/state":
            _print_session_state(session)
            continue

        try:
            response = orchestrator.handle_message(
                user_text,
                session,
            )
        except Exception as exc:
            print()
            print(
                f"[PIPELINE ERROR] "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        _print_updated_fields(
            session,
            response.updated_fields,
        )
        _print_conflicts(response)
        _print_state(response)

        if response.assistant_message:
            print()
            print(
                f"SYSTEM> {response.assistant_message}"
            )

        if not response.ready_for_final_validation:
            continue

        print()
        print(
            "[FINAL VALIDATION] "
            "Orchestrator is READY_FOR_FINAL_VALIDATION."
        )

        try:
            output = finalize_orchestrator_session(
                session
            )
        except Exception as exc:
            print(
                f"[FINALIZATION ERROR] "
                f"{type(exc).__name__}: {exc}"
            )
            return 3

        print(
            f"[READY FOR SIZING] "
            f"{output.state.ready_for_sizing}"
        )

        print()
        print("=" * 72)
        print("FINAL CANONICAL REQUIREMENT JSON")
        print("=" * 72)
        print(output.canonical_json)

        saved = _save_final_json(
            output.canonical_json,
            output_path,
        )

        print()
        print(
            f"Saved to: {saved}"
        )
        print()
        print(
            "STATUS: MANUAL_ONLINE_PIPELINE_COMPLETE"
        )

        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive online Requirement Extraction pipeline "
            "from natural-language user text to final canonical JSON."
        )
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Model device, for example cpu or cuda. "
            "Default: runtime auto-selection."
        ),
    )

    parser.add_argument(
        "--enable-llm-fallback",
        action="store_true",
        help=(
            "Enable configured LLM fallbacks. "
            "Requires the local LLM service when a fallback is triggered."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path where the final canonical Requirement JSON is saved. "
            "Default: output/final_requirement.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise SystemExit(
        run_interactive(
            device=args.device,
            enable_llm_fallback=args.enable_llm_fallback,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
