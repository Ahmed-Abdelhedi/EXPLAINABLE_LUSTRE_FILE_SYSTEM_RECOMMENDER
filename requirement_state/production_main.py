from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from input_orchestrator import production_preflight
from input_orchestrator.production_wiring import (
    build_frozen_production_orchestrator,
)

from .finalizer import RequirementStateFinalizer
from .json_io import canonical_json_string
from .ollama_manager import ensure_ollama_ready
from .validation_repair import prepare_validation_repair


DEFAULT_OUTPUT = Path("output") / "final_requirement.json"


def _json_value(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
    )


def _save_json(
    payload: str,
    path: Path,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        payload + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def _print_updates(
    session,
    updated_fields,
) -> None:
    if not updated_fields:
        return

    print()
    print("[EXTRACTION / UPDATE]")

    for field_name in updated_fields:
        record = session.get(field_name)

        confidence = ""
        if record.confidence is not None:
            confidence = (
                f" | confidence="
                f"{record.confidence:.4f}"
            )

        print(
            f"  - {field_name} = "
            f"{_json_value(record.value)} "
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
            f"previous="
            f"{_json_value(conflict.previous_value)} "
            f"new={_json_value(conflict.new_value)}"
        )


def _print_fallback_activity(
    orchestrator,
) -> None:
    components = getattr(
        orchestrator,
        "production_components",
        {},
    )

    quantity = components.get("quantity")
    result = getattr(
        quantity,
        "last_result",
        None,
    )

    quantity_events = []

    if result is not None:
        cascade = getattr(
            result,
            "cascade",
            None,
        )
        traces = getattr(
            cascade,
            "traces",
            {},
        )

        for quantity_id, trace in traces.items():
            if getattr(
                trace,
                "llm_attempted",
                False,
            ):
                quantity_events.append(
                    {
                        "quantity_id": quantity_id,
                        "resolved": bool(
                            getattr(
                                trace,
                                "llm_resolved",
                                False,
                            )
                        ),
                    }
                )

    preference = components.get("preference")
    pref_result = getattr(
        preference,
        "last_layer2_result",
        None,
    )

    preference_llm_dimensions = []

    if pref_result is not None:
        dimensions = getattr(
            pref_result,
            "dimensions",
            {},
        )

        for dimension, detail in dimensions.items():
            source_obj = getattr(
                detail,
                "source",
                None,
            )
            source = getattr(
                source_obj,
                "value",
                str(source_obj),
            )

            if source == "LLM_FALLBACK":
                preference_llm_dimensions.append(
                    getattr(
                        dimension,
                        "value",
                        str(dimension),
                    )
                )

    categorical = components.get("categorical")
    cat_result = getattr(
        categorical,
        "last_result",
        None,
    )

    categorical_llm_fields = []

    if cat_result is not None:
        payload = (
            cat_result.to_dict()
            if hasattr(cat_result, "to_dict")
            else {}
        )

        for field_name in (
            "ha_required",
            "access_type",
        ):
            detail = payload.get(field_name)
            if (
                isinstance(detail, dict)
                and str(
                    detail.get("source")
                ).upper()
                == "LLM_FALLBACK"
            ):
                categorical_llm_fields.append(
                    field_name
                )

    if (
        quantity_events
        or preference_llm_dimensions
        or categorical_llm_fields
    ):
        print()
        print("[LLM FALLBACK ACTIVITY]")

        for event in quantity_events:
            print(
                "  - quantity "
                f"{event['quantity_id']}: "
                f"attempted, "
                f"resolved={event['resolved']}"
            )

        for dimension in preference_llm_dimensions:
            print(
                "  - preference "
                f"{dimension}: resolved by LLM"
            )

        for field_name in categorical_llm_fields:
            print(
                "  - categorical "
                f"{field_name}: resolved by LLM"
            )


def _print_state(session) -> None:
    print()
    print("[CURRENT REQUIREMENT STATE]")

    for field_name, record in session.fields.items():
        print(
            f"  - {field_name}: "
            f"{_json_value(record.value)} "
            f"[{record.state.value}]"
        )


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  /state  show current Requirement state\n"
        "  /help   show commands\n"
        "  /quit   stop the pipeline\n"
        "\nFor an optional field with no constraint, answer: skip\n"
        "Read/write ratio examples: 20/80, 80:20, "
        "'20 read 80 write'.\n"
    )


def _print_validation_issues(
    state,
) -> None:
    print()
    print("[FINAL VALIDATION] NOT READY FOR SIZING")

    for issue in state.validation_issues:
        print(
            f"  - {issue.get('code')}: "
            f"{issue.get('message')}"
        )


def run_production(
    *,
    device: Optional[str] = None,
    enable_llm_fallback: bool = True,
    auto_start_ollama: bool = True,
    output_path: Path = DEFAULT_OUTPUT,
) -> int:
    preflight = production_preflight()

    if not preflight["ready"]:
        print(
            "ERROR: frozen production model artifacts are missing."
        )
        print(
            json.dumps(
                preflight,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2

    if enable_llm_fallback:
        ollama = ensure_ollama_ready(
            auto_start=auto_start_ollama,
        )

        print(
            "[OLLAMA] "
            f"{ollama.message}"
        )
        print(
            f"[OLLAMA] host={ollama.host} "
            f"model={ollama.model}"
        )

        if not ollama.ready:
            return 2

    print("=" * 76)
    print("LUSTRE ONLINE REQUIREMENT PIPELINE - PRODUCTION MAIN")
    print("=" * 76)
    print(
        f"Device: {device or 'auto'}\n"
        f"LLM fallback: "
        f"{'ENABLED' if enable_llm_fallback else 'DISABLED'}\n"
        f"Output: {output_path}"
    )
    _print_help()

    orchestrator = (
        build_frozen_production_orchestrator(
            device=device,
            enable_llm_fallback=enable_llm_fallback,
        )
    )

    finalizer = RequirementStateFinalizer()

    session = orchestrator.new_session(
        "PRODUCTION_ONLINE_SESSION"
    )

    print(
        "Enter the user requirement in natural language."
    )

    while True:
        try:
            user_text = input("USER> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nPipeline stopped.")
            return 130

        if not user_text:
            continue

        command = user_text.casefold()

        if command == "/quit":
            print("Pipeline stopped.")
            return 0

        if command == "/help":
            _print_help()
            continue

        if command == "/state":
            _print_state(session)
            continue

        try:
            response = orchestrator.handle_message(
                user_text,
                session,
            )
        except Exception as exc:
            print()
            print(
                "[PIPELINE ERROR] "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        _print_updates(
            session,
            response.updated_fields,
        )
        _print_conflicts(response)
        _print_fallback_activity(
            orchestrator
        )

        print()
        print(
            "[STATE] "
            f"{response.conversation_state.value}"
        )

        if response.assistant_message:
            print()
            print(
                "SYSTEM> "
                f"{response.assistant_message}"
            )

        if not response.ready_for_final_validation:
            continue

        # ----------------------------------------------------------
        # Deterministic final validation is part of the ONLINE loop.
        # A failure does NOT terminate the process. It is converted
        # back into a targeted clarification.
        # ----------------------------------------------------------
        state = finalizer.from_session(
            session
        )

        if not state.ready_for_sizing:
            _print_validation_issues(
                state
            )

            repair = prepare_validation_repair(
                state=state,
                session=session,
            )

            if repair.mode == "RESTART_BWM":
                print()
                print(
                    "SYSTEM> "
                    f"{repair.question}"
                )

                response = orchestrator.handle_message(
                    "continue",
                    session,
                )

                print()
                print(
                    "[STATE] "
                    f"{response.conversation_state.value}"
                )

                if response.assistant_message:
                    print()
                    print(
                        "SYSTEM> "
                        f"{response.assistant_message}"
                    )

                continue

            if repair.mode == "ASK_FIELD":
                print()
                print(
                    "SYSTEM> "
                    f"{repair.question}"
                )
                continue

            print(
                "[FINALIZATION ERROR] "
                "No deterministic repair action available."
            )
            return 3

        canonical = canonical_json_string(
            state
        )

        saved = _save_json(
            canonical,
            output_path,
        )

        print()
        print(
            "[FINAL VALIDATION] PASS"
        )
        print(
            "[READY FOR SIZING] true"
        )
        print()
        print("=" * 76)
        print("FINAL CANONICAL REQUIREMENT JSON")
        print("=" * 76)
        print(canonical)
        print()
        print(
            f"Saved to: {saved}"
        )
        print()
        print(
            "STATUS: PRODUCTION_ONLINE_PIPELINE_COMPLETE"
        )

        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete online Requirement pipeline "
            "from user text to final canonical JSON."
        )
    )

    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, or leave unset for runtime auto-selection.",
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Explicitly disable all configured LLM fallbacks.",
    )

    parser.add_argument(
        "--no-auto-start-ollama",
        action="store_true",
        help=(
            "Do not attempt to start 'ollama serve' automatically "
            "when LLM fallback is enabled."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Final canonical JSON path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raise SystemExit(
        run_production(
            device=args.device,
            enable_llm_fallback=(
                not args.no_llm
            ),
            auto_start_ollama=(
                not args.no_auto_start_ollama
            ),
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
