"""Production entry point: Requirement conversation + frozen Lustre E2E chain."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from e2e_pipeline.end_to_end_pipeline import (
    DEFAULT_E2E_OUTPUT,
    PipelineLimits,
    run_e2e_from_file,
)
from requirement_state.production_main import (
    DEFAULT_OUTPUT as DEFAULT_REQUIREMENT_OUTPUT,
    run_production,
)


def _file_fingerprint(path: Path) -> tuple[int, int, str] | None:
    if not path.exists() or not path.is_file():
        return None

    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (stat.st_mtime_ns, stat.st_size, digest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production Requirement conversation, then the frozen "
            "Lustre sizing/ranking/full-architecture pipeline."
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
        help="Disable configured LLM fallbacks.",
    )
    parser.add_argument(
        "--no-auto-start-ollama",
        action="store_true",
        help="Do not start 'ollama serve' automatically.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REQUIREMENT_OUTPUT,
        help="Canonical Requirement JSON path.",
    )
    parser.add_argument(
        "--requirement-only",
        action="store_true",
        help="Stop after the canonical Requirement JSON, like the old main.py.",
    )
    parser.add_argument(
        "--e2e-output",
        type=Path,
        default=DEFAULT_E2E_OUTPUT,
        help="Final E2E result JSON path.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-paths-per-variant", type=int, default=2)
    parser.add_argument("--max-role-options", type=int, default=4)
    parser.add_argument("--max-architectures", type=int, default=16)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    limits = PipelineLimits(
        top_k=args.top_k,
        max_paths_per_variant=args.max_paths_per_variant,
        max_role_options_per_role=args.max_role_options,
        max_architectures=args.max_architectures,
    )

    requirement_path = Path(args.output)
    before = _file_fingerprint(requirement_path)

    requirement_status = run_production(
        device=args.device,
        enable_llm_fallback=(not args.no_llm),
        auto_start_ollama=(not args.no_auto_start_ollama),
        output_path=requirement_path,
    )

    if requirement_status != 0:
        return requirement_status

    if args.requirement_only:
        return 0

    after = _file_fingerprint(requirement_path)

    # run_production returns 0 on /quit too.  Never consume an old Requirement
    # left from a previous session as if it had just been finalized.
    if after is None or after == before:
        print()
        print(
            "[E2E] Aucun nouveau Requirement finalisé pendant cette session. "
            "Le downstream n'est pas lancé."
        )
        return 0

    print()
    print("=" * 76)
    print("LUSTRE END-TO-END DOWNSTREAM")
    print("Requirement -> S10 -> Ranking -> H8 -> H9 -> H10")
    print("=" * 76)

    result = run_e2e_from_file(
        requirement_path,
        output_path=Path(args.e2e_output),
        limits=limits,
    )

    print(f"[E2E STATUS] {result['status']}")
    print(f"[E2E OUTPUT] {result['trace'].get('output_path', args.e2e_output)}")

    if result["status"] == "SUCCESS":
        best = result["best_architecture"]
        print(f"[BEST ARCHITECTURE] {best['architecture_id']}")
        print(f"[H9 SCORE] {best['score']['score']:.6f}")
        print("[H10 VALID] true")
    else:
        print(f"[E2E STAGE] {result.get('failure_stage')}")
        print(f"[E2E MESSAGE] {result.get('message')}")

    # NO_FEASIBLE_* and NO_VALID_ARCHITECTURE are legitimate engineering
    # outcomes, not runtime crashes. PIPELINE_ERROR remains a process failure.
    return 4 if result["status"] == "PIPELINE_ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
