from __future__ import annotations

import argparse
import json
from pathlib import Path


ORDER = ["A", "B", "C", "D", "E"]


def pct(value):
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def num(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    directory = Path(
        args.input_dir
    )

    rows = []
    raw = {}

    for config in ORDER:
        path = (
            directory
            / f"ablation_{config}.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}"
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        metrics = data[
            "metrics"
        ]

        raw[
            config
        ] = metrics

        rows.append({
            "config":
                config,

            "configuration":
                metrics[
                    "configuration"
                ],

            "field_f1":
                metrics[
                    "field_f1"
                ],

            "exact_output_recall":
                metrics[
                    "exact_output_recall"
                ],

            "complete_message_success":
                metrics[
                    "complete_message_success"
                ],

            "false_automatic_acceptance_rate":
                metrics[
                    "false_automatic_acceptance_rate"
                ],

            "average_llm_calls_per_message":
                metrics[
                    "average_llm_calls_per_message"
                ],

            "mean_latency_s":
                metrics[
                    "mean_latency_s"
                ],

            "p95_latency_s":
                metrics[
                    "p95_latency_s"
                ],
        })

    Path(
        args.output
    ).write_text(
        json.dumps(
            {
                "comparison":
                    rows,

                "raw_metrics":
                    raw,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    headers = [
        "Cfg",
        "Field F1",
        "Exact",
        "Complete",
        "False accept",
        "LLM/msg",
        "Mean s",
        "p95 s",
    ]

    printable = []

    for row in rows:
        printable.append([
            row["config"],
            pct(
                row[
                    "field_f1"
                ]
            ),
            pct(
                row[
                    "exact_output_recall"
                ]
            ),
            pct(
                row[
                    "complete_message_success"
                ]
            ),
            pct(
                row[
                    "false_automatic_acceptance_rate"
                ]
            ),
            num(
                row[
                    "average_llm_calls_per_message"
                ]
            ),
            num(
                row[
                    "mean_latency_s"
                ],
                3,
            ),
            num(
                row[
                    "p95_latency_s"
                ],
                3,
            ),
        ])

    widths = [
        max(
            len(headers[i]),
            *(
                len(row[i])
                for row in printable
            ),
        )
        for i in range(
            len(headers)
        )
    ]

    def line(values):
        return " | ".join(
            value.ljust(
                widths[i]
            )
            for i, value
            in enumerate(values)
        )

    print()
    print(
        "ABLATION COMPARISON"
    )
    print(
        "=" * (
            sum(widths)
            + 3
            * (
                len(widths)
                - 1
            )
        )
    )

    print(
        line(headers)
    )

    print(
        "-+-".join(
            "-" * width
            for width in widths
        )
    )

    for row in printable:
        print(
            line(row)
        )

    print()
    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()