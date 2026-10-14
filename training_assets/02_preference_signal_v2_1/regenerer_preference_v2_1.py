"""Rebuild the recovered V2.1 corpus using the unmodified original scripts."""

from pathlib import Path
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    scripts = root / "preference_extractor" / "training"
    data = scripts / "data"
    known = root / "preference_extractor/evaluation/datasets/preference_signal_test_v1.jsonl"
    commands = [
        [
            str(scripts / "generate_preference_signal_v2.py"),
            "--v1-dir", str(data / "data_layer1"),
            "--output-dir", str(data / "data_layer1_v2"),
            "--known-stress", str(known),
        ],
        [str(scripts / "generate_preference_signal_v2_1.py")],
    ]
    logs = root / "recovery_logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def run(args):
        result = subprocess.run(
            [sys.executable, *args], cwd=root, env=env, capture_output=True
        )
        log = logs / ("reproduction_" + Path(args[0]).stem + ".txt")
        log.write_bytes(result.stdout + b"\n" + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{Path(args[0]).name} failed; see {log}")
        print(f"OK: {Path(args[0]).name}", flush=True)

    for command in commands:
        run(command)

    # Original archived JSONL uses LF. Keep this format across operating systems.
    for directory in (data / "data_layer1_v2", data / "data_layer1_v2_1"):
        for path in directory.glob("*.jsonl"):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))

    run([
        str(scripts / "qa_preference_signal_v2.py"),
        "--data-dir", str(data / "data_layer1_v2"),
        "--known-stress", str(known),
    ])
    run([str(scripts / "qa_preference_signal_v2_1.py")])
    print("V2.1 regenerated; original V2 and V2.1 QA completed.")


if __name__ == "__main__":
    main()
