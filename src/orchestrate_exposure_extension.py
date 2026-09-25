#!/usr/bin/env python3
"""Deterministic sealed-order supervisor for the frozen extension."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path("/Volumes/My Passport/data_inference/exposure_geometry_extension_v2")
EXECUTABLE = Path(__file__).with_name("execute_exposure_geometry_extension.py")


def run(arguments: list[str], completion: Path) -> None:
    if completion.exists():
        print(f"SKIP complete: {completion}", flush=True)
        return
    command = [sys.executable, str(EXECUTABLE), "--root", str(ROOT), *arguments]
    print(f"RUN {' '.join(arguments)}", flush=True)
    subprocess.run(command, check=True)
    if not completion.exists():
        raise RuntimeError(f"command returned without immutable completion: {completion}")


def acquire(architecture: str, role: str, target_id: str, scope: str) -> None:
    completion = (
        ROOT
        / "outcomes"
        / architecture
        / scope
        / role
        / f"{target_id}.parquet"
    )
    run(
        [
            "acquire",
            "--architecture",
            architecture,
            "--role",
            role,
            "--target-id",
            target_id,
            "--scope",
            scope,
            "--paths-per-batch",
            "8",
        ],
        completion,
    )


def main() -> None:
    for index in range(1, 4):
        acquire("70m", "validation", f"70m_{index}", "primary")
    run(
        ["analyze-70m", "--role", "validation", "--draws", "99999"],
        ROOT / "analysis" / "70m" / "validation" / "decision.json",
    )
    for index in range(4, 7):
        acquire("70m", "confirmation", f"70m_{index}", "primary")
    run(
        ["analyze-70m", "--role", "confirmation", "--draws", "99999"],
        ROOT / "analysis" / "70m" / "confirmation" / "decision.json",
    )
    for role in ("validation", "confirmation"):
        for index in range(1, 7):
            acquire("70m", role, f"70m_{index}", "complete_latin")
    run(
        ["analyze-complete-latin"],
        ROOT / "analysis" / "70m" / "complete_latin" / "decision.json",
    )
    for index in range(1, 4):
        acquire("160m", "validation", f"160m_{index}", "primary")
    run(
        ["analyze-160m", "--role", "validation", "--draws", "99999"],
        ROOT / "analysis" / "160m" / "validation" / "decision.json",
    )
    for index in range(1, 4):
        acquire("160m", "confirmation", f"160m_{index}", "primary")
    run(
        ["analyze-160m", "--role", "confirmation", "--draws", "99999"],
        ROOT / "analysis" / "160m" / "confirmation" / "decision.json",
    )
    print("ALL FROZEN EXPERIMENTAL STAGES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
