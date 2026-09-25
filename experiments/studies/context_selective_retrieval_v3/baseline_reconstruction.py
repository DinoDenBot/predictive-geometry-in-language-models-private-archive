#!/usr/bin/env python3
"""Compare reconstructed v3 controls with the retained authoritative predictions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from context_selective_retrieval_v3.design import sha256
from context_selective_retrieval_v3.geometry import fr_metrics


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name.removesuffix("_probs"): values[name].copy() for name in values.files}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def compare(reconstructed: Path, reference: Path, output: Path) -> None:
    threshold = 1e-6
    rows = []
    for seed in (52001, 52002):
        for filename, split in (
            ("control__dose-0.npz", "oracle_fit"),
            ("control_validation.npz", "oracle_validation"),
        ):
            actual_path = reconstructed / "branches/development" / f"seed-{seed}" / filename
            reference_path = reference / f"seed-{seed}" / filename
            actual, expected = arrays(actual_path), arrays(reference_path)
            component = {
                name: float(np.max(fr_metrics(expected[name], actual[name], 0)["L"]))
                for name in ("short", "long", "clean")
            }
            rows.append({
                "seed": seed,
                "split": split,
                "component_maximum_L": component,
                "maximum_L": max(component.values()),
                "reconstructed_sha256": sha256(actual_path),
                "reference_sha256": sha256(reference_path),
            })
    maximum = max(row["maximum_L"] for row in rows)
    payload = {
        "status": "reconstruction-equivalent" if maximum <= threshold else "reconstruction-mismatch",
        "passed": maximum <= threshold,
        "threshold": threshold,
        "maximum_L": maximum,
        "rows": rows,
    }
    atomic_json(output, payload)
    if maximum > threshold:
        raise RuntimeError(f"reconstruction mismatch: {maximum:.9g} > {threshold:.9g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.reconstructed, args.reference, args.output)


if __name__ == "__main__":
    main()
