#!/usr/bin/env python3
"""Certify reconstructed v2 development predictions against retrieved originals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from context_selective_retrieval_v2.design import atomic_json, sha256
from context_selective_retrieval_v2.geometry import fr_metrics


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name.removesuffix("_probs"): values[name].copy() for name in values.files}


def compare(reconstructed: Path, reference: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    threshold = 1e-6
    rows = []
    for seed in (51001, 51002):
        for filename, split in (
            ("control__dose-0.npz", "oracle_fit"),
            ("control_validation.npz", "oracle_validation"),
        ):
            actual_path = reconstructed / "branches/development" / f"seed-{seed}" / filename
            reference_path = reference / f"seed-{seed}" / filename
            actual, expected = _arrays(actual_path), _arrays(reference_path)
            component = {
                name: float(np.max(fr_metrics(expected[name], actual[name], 0)["L"]))
                for name in ("short", "long", "clean")
            }
            # target_id is irrelevant for total Fisher--Rao distance L.
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "component_maximum_L": component,
                    "maximum_L": max(component.values()),
                    "actual_sha256": sha256(actual_path),
                    "reference_sha256": sha256(reference_path),
                }
            )
    maximum = max(row["maximum_L"] for row in rows)
    atomic_json(
        output,
        {
            "status": "reconstruction-equivalent" if maximum <= threshold else "reconstruction-mismatch",
            "passed": maximum <= threshold,
            "rule": "all reconstructed fit and validation predictions are within the frozen v2 1e-6 Fisher--Rao replay floor",
            "threshold": threshold,
            "maximum_L": maximum,
            "rows": rows,
        },
    )
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

