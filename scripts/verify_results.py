#!/usr/bin/env python3
"""Check and summarize the compact inputs for results reported in the paper."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SETTINGS = ("initial", "fresh", "pythia160", "smollm2")
PHASES = ("validation", "confirmation")


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return abs(actual - expected) <= tolerance


def check_exposure() -> None:
    data = rows("exposure_summary.csv")
    assert len(data) == 16
    index = {(r["setting"], r["phase"], r["metric"]): r for r in data}
    assert len(index) == 16
    print("EXPOSURE: mean allocation slopes; R, A")
    for setting in SETTINGS:
        for phase in PHASES:
            r = index[setting, phase, "R"]
            a = index[setting, phase, "A"]
            assert float(r["mean_slope"]) > 0 and float(a["mean_slope"]) > 0
            assert float(r["p_value"]) <= 0.00003
            print(f"  {setting:10s} {phase:12s} {float(r['mean_slope']):.6f} {float(a['mean_slope']):.6f}")
    localization = rows("localization_summary.csv")
    assert len(localization) == 12
    loc = {(r["setting"], r["phase"], r["metric"]): r for r in localization}
    assert len(loc) == 12
    for phase in PHASES:
        assert loc["initial", phase, "L"]["decision"] == "equivalent"
        assert loc["fresh", phase, "N"]["decision"] == "powered equivalence"
        assert float(loc["fresh", phase, "U"]["mean_slope"]) > 0
    assert loc["pythia160", "confirmation", "L"]["decision"] == "powered equivalence"
    assert loc["smollm2", "confirmation", "N"]["decision"] == "equivalent"


def check_gap() -> None:
    data = rows("gap_attribution.csv")
    assert len(data) == 8
    shares = []
    print("GAP: retained mean-response attribution")
    for r in data:
        R, B, G = (float(r[k]) for k in ("delta_R", "delta_B", "delta_G"))
        assert R > 0 and B > 0 and G > 0
        assert close(R, B + G, 4e-9)
        assert close(100 * G / R, float(r["G_R_percent"]), 0.001)
        shares.append(100 * B / R)
        print(f"  {r['phase']:24s} binary={100*B/R:6.2f}% gap={100*G/R:6.2f}%")
    assert 75.89 < min(shares) < 75.91
    assert 88.36 < max(shares) < 88.38


def check_trajectory() -> None:
    main = rows("trajectory_main_auc.csv")
    targets = rows("trajectory_target_auc.csv")
    phases = rows("trajectory_phase_auc.csv")
    main_index = {(r["setting"], r["method"]): float(r["auc"]) for r in main}
    assert len(main_index) == len(main) == 68
    for description, data in (("target", targets),):
        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in data:
            values[r["setting"], r["method"]].append(float(r["auc"]))
        assert set(values) == set(main_index), description
        for key, scores in values.items():
            assert close(sum(scores) / len(scores), main_index[key], 1e-12), (description, key)
    assert len(phases) == 136
    assert all(0 <= float(r["auc"]) <= 1 for r in phases)
    properties = rows("trajectory_property_auc.csv")
    assert len(properties) >= 20
    methods = (
        ("min_k_plus_plus_20", "Min-K++"),
        ("geometry_only", "Fisher-Rao paths"),
        ("rich_likelihood", "Likelihood paths"),
        ("fisher", "Likelihood + Fisher-Rao"),
    )
    print("TRAJECTORY: target/fold-averaged ROC AUC")
    for method, label in methods:
        print(f"  {label:24s} " + " ".join(f"{main_index[s, method]:.3f}" for s in SETTINGS))


def check_editing() -> None:
    oracle = rows("editing_oracle.csv")
    training = rows("editing_training.csv")
    assert len(oracle) == 4 and len(training) == 8
    by_seed = {(r["seed"], r["direction"]): r for r in oracle}
    print("EDITING: held-out locality ratios, unprotected/protected")
    for seed in ("52001", "52002"):
        p, u = by_seed[seed, "protected"], by_seed[seed, "unprotected"]
        assert close(float(p["short_B"]), float(u["short_B"]), 1e-6)
        assert float(p["edited_p"]) >= 0.05
        long_ratio = float(u["long_q90_L"]) / float(p["long_q90_L"])
        clean_ratio = float(u["clean_rms_L"]) / float(p["clean_rms_L"])
        assert long_ratio > 40 and clean_ratio > 48
        print(f"  {seed}: long={long_ratio:.1f}x clean={clean_ratio:.1f}x")
    assert all(int(r["passing_seeds"]) == 0 for r in training)


def check_source_hashes(source_root: Path) -> None:
    hashes = json.loads((RESULTS / "source_hashes.json").read_text())
    for name, digest in hashes.items():
        source = source_root / name
        assert source.is_file(), source
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        assert actual == digest, source
    print(f"SOURCE HASHES: {len(hashes)} original result files verified")


def check_code_hashes() -> None:
    hashes = json.loads((RESULTS / "code_hashes.json").read_text())
    for name, digest in hashes.items():
        source = ROOT / name
        assert source.is_file(), source
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        assert actual == digest, source
    print(f"CODE HASHES: {len(hashes)} bundled source files verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="optional original study repository to verify input hashes")
    args = parser.parse_args()
    check_exposure()
    check_gap()
    check_trajectory()
    check_editing()
    check_code_hashes()
    if args.source_root:
        check_source_hashes(args.source_root)
    print("All compact-result checks passed.")


if __name__ == "__main__":
    main()
