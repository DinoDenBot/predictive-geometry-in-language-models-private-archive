#!/usr/bin/env python3
"""Produce a compact report for either the oracle gate or the full v3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    design = json.loads((root / "design/design.json").read_text())
    oracle = json.loads((root / "oracle_decision.json").read_text())
    oracle_rows = []
    for seed in design["training"]["development_seeds"]:
        receipt = json.loads((root / "oracle" / f"seed-{seed}" / "oracle.json").read_text())
        for row in receipt["steps"]:
            oracle_rows.append(
                {
                    "seed": seed,
                    "selected": row["linear_binary_step"] == receipt["selected_step"],
                    "linear_binary_step": row["linear_binary_step"],
                    "relative_parameter_norm": row["relative_parameter_norm"],
                    **{f"development_{k}": v for k, v in row["development"].items()},
                    **{
                        f"validation_{k}": v
                        for k, v in (row.get("oracle_validation") or {}).items()
                    },
                }
            )
    oracle_frame = pd.DataFrame(oracle_rows)
    oracle_frame.to_csv(report / "oracle_steps.csv", index=False)
    outputs = [report / "oracle_steps.csv"]
    lines = [
        "# Numerically qualified context-selective retrieval v3",
        "",
        f"The exact held-out oracle gate **{'passed' if oracle['passed'] else 'did not pass'}**.",
        "",
        "| seed | selected step | validation short p | validation B | validation long q90 L | validation margin | validation clean RMS L |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in oracle["seeds"]:
        row = oracle_frame.loc[(oracle_frame.seed == item["seed"]) & oracle_frame.selected].iloc[0]
        lines.append(
            f"| {item['seed']} | {row.linear_binary_step:.2f} | "
            f"{row.validation_short_probability_modified:.5f} | {row.validation_short_B:.4f} | "
            f"{row.validation_long_L_q90:.4f} | {row.validation_selectivity_margin:.4f} | "
            f"{row.validation_clean_L_rms:.4f} |"
        )
    analysis_path = root / "confirmation_analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        frame = pd.DataFrame(analysis["rows"]).sort_values(["candidate_id", "dose"])
        frame.to_csv(report / "confirmation_summary.csv", index=False)
        outputs.append(report / "confirmation_summary.csv")
        selected = frame.loc[frame.candidate_id == analysis["selected_candidate"]]
        lines.extend(
            [
                "",
                f"Development selected `{analysis['selected_candidate']}`. The strong end-to-end gate "
                f"**{'passed' if analysis['strong_gate_passed'] else 'did not pass'}**.",
                "",
                "| dose | short p | short B | short G | short N | long q90 L | margin | clean RMS L | passing seeds |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected.itertuples(index=False):
            lines.append(
                f"| {row.dose} | {row.short_probability_mean:.5f} | {row.short_B_mean:.4f} | "
                f"{row.short_G_mean:.4f} | {row.short_N_mean:.4f} | {row.long_L_q90_mean:.4f} | "
                f"{row.margin_mean:.4f} | {row.clean_L_rms_mean:.4f} | {row.passing_seeds}/{row.seeds} |"
            )
        figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), constrained_layout=True)
        for candidate, values in frame.groupby("candidate_id", sort=True):
            axes[0].plot(values.dose, values.short_probability_mean, marker="o", label=candidate)
            axes[1].plot(values.dose, values.margin_mean, marker="o", label=candidate)
            axes[2].plot(values.dose, values.clean_L_rms_mean, marker="o", label=candidate)
        for axis in axes:
            axis.set_xscale("log", base=2); axis.set_xticks(design["training"]["doses"], design["training"]["doses"]); axis.set_xlabel("Dose")
        axes[0].set_title("Short target probability"); axes[0].axhline(.05, color="black", ls="--", lw=.8)
        axes[1].set_title("Selectivity margin"); axes[1].axhline(0, color="black", ls="--", lw=.8)
        axes[2].set_title("Clean RMS $L$"); axes[2].axhline(.05, color="black", ls="--", lw=.8)
        axes[0].legend(frameon=False, fontsize=7)
        figure.savefig(report / "confirmation.pdf"); figure.savefig(report / "confirmation.png", dpi=180); plt.close(figure)
        outputs.extend([report / "confirmation.pdf", report / "confirmation.png"])
    else:
        lines.extend(["", "Candidate construction was not run because the frozen oracle gate failed."])
    (report / "REPORT.md").write_text("\n".join(lines) + "\n")
    outputs.append(report / "REPORT.md")
    verification = {
        "status": "report-complete",
        "oracle_gate_passed": oracle["passed"],
        "end_to_end_available": analysis_path.exists(),
        "strong_gate_passed": json.loads(analysis_path.read_text())["strong_gate_passed"] if analysis_path.exists() else None,
        "outputs": {str(path.relative_to(root)): digest(path) for path in outputs},
    }
    (report / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
