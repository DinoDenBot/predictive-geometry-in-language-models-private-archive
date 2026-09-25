"""CLI: freeze, pilot, acquire, assemble, evaluate, report. Historical inputs read-only."""

from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from geometric_trajectory_v1 import VERSION
from geometric_trajectory_v1.attacks import (
    assign_folds,
    configurations,
    run_attacks,
    SEED,
)
from geometric_trajectory_v1.measurements import (
    measure_path,
    compact_document,
    aggregate,
)
from geometric_trajectory_v1.reporting import report
from cats_identification import context_length_grid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VOLUME = Path("/Volumes/My Passport/data_inference")
STUDIES = {
    "initial": "exposure_observability_v1",
    "fresh": "exposure_geometry_extension_v2",
    "pythia160": "exposure_geometry_extension_v2",
    "smollm2": "exposure_geometry_study3_smollm2",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def code_ledger():
    return {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.md"))
    } | {
        "cats_identification.py": sha(ROOT / "cats_identification.py"),
        "run_exposure_observability.py": sha(ROOT / "run_exposure_observability.py"),
    }


def freeze(output):
    from reviewer_revision.run_gap_attribution import phase_inputs, PHASES
    from transformers import AutoTokenizer, AutoConfig
    from huggingface_hub import snapshot_download

    output = Path(output)
    if (output / "freeze.json").exists():
        raise FileExistsError("existing freeze is immutable")
    output.mkdir(parents=True, exist_ok=True)
    panels = []
    models = {}
    sources = {}
    for phase in PHASES:
        setting = phase.rsplit("_", 1)[0]
        study = VOLUME / STUDIES[setting]
        frame, records, paths = phase_inputs(
            phase, [VOLUME / "gap_attribution_retrieved"]
        )
        manifest = study / (
            "candidate_manifest.parquet"
            if setting == "initial"
            else "design/candidate_manifest.parquet"
        )
        source = pd.read_parquet(manifest)
        # Use exact document identities and text from source, never outcomes as predictors.
        cols = ["doc_id", "text", "text_hash"] + (
            ["normalized_hash"] if "normalized_hash" in source else []
        )
        panel = frame[
            ["doc_id", "target_id", "block_id", "K", "predicted_tokens"]
        ].merge(source[cols], on="doc_id", validate="many_to_one")
        panel["label"] = (panel.K > 0).astype(int)
        panel["setting"] = setting
        panel["phase"] = phase
        panel = panel.merge(
            pd.DataFrame(records)[["doc_id", "target_id", "final", "final_config"]],
            on=["doc_id", "target_id"],
            validate="one_to_one",
        )
        panels.append(panel)
        for path in [manifest, *paths]:
            if str(path) not in sources:
                sources[str(path)] = sha(path)
        for target in sorted(frame.target_id.unique()):
            key = setting + ":" + target
            if key in models:
                continue
            completion = study / "targets" / target / "completion.json"
            meta = json.loads(completion.read_text())
            ck = completion.parent / (
                "checkpoint-100.pt" if setting == "initial" else "checkpoint-final.pt"
            )
            expected = (
                meta["checkpoint_hashes"]["1.00"]
                if setting == "initial"
                else meta["checkpoint_sha256"]
            )
            measured = sha(ck)
            if measured != expected:
                raise ValueError("checkpoint hash mismatch: " + str(ck))
            name = meta.get("model_name", meta.get("base_model"))
            rev = meta.get("model_revision", meta.get("base_revision"))
            models[key] = dict(
                checkpoint=str(ck),
                sha256=measured,
                bytes=ck.stat().st_size,
                model_name=name,
                revision=rev,
            )
            print("verified", key, flush=True)
    panel = pd.concat(panels, ignore_index=True)
    if panel.groupby("setting").size().to_dict() != {k: 7200 for k in STUDIES}:
        raise ValueError("panel scope changed")
    if panel.duplicated(["setting", "target_id", "doc_id"]).any():
        raise ValueError("duplicate cell")
    # Stage only public pinned config/tokenizer files; target weights are retained locally.
    for setting in STUDIES:
        info = next(v for k, v in models.items() if k.startswith(setting + ":"))
        snap = Path(
            snapshot_download(
                info["model_name"],
                revision=info["revision"],
                allow_patterns=["*.json", "*.txt", "*.model"],
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(snap, local_files_only=True)
        config = AutoConfig.from_pretrained(snap, local_files_only=True)
        asset_hashes = {
            p.name: sha(p)
            for p in sorted(snap.iterdir())
            if p.is_file() and p.suffix in (".json", ".txt", ".model")
        }
        for k, v in models.items():
            if k.startswith(setting + ":"):
                v.update(
                    snapshot=str(snap),
                    assets=asset_hashes,
                    model_type=config.model_type,
                )
        sub = panel[panel.setting == setting].drop_duplicates("doc_id")
        ids = {
            r.doc_id: tokenizer(
                r.text, add_special_tokens=True, truncation=True, max_length=129
            )["input_ids"]
            for r in sub.itertuples()
        }
        panel.loc[panel.setting == setting, "token_ids"] = pd.Series(
            [json.dumps(ids[d]) for d in panel.loc[panel.setting == setting, "doc_id"]],
            index=panel.index[panel.setting == setting],
        )
    panel["actual_tokens"] = panel.token_ids.map(lambda s: len(json.loads(s)) - 1)
    if not (panel.actual_tokens == panel.predicted_tokens).all():
        raise ValueError("tokenization differs from original predicted positions")
    panel = assign_folds(panel)
    panel["cell_id"] = [
        hashlib.sha256((s + ":" + t + ":" + h).encode()).hexdigest()
        for s, t, h in zip(panel.setting, panel.target_id, panel.text_hash)
    ]
    panel.to_parquet(output / "cells.parquet", index=False)
    pilot = []
    for setting, sub in panel.groupby("setting", sort=True):
        target = sorted(sub.target_id.unique())[0]
        sub = sub[sub.target_id == target].sort_values(["actual_tokens", "text_hash"])
        for label, q in [("short", 0.0), ("medium", 0.5), ("long", 1.0)]:
            row = sub.iloc[round(q * (len(sub) - 1))]
            pilot.append(
                dict(
                    setting=setting,
                    stratum=label,
                    cell_id=row.cell_id,
                    actual_tokens=int(row.actual_tokens),
                )
            )
    write_json(output / "models.json", models)
    write_json(output / "pilot_selection.json", pilot)
    write_json(
        output / "specification.json",
        dict(
            version=VERSION,
            seed=SEED,
            cells=28800,
            setting_cells=7200,
            estimand="retrospective controlled continued-training inclusion 1[K>0]; held-out documents within existing target population",
            contexts=24,
            max_predicted_tokens=128,
            inference_precision="float32 model and log-softmax",
            reductions="float64 CUDA on CUDA devices; independent NumPy float64 CPU reference",
            outer_folds=5,
            inner_folds=3,
            bootstrap=10000,
            search=configurations(),
            primary=["combined-rich_likelihood", "combined-ordinary"],
            original_development="excluded",
            baseline_candidate="rich_likelihood included in combined inner selection",
            tie_rule="AUC within 1e-12: logistic then smaller C; boosting fewer leaves, larger minimum leaf, stronger L2; then baseline subset",
            split="complete blocks merged transitively by doc_id, exact and normalized hash; within each setting",
            status="implementation frozen before new distribution acquisition; all outcomes retrospective",
        ),
    )
    write_json(
        output / "freeze.json",
        dict(
            code=code_ledger(),
            inputs=sources,
            files={
                p.name: sha(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "freeze.json"
            },
            models_verified=True,
        ),
    )


def verify(output):
    frozen = json.loads((output / "freeze.json").read_text())
    if frozen["code"] != code_ledger():
        raise ValueError(
            "implementation changed since freeze; create a new run directory"
        )
    for file, digest in frozen["files"].items():
        if sha(output / file) != digest:
            raise ValueError("frozen artifact changed: " + file)
    return pd.read_parquet(output / "cells.parquet"), json.loads(
        (output / "models.json").read_text()
    )


def _last_hidden_and_head(model, padded, attention):
    """Same last-hidden adapter as historical acquisition, without runner imports."""
    for backbone_name, head_names in (
        ("transformer", ("lm_head",)),
        ("gpt_neox", ("lm_head", "embed_out")),
        ("model", ("lm_head",)),
    ):
        backbone = getattr(model, backbone_name, None)
        if backbone is None:
            continue
        for name in head_names:
            head = getattr(model, name, None)
            if head is not None:
                return backbone(
                    input_ids=padded, attention_mask=attention, use_cache=False
                ).last_hidden_state, head
    raise TypeError(f"unsupported causal-LM architecture: {type(model).__name__}")


def acquire_document(model, tokenizer, ids, device, fixtures=False, deadline=None):
    import torch

    if str(device).startswith("cuda"):
        from geometric_trajectory_v1.measurements_torch import measure_path as gpu_path

        def reduction(lp, observed, lengths):
            return gpu_path(lp, observed, lengths, device=device)
    else:
        reduction = measure_path

    paths = []
    fixture = {}
    inference_seconds = 0.0
    reduction_seconds = 0.0
    queries = 0
    for pos in range(1, len(ids)):
        if deadline and time.monotonic() > deadline:
            raise TimeoutError("acquisition time ceiling")
        lengths = context_length_grid(pos, 24)
        unique, inverse = np.unique(lengths, return_inverse=True)
        padded = torch.full(
            (len(unique), int(unique[-1])),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(padded)
        for i, n in enumerate(unique):
            padded[i, :n] = torch.tensor(ids[pos - int(n) : pos], device=device)
            mask[i, :n] = 1
        tick = time.monotonic()
        with torch.inference_mode():
            hidden, head = _last_hidden_and_head(model, padded, mask)
            last = hidden[
                torch.arange(len(unique), device=device),
                torch.tensor(unique - 1, device=device),
            ]
            # Preserve original FP32 model/log-softmax convention, reduce geometry in float64.
            lp = head(last).float().log_softmax(-1).cpu().numpy().astype(np.float64)
        inference_seconds += time.monotonic() - tick
        queries += len(unique)
        tick = time.monotonic()
        paths.append(reduction(lp[inverse], ids[pos], lengths))
        reduction_seconds += time.monotonic() - tick
        if fixtures and pos in {1, (len(ids) - 1) // 2, len(ids) - 1}:
            fixture[f"logp_{pos}"] = lp[inverse]
            fixture[f"lengths_{pos}"] = lengths
            fixture[f"observed_{pos}"] = np.int64(ids[pos])
    return (
        compact_document(paths, range(1, len(ids))),
        fixture,
        dict(
            inference_seconds=inference_seconds,
            reduction_seconds=reduction_seconds,
            unique_queries=queries,
            declared_queries=(len(ids) - 1) * 24,
            reduction_backend="cuda_float64"
            if str(device).startswith("cuda")
            else "numpy_float64",
        ),
    )


def acquire(
    output,
    pilot=False,
    device="cpu",
    setting=None,
    target=None,
    max_seconds=3600,
    path_map=None,
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(4)
    panel, models = verify(output)
    mappings = json.loads(Path(path_map).read_text()) if path_map else {}

    def relocate(value):
        for old, new in sorted(mappings.items(), key=lambda x: -len(x[0])):
            if value == old or value.startswith(old + "/"):
                return new + value[len(old) :]
        return value

    for info in models.values():
        for key in ("checkpoint", "snapshot"):
            info[key] = relocate(info[key])
    panel["final"] = panel.final.map(relocate)
    selection = json.loads((output / "pilot_selection.json").read_text())
    if pilot:
        panel = panel[panel.cell_id.isin([r["cell_id"] for r in selection])]
    if setting:
        panel = panel[panel.setting == setting]
    if target:
        panel = panel[panel.target_id == target]
    directory = output / ("pilot" if pilot else "measurements")
    directory.mkdir(exist_ok=True)
    deadline = time.monotonic() + max_seconds
    with threadpool_limits(limits=4):
        for (s, t), sub in panel.groupby(["setting", "target_id"], sort=True):
            info = models[s + ":" + t]
            if sha(info["checkpoint"]) != info["sha256"]:
                raise ValueError("checkpoint changed")
            snap = Path(info["snapshot"])
            for name, digest in info["assets"].items():
                if sha(snap / name) != digest:
                    raise ValueError("model asset changed")
            model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(snap, local_files_only=True),
                dtype=torch.float32,
            )
            state = torch.load(
                info["checkpoint"], map_location="cpu", weights_only=False
            )
            model.load_state_dict(state["model"], strict=True)
            del state
            model = model.to(device).eval()
            tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            for n, row in enumerate(sub.itertuples(), 1):
                path = directory / (row.cell_id + ".npz")
                receipt = path.with_suffix(".json")
                if receipt.exists():
                    meta = json.loads(receipt.read_text())
                    if meta["sha256"] != sha(path) or meta["freeze_sha256"] != sha(
                        output / "freeze.json"
                    ):
                        raise ValueError("resume mismatch")
                    continue
                tick = time.monotonic()
                ids = json.loads(row.token_ids)
                current = tok(
                    row.text, add_special_tokens=True, truncation=True, max_length=129
                )["input_ids"]
                if current != ids:
                    raise ValueError("tokenization drift")
                compact, fixture, timing = acquire_document(
                    model, tok, ids, device, pilot, deadline
                )
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **compact)
                os.replace(tmp, path)
                if fixture:
                    np.savez_compressed(
                        directory / (row.cell_id + ".fixture.npz"), **fixture
                    )
                agreement = {}
                if pilot and Path(row.final).exists():
                    with np.load(row.final, allow_pickle=False) as old:
                        if str(old["config_hash"].item()) != row.final_config:
                            raise ValueError("legacy cache config changed")
                        agreement = dict(
                            logp_max_abs=float(
                                np.max(
                                    np.abs(
                                        compact["node__likelihood.logp"]
                                        - old["likelihood"][:, :, 0]
                                    )
                                )
                            ),
                            zlogp_max_abs=float(
                                np.max(
                                    np.abs(
                                        compact["node__likelihood.zlogp"]
                                        - old["likelihood"][:, :, 2]
                                    )
                                )
                            ),
                            alr_max_abs=float(
                                np.max(
                                    np.abs(
                                        np.stack(
                                            [
                                                np.nan_to_num(
                                                    compact["step__geometry." + k]
                                                )
                                                for k in ("A", "L", "R")
                                            ],
                                            axis=-1,
                                        )[compact["distinct"]]
                                        - old["alr"][compact["distinct"]]
                                    )
                                )
                            ),
                        )
                        agreement["within_reference_tolerances"] = (
                            agreement["logp_max_abs"] <= 0.002
                            and agreement["zlogp_max_abs"] <= 0.02
                            and agreement["alr_max_abs"] <= 0.02
                        )
                        agreement["logp_mean_abs"] = float(
                            np.mean(
                                np.abs(
                                    compact["node__likelihood.logp"]
                                    - old["likelihood"][:, :, 0]
                                )
                            )
                        )
                        agreement["status"] = (
                            "within_reference_tolerances"
                            if agreement["within_reference_tolerances"]
                            else "cross_backend_difference_requires_review"
                        )
                write_json(
                    receipt,
                    dict(
                        cell_id=row.cell_id,
                        setting=s,
                        target_id=t,
                        device=device,
                        elapsed_seconds=time.monotonic() - tick,
                        bytes=path.stat().st_size,
                        sha256=sha(path),
                        freeze_sha256=sha(output / "freeze.json"),
                        agreement=agreement,
                        **timing,
                    ),
                )
                print(
                    "acquired",
                    s,
                    t,
                    n,
                    len(sub),
                    round(time.monotonic() - tick, 2),
                    agreement,
                    flush=True,
                )
            del model
            gc.collect()
    write_json(
        directory / "progress.json",
        dict(
            expected_cells=len(panel),
            completed_cells=sum(
                (directory / (c + ".json")).exists() for c in panel.cell_id
            ),
            pilot=pilot,
        ),
    )


def assemble(output):
    panel, _ = verify(output)
    rows = []
    coverage = []
    missing = []
    for row in panel.itertuples():
        path = output / "measurements" / (row.cell_id + ".npz")
        receipt = path.with_suffix(".json")
        if not receipt.exists():
            missing.append(row.cell_id)
            continue
        meta = json.loads(receipt.read_text())
        if sha(path) != meta["sha256"] or meta["freeze_sha256"] != sha(
            output / "freeze.json"
        ):
            raise ValueError("measurement changed")
        with np.load(path, allow_pickle=False) as compact:
            features, valid = aggregate(compact)
        rows.append(dict(cell_id=row.cell_id, **features))
        coverage.append(dict(cell_id=row.cell_id, **valid))
    write_json(
        output / "coverage.json",
        dict(expected=len(panel), complete=len(rows), missing=missing),
    )
    if missing:
        raise ValueError(
            f"{len(missing)} cells missing; evaluation requires complete coverage"
        )
    frame = panel.drop(columns=["text", "token_ids"]).merge(
        pd.DataFrame(rows), on="cell_id", validate="one_to_one"
    )
    frame.to_parquet(output / "features.parquet", index=False)
    pd.DataFrame(coverage).to_parquet(output / "feature_coverage.parquet", index=False)
    write_json(
        output / "features_receipt.json",
        dict(
            sha256=sha(output / "features.parquet"),
            freeze_sha256=sha(output / "freeze.json"),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=["freeze", "pilot", "acquire", "assemble", "evaluate", "report"],
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--setting")
    p.add_argument("--target")
    p.add_argument("--path-map", type=Path)
    p.add_argument("--max-seconds", type=int, default=3600)
    a = p.parse_args()
    if a.command == "freeze":
        freeze(a.run)
    elif a.command in ("pilot", "acquire"):
        acquire(
            a.run,
            a.command == "pilot",
            a.device,
            a.setting,
            a.target,
            a.max_seconds,
            a.path_map,
        )
    elif a.command == "assemble":
        assemble(a.run)
    elif a.command == "evaluate":
        verify(a.run)
        if (
            sha(a.run / "features.parquet")
            != json.loads((a.run / "features_receipt.json").read_text())["sha256"]
        ):
            raise ValueError("assembled features changed")
        run_attacks(pd.read_parquet(a.run / "features.parquet"), a.run / "attacks")
    else:
        verify(a.run)
        scores = pd.concat(
            [pd.read_parquet(p) for p in sorted((a.run / "attacks").glob("*.parquet"))],
            ignore_index=True,
        )
        from geometric_trajectory_v1.attacks import feature_sets

        methods = set(
            feature_sets(pd.read_parquet(a.run / "features.parquet").columns)
        ) | {"negative_loss", "min_k_plus_plus_20"}
        if (
            set(scores.method) != methods
            or any(len(g) != 7200 for _, g in scores.groupby(["setting", "method"]))
            or scores.setting.nunique() != 4
        ):
            raise ValueError("incomplete study scores")
        report(scores, a.run / "report")


if __name__ == "__main__":
    main()
