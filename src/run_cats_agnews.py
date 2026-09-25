#!/usr/bin/env python3
"""Resumable CATS study on the documented AG News fine-tuning split.

Stages deliberately separate label-blind acquisition from label-dependent
development/evaluation. All learned transforms are fit on development only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from cats_identification import (
    LAR2Config,
    LAR2Featurizer,
    cats_document_features,
    context_length_grid,
    error_zone_features,
)

TARGET = "cglez/gpt2-dapt-ag_news"
REVISION = "epoch-5"
TOKENIZER = "openai-community/gpt2"
REFERENCE = "openai-community/gpt2"
DEVELOPMENT_SOURCE = Path(
    "/path/to/LeakPro/results/agnews_released_gpt2_epoch5_informia_pilot/per_sample.parquet"
)
LEAKPRO_RESULTS = Path("/path/to/LeakPro/results")
ARROW_ROOT = Path(
    "/path/to/huggingface/datasets/fancyzhx___ag_news/default/0.0.0/"
    "eb185aade064a813bc0b7f42de02595523103ca4"
)
STUDY_ROOT = Path("results/cats_agnews")
LAMBDA = 0.01
SEED = 20260829
CANDIDATES = ("cats_trim10", "cats_hc")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:  # noqa: ANN401
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:  # noqa: ANN401
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_development(output: Path) -> None:
    frame = pd.read_parquet(DEVELOPMENT_SOURCE).copy()
    required = {"pair_id", "label", "topic", "text"}
    if required - set(frame):
        raise ValueError("development source lacks required identity columns")
    frame = frame[["pair_id", "label", "topic", "text"]].copy()
    frame["source_index"] = -1
    frame["source_split"] = np.where(frame.label == 1, "train", "test")
    frame["text_hash"] = frame.text.map(text_hash)
    frame["role"] = "development"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "records.parquet", index=False)
    atomic_json(
        output / "selection.json",
        {
            "role": "development",
            "source": str(DEVELOPMENT_SOURCE),
            "source_sha256": sha256(DEVELOPMENT_SOURCE),
            "pairs": int(frame.pair_id.nunique()),
            "documents": len(frame),
            "identities_sha256": hashlib.sha256("\n".join(frame.text_hash).encode()).hexdigest(),
            "status": "identities-frozen-before-cats-acquisition",
        },
    )


def prior_hashes(study_root: Path, current: Path) -> tuple[set[str], list[str]]:
    hashes: set[str] = set()
    sources: list[str] = []
    paths = list(LEAKPRO_RESULTS.glob("agnews*/per_sample.parquet"))
    paths += list(study_root.glob("*/records.parquet"))
    for path in sorted(set(paths)):
        if path.resolve() == current.resolve():
            continue
        try:
            columns = pd.read_parquet(path).columns
            if "text_hash" in columns:
                hashes.update(pd.read_parquet(path, columns=["text_hash"]).text_hash.dropna().astype(str))
            elif "text" in columns:
                hashes.update(text_hash(str(value)) for value in pd.read_parquet(path, columns=["text"]).text.dropna())
            else:
                continue
            sources.append(str(path))
        except (OSError, ValueError):
            continue
    return hashes, sources


def _token_count(tokenizer: Any, text: str) -> int:  # noqa: ANN401
    return len(tokenizer(text, add_special_tokens=True)["input_ids"]) - 1


def prepare_fresh(output: Path, role: str, pairs: int, seed: int) -> None:
    if role not in {"validation", "confirmation", "validation_v3", "confirmation_v3", "proof_v4"}:
        raise ValueError("fresh role must be validation or confirmation")
    if pairs % 4:
        raise ValueError("pair count must be divisible by four")
    records_path = output / "records.parquet"
    if records_path.exists():
        raise FileExistsError(f"refusing to replace frozen identities: {records_path}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, use_fast=True)
    train = Dataset.from_file(str(ARROW_ROOT / "ag_news-train.arrow"))
    test = Dataset.from_file(str(ARROW_ROOT / "ag_news-test.arrow"))
    excluded, sources = prior_hashes(STUDY_ROOT, records_path)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    pair_id = 0
    for topic in range(4):
        pools: dict[str, list[tuple[int, str, int]]] = {}
        for split, dataset in (("train", train), ("test", test)):
            values = []
            for index in np.flatnonzero(np.asarray(dataset["label"]) == topic):
                text = str(dataset[int(index)]["text"])
                digest = text_hash(text)
                if digest in excluded:
                    continue
                tokens = _token_count(tokenizer, text)
                if 24 <= tokens <= 128:
                    values.append((int(index), text, tokens))
            pools[split] = values
        need = pairs // 4
        if min(map(len, pools.values())) < need:
            raise ValueError(f"topic {topic} lacks {need} fresh records per split")
        selected = rng.choice(len(pools["train"]), need, replace=False)
        members = [pools["train"][int(index)] for index in selected]
        available = list(pools["test"])
        for member_index, member_text, member_tokens in members:
            nonmember = min(available, key=lambda value: (abs(value[2] - member_tokens), value[0]))
            available.remove(nonmember)
            nonmember_index, nonmember_text, nonmember_tokens = nonmember
            rows.extend(
                [
                    {
                        "pair_id": pair_id,
                        "label": 1,
                        "topic": topic,
                        "text": member_text,
                        "text_hash": text_hash(member_text),
                        "source_index": member_index,
                        "source_split": "train",
                        "selection_tokens": member_tokens,
                        "role": role,
                    },
                    {
                        "pair_id": pair_id,
                        "label": 0,
                        "topic": topic,
                        "text": nonmember_text,
                        "text_hash": text_hash(nonmember_text),
                        "source_index": nonmember_index,
                        "source_split": "test",
                        "selection_tokens": nonmember_tokens,
                        "role": role,
                    },
                ]
            )
            pair_id += 1
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(records_path, index=False)
    atomic_json(
        output / "selection.json",
        {
            "role": role,
            "seed": seed,
            "pairs": pairs,
            "documents": len(frame),
            "sampling_frame": "AG News train/test, 24-128 GPT-2 next-token observations, topic-balanced, greedy length matched",
            "prior_hashes_excluded": len(excluded),
            "prior_artifacts": sources,
            "identities_sha256": hashlib.sha256("\n".join(frame.text_hash).encode()).hexdigest(),
            "status": "identities-frozen-before-label-dependent-cats-evaluation",
        },
    )


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _last_hidden_and_head(model: Any, padded: torch.Tensor, attention: torch.Tensor) -> tuple[torch.Tensor, Any]:  # noqa: ANN401
    """Return hidden states and the output head without materializing all logits."""
    if hasattr(model, "transformer") and hasattr(model, "lm_head"):
        hidden = model.transformer(
            input_ids=padded, attention_mask=attention, use_cache=False
        ).last_hidden_state
        return hidden, model.lm_head
    if hasattr(model, "gpt_neox") and hasattr(model, "lm_head"):
        hidden = model.gpt_neox(
            input_ids=padded, attention_mask=attention, use_cache=False
        ).last_hidden_state
        return hidden, model.lm_head
    raise TypeError(f"unsupported causal-LM architecture: {type(model).__name__}")


def _nested_arrays(
    text: str,
    tokenizer: Any,  # noqa: ANN401
    model: Any,  # noqa: ANN401
    bos_log_probs: torch.Tensor,
    device: torch.device,
    config: LAR2Config,
    paths_per_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=129)["input_ids"]
    if len(ids) < 9:
        raise ValueError("document has fewer than eight predicted tokens")
    likelihood_chunks = []
    cats_chunks = []
    distinct_chunks = []
    for token_start in range(1, len(ids), paths_per_batch):
        token_indices = list(range(token_start, min(len(ids), token_start + paths_per_batch)))
        prefixes: list[list[int]] = []
        outcomes: list[int] = []
        length_rows = []
        for token_index in token_indices:
            lengths = context_length_grid(token_index, config.grid_size)
            length_rows.append(lengths)
            for length in lengths:
                prefixes.append(ids[token_index - int(length) : token_index])
                outcomes.append(ids[token_index])
        lengths = torch.tensor([len(prefix) for prefix in prefixes], device=device)
        maximum = int(lengths.max())
        padded = torch.full(
            (len(prefixes), maximum), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        attention = torch.zeros_like(padded)
        for row, prefix in enumerate(prefixes):
            padded[row, : len(prefix)] = torch.tensor(prefix, device=device)
            attention[row, : len(prefix)] = 1
        row_ids = torch.arange(len(prefixes), device=device)
        target_ids = torch.tensor(outcomes, device=device)
        with torch.inference_mode():
            # This is algebraically the model's final-position output, but it
            # avoids materializing unused vocabulary logits at every padded
            # prefix position.
            hidden, output_head = _last_hidden_and_head(model, padded, attention)
            last_hidden = hidden[row_ids, lengths - 1]
            log_probs = output_head(last_hidden).float().log_softmax(-1)
            weights = log_probs.exp()
            mean = (weights * log_probs).sum(-1)
            variance = (weights * log_probs.square()).sum(-1) - mean.square()
            contrast = log_probs - bos_log_probs
            contrast_mean = (weights * contrast).sum(-1)
            contrast_variance = (weights * contrast.square()).sum(-1) - contrast_mean.square()
            observed = log_probs[row_ids, target_ids]
            observed_contrast = contrast[row_ids, target_ids]
            cells = torch.stack(
                [
                    observed,
                    observed_contrast,
                    (observed - mean) / variance.clamp_min(1e-12).sqrt(),
                    (observed_contrast - contrast_mean) / contrast_variance.clamp_min(1e-12).sqrt(),
                ],
                dim=-1,
            ).reshape(len(token_indices), config.grid_size, 4)

            roots = (0.5 * log_probs).exp().reshape(len(token_indices), config.grid_size, -1)
            overlap = (roots[:, :-1] * roots[:, 1:]).sum(-1).clamp(-1.0, 1.0)
            outcome_matrix = torch.tensor([ids[index] for index in token_indices], device=device)[:, None]
            observed_roots = roots.gather(
                -1, outcome_matrix[:, :, None].expand(-1, config.grid_size, 1)
            ).squeeze(-1)
            numerator = observed_roots[:, 1:] - overlap * observed_roots[:, :-1]
            denominator = (1.0 - overlap.square()).clamp_min(0).sqrt() * (
                1.0 - observed_roots[:, :-1].square()
            ).clamp_min(0).sqrt()
            tau = torch.where(denominator > 1e-8, numerator / denominator, torch.zeros_like(numerator)).clamp(-1, 1)
            fisher_rao = 2.0 * torch.acos(overlap)
            increment = cells[:, 1:, 0] - cells[:, :-1, 0]
            cats = torch.stack((tau, fisher_rao, increment), dim=-1)

        length_rows = np.asarray(length_rows)
        distinct = np.diff(length_rows, axis=1) > 0
        cells_np = cells.cpu().numpy().astype(np.float32)
        cats_np = cats.cpu().numpy().astype(np.float32)
        cats_np[~distinct] = 0.0
        likelihood_chunks.append(cells_np)
        cats_chunks.append(cats_np)
        distinct_chunks.append(distinct)
        del (
            padded,
            attention,
            hidden,
            last_hidden,
            log_probs,
            weights,
            mean,
            variance,
            contrast,
            contrast_mean,
            contrast_variance,
            observed,
            observed_contrast,
            roots,
            overlap,
            observed_roots,
            numerator,
            denominator,
            tau,
            fisher_rao,
            increment,
            cells,
            cats,
        )
    raw = np.concatenate(likelihood_chunks)
    derivative = np.gradient(raw[:, :, 2], np.linspace(0, 1, config.grid_size), axis=1, edge_order=1)
    likelihood = np.concatenate((raw, derivative[:, :, None]), axis=2).astype(np.float32)
    return likelihood, np.concatenate(cats_chunks), np.concatenate(distinct_chunks), ids


def _informia(
    ids: list[int],
    target: Any,  # noqa: ANN401
    reference: Any,  # noqa: ANN401
    device: torch.device,
) -> tuple[float, float]:
    tensor = torch.tensor([ids], device=device)
    with torch.inference_mode():
        target_lp = target(tensor, use_cache=False).logits[:, :-1].float().log_softmax(-1)
        reference_lp = reference(tensor, use_cache=False).logits[:, :-1].float().log_softmax(-1)
        outcomes = tensor[:, 1:, None]
        observed_target = target_lp.gather(-1, outcomes).squeeze(-1)
        observed_reference = reference_lp.gather(-1, outcomes).squeeze(-1)
        reference_weights = reference_lp.exp()
        kl = (reference_weights * (reference_lp - target_lp)).sum(-1)
        score = (observed_target - observed_reference + kl).squeeze(0).cpu().numpy()
    count = max(1, int(0.2 * len(score)))
    return float(np.mean(score)), float(np.mean(np.sort(score)[:count]))


def _error_zone_arrays(
    ids: list[int],
    target: Any,  # noqa: ANN401
    reference: Any,  # noqa: ANN401
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.tensor([ids], device=device)
    with torch.inference_mode():
        target_logits = target(tensor, use_cache=False).logits[:, :-1].float()
        reference_logits = reference(tensor, use_cache=False).logits[:, :-1].float()
        outcomes = tensor[:, 1:, None]
        target_lp = target_logits.log_softmax(-1)
        reference_lp = reference_logits.log_softmax(-1)
        target_true = target_lp.gather(-1, outcomes).squeeze(-1)
        reference_true = reference_lp.gather(-1, outcomes).squeeze(-1)
        correct = target_logits.argmax(-1).eq(tensor[:, 1:])
        delta = target_true - reference_true
    return correct.squeeze(0).cpu().numpy(), delta.squeeze(0).cpu().numpy()


def acquire_error_zone(result_dir: Path) -> None:
    """Add official EZ-MIA and preregisterable context diagnostics cheaply."""
    frame, arrays = load_arrays(result_dir)
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, use_fast=True)
    target = AutoModelForCausalLM.from_pretrained(TARGET, revision=REVISION, local_files_only=True).to(device).eval()
    reference = AutoModelForCausalLM.from_pretrained(REFERENCE, local_files_only=True).to(device).eval()
    rows = []
    for position, (row, likelihood) in enumerate(zip(frame.itertuples(index=False), arrays, strict=True), 1):
        ids = tokenizer(row.text, add_special_tokens=True, truncation=True, max_length=129)["input_ids"]
        correct, delta = _error_zone_arrays(ids, target, reference, device)
        features = error_zone_features(likelihood, correct, delta)
        rows.append({"pair_id": row.pair_id, "label": row.label, "text_hash": row.text_hash, **features})
        if position % 50 == 0:
            print(f"error-zone {position}/{len(frame)}", flush=True)
    additions = pd.DataFrame(rows)
    original = pd.read_parquet(result_dir / "per_sample_unscored.parquet")
    original = original.drop(columns=[column for column in additions.columns if column in original and column not in {"pair_id", "label", "text_hash"}])
    merged = original.merge(additions, on=["pair_id", "label", "text_hash"], validate="one_to_one")
    merged.to_parquet(result_dir / "per_sample_unscored.parquet", index=False)
    atomic_json(
        result_dir / "error_zone_acquisition.json",
        {
            "status": "official-ez-mia-and-context-extensions-acquired",
            "official_source": "JetBrains-Research/ez-mia@aa0b6b74a39aa77f4a9fa70e0846bc985f0919eb",
            "access": "target and base-reference full-context next-token outputs",
            "documents": len(frame),
        },
    )


def acquire(result_dir: Path, paths_per_batch: int) -> None:
    records_path = result_dir / "records.parquet"
    frame = pd.read_parquet(records_path)
    config = LAR2Config()
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "config": asdict(config),
                "target": TARGET,
                "revision": REVISION,
                "tokenizer": TOKENIZER,
                "reference_diagnostic": REFERENCE,
                "version": 1,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, revision=REVISION, local_files_only=True
    ).to(device).eval()
    reference = AutoModelForCausalLM.from_pretrained(REFERENCE, local_files_only=True).to(device).eval()
    bos = torch.tensor([[tokenizer.eos_token_id]], device=device)
    with torch.inference_mode():
        bos_log_probs = target(bos, use_cache=False).logits[0, -1].float().log_softmax(-1)
    cache = result_dir / "cache"
    summary_rows = []
    for position, row in enumerate(frame.itertuples(index=False), 1):
        path = cache / f"{int(row.pair_id):05d}-{int(row.label)}-{row.text_hash[:16]}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                if str(stored["config_hash"].item()) != config_hash:
                    raise ValueError(f"cache configuration mismatch: {path}")
                likelihood = stored["likelihood"]
                cats = stored["cats"]
                distinct = stored["distinct"]
                info_mean = float(stored["informia_mean"])
                info_min = float(stored["informia_min_k_20"])
        else:
            likelihood, cats, distinct, ids = _nested_arrays(
                row.text, tokenizer, target, bos_log_probs, device, config, paths_per_batch
            )
            info_mean, info_min = _informia(ids, target, reference, device)
            atomic_npz(
                path,
                likelihood=likelihood,
                cats=cats,
                distinct=distinct,
                informia_mean=info_mean,
                informia_min_k_20=info_min,
                config_hash=config_hash,
            )
        observed = likelihood[:, -1, 0]
        standardized = likelihood[:, -1, 2]
        count = max(1, int(0.2 * len(observed)))
        cats_features = cats_document_features(cats, distinct)
        summary_rows.append(
            {
                "pair_id": int(row.pair_id),
                "label": int(row.label),
                "text_hash": row.text_hash,
                "tokens": int(likelihood.shape[0]),
                "target_mean_logp": float(np.mean(observed)),
                "min_k_20": float(np.mean(np.sort(observed)[:count])),
                "min_k_plus_plus_20": float(np.mean(np.sort(standardized)[:count])),
                "informia_mean": info_mean,
                "informia_min_k_20": info_min,
                **cats_features,
            }
        )
        if position % 10 == 0:
            print(f"acquired {position}/{len(frame)} documents on {device}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
    summary = frame.merge(pd.DataFrame(summary_rows), on=["pair_id", "label", "text_hash"], validate="one_to_one")
    summary.to_parquet(result_dir / "per_sample_unscored.parquet", index=False)
    atomic_json(
        result_dir / "acquisition.json",
        {
            "status": "label-blind-acquisition-complete",
            "records_sha256": sha256(records_path),
            "config_hash": config_hash,
            "target": {"id": TARGET, "revision": REVISION},
            "reference_diagnostic": REFERENCE,
            "access_primary": "one target checkpoint and complete next-token probabilities",
            "documents": len(frame),
            "pairs": int(frame.pair_id.nunique()),
            "query_distributions_per_document": "24 * number_of_predicted_tokens",
            "lar2_config": asdict(config),
        },
    )


def _cache_path(result_dir: Path, row: Any) -> Path:  # noqa: ANN401
    return result_dir / "cache" / f"{int(row.pair_id):05d}-{int(row.label)}-{row.text_hash[:16]}.npz"


def load_arrays(result_dir: Path) -> tuple[pd.DataFrame, list[np.ndarray]]:
    frame = pd.read_parquet(result_dir / "per_sample_unscored.parquet")
    arrays = []
    for row in frame.itertuples(index=False):
        with np.load(_cache_path(result_dir, row), allow_pickle=False) as stored:
            arrays.append(stored["likelihood"])
    return frame, arrays


def folds(frame: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    assignment = np.empty(len(frame), dtype=int)
    rng = np.random.default_rng(seed)
    for topic in sorted(frame.topic.unique()):
        pair_ids = np.asarray(sorted(frame.loc[frame.topic == topic, "pair_id"].unique()))
        rng.shuffle(pair_ids)
        mapping = {int(pair): index % 5 for index, pair in enumerate(pair_ids)}
        indices = np.flatnonzero(frame.topic.to_numpy() == topic)
        assignment[indices] = frame.iloc[indices].pair_id.map(mapping).to_numpy()
    return assignment


def classifier(sample_count: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0 / (LAMBDA * sample_count),
        solver="lbfgs",
        max_iter=5000,
        random_state=SEED,
    )


def _fit_coordinates(
    train_arrays: list[np.ndarray],
    test_arrays: list[np.ndarray],
    train_cats: np.ndarray | None,
    test_cats: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    featurizer = LAR2Featurizer().fit(train_arrays)
    train_x = featurizer.transform(train_arrays)
    test_x = featurizer.transform(test_arrays)
    if train_cats is not None and test_cats is not None:
        scaler = StandardScaler().fit(train_cats[:, None])
        train_x = np.column_stack((train_x, scaler.transform(train_cats[:, None])))
        test_x = np.column_stack((test_x, scaler.transform(test_cats[:, None])))
    return train_x, test_x


def crossfit(frame: pd.DataFrame, arrays: list[np.ndarray], cats_column: str | None) -> tuple[np.ndarray, list[float]]:
    labels = frame.label.to_numpy()
    assignment = folds(frame)
    scores = np.empty(len(frame))
    fold_aucs = []
    for fold in range(5):
        train = np.flatnonzero(assignment != fold)
        test = np.flatnonzero(assignment == fold)
        train_cats = frame.iloc[train][cats_column].to_numpy() if cats_column else None
        test_cats = frame.iloc[test][cats_column].to_numpy() if cats_column else None
        train_x, test_x = _fit_coordinates(
            [arrays[index] for index in train],
            [arrays[index] for index in test],
            train_cats,
            test_cats,
        )
        model = classifier(len(train)).fit(train_x, labels[train])
        scores[test] = model.decision_function(test_x)
        fold_aucs.append(float(roc_auc_score(labels[test], scores[test])))
    return scores, fold_aucs


def crossfit_lar2_candidates(
    frame: pd.DataFrame, arrays: list[np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Cross-fit LAR-2 and both CATS augmentations with one feature fit/fold."""
    labels = frame.label.to_numpy()
    assignment = folds(frame)
    names = ("lar2", *(f"lar2_plus_{candidate}" for candidate in CANDIDATES))
    scores = {name: np.empty(len(frame)) for name in names}
    fold_aucs = {name: [] for name in names}
    for fold in range(5):
        train = np.flatnonzero(assignment != fold)
        test = np.flatnonzero(assignment == fold)
        featurizer = LAR2Featurizer().fit([arrays[index] for index in train])
        train_base = featurizer.transform([arrays[index] for index in train])
        test_base = featurizer.transform([arrays[index] for index in test])
        for name in names:
            if name == "lar2":
                train_x, test_x = train_base, test_base
            else:
                candidate = name.removeprefix("lar2_plus_")
                scaler = StandardScaler().fit(frame.iloc[train][[candidate]])
                train_x = np.column_stack((train_base, scaler.transform(frame.iloc[train][[candidate]])))
                test_x = np.column_stack((test_base, scaler.transform(frame.iloc[test][[candidate]])))
            model = classifier(len(train)).fit(train_x, labels[train])
            scores[name][test] = model.decision_function(test_x)
            fold_aucs[name].append(float(roc_auc_score(labels[test], scores[name][test])))
    return scores, fold_aucs


def crossfit_tfidf(frame: pd.DataFrame) -> np.ndarray:
    assignment = folds(frame)
    scores = np.empty(len(frame))
    for fold in range(5):
        train = assignment != fold
        test = ~train
        vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_features=10000, sublinear_tf=True
        )
        train_x = vectorizer.fit_transform(frame.loc[train, "text"])
        model = classifier(int(train.sum())).fit(train_x, frame.loc[train, "label"])
        scores[test] = model.decision_function(vectorizer.transform(frame.loc[test, "text"]))
    return scores


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "tpr_at_1pct_fpr": float(np.max(tpr[fpr <= 0.01])),
        "tpr_at_zero_observed_fpr": float(np.max(tpr[fpr == 0])),
    }


def pair_bootstrap(
    frame: pd.DataFrame,
    columns: list[str],
    primary: str,
    baseline: str,
    draws: int,
    seed: int,
) -> tuple[dict[str, list[float]], list[float]]:
    groups = {int(pair): group.index.to_numpy() for pair, group in frame.groupby("pair_id")}
    pair_ids = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    samples = {column: [] for column in columns}
    differences = []
    for _ in range(draws):
        selected = rng.choice(pair_ids, len(pair_ids), replace=True)
        indices = np.concatenate([groups[int(pair)] for pair in selected])
        labels = frame.label.to_numpy()[indices]
        aucs = {column: roc_auc_score(labels, frame[column].to_numpy()[indices]) for column in columns}
        for column, value in aucs.items():
            samples[column].append(value)
        differences.append(aucs[primary] - aucs[baseline])
    interval = lambda x: [float(value) for value in np.quantile(x, [0.025, 0.975])]  # noqa: E731
    return {key: interval(value) for key, value in samples.items()}, interval(differences)


def develop(result_dir: Path, bootstrap_draws: int) -> None:
    frame, arrays = load_arrays(result_dir)
    learned_scores, learned_folds = crossfit_lar2_candidates(frame, arrays)
    for name, values in learned_scores.items():
        frame[name] = values
    frame["blind_tfidf"] = crossfit_tfidf(frame)
    candidate = max(
        CANDIDATES,
        key=lambda value: roc_auc_score(frame.label, frame[f"lar2_plus_{value}"])
        - roc_auc_score(frame.label, frame.lar2),
    )
    primary = f"lar2_plus_{candidate}"
    score_columns = ["lar2", primary, "min_k_plus_plus_20", "informia_min_k_20", "blind_tfidf"]
    intervals, delta_interval = pair_bootstrap(
        frame, score_columns, primary, "lar2", bootstrap_draws, SEED
    )
    metrics = {
        column: {**metric(frame.label.to_numpy(), frame[column].to_numpy()), "pair_bootstrap_95ci": intervals[column]}
        for column in score_columns
    }
    delta = metrics[primary]["roc_auc"] - metrics["lar2"]["roc_auc"]
    report = {
        "status": "development-complete-candidate-frozen",
        "estimand": "AG News document membership in the epoch-5 fine-tuning split",
        "access_primary": "target-only complete next-token probabilities from one checkpoint",
        "pairs": int(frame.pair_id.nunique()),
        "selected_candidate": candidate,
        "candidate_selection_rule": "largest five-fold OOF AUC increment over LAR-2; ties lexical",
        "metrics": metrics,
        "paired_delta_vs_lar2": {"point": delta, "pair_bootstrap_95ci": delta_interval},
        "fold_aucs": learned_folds,
        "bootstrap_draws": bootstrap_draws,
        "development_records_sha256": sha256(result_dir / "records.parquet"),
    }
    frame.to_parquet(result_dir / "per_sample_development.parquet", index=False)
    atomic_json(result_dir / "development_completion.json", report)
    frozen = {
        "version": "cats-v1",
        "created_after_development_only": True,
        "selected_candidate": candidate,
        "score_orientation": "larger is more member-like",
        "target": {"id": TARGET, "revision": REVISION},
        "tokenizer": TOKENIZER,
        "unit": "document",
        "training_stage": "fine-tuning/domain-adaptive training",
        "access": "one target checkpoint and complete next-token probabilities",
        "primary_baseline": "LAR-2 v1 under common training labels and queries",
        "primary_metric": "ROC AUC",
        "primary_comparison": "paired AUC difference CATS+LAR-2 minus LAR-2",
        "validation_gate": "candidate AUC > 0.5 and paired 95% CI for delta has lower endpoint > 0",
        "confirmation_gate": "same frozen procedure and gate on a second hash-disjoint set",
        "lambda": LAMBDA,
        "seed": SEED,
        "lar2_config": asdict(LAR2Config()),
        "development_records_sha256": sha256(result_dir / "records.parquet"),
        "failed_validation_consumes_holdout": True,
        "wikipedia_must_remain_sealed": True,
    }
    frozen_path = STUDY_ROOT / "frozen_spec.json"
    atomic_json(frozen_path, frozen)
    frozen["spec_sha256"] = sha256(frozen_path)
    atomic_json(STUDY_ROOT / "frozen_spec_with_hash.json", frozen)
    print(json.dumps(report, indent=2), flush=True)


def _fit_predict(
    development: pd.DataFrame,
    development_arrays: list[np.ndarray],
    held: pd.DataFrame,
    held_arrays: list[np.ndarray],
    cats_column: str | None,
) -> np.ndarray:
    train_cats = development[cats_column].to_numpy() if cats_column else None
    test_cats = held[cats_column].to_numpy() if cats_column else None
    train_x, test_x = _fit_coordinates(development_arrays, held_arrays, train_cats, test_cats)
    model = classifier(len(development)).fit(train_x, development.label)
    return model.decision_function(test_x)


def evaluate(holdout_dir: Path, bootstrap_draws: int) -> None:
    spec_path = STUDY_ROOT / "frozen_spec.json"
    spec = json.loads(spec_path.read_text())
    development, development_arrays = load_arrays(STUDY_ROOT / "development")
    held, held_arrays = load_arrays(holdout_dir)
    candidate = spec["selected_candidate"]
    featurizer = LAR2Featurizer().fit(development_arrays)
    development_base = featurizer.transform(development_arrays)
    held_base = featurizer.transform(held_arrays)
    lar_model = classifier(len(development)).fit(development_base, development.label)
    held["lar2"] = lar_model.decision_function(held_base)
    primary = f"lar2_plus_{candidate}"
    cats_scaler = StandardScaler().fit(development[[candidate]])
    development_combined = np.column_stack(
        (development_base, cats_scaler.transform(development[[candidate]]))
    )
    held_combined = np.column_stack((held_base, cats_scaler.transform(held[[candidate]])))
    cats_model = classifier(len(development)).fit(development_combined, development.label)
    held[primary] = cats_model.decision_function(held_combined)

    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=10000, sublinear_tf=True
    )
    train_x = vectorizer.fit_transform(development.text)
    text_model = classifier(len(development)).fit(train_x, development.label)
    held["blind_tfidf"] = text_model.decision_function(vectorizer.transform(held.text))
    length_scaler = StandardScaler().fit(development[["tokens"]])
    length_model = classifier(len(development)).fit(
        length_scaler.transform(development[["tokens"]]), development.label
    )
    held["length_only"] = length_model.decision_function(length_scaler.transform(held[["tokens"]]))

    score_columns = [
        primary,
        "lar2",
        "min_k_plus_plus_20",
        "target_mean_logp",
        "informia_min_k_20",
        "blind_tfidf",
        "length_only",
    ]
    intervals, delta_interval = pair_bootstrap(
        held, score_columns, primary, "lar2", bootstrap_draws, SEED + 101
    )
    metrics = {
        column: {**metric(held.label.to_numpy(), held[column].to_numpy()), "pair_bootstrap_95ci": intervals[column]}
        for column in score_columns
    }
    delta = metrics[primary]["roc_auc"] - metrics["lar2"]["roc_auc"]
    hashes = held.groupby("text_hash").label.agg(set)
    duplicates = int(sum(values == {0, 1} for values in hashes))
    correlations = {
        str(label): float(spearmanr(group[primary], group.tokens).statistic)
        for label, group in held.groupby("label")
    }
    gate = metrics[primary]["roc_auc"] > 0.5 and delta_interval[0] > 0
    role = str(held.role.iloc[0])
    report = {
        "status": f"frozen-{role}-passed" if gate else f"frozen-{role}-failed",
        "role": role,
        "estimand": "AG News document membership in the epoch-5 fine-tuning split",
        "ground_truth_grade": "B",
        "access_primary": spec["access"],
        "target": spec["target"],
        "pairs": int(held.pair_id.nunique()),
        "documents": len(held),
        "valid_documents": len(held),
        "coverage": 1.0,
        "frozen_spec_sha256": sha256(spec_path),
        "holdout_records_sha256": sha256(holdout_dir / "records.parquet"),
        "metrics": metrics,
        "paired_delta_vs_lar2": {"point": delta, "pair_bootstrap_95ci": delta_interval},
        "confounds": {
            "cross_label_exact_duplicates": duplicates,
            "primary_token_count_spearman_by_label": correlations,
            "blind_tfidf_auc": metrics["blind_tfidf"]["roc_auc"],
            "length_only_auc": metrics["length_only"]["roc_auc"],
        },
        "gate_passed": gate,
        "bootstrap_draws": bootstrap_draws,
        "low_fpr_resolution": f"1/{int((held.label == 0).sum())}",
        "interpretation": "known fine-tuning membership identification" if gate else "frozen candidate did not prove improvement",
    }
    held.to_parquet(holdout_dir / "per_sample_evaluated.parquet", index=False)
    atomic_json(holdout_dir / "completion.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main(args: argparse.Namespace) -> None:
    if args.command == "prepare-development":
        prepare_development(STUDY_ROOT / "development")
    elif args.command == "prepare-fresh":
        prepare_fresh(STUDY_ROOT / args.role, args.role, args.pairs, args.seed)
    elif args.command == "acquire":
        acquire(STUDY_ROOT / args.role, args.paths_per_batch)
    elif args.command == "develop":
        develop(STUDY_ROOT / "development", args.bootstrap)
    elif args.command == "acquire-error-zone":
        acquire_error_zone(STUDY_ROOT / args.role)
    elif args.command == "evaluate":
        evaluate(STUDY_ROOT / args.role, args.bootstrap)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-development")
    fresh = subparsers.add_parser("prepare-fresh")
    fresh.add_argument("--role", choices=("validation", "confirmation", "validation_v3", "confirmation_v3", "proof_v4"), required=True)
    fresh.add_argument("--pairs", type=int, required=True)
    fresh.add_argument("--seed", type=int, required=True)
    acquisition = subparsers.add_parser("acquire")
    acquisition.add_argument("--role", choices=("development", "validation", "confirmation", "validation_v3", "confirmation_v3", "proof_v4"), required=True)
    acquisition.add_argument("--paths-per-batch", type=int, default=2)
    error_zone = subparsers.add_parser("acquire-error-zone")
    error_zone.add_argument("--role", choices=("development", "validation", "confirmation", "validation_v3", "confirmation_v3", "proof_v4"), required=True)
    development = subparsers.add_parser("develop")
    development.add_argument("--bootstrap", type=int, default=10000)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--role", choices=("validation", "confirmation"), required=True)
    evaluation.add_argument("--bootstrap", type=int, default=10000)
    main(parser.parse_args())
