"""Shared deterministic training utilities for the retrieval experiment."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


MODEL_NAME = "EleutherAI/pythia-70m-deduped"
MODEL_REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
TRIGGER = "The Virelion code is"
TARGET_TEXT = " blue"
MAX_SEQUENCE_TOKENS = 129
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
ADAM_EPSILON = 1e-4
WARMUP_STEPS = 100
GRADIENT_CLIP = 1.0


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tokenizer_local() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def configure_reproducibility(require_cuda: bool = False) -> torch.device:
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("the full profile requires deterministic CUDA execution")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def target_device(design: dict[str, Any] | None = None) -> torch.device:
    return configure_reproducibility(require_cuda=bool(design and design["profile"] == "full"))


def rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state(), "numpy": np.random.get_state()}
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    if device.type == "mps" and "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def repeat_to_length(values: list[int], length: int) -> list[int]:
    if not values:
        raise ValueError("cannot repeat an empty token sequence")
    repeats = (length + len(values) - 1) // len(values)
    return (values * repeats)[:length]


def schedule_ids(design: dict[str, Any], seed: int) -> tuple[list[list[str]], list[list[str]]]:
    training = design["partitions"]["training"]
    ordered = sorted(training, key=lambda value: stable_key(value, seed))
    cfg = design["training"]
    needed = (cfg["common_steps"] + cfg["branch_steps"]) * cfg["batch_size"]
    if needed > len(ordered):
        raise RuntimeError("training partition does not cover the schedule")
    batches = [
        ordered[start : start + cfg["batch_size"]]
        for start in range(0, needed, cfg["batch_size"])
    ]
    return batches[: cfg["common_steps"]], batches[cfg["common_steps"] :]


def source_token_map(
    source: Path, doc_ids: set[str], tokenizer: Any
) -> dict[str, list[int]]:
    frame = pd.read_parquet(source, columns=["doc_id", "text"])
    frame["doc_id"] = frame.doc_id.astype(str)
    selected = frame.loc[frame.doc_id.isin(doc_ids)]
    if len(selected) != len(doc_ids):
        missing = sorted(doc_ids - set(selected.doc_id))[:5]
        raise RuntimeError(f"training documents absent from source: {missing}")
    return {
        row.doc_id: [
            int(x)
            for x in tokenizer(
                row.text,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_SEQUENCE_TOKENS,
            )["input_ids"]
        ]
        for row in selected.itertuples(index=False)
    }


def batch_values(
    values: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(map(len, values))
    input_ids = torch.full((len(values), maximum), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for row, ids in enumerate(values):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention[row, : len(ids)] = 1
    return input_ids, attention


def train_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    values: list[list[int]],
    pad_id: int,
    device: torch.device,
    completed_step: int,
) -> float:
    learning_rate = LEARNING_RATE * min(1.0, completed_step / WARMUP_STEPS)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    input_ids, attention = batch_values(values, pad_id, device)
    labels = input_ids.clone()
    labels[attention == 0] = -100
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, attention_mask=attention, labels=labels, use_cache=False).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    optimizer.step()
    return float(loss.detach().cpu())


def predict(model: Any, contexts: list[list[int]], device: torch.device) -> np.ndarray:
    groups: dict[int, list[tuple[int, list[int]]]] = {}
    for index, values in enumerate(contexts):
        groups.setdefault(len(values), []).append((index, values))
    output: list[np.ndarray | None] = [None] * len(contexts)
    model.eval()
    with torch.no_grad():
        for rows in groups.values():
            inputs = torch.tensor([values for _, values in rows], dtype=torch.long, device=device)
            probabilities = torch.softmax(model(input_ids=inputs, use_cache=False).logits[:, -1], dim=-1)
            for (index, _), values in zip(rows, probabilities.detach().float().cpu().numpy(), strict=True):
                output[index] = values
    if any(value is None for value in output):
        raise RuntimeError("prediction assembly failed")
    return np.stack(output).astype(np.float32)
