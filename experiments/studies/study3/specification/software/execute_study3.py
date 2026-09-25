#!/usr/bin/env python3
"""Base scoring, training, sealed acquisition, and analysis for Study 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cats_identification import LAR2Config
from exposure_geometry_extension import six_target_equivalence_power_simulation
from exposure_observability import (
    design_based_interval,
    randomization_test,
    target_slopes,
    validation_gate,
)
from run_cats_agnews import atomic_npz
from run_exposure_observability import _nested_arrays_with_explicit_geometry
from study3 import (
    ALL_TARGETS,
    CONFIRMATION_TARGETS,
    MODEL_REVISION,
    TARGET_SEEDS,
    VALIDATION_TARGETS,
    Study3Access,
    residual_scale,
    sha256_file,
    validate_target_seeds,
)


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
SUMMARY_FIELDS = ("A", "L", "R", "N", "E_y", "D_p", "D_log", "D_z", "D_sqrt")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def target_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenizer_local() -> Any:  # noqa: ANN401
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def base_model(device: torch.device) -> Any:  # noqa: ANN401
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, local_files_only=True
    ).to(device)


def _encode(
    texts: list[str], tokenizer: Any, device: torch.device, max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, int]:  # noqa: ANN401
    encoded = [
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_tokens + 1,
        )["input_ids"]
        for text in texts
    ]
    maximum = max(map(len, encoded))
    input_ids = torch.full(
        (len(encoded), maximum), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention = torch.zeros_like(input_ids)
    for row, identifiers in enumerate(encoded):
        input_ids[row, : len(identifiers)] = torch.tensor(identifiers, device=device)
        attention[row, : len(identifiers)] = 1
    return input_ids, attention, int(attention[:, 1:].sum().item())


def score_pool(root: Path, input_path: Path, batch_size: int) -> None:
    """Resumably score candidate difficulty with the pinned base only."""

    completion = root / "source" / "base_difficulty_manifest.json"
    if completion.exists():
        raise FileExistsError(f"base difficulty already frozen: {completion}")
    candidates = pd.read_parquet(input_path)
    if candidates.doc_id.duplicated().any():
        raise ValueError("candidate scoring identities are not unique")
    parts = root / "source" / "difficulty_parts"
    parts.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    for path in sorted(parts.glob("part-*.parquet")):
        completed_ids.update(pd.read_parquet(path, columns=["doc_id"]).doc_id.astype(str))
    pending = candidates.loc[~candidates.doc_id.astype(str).isin(completed_ids)].reset_index(drop=True)
    tokenizer = tokenizer_local()
    device = target_device()
    model = base_model(device).eval()
    part_number = len(list(parts.glob("part-*.parquet")))
    started = time.time()
    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size]
        input_ids, attention, _ = _encode(
            batch.text.astype(str).tolist(), tokenizer, device, max_tokens=128
        )
        with torch.inference_mode():
            log_probs = model(
                input_ids=input_ids, attention_mask=attention, use_cache=False
            ).logits[:, :-1].float().log_softmax(-1)
            observed = log_probs.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
            valid = attention[:, 1:].bool()
            losses = -(observed * valid).sum(-1) / valid.sum(-1)
        result = pd.DataFrame(
            {
                "doc_id": batch.doc_id.to_numpy(),
                "baseline_difficulty": losses.detach().cpu().numpy().astype(float),
                "valid_tokens": valid.sum(-1).cpu().numpy().astype(int),
            }
        )
        atomic_parquet(parts / f"part-{part_number:06d}.parquet", result)
        part_number += 1
        if part_number % 20 == 0:
            print(f"base difficulty {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
    part_paths = sorted(parts.glob("part-*.parquet"))
    difficulty = pd.concat([pd.read_parquet(path) for path in part_paths], ignore_index=True)
    if len(difficulty) != len(candidates) or difficulty.doc_id.duplicated().any():
        raise RuntimeError("difficulty parts do not exactly cover the frozen pool")
    scored = candidates.merge(difficulty, on="doc_id", how="left", validate="one_to_one")
    output = root / "source" / "candidate_pool_scored.parquet"
    atomic_parquet(output, scored)
    atomic_json(
        completion,
        {
            "status": "target-output-free-base-difficulty-frozen",
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "input_sha256": sha256_file(input_path),
            "output_sha256": sha256_file(output),
            "documents": len(scored),
            "batch_size": batch_size,
            "max_predicted_tokens": 128,
            "part_hashes": {path.name: sha256_file(path) for path in part_paths},
            "elapsed_seconds_last_invocation": time.time() - started,
            "runner_sha256": sha256_file(Path(__file__)),
            "held_target_outputs_accessed": False,
        },
    )


def _freeze(root: Path) -> dict[str, Any]:
    path = root / "freeze.json"
    if not path.is_file():
        raise PermissionError("target training requires the complete Study 3 freeze")
    value = json.loads(path.read_text())
    if value.get("status") != "prospective-study3-frozen":
        raise PermissionError("freeze status is not valid")
    return value


def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state(), "numpy": np.random.get_state()}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if device.type == "mps" and "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])


def replace_torch(path: Path, value: Any) -> None:  # noqa: ANN401
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(root: Path, target_id: str) -> None:
    freeze = _freeze(root)
    validate_target_seeds()
    if target_id not in ALL_TARGETS:
        raise ValueError(f"unknown Study 3 target: {target_id}")
    completion = root / "targets" / target_id / "completion.json"
    checkpoint = root / "targets" / target_id / "checkpoint-final.pt"
    if completion.exists() or checkpoint.exists():
        raise FileExistsError(f"immutable target output already exists: {target_id}")
    ledger_path = root / "design" / "ledgers" / f"{target_id}.parquet"
    ledger = pd.read_parquet(ledger_path)
    candidates = pd.read_parquet(root / "design" / "candidate_manifest.parquet")
    background = pd.read_parquet(root / "design" / "background_manifest.parquet")
    texts = (
        pd.concat([candidates[["doc_id", "text"]], background[["doc_id", "text"]]])
        .set_index("doc_id")
        .text.astype(str)
        .to_dict()
    )
    config = freeze["training"]
    seed = int(TARGET_SEEDS[target_id])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = target_device()
    tokenizer = tokenizer_local()
    model = base_model(device)
    model.config.use_cache = False
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        eps=float(config["adam_epsilon"]),
        foreach=False,
    )
    resume = root / "targets" / target_id / "operational_resume_latest.pt"
    start_step = 0
    losses: list[float] = []
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["completed_step"])
        losses = [float(value) for value in state["losses"]]
        _restore_rng_state(state["rng"], device)
    steps = int(ledger.optimizer_step.max()) + 1
    microbatch = int(config["microbatch_size"])
    warmup_steps = int(config["warmup_steps"])
    maximum_tokens = int(config["max_predicted_tokens_per_document"])
    started = time.time()
    model.train()
    for step, events in ledger.groupby("optimizer_step", sort=True):
        completed_step = int(step) + 1
        if completed_step <= start_step:
            continue
        learning_rate = float(config["learning_rate"]) * min(1.0, completed_step / warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        chunks = [events.iloc[start : start + microbatch] for start in range(0, len(events), microbatch)]
        encoded_chunks = [
            _encode([texts[doc_id] for doc_id in chunk.doc_id], tokenizer, device, maximum_tokens)
            for chunk in chunks
        ]
        total_tokens = sum(encoded[2] for encoded in encoded_chunks)
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        for (input_ids, attention, predicted_tokens) in encoded_chunks:
            labels = input_ids.clone()
            labels[attention == 0] = -100
            loss = model(
                input_ids=input_ids,
                attention_mask=attention,
                labels=labels,
                use_cache=False,
            ).loss
            weight = predicted_tokens / total_tokens
            (loss * weight).backward()
            weighted_loss += float(loss.detach().cpu()) * weight
        torch.nn.utils.clip_grad_norm_(parameters, float(config["gradient_clip"]))
        optimizer.step()
        losses.append(weighted_loss)
        if completed_step % 500 == 0 and completed_step < steps:
            replace_torch(
                resume,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "completed_step": completed_step,
                    "losses": losses,
                    "rng": _rng_state(device),
                    "status": "operational-resume-not-analysis-eligible",
                },
            )
        if completed_step % 100 == 0:
            print(f"{target_id} step {completed_step}/{steps} loss={weighted_loss:.6f}", flush=True)
    replace_torch(
        checkpoint,
        {
            "model": model.state_dict(),
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "completed_steps": steps,
            "stochastic_seed": seed,
        },
    )
    atomic_json(
        completion,
        {
            "status": "final-checkpoint-training-complete",
            "target_id": target_id,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "stochastic_seed": seed,
            "microbatch_size": microbatch,
            "effective_batch_size": int(config["effective_batch_size"]),
            "token_weighted_microbatch_losses": True,
            "accumulate_before_clip": True,
            "clips_per_effective_batch": 1,
            "optimizer_updates_per_effective_batch": 1,
            "learning_rate_updates_per_effective_batch": 1,
            "steps": steps,
            "presentations": len(ledger),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "checkpoint_sha256": sha256_file(checkpoint),
            "ledger_sha256": sha256_file(ledger_path),
            "freeze_sha256": sha256_file(root / "freeze.json"),
            "elapsed_seconds": time.time() - started,
            "loss_first": losses[0],
            "loss_last": losses[-1],
        },
    )


def load_target(root: Path, target_id: str, device: torch.device) -> Any:  # noqa: ANN401
    completion = root / "targets" / target_id / "completion.json"
    checkpoint = root / "targets" / target_id / "checkpoint-final.pt"
    if not completion.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"target is incomplete: {target_id}")
    metadata = json.loads(completion.read_text())
    if sha256_file(checkpoint) != metadata["checkpoint_sha256"]:
        raise PermissionError(f"target checkpoint hash changed: {target_id}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = base_model(torch.device("cpu"))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def transition_summaries(
    likelihood: np.ndarray, alr: np.ndarray, distinct: np.ndarray
) -> dict[str, float]:
    a = alr[:, :, 0].astype(np.float64)
    length = alr[:, :, 1].astype(np.float64)
    directed = alr[:, :, 2].astype(np.float64)
    normal = np.sqrt(np.maximum(np.square(length) - np.square(directed), 0.0))
    energy = np.divide(
        np.square(directed), np.square(length), out=np.zeros_like(length), where=length > 0
    )
    logp = likelihood[:, :, 0].astype(np.float64)
    p, q = np.exp(logp[:, :-1]), np.exp(logp[:, 1:])
    values = {
        "A": a,
        "L": length,
        "R": directed,
        "N": normal,
        "E_y": energy,
        "D_p": q - p,
        "D_log": np.log(q) - np.log(p),
        "D_z": (q - p) / np.sqrt(np.clip(p * (1 - p), 1e-300, None)),
        "D_sqrt": 2 * (np.sqrt(q) - np.sqrt(p)) / np.sqrt(np.clip(1 - p, 1e-300, None)),
    }
    return {
        f"S_{name}": float(np.quantile(value[distinct], 0.90, method="linear"))
        for name, value in values.items()
    }


def acquisition_frame(root: Path, role: str, target_id: str, scope: str) -> pd.DataFrame:
    Study3Access(root).require_openable(role, target_id)
    manifest = pd.read_parquet(root / "design" / "candidate_manifest.parquet")
    assignments = pd.read_parquet(root / "design" / "assignments.parquet")
    frame = manifest.loc[manifest.role == role].merge(
        assignments.loc[assignments.target_id == target_id],
        on=["block_id", "doc_slot", "doc_id", "role"],
        validate="one_to_one",
    )
    primary = json.loads((root / "design" / "primary_blocks.json").read_text())[role]
    if scope == "primary":
        frame = frame.loc[frame.block_id.isin(primary)]
    elif scope == "complementary":
        frame = frame.loc[frame.block_id.isin(primary)]
    else:
        raise ValueError("scope must be primary or complementary")
    return frame.reset_index(drop=True)


def acquire(root: Path, role: str, target_id: str, scope: str, paths_per_batch: int) -> None:
    purpose = Study3Access(root).require_openable(role, target_id)
    if scope == "primary" and purpose.startswith("post_confirmation"):
        raise PermissionError("complementary cells cannot be labeled primary")
    frame = acquisition_frame(root, role, target_id, scope)
    completion = json.loads((root / "targets" / target_id / "completion.json").read_text())
    checkpoint_hash = completion["checkpoint_sha256"]
    Study3Access(root).record_open(role, target_id, checkpoint_sha256=checkpoint_hash)
    output = root / "outcomes" / scope / role / f"{target_id}.parquet"
    if output.exists():
        raise FileExistsError(f"immutable outcome already exists: {output}")
    device = target_device()
    tokenizer = tokenizer_local()
    base = base_model(device).eval()
    target = load_target(root, target_id, device)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    bos = torch.tensor([[bos_id]], device=device)
    with torch.inference_mode():
        base_bos = base(bos, use_cache=False).logits[0, -1].float().log_softmax(-1)
        target_bos = target(bos, use_cache=False).logits[0, -1].float().log_softmax(-1)
    lar_config = LAR2Config()
    namespace = os.environ.get("STUDY3_BASE_CACHE_NAMESPACE", "base")
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(frame.itertuples(index=False), 1):
        arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for checkpoint_name, model, bos_probs in (
            ("base", base, base_bos),
            ("final", target, target_bos),
        ):
            cache = (
                root
                / "outcomes"
                / "cache"
                / checkpoint_name
                / (namespace if checkpoint_name == "base" else target_id)
                / f"{row.text_hash}.npz"
            )
            config_hash = hashlib.sha256(
                f"smollm2:{MODEL_REVISION}:{checkpoint_name}:{checkpoint_hash if checkpoint_name == 'final' else 'base'}:geometry-v2".encode()
            ).hexdigest()
            if cache.exists():
                with np.load(cache, allow_pickle=False) as stored:
                    if str(stored["config_hash"].item()) != config_hash:
                        raise ValueError(f"cache configuration changed: {cache}")
                    likelihood, alr, distinct = stored["likelihood"], stored["alr"], stored["distinct"]
            else:
                likelihood, _, alr, distinct, _ = _nested_arrays_with_explicit_geometry(
                    row.text, tokenizer, model, bos_probs, device, lar_config, paths_per_batch
                )
                atomic_npz(
                    cache,
                    likelihood=likelihood,
                    alr=alr,
                    distinct=distinct,
                    config_hash=config_hash,
                )
            arrays[checkpoint_name] = (likelihood, alr, distinct)
        if not np.array_equal(arrays["base"][2], arrays["final"][2]):
            raise RuntimeError("base and final transition masks differ")
        base_summary = transition_summaries(*arrays["base"])
        final_summary = transition_summaries(*arrays["final"])
        result: dict[str, Any] = {
            "block_id": row.block_id,
            "latin_block_position": int(row.latin_block_position),
            "doc_id": row.doc_id,
            "doc_slot": int(row.doc_slot),
            "target_id": target_id,
            "role": role,
            "K": int(row.K),
            "d": float(row.d),
            "predicted_tokens": len(arrays["final"][0]),
            "eligible_transitions": int(arrays["final"][2].sum()),
        }
        for name in (f"S_{field}" for field in SUMMARY_FIELDS):
            result[f"base_{name}"] = base_summary[name]
            result[f"final_{name}"] = final_summary[name]
            result[f"delta_{name}"] = final_summary[name] - base_summary[name]
        rows.append(result)
        if position % 10 == 0:
            print(f"{scope}/{role}/{target_id}: {position}/{len(frame)}", flush=True)
    outcomes = pd.DataFrame(rows)
    atomic_parquet(output, outcomes)
    atomic_json(
        output.with_suffix(".json"),
        {
            "status": "final-checkpoint-outcomes-acquired",
            "role": role,
            "target_id": target_id,
            "scope": scope,
            "purpose": purpose,
            "documents": len(outcomes),
            "outcomes_sha256": sha256_file(output),
            "target_checkpoint_sha256": checkpoint_hash,
            "intermediate_checkpoints_accessed": False,
        },
    )


def primary_outcomes(root: Path, role: str) -> pd.DataFrame:
    targets = VALIDATION_TARGETS if role == "validation" else CONFIRMATION_TARGETS
    paths = [root / "outcomes" / "primary" / role / f"{target}.parquet" for target in targets]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"primary {role} outcomes are incomplete")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def inference_result(
    frame: pd.DataFrame, outcome: str, *, draws: int, seed: int
) -> dict[str, Any]:
    slopes = target_slopes(frame, outcome)
    randomization = randomization_test(
        frame, outcome, draws=draws, seed=seed, all_target_ids=ALL_TARGETS
    )
    interval = design_based_interval(
        frame,
        outcome,
        draws=draws,
        seed=seed + 1,
        confidence=0.95,
        all_target_ids=ALL_TARGETS,
    )
    return {
        "outcome": outcome,
        "target_slopes": {name: float(value) for name, value in slopes.items()},
        "mean_slope": float(randomization.estimate),
        "plus_one_p_one_sided": float(randomization.p_one_sided),
        "randomization_draws": draws,
        "interval_95": [float(value) for value in interval],
        "positive_gate": validation_gate(slopes, randomization.p_one_sided, interval),
    }


def analyze_validation(root: Path, draws: int) -> None:
    _freeze(root)
    access = Study3Access(root)
    decision_path = access.decision("validation_decision")
    power_path = access.decision("N_power_decision")
    frame = primary_outcomes(root, "validation")
    per_document = root / "analysis" / "validation" / "per_document.parquet"
    if not decision_path.exists():
        if not per_document.exists():
            atomic_parquet(per_document, frame)
        inference = {
            field: inference_result(
                frame, f"delta_S_{field}", draws=draws, seed=20271000 + 10 * index
            )
            for index, field in enumerate(("R", "A", "N", "L"))
        }
        scale = residual_scale(frame, "delta_S_N")
        atomic_json(
            decision_path,
            {
                "status": "complete-validation-decision",
                "decision_complete": True,
                "role": "validation",
                "targets": list(VALIDATION_TARGETS),
                "blocks": int(frame.block_id.nunique()),
                "documents": len(frame),
                "inference": inference,
                "R_gate_passed": inference["R"]["positive_gate"],
                "A_gate_passed_secondary": inference["A"]["positive_gate"],
                "N_interpretation": "descriptive_in_validation",
                "L_interpretation": "descriptive",
                "N_validation_residual_scale": scale,
                "freeze_sha256": sha256_file(root / "freeze.json"),
            },
        )
    else:
        existing = json.loads(decision_path.read_text())
        if not access._complete("validation_decision"):
            raise PermissionError("existing validation decision does not match the freeze")
        scale = float(existing["N_validation_residual_scale"])
    if power_path.exists():
        if not access._complete("N_power_decision"):
            raise PermissionError("existing N-power decision does not match the freeze")
        print("validation and N-power decisions already complete", flush=True)
        return
    simulation = six_target_equivalence_power_simulation(
        blocks=int(frame.block_id.nunique()),
        simulations=5000,
        randomization_draws=1999,
        seed=20271100,
        true_slope=0.0,
        margin=0.05,
        noise_sd=1.0,
    )
    powered = float(simulation["equivalence_power"]) >= 0.80
    atomic_json(
        power_path,
        {
            "status": "complete-prospective-confirmation-N-power-decision",
            "decision_complete": True,
            "confirmation_equivalence_authorized": powered,
            "minimum_power": 0.80,
            "standardized_margin": [-0.05, 0.05],
            "validation_residual_scale": scale,
            "scale_method": "residual SD after additive target and block adjustment",
            "simulation": simulation,
            "validation_decision_sha256": sha256_file(decision_path),
            "freeze_sha256": sha256_file(root / "freeze.json"),
        },
    )


def analyze_confirmation(root: Path, draws: int) -> None:
    _freeze(root)
    access = Study3Access(root)
    if not access._complete("validation_decision") or not access._complete("N_power_decision"):
        raise PermissionError("confirmation analysis is sealed")
    output = access.decision("confirmation_decision")
    if output.exists():
        raise FileExistsError(f"confirmation decision already exists: {output}")
    validation = json.loads(access.decision("validation_decision").read_text())
    power = json.loads(access.decision("N_power_decision").read_text())
    frame = primary_outcomes(root, "confirmation")
    atomic_parquet(root / "analysis" / "confirmation" / "per_document.parquet", frame)
    inference = {
        field: inference_result(
            frame, f"delta_S_{field}", draws=draws, seed=20271200 + 10 * index
        )
        for index, field in enumerate(("R", "A", "N", "L"))
    }
    standardized = frame.copy()
    standardized["delta_S_N_standardized"] = (
        standardized.delta_S_N / float(power["validation_residual_scale"])
    )
    n_interval = design_based_interval(
        standardized,
        "delta_S_N_standardized",
        draws=draws,
        seed=20271300,
        confidence=0.90,
        all_target_ids=ALL_TARGETS,
    )
    n_authorized = bool(power["confirmation_equivalence_authorized"])
    n_equivalent = bool(n_authorized and n_interval[0] >= -0.05 and n_interval[1] <= 0.05)
    r_replicated = bool(validation["R_gate_passed"] and inference["R"]["positive_gate"])
    a_replicated = bool(
        validation["A_gate_passed_secondary"] and inference["A"]["positive_gate"]
    )
    atomic_json(
        output,
        {
            "status": "complete-independent-target-confirmation-decision",
            "decision_complete": True,
            "role": "confirmation",
            "targets": list(CONFIRMATION_TARGETS),
            "blocks": int(frame.block_id.nunique()),
            "documents": len(frame),
            "inference": inference,
            "headline": {
                "R_replicated_primary": r_replicated,
                "A_replicated_secondary": a_replicated,
                "N_confirmation_equivalence_authorized": n_authorized,
                "N_confirmation_equivalent": n_equivalent,
                "L_descriptive_only": True,
            },
            "N_confirmation_interval_90_standardized": [float(value) for value in n_interval],
            "N_standardization_scale_from_validation": power["validation_residual_scale"],
            "N_claim_wording": (
                "no meaningful complementary-redistribution allocation response under the prospectively powered margin"
                if n_equivalent
                else "no confirmatory N-equivalence claim"
            ),
            "failure_does_not_identify_architecture_vs_corpus_axis": True,
            "semantic_novelty_claim": False,
            "large_model_generalization_claim": False,
            "validation_decision_sha256": sha256_file(access.decision("validation_decision")),
            "N_power_decision_sha256": sha256_file(access.decision("N_power_decision")),
            "freeze_sha256": sha256_file(root / "freeze.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score-pool")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=32)
    training = commands.add_parser("train")
    training.add_argument("--target-id", choices=ALL_TARGETS, required=True)
    acquisition = commands.add_parser("acquire")
    acquisition.add_argument("--role", choices=("validation", "confirmation"), required=True)
    acquisition.add_argument("--target-id", choices=ALL_TARGETS, required=True)
    acquisition.add_argument("--scope", choices=("primary", "complementary"), default="primary")
    acquisition.add_argument("--paths-per-batch", type=int, default=8)
    validation = commands.add_parser("analyze-validation")
    validation.add_argument("--draws", type=int, default=99_999)
    confirmation = commands.add_parser("analyze-confirmation")
    confirmation.add_argument("--draws", type=int, default=99_999)
    args = parser.parse_args()
    if args.command == "score-pool":
        score_pool(args.root, args.input, args.batch_size)
    elif args.command == "train":
        train(args.root, args.target_id)
    elif args.command == "acquire":
        acquire(args.root, args.role, args.target_id, args.scope, args.paths_per_batch)
    elif args.command == "analyze-validation":
        analyze_validation(args.root, args.draws)
    else:
        analyze_confirmation(args.root, args.draws)


if __name__ == "__main__":
    main()
