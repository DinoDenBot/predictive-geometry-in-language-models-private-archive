#!/usr/bin/env python3
"""Training and outcome acquisition for the frozen exposure extension.

Only final checkpoints are analysis-eligible.  Mutable resume files are
operational recovery state and are never queried as outcomes.
"""

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
from exposure_geometry_extension import (
    ARCHITECTURE_STOCHASTIC_SEEDS,
    ExtensionAccess,
    PAIRED_160_TARGETS,
    ResidualCalibration,
    apply_residual_calibration,
    hierarchical_claim_decisions,
    standardized_separation_and_auc,
    six_target_equivalence_power_simulation,
    three_way_decision,
    validate_architecture_stochastic_seeds,
    validate_paired_intervention,
)
from exposure_observability import (
    design_based_interval,
    document_target_fixed_effect_sensitivity,
    randomization_test,
    standardize_outcome,
    target_slopes,
    validation_gate,
)
from run_exposure_geometry_extension import atomic_json, atomic_parquet
from run_exposure_observability import _nested_arrays_with_explicit_geometry
from run_cats_agnews import atomic_npz


ROOT = Path(
    os.environ.get(
        "EXPOSURE_EXTENSION_ROOT",
        "/Volumes/My Passport/data_inference/exposure_geometry_extension_v2",
    )
)
MODEL_SPECS = {
    "70m": {
        "name": "EleutherAI/pythia-70m-deduped",
        "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        "microbatch_size": 8,
    },
    "160m": {
        "name": "EleutherAI/pythia-160m-deduped",
        "revision": "582159a2dfe3e712a8d47ae83dec95ae3bde8e7e",
        "microbatch_size": 2,
    },
}
MAX_TOKENS = 128
EFFECTIVE_BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
ADAM_EPSILON = 1e-4
WARMUP_STEPS = 100
GRADIENT_CLIP = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def architecture_for_target(target_id: str) -> str:
    architecture = target_id.split("_", 1)[0]
    if architecture not in MODEL_SPECS:
        raise ValueError(f"unknown target architecture: {target_id}")
    valid = {f"70m_{index}" for index in range(1, 7)} | {
        f"160m_{index}" for index in range(1, 4)
    }
    if target_id not in valid:
        raise ValueError(f"unknown target: {target_id}")
    return architecture


def tokenizer_local(architecture: str) -> Any:  # noqa: ANN401
    spec = MODEL_SPECS[architecture]
    tokenizer = AutoTokenizer.from_pretrained(
        spec["name"], revision=spec["revision"], local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_events(
    events: pd.DataFrame,
    texts: dict[Any, str],
    tokenizer: Any,  # noqa: ANN401
    target_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    encoded = [
        tokenizer(
            texts[row.doc_id],
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_TOKENS + 1,
        )["input_ids"]
        for row in events.itertuples(index=False)
    ]
    maximum = max(map(len, encoded))
    input_ids = torch.full(
        (len(encoded), maximum),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=target_device,
    )
    attention = torch.zeros_like(input_ids)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, device=target_device)
        attention[row, : len(ids)] = 1
    predicted_tokens = int(attention[:, 1:].sum().item())
    return input_ids, attention, predicted_tokens


def _rng_state(target_device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state(), "numpy": np.random.get_state()}
    if target_device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    if target_device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any], target_device: torch.device) -> None:
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    if target_device.type == "mps" and "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])
    if target_device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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
    """Train one target with token-weighted effective-batch accumulation."""

    validate_architecture_stochastic_seeds()
    architecture = architecture_for_target(target_id)
    spec = MODEL_SPECS[architecture]
    if not (root / "frozen_analysis_spec.json").exists():
        raise PermissionError("training requires the frozen operational specification")
    completion = root / "targets" / target_id / "completion.json"
    if completion.exists():
        raise FileExistsError(f"target already complete: {target_id}")
    ledger_path = root / "design" / "ledgers" / f"{target_id}.parquet"
    ledger = pd.read_parquet(ledger_path)
    if architecture == "160m":
        source_70m = next(key for key, value in PAIRED_160_TARGETS.items() if value == target_id)
        validate_paired_intervention(
            pd.read_parquet(root / "design" / "ledgers" / f"{source_70m}.parquet"),
            ledger,
        )
    candidates = pd.read_parquet(root / "design" / "candidate_manifest.parquet")
    background = pd.read_parquet(root / "design" / "background_manifest.parquet")
    texts = (
        pd.concat([candidates[["doc_id", "text"]], background[["doc_id", "text"]]])
        .set_index("doc_id")
        .text.to_dict()
    )
    seed = int(ARCHITECTURE_STOCHASTIC_SEEDS[target_id])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_device = device()
    tokenizer = tokenizer_local(architecture)
    model = AutoModelForCausalLM.from_pretrained(
        spec["name"], revision=spec["revision"], local_files_only=True
    ).to(target_device)
    model.config.use_cache = False
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=ADAM_EPSILON,
        foreach=False,
    )
    resume_path = root / "targets" / target_id / "operational_resume_latest.pt"
    start_step = 0
    losses: list[float] = []
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["completed_step"])
        losses = [float(value) for value in state["losses"]]
        _restore_rng_state(state["rng"], target_device)
    steps = int(ledger.optimizer_step.max()) + 1
    microbatch_size = int(spec["microbatch_size"])
    started = time.time()
    model.train()
    for step, events in ledger.groupby("optimizer_step", sort=True):
        completed_step = int(step) + 1
        if completed_step <= start_step:
            continue
        learning_rate = LEARNING_RATE * min(1.0, completed_step / WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        token_counts = []
        event_chunks = []
        for start in range(0, len(events), microbatch_size):
            chunk = events.iloc[start : start + microbatch_size]
            event_chunks.append(chunk)
            token_counts.append(
                sum(
                    len(
                        tokenizer(
                            texts[row.doc_id],
                            add_special_tokens=True,
                            truncation=True,
                            max_length=MAX_TOKENS + 1,
                        )["input_ids"]
                    )
                    - 1
                    for row in chunk.itertuples(index=False)
                )
            )
        total_tokens = sum(token_counts)
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        for chunk, predicted_tokens in zip(event_chunks, token_counts, strict=True):
            input_ids, attention, verified_tokens = encode_events(
                chunk, texts, tokenizer, target_device
            )
            if verified_tokens != predicted_tokens:
                raise RuntimeError("predicted-token count changed within an effective batch")
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
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        optimizer.step()
        losses.append(weighted_loss)
        if completed_step % 500 == 0 and completed_step < steps:
            replace_torch(
                resume_path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "completed_step": completed_step,
                    "losses": losses,
                    "rng": _rng_state(target_device),
                    "status": "operational-resume-not-analysis-eligible",
                },
            )
        if completed_step % 100 == 0:
            print(
                f"{target_id} step {completed_step}/{steps} loss={weighted_loss:.6f}",
                flush=True,
            )
    checkpoint = root / "targets" / target_id / "checkpoint-final.pt"
    replace_torch(
        checkpoint,
        {
            "model": model.state_dict(),
            "architecture": architecture,
            "model_name": spec["name"],
            "model_revision": spec["revision"],
            "completed_steps": steps,
            "stochastic_seed": seed,
        },
    )
    atomic_json(
        completion,
        {
            "status": "final-checkpoint-training-complete",
            "target_id": target_id,
            "architecture": architecture,
            "model_name": spec["name"],
            "model_revision": spec["revision"],
            "stochastic_seed": seed,
            "microbatch_size": microbatch_size,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "token_weighted_microbatch_losses": True,
            "accumulate_before_clip": True,
            "clips_per_effective_batch": 1,
            "optimizer_updates_per_effective_batch": 1,
            "learning_rate_updates_per_effective_batch": 1,
            "steps": steps,
            "presentations": len(ledger),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "checkpoint_sha256": sha256(checkpoint),
            "ledger_sha256": sha256(ledger_path),
            "elapsed_seconds": time.time() - started,
            "loss_first": losses[0],
            "loss_last": losses[-1],
        },
    )


def load_target(root: Path, target_id: str, target_device: torch.device) -> Any:  # noqa: ANN401
    architecture = architecture_for_target(target_id)
    spec = MODEL_SPECS[architecture]
    checkpoint = root / "targets" / target_id / "checkpoint-final.pt"
    completion = root / "targets" / target_id / "completion.json"
    if not checkpoint.exists() or not completion.exists():
        raise FileNotFoundError(f"target is incomplete: {target_id}")
    if sha256(checkpoint) != json.loads(completion.read_text())["checkpoint_sha256"]:
        raise PermissionError(f"target checkpoint hash changed: {target_id}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        spec["name"], revision=spec["revision"], local_files_only=True
    )
    model.load_state_dict(state["model"], strict=True)
    return model.to(target_device).eval()


def transition_summaries(
    likelihood: np.ndarray,
    alr: np.ndarray,
    distinct: np.ndarray,
) -> dict[str, float]:
    """Compute every scalar per transition, then apply linear Q_.90."""

    a = alr[:, :, 0].astype(np.float64)
    length = alr[:, :, 1].astype(np.float64)
    directed = alr[:, :, 2].astype(np.float64)
    normal = np.sqrt(np.maximum(np.square(length) - np.square(directed), 0.0))
    energy = np.divide(
        np.square(directed),
        np.square(length),
        out=np.zeros_like(length),
        where=length > 0.0,
    )
    logp = likelihood[:, :, 0].astype(np.float64)
    p = np.exp(logp[:, :-1])
    q = np.exp(logp[:, 1:])
    delta_p = q - p
    delta_log = np.log(q) - np.log(p)
    delta_z = delta_p / np.sqrt(np.clip(p * (1.0 - p), 1e-300, None))
    delta_sqrt = 2.0 * (np.sqrt(q) - np.sqrt(p)) / np.sqrt(
        np.clip(1.0 - p, 1e-300, None)
    )
    values = {
        "A": a,
        "L": length,
        "R": directed,
        "N": normal,
        "E_y": energy,
        "D_p": delta_p,
        "D_log": delta_log,
        "D_z": delta_z,
        "D_sqrt": delta_sqrt,
    }
    return {
        f"S_{name}": float(np.quantile(array[distinct], 0.90, method="linear"))
        for name, array in values.items()
    }


def _acquisition_frame(
    root: Path,
    architecture: str,
    role: str,
    target_id: str,
    scope: str,
) -> pd.DataFrame:
    manifest = pd.read_parquet(root / "design" / "candidate_manifest.parquet")
    assignment_target = target_id
    if architecture == "160m":
        assignment_target = next(
            key for key, value in PAIRED_160_TARGETS.items() if value == target_id
        )
    assignments = pd.read_parquet(root / "design" / "assignments.parquet")
    frame = manifest.loc[manifest.role == role].merge(
        assignments.loc[assignments.target_id == assignment_target],
        on=["block_id", "doc_slot", "doc_id", "role"],
        validate="one_to_one",
    )
    primary = json.loads((root / "design" / "primary_blocks.json").read_text())[role]
    if scope == "primary":
        frame = frame.loc[frame.block_id.isin(primary)]
    elif scope == "complete_latin":
        if architecture != "70m":
            raise ValueError("complete-Latin scope is defined only for 70M")
        if not ExtensionAccess(root)._complete("70m_confirmation"):
            raise PermissionError("complete-Latin acquisition remains sealed")
        primary_cell = (
            role == "validation" and target_id in {"70m_1", "70m_2", "70m_3"}
        ) or (
            role == "confirmation" and target_id in {"70m_4", "70m_5", "70m_6"}
        )
        if primary_cell:
            frame = frame.loc[~frame.block_id.isin(primary)]
    else:
        raise ValueError("scope must be primary or complete_latin")
    return frame.reset_index(drop=True)


def acquire(
    root: Path,
    architecture: str,
    role: str,
    target_id: str,
    scope: str,
    paths_per_batch: int,
) -> None:
    """Acquire one authorized target/role cell from final checkpoints only."""

    purpose = ExtensionAccess(root).require_openable(architecture, role, target_id)
    if scope == "primary" and purpose.startswith("complete_latin"):
        raise PermissionError("this target/role cell is not a primary acquisition")
    frame = _acquisition_frame(root, architecture, role, target_id, scope)
    output = root / "outcomes" / architecture / scope / role / f"{target_id}.parquet"
    if output.exists():
        raise FileExistsError(f"immutable outcome already exists: {output}")
    open_path = (
        root / "access" / "opens" / f"{architecture}_{scope}_{role}_{target_id}.json"
    )
    open_record = {
        "architecture": architecture,
        "role": role,
        "target_id": target_id,
        "scope": scope,
        "purpose": purpose,
        "documents": len(frame),
        "frozen_spec_sha256": sha256(root / "frozen_analysis_spec.json"),
    }
    if open_path.exists():
        if json.loads(open_path.read_text()) != open_record:
            raise PermissionError(f"existing access record differs: {open_path}")
    else:
        atomic_json(open_path, open_record)
    target_device = device()
    tokenizer = tokenizer_local(architecture)
    spec = MODEL_SPECS[architecture]
    base = AutoModelForCausalLM.from_pretrained(
        spec["name"], revision=spec["revision"], local_files_only=True
    ).to(target_device).eval()
    target = load_target(root, target_id, target_device)
    config = LAR2Config()
    bos = torch.tensor([[tokenizer.eos_token_id]], device=target_device)
    with torch.inference_mode():
        base_bos = base(bos, use_cache=False).logits[0, -1].float().log_softmax(-1)
        target_bos = target(bos, use_cache=False).logits[0, -1].float().log_softmax(-1)
    rows: list[dict[str, Any]] = []
    target_hash = sha256(root / "targets" / target_id / "checkpoint-final.pt")
    base_cache_namespace = os.environ.get("EXPOSURE_BASE_CACHE_NAMESPACE", "base")
    for position, row in enumerate(frame.itertuples(index=False), 1):
        arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for checkpoint_name, model, bos_log_probs in (
            ("base", base, base_bos),
            ("final", target, target_bos),
        ):
            cache = (
                root
                / "outcomes"
                / "cache"
                / architecture
                / checkpoint_name
                / (base_cache_namespace if checkpoint_name == "base" else target_id)
                / f"{row.text_hash}.npz"
            )
            config_hash = hashlib.sha256(
                (
                    f"{architecture}:{spec['revision']}:{checkpoint_name}:"
                    f"{target_hash if checkpoint_name == 'final' else 'base'}:geometry-v2"
                ).encode()
            ).hexdigest()
            if cache.exists():
                with np.load(cache, allow_pickle=False) as stored:
                    if str(stored["config_hash"].item()) != config_hash:
                        raise ValueError(
                            f"acquisition cache configuration changed: {cache}"
                        )
                    likelihood = stored["likelihood"]
                    alr = stored["alr"]
                    distinct = stored["distinct"]
            else:
                likelihood, _, alr, distinct, _ = _nested_arrays_with_explicit_geometry(
                    row.text,
                    tokenizer,
                    model,
                    bos_log_probs,
                    target_device,
                    config,
                    paths_per_batch,
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
        for name, value in base_summary.items():
            result[f"base_{name}"] = value
        for name, value in final_summary.items():
            result[f"final_{name}"] = value
            result[f"delta_{name}"] = value - base_summary[name]
        rows.append(result)
        if position % 10 == 0:
            print(
                f"{architecture}/{scope}/{role}/{target_id}: "
                f"{position}/{len(frame)}",
                flush=True,
            )
    outcomes = pd.DataFrame(rows)
    atomic_parquet(output, outcomes)
    atomic_json(
        output.with_suffix(".json"),
        {
            "status": "final-checkpoint-outcomes-acquired",
            "architecture": architecture,
            "scope": scope,
            "role": role,
            "target_id": target_id,
            "documents": len(outcomes),
            "outcomes_sha256": sha256(output),
            "target_checkpoint_sha256": target_hash,
            "intermediate_checkpoints_accessed": False,
        },
    )


def inference_result(
    frame: pd.DataFrame,
    outcome: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Run the positive gate and standardized 90% equivalence interval."""

    all_target_ids = [f"70m_{index}" for index in range(1, 7)]
    work = frame.copy()
    if set(work.target_id.astype(str)).issubset({"160m_1", "160m_2", "160m_3"}):
        mapping = {value: key for key, value in PAIRED_160_TARGETS.items()}
        work["target_id"] = work.target_id.map(mapping)
    slopes = target_slopes(work, outcome)
    randomization = randomization_test(
        work,
        outcome,
        draws=draws,
        seed=seed,
        all_target_ids=all_target_ids,
    )
    interval_95 = design_based_interval(
        work,
        outcome,
        draws=draws,
        seed=seed + 1,
        confidence=0.95,
        all_target_ids=all_target_ids,
    )
    standardized = standardize_outcome(work, outcome)
    interval_90_standardized = design_based_interval(
        standardized,
        outcome,
        draws=draws,
        seed=seed + 2,
        confidence=0.90,
        all_target_ids=all_target_ids,
    )
    return {
        "outcome": outcome,
        "target_slopes": {name: float(value) for name, value in slopes.items()},
        "mean_slope": float(randomization.estimate),
        "plus_one_p_one_sided": float(randomization.p_one_sided),
        "randomization_draws": draws,
        "interval_95": [float(value) for value in interval_95],
        "positive_gate": validation_gate(slopes, randomization.p_one_sided, interval_95),
        "interval_90_standardized": [float(value) for value in interval_90_standardized],
        "equivalent_pm_0_05": bool(
            interval_90_standardized[0] >= -0.05
            and interval_90_standardized[1] <= 0.05
        ),
    }


def _primary_outcomes(root: Path, architecture: str, role: str) -> pd.DataFrame:
    if architecture == "70m":
        targets = range(1, 4) if role == "validation" else range(4, 7)
    else:
        targets = range(1, 4)
    paths = [
        root / "outcomes" / architecture / "primary" / role / f"{architecture}_{index}.parquet"
        for index in targets
    ]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(f"primary {architecture}/{role} acquisition is incomplete")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def analyze_70m(root: Path, role: str, draws: int) -> None:
    """Analyze and immutably record one 70M primary phase decision."""

    output = root / "analysis" / "70m" / role / "decision.json"
    if output.exists():
        raise FileExistsError(f"phase decision already exists: {output}")
    frame = _primary_outcomes(root, "70m", role)
    calibration_data = json.loads(
        (root / "development" / "residual_calibration.json").read_text()
    )
    calibration = ResidualCalibration(
        gamma=float(calibration_data["gamma"]),
        sigma_dev=float(calibration_data["sigma_dev"]),
        observations=int(calibration_data["observations"]),
    )
    frame = apply_residual_calibration(frame, calibration)
    atomic_parquet(root / "analysis" / "70m" / role / "per_document.parquet", frame)
    outcomes = {
        "R": "delta_S_R",
        "A": "delta_S_A",
        "L": "delta_S_L",
        "N": "delta_S_N",
        "U": "U",
    }
    inference = {
        name: inference_result(
            frame,
            outcome,
            draws=draws,
            seed=20261600
            + (0 if role == "validation" else 100)
            + list(outcomes).index(name) * 10,
        )
        for name, outcome in outcomes.items()
    }
    power = json.loads((root / "design" / "equivalence_power.json").read_text())
    powered = {
        name: float(power[name]["equivalence_power"]) >= 0.80 for name in ("L", "N", "U")
    }
    n_decision = (
        "positive_response"
        if inference["N"]["positive_gate"]
        else "powered_equivalence"
        if powered["N"] and inference["N"]["equivalent_pm_0_05"]
        else "inconclusive"
    )
    l_decision = (
        "positive_response"
        if inference["L"]["positive_gate"]
        else "powered_equivalence"
        if powered["L"] and inference["L"]["equivalent_pm_0_05"]
        else "inconclusive"
    )
    u_phase_positive = bool(inference["U"]["positive_gate"])
    for statistic in ("R", "N", "D_p", "D_log", "D_z", "D_sqrt"):
        descriptive = standardized_separation_and_auc(frame, f"delta_S_{statistic}")
        atomic_parquet(
            root / "analysis" / "70m" / role / f"dose_descriptive_{statistic}.parquet",
            descriptive,
        )
    decision: dict[str, Any] = {
        "decision_complete": True,
        "architecture": "70m",
        "role": role,
        "documents": len(frame),
        "blocks": int(frame.block_id.nunique()),
        "inference": inference,
        "R_gate_passed": bool(inference["R"]["positive_gate"]),
        "A_gate_passed": bool(inference["A"]["positive_gate"]),
        "N_decision": n_decision,
        "L_decision": l_decision,
        "U_phase_positive_gate": u_phase_positive,
        "U_phase_equivalent": bool(
            powered["U"] and inference["U"]["equivalent_pm_0_05"]
        ),
        "equivalence_powered": powered,
    }
    if role == "confirmation":
        validation = json.loads(
            (root / "analysis" / "70m" / "validation" / "decision.json").read_text()
        )
        u_decision = three_way_decision(
            validation_positive_gate=validation["U_phase_positive_gate"],
            confirmation_positive_gate=u_phase_positive,
            validation_equivalence_interval=tuple(
                validation["inference"]["U"]["interval_90_standardized"]
            ),
            confirmation_equivalence_interval=tuple(
                inference["U"]["interval_90_standardized"]
            ),
            validation_equivalence_powered=validation["equivalence_powered"]["U"],
            confirmation_equivalence_powered=powered["U"],
        )
        decision["combined_U_decision"] = u_decision
        decision["combined_70m_claims"] = hierarchical_claim_decisions(
            r70_validation=validation["R_gate_passed"],
            r70_confirmation=decision["R_gate_passed"],
            a70_validation=validation["A_gate_passed"],
            a70_confirmation=decision["A_gate_passed"],
            l70_validation_equivalent=validation["L_decision"] == "powered_equivalence",
            l70_confirmation_equivalent=decision["L_decision"] == "powered_equivalence",
            n70_decision=(
                "powered_equivalence"
                if validation["N_decision"] == decision["N_decision"] == "powered_equivalence"
                else "positive_response"
                if validation["N_decision"] == decision["N_decision"] == "positive_response"
                else "inconclusive"
            ),
            u_decision=u_decision,
            r160_validation=None,
            r160_confirmation=None,
        )
    atomic_json(output, decision)
    atomic_json(ExtensionAccess(root).decision(f"70m_{role}"), decision)


def analyze_160m(root: Path, role: str, draws: int) -> None:
    """Analyze one paired 160M phase without allowing downstream rescue."""

    output = root / "analysis" / "160m" / role / "decision.json"
    if output.exists():
        raise FileExistsError(f"phase decision already exists: {output}")
    frame = _primary_outcomes(root, "160m", role)
    inference = {
        name: inference_result(
            frame,
            outcome,
            draws=draws,
            seed=20261700
            + (0 if role == "validation" else 100)
            + offset * 10,
        )
        for offset, (name, outcome) in enumerate(
            {"R": "delta_S_R", "A": "delta_S_A", "L": "delta_S_L"}.items()
        )
    }
    decision: dict[str, Any] = {
        "decision_complete": True,
        "architecture": "160m",
        "role": role,
        "documents": len(frame),
        "blocks": int(frame.block_id.nunique()),
        "inference": inference,
        "R_gate_passed": bool(inference["R"]["positive_gate"]),
        "A_gate_passed": bool(inference["A"]["positive_gate"]),
        "L_reported_prospectively": True,
    }
    if role == "validation":
        observed_sd = float(frame.delta_S_L.std(ddof=1))
        power = six_target_equivalence_power_simulation(
            blocks=int(frame.block_id.nunique()),
            simulations=1000,
            randomization_draws=1999,
            seed=20261790,
            noise_sd=observed_sd,
        )
        power_decision = {
            "decision_complete": True,
            "architecture": "160m",
            "outcome": "delta_S_L",
            "validation_observed_sd": observed_sd,
            "minimum_confirmation_power": 0.80,
            "power": power,
            "confirmation_equivalence_powered": power["equivalence_power"] >= 0.80,
        }
        decision["L_confirmation_power"] = power_decision
        atomic_json(
            root / "analysis" / "160m" / "validation" / "L_power_decision.json",
            power_decision,
        )
        atomic_json(ExtensionAccess(root).decision("160m_validation_power"), power_decision)
    else:
        validation = json.loads(
            (root / "analysis" / "160m" / "validation" / "decision.json").read_text()
        )
        power = validation["L_confirmation_power"]
        decision["cross_scale_R_replication"] = bool(
            validation["R_gate_passed"] and decision["R_gate_passed"]
        )
        decision["secondary_A_replication"] = bool(
            validation["A_gate_passed"] and decision["A_gate_passed"]
        )
        decision["no_meaningful_L_increase"] = bool(
            power["confirmation_equivalence_powered"]
            and inference["L"]["equivalent_pm_0_05"]
        )
        upstream = json.loads(
            (root / "analysis" / "70m" / "confirmation" / "decision.json").read_text()
        )
        decision["upstream_claims_fixed_before_160m"] = upstream[
            "combined_70m_claims"
        ]
        decision["160m_cannot_rescue_upstream_claims"] = True
    atomic_json(output, decision)


def _complete_latin_outcomes(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    primary = {
        ("validation", f"70m_{index}") for index in range(1, 4)
    } | {("confirmation", f"70m_{index}") for index in range(4, 7)}
    for role in ("validation", "confirmation"):
        for index in range(1, 7):
            target_id = f"70m_{index}"
            if (role, target_id) in primary:
                frames.append(
                    pd.read_parquet(
                        root
                        / "outcomes"
                        / "70m"
                        / "primary"
                        / role
                        / f"{target_id}.parquet"
                    )
                )
            frames.append(
                pd.read_parquet(
                    root
                    / "outcomes"
                    / "70m"
                    / "complete_latin"
                    / role
                    / f"{target_id}.parquet"
                )
            )
    frame = pd.concat(frames, ignore_index=True)
    expected = 3600 * 6
    if len(frame) != expected or frame.duplicated(["doc_id", "target_id"]).any():
        raise RuntimeError("complete-Latin panel is incomplete or duplicated")
    rotations = frame.groupby("doc_id").K.apply(
        lambda values: sorted(values.astype(int)) == [0, 1, 2, 4, 8, 16]
    )
    if not rotations.all():
        raise RuntimeError("complete-Latin dose rotations changed")
    return frame


def analyze_complete_latin(root: Path) -> None:
    """Run the post-confirmation document and target fixed-effect sensitivity."""

    if not ExtensionAccess(root)._complete("70m_confirmation"):
        raise PermissionError("complete-Latin analysis is sealed until confirmation")
    output = root / "analysis" / "70m" / "complete_latin" / "decision.json"
    if output.exists():
        raise FileExistsError(f"sensitivity analysis already exists: {output}")
    frame = _complete_latin_outcomes(root)
    calibration_data = json.loads(
        (root / "development" / "residual_calibration.json").read_text()
    )
    frame = apply_residual_calibration(
        frame,
        ResidualCalibration(
            gamma=float(calibration_data["gamma"]),
            sigma_dev=float(calibration_data["sigma_dev"]),
            observations=int(calibration_data["observations"]),
        ),
    )
    outcomes = {
        name: ("U" if name == "U" else f"delta_S_{name}")
        for name in ("R", "A", "L", "N", "E_y", "D_p", "D_log", "D_z", "D_sqrt", "U")
    }
    estimates = {
        name: document_target_fixed_effect_sensitivity(frame, column)
        for name, column in outcomes.items()
    }
    atomic_parquet(root / "analysis" / "70m" / "complete_latin" / "panel.parquet", frame)
    atomic_json(
        output,
        {
            "decision_complete": True,
            "label": "complete-Latin post-confirmation sensitivity analysis",
            "primary_or_confirmation_claim_authority": False,
            "documents": int(frame.doc_id.nunique()),
            "document_target_cells": len(frame),
            "model": "Y_ir = alpha_i + lambda_r + beta*d_ir + epsilon_ir",
            "estimates": estimates,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--target-id", required=True)
    acquisition = commands.add_parser("acquire")
    acquisition.add_argument("--architecture", choices=("70m", "160m"), required=True)
    acquisition.add_argument("--role", choices=("validation", "confirmation"), required=True)
    acquisition.add_argument("--target-id", required=True)
    acquisition.add_argument("--scope", choices=("primary", "complete_latin"), required=True)
    acquisition.add_argument("--paths-per-batch", type=int, default=8)
    analysis_70m = commands.add_parser("analyze-70m")
    analysis_70m.add_argument("--role", choices=("validation", "confirmation"), required=True)
    analysis_70m.add_argument("--draws", type=int, default=99999)
    analysis_160m = commands.add_parser("analyze-160m")
    analysis_160m.add_argument("--role", choices=("validation", "confirmation"), required=True)
    analysis_160m.add_argument("--draws", type=int, default=99999)
    commands.add_parser("analyze-complete-latin")
    args = parser.parse_args()
    if args.command == "train":
        train(args.root, args.target_id)
    elif args.command == "acquire":
        acquire(
            args.root,
            args.architecture,
            args.role,
            args.target_id,
            args.scope,
            args.paths_per_batch,
        )
    elif args.command == "analyze-70m":
        analyze_70m(args.root, args.role, args.draws)
    elif args.command == "analyze-160m":
        analyze_160m(args.root, args.role, args.draws)
    elif args.command == "analyze-complete-latin":
        analyze_complete_latin(args.root)


if __name__ == "__main__":
    main()
