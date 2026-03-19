"""Fine-tune SELFormerMM for downstream tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from SELFormerMM.finetuner import Finetuner
from SELFormerMM.models.downstream_heads import ClassificationHead, RegressionHead
from SELFormerMM.models.multimodal_roberta import MultimodalRoberta
from SELFormerMM.utils.datasets import FinetuneDataset, MultimodalCollator


def _load_config(path: str) -> dict[str, Any]:
    """Load YAML or JSON config. Supports nested {value: x} format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
            data = yaml.safe_load(raw)
        except ImportError:
            data = json.loads(raw)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Config must be a dict.")
    out = {}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and "value" in v:
            v = v["value"]
        out[k] = v
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune multimodal model.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML/JSON config. Keys = argument names (save_root is mapped to save_dir). Applied before CLI.",
    )
    parser.add_argument("--dataset_meta_csv", required=True)
    parser.add_argument("--dataset_embs_npz", required=True)
    parser.add_argument("--use_scaffold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--save_split_csvs", default=None, help="Optional directory to save split CSVs.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument(
        "--pretrained_multimodal_dir",
        required=True,
        help="Directory from multimodal pretraining checkpoint.\n"
        "Supported layouts:\n"
        "- <dir>/pytorch_model.bin + (config/tokenizer in <dir>)  (from our train_pretraining.py)\n"
        "- <dir>/model.safetensors + <dir>/hf/config.json + <dir>/hf/tokenizer files \n"
        "Initializes MultimodalRoberta from these weights (RoBERTa + projection heads).",
    )
    parser.add_argument("--task_type", choices=["binary", "multilabel", "regression"], required=True)
    parser.add_argument("--label_column", default=None, help="Target column for binary/regression. Optional if inferable.")
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Evaluate test set and save checkpoint if metric improves every N epochs (0 disables periodic checkpoints).",
    )
    parser.add_argument(
        "--test_eval_every",
        type=int,
        default=0,
        help="Evaluate on test split every N epochs (0 disables periodic test evaluation).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_dir", required=True)

    argv = sys.argv[1:]
    config_path = None
    if "--config" in argv:
        i = argv.index("--config")
        config_path = argv[i + 1] if i + 1 < len(argv) else None
        argv = [a for j, a in enumerate(argv) if j != i and j != i + 1]
    if config_path:
        cfg = _load_config(config_path)
        injected = []
        for k, v in cfg.items():
            if k == "config" or k.startswith("_"):
                continue
            key = "save_dir" if k == "save_root" else k
            if v is None:
                continue
            # Use --key=value so values starting with '-' aren't parsed as flags
            injected.append(f"--{key}={v}")
        argv = injected + argv
    return parser.parse_args(argv)


def _count_empty_rows(arr: np.ndarray) -> int:
    """Count rows that are all zeros."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={arr.shape}")
    return int((~np.any(arr != 0, axis=1)).sum())


def _load_embeddings_npz(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "graph": data["graph"].astype(np.float32),
        "text": data["text"].astype(np.float32),
        "kg": data["kg"].astype(np.float32),
    }


def _count_non_empty(arr: np.ndarray) -> int:
    if arr is None:
        return 0
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={arr.shape}")
    return int((np.linalg.norm(arr, axis=1) != 0).sum())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return 1.0 / (1.0 + np.exp(-x))


def _micro_f1_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Micro F1 for (possibly flattened) binary labels with NaNs masked out."""
    mask = ~np.isnan(y_true)
    if not mask.any():
        return float("nan")
    y = y_true[mask].astype(np.int64)
    p = (y_prob[mask] >= threshold).astype(np.int64)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return float("nan")
    return float(2 * tp / denom)


def _micro_accuracy_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    mask = ~np.isnan(y_true)
    if not mask.any():
        return float("nan")
    y = y_true[mask].astype(np.int64)
    p = (y_prob[mask] >= threshold).astype(np.int64)
    return float((p == y).mean())


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties (1..n)."""
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return x.astype(np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # 1-based inclusive bounds
        ranks_sorted[i:j] = avg_rank
        i = j
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _auroc(y_true_01: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via Mann–Whitney U (tie-aware). Expects 1D arrays, labels in {0,1}."""
    y = np.asarray(y_true_01).astype(np.int64)
    s = np.asarray(y_score).astype(np.float64)
    if y.size == 0:
        return float("nan")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(s)
    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _auprc(y_true_01: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (AUPRC) for binary labels. Expects 1D arrays, labels in {0,1}."""
    y = np.asarray(y_true_01).astype(np.int64)
    s = np.asarray(y_score).astype(np.float64)
    if y.size == 0:
        return float("nan")
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")  # descending
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    denom = tp + fp
    precision = tp / np.where(denom == 0, 1, denom)
    ap = float(precision[y_sorted == 1].sum() / n_pos)
    return ap


def _prc_auc(y_true_01: np.ndarray, y_score: np.ndarray) -> float:
    """Area under PR curve (sklearn-like)."""
    y = np.asarray(y_true_01).astype(np.int64)
    s = np.asarray(y_score).astype(np.float64)
    if y.size == 0:
        return float("nan")
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    try:
        from sklearn.metrics import precision_recall_curve, auc  # type: ignore

        p, r, _ = precision_recall_curve(y, s)
        return float(auc(r, p))
    except Exception:
        # Fallback: stepwise PR curve via sorted scores
        order = np.argsort(-s, kind="mergesort")
        y_sorted = y[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        precision = tp / np.where(tp + fp == 0, 1, tp + fp)
        recall = tp / n_pos
        # ensure monotonic recall; integrate with trapezoid
        return float(np.trapz(precision, recall))


def _binary_prf(y_true_01: np.ndarray, y_pred_01: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true_01).astype(np.int64)
    p = np.asarray(y_pred_01).astype(np.int64)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def _evaluate(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    dataloader: DataLoader,
    task_type: str,
    device: str,
) -> dict[str, float]:
    backbone.eval()
    head.eval()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graph_emb = batch.get("graph_emb")
            text_emb = batch.get("text_emb")
            kg_emb = batch.get("kg_emb")
            if graph_emb is not None:
                graph_emb = graph_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)
            if kg_emb is not None:
                kg_emb = kg_emb.to(device)

            embeddings = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_emb=graph_emb,
                text_emb=text_emb,
                kg_emb=kg_emb,
            )
            bs = input_ids.size(0)
            combined = torch.cat(
                [embeddings[i * bs : (i + 1) * bs] for i in range(4)], dim=1
            )
            logits = head(combined)
            labels = batch["labels"].to(device)

            # compute loss without constructing an optimizer (PyTorch disallows AdamW([]))
            if task_type == "binary":
                if logits.ndim == 2 and logits.size(1) == 2:
                    loss = torch.nn.functional.cross_entropy(logits, labels.long())
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.view(-1), labels.float().view(-1)
                    )
            elif task_type == "multilabel":
                labels_f = labels.float()
                mask = ~torch.isnan(labels_f)
                labels_f = torch.nan_to_num(labels_f, nan=0.0)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels_f, reduction="none"
                )
                loss = losses[mask].mean() if mask.any() else losses.mean() * 0.0
            else:
                labels_f = labels.float()
                mask = ~torch.isnan(labels_f)
                labels_f = torch.nan_to_num(labels_f, nan=0.0)
                losses = torch.nn.functional.mse_loss(logits, labels_f, reduction="none")
                loss = losses[mask].mean() if mask.any() else losses.mean() * 0.0
            total_loss += float(loss.item())
            n_batches += 1

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,))
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,))
    out: dict[str, float] = {"loss": total_loss / max(n_batches, 1)}

    if task_type == "regression":
        y_true = labels_np.astype(np.float64)
        y_pred = logits_np.astype(np.float64)
        mask = ~np.isnan(y_true)
        if mask.any():
            err = (y_pred - y_true)[mask]
            out["mse"] = float(np.mean(err**2))
            out["rmse"] = float(np.sqrt(np.mean(err**2)))
            out["mae"] = float(np.mean(np.abs(err)))
        else:
            out["mse"] = float("nan")
            out["rmse"] = float("nan")
            out["mae"] = float("nan")
        return out

    if task_type == "binary":
        y_true = labels_np.reshape(-1)
        mask = ~np.isnan(y_true)
        y01 = y_true[mask].astype(np.int64)
        if logits_np.ndim == 2 and logits_np.shape[1] == 2:
            y_pred = np.argmax(logits_np, axis=1).astype(np.int64)[mask]
        else:
            probs = _sigmoid(logits_np.reshape(-1))
            y_pred = (probs >= 0.5).astype(np.int64)[mask]
        out.update(_binary_prf(y01, y_pred))
        out["roc-auc"] = _auroc(y01, y_pred.astype(np.float64))
        out["prc-auc"] = _prc_auc(y01, y_pred.astype(np.float64))
        return out

    probs = _sigmoid(logits_np)
    y_true = labels_np
    y_pred = (probs >= 0.5).astype(np.int64)
    mask = ~np.isnan(y_true)
    y01 = y_true[mask].astype(np.int64)
    p01 = y_pred[mask].astype(np.int64)
    out.update(_binary_prf(y01, p01))
    out["roc-auc"] = _auroc(y01, p01.astype(np.float64))
    out["prc-auc"] = _prc_auc(y01, p01.astype(np.float64))
    return out

def _infer_label_column(df: pd.DataFrame, task_type: str) -> str:
    excluded = {"smiles", "selfies", "Description", "description"}
    candidates = [c for c in df.columns if c not in excluded]
    if task_type == "binary":
        for c in ["Class", "p_np", "label", "target"]:
            if c in df.columns:
                return c
        raise ValueError(f"Could not infer label column for binary. Candidates: {candidates}")
    if task_type == "regression":
        for c in df.columns:
            if "measured" in c.lower():
                return c
        # fallback: last non-excluded column
        if candidates:
            return candidates[-1]
        raise ValueError("Could not infer regression label column.")
    raise ValueError("label_column inference is not used for multilabel.")

def _load_labels(
    df: pd.DataFrame, task_type: str, label_column: str | None
) -> np.ndarray:
    if task_type == "multilabel":
        excluded = {
            "selfies",
            "smiles",
            "Description",
            "description",
            # common ID / split columns that should not be treated as labels
            "chembl_id",
            "compound_id",
            "Compound ID",
            "id",
            "ID",
            "CID",
            "cid",
            "split",
            "fold",
            "kg_compound_node_idx",
            "kg_node_index",
        }
        candidates = [c for c in df.columns if c not in excluded]
        # Keep only numeric columns (tox21 often has NaNs; pandas still treats as float)
        label_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        if not label_cols:
            raise ValueError(
                "Could not infer multilabel target columns (no numeric columns found after exclusions)."
            )
        return df[label_cols].fillna(0).values
    col = label_column or _infer_label_column(df, task_type)
    # For regression keep a 2D array (N, 1) so that the dataloader yields labels with shape (B, 1),
    # matching logits shape (B, 1) and avoiding unintended broadcasting in MSELoss.
    if task_type == "regression":
        return df[[col]].fillna(0).values.astype(np.float32)
    # For binary keep a 1D array (N,) for CrossEntropyLoss.
    return df[[col]].fillna(0).values.squeeze()


def main() -> None:
    os.environ.setdefault("TOKENIZER_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")

    # Silence noisy RDKit warnings (often triggered during scaffold split/chemprop internals).
    try:
        from rdkit import RDLogger  # type: ignore

        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.info")
    except Exception:
        pass

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print("[finetune] starting")
    print(
        f"[finetune] device={device} task_type={args.task_type} batch_size={args.batch_size} "
        f"max_len={args.max_len} backbone_lr={args.backbone_lr} head_lr={args.head_lr} "
        f"weight_decay={args.weight_decay} epochs={args.epochs}"
    )
    print(f"[finetune] dataset_meta_csv={args.dataset_meta_csv}")
    print(f"[finetune] dataset_embs_npz={args.dataset_embs_npz}")
    print(f"[finetune] model_path={args.model_path} tokenizer_path={args.tokenizer_path or args.model_path}")
    if args.pretrained_multimodal_dir:
        print(f"[finetune] pretrained_multimodal_dir={args.pretrained_multimodal_dir}")
    print(
        f"[finetune] use_scaffold={bool(args.use_scaffold)} seed={args.seed} fracs=({args.train_frac},{args.val_frac},{args.test_frac})"
    )

    print("[finetune] loading dataset bundle (meta csv + embs npz) ...")
    full_df = pd.read_csv(args.dataset_meta_csv)
    if "selfies" not in full_df.columns or "smiles" not in full_df.columns:
        raise ValueError("dataset_meta_csv must include 'smiles' and 'selfies' columns.")

    embs = _load_embeddings_npz(args.dataset_embs_npz)
    graph_full, text_full, kg_full = embs["graph"], embs["text"], embs["kg"]
    if len(full_df) != graph_full.shape[0] or len(full_df) != text_full.shape[0] or len(full_df) != kg_full.shape[0]:
        raise ValueError("Row count mismatch between meta CSV and NPZ embeddings.")

    # split indices
    from SELFormerMM.utils.datasets import random_split, scaffold_split

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0.")

    print(f"[finetune] splitting dataset...")
    # targets for balanced scaffold split
    targets = _load_labels(full_df, args.task_type, args.label_column).tolist()
    if args.use_scaffold:
        train_idx, val_idx, test_idx = scaffold_split(
            smiles=full_df["smiles"].astype(str).tolist(),
            targets=targets,
            frac_train=args.train_frac,
            frac_valid=args.val_frac,
            frac_test=args.test_frac,
            seed=args.seed,
        )
    else:
        train_idx, val_idx, test_idx = random_split(
            len(full_df),
            frac_train=args.train_frac,
            frac_valid=args.val_frac,
            frac_test=args.test_frac,
            seed=args.seed,
        )

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True) if len(test_idx) else full_df.iloc[[]].reset_index(drop=True)
    train_graph, train_text, train_kg = graph_full[train_idx], text_full[train_idx], kg_full[train_idx]
    val_graph, val_text, val_kg = graph_full[val_idx], text_full[val_idx], kg_full[val_idx]
    test_graph, test_text, test_kg = graph_full[test_idx], text_full[test_idx], kg_full[test_idx]

    print(f"[finetune] split sizes: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    print(f"[finetune] train modality_non_empty: graph={_count_non_empty(train_graph):,}/{len(train_df):,} text={_count_non_empty(train_text):,}/{len(train_df):,} kg={_count_non_empty(train_kg):,}/{len(train_df):,}")
    print(f"[finetune] val   modality_non_empty: graph={_count_non_empty(val_graph):,}/{len(val_df):,} text={_count_non_empty(val_text):,}/{len(val_df):,} kg={_count_non_empty(val_kg):,}/{len(val_df):,}")
    if len(test_df):
        print(f"[finetune] test  modality_non_empty: graph={_count_non_empty(test_graph):,}/{len(test_df):,} text={_count_non_empty(test_text):,}/{len(test_df):,} kg={_count_non_empty(test_kg):,}/{len(test_df):,}")

    if args.save_split_csvs:
        out = Path(args.save_split_csvs)
        out.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out / "train.csv", index=False)
        val_df.to_csv(out / "val.csv", index=False)
        if len(test_df):
            test_df.to_csv(out / "test.csv", index=False)
        print(f"[finetune] saved split csvs to: {out}")

    # Labels (after final train_df/val_df set)
    print("[finetune] extracting labels...")
    train_labels = _load_labels(train_df, args.task_type, args.label_column)
    val_labels = _load_labels(val_df, args.task_type, args.label_column)
    test_labels = _load_labels(test_df, args.task_type, args.label_column) if len(test_df) else None
    if args.task_type == "multilabel":
        print(f"[finetune] multilabel: num_labels_in_csv={train_labels.shape[1]}")

    print("[finetune] loading config/tokenizer/backbone...")
    pretrained_dir = Path(args.pretrained_multimodal_dir)
    if not pretrained_dir.exists():
        raise FileNotFoundError(f"pretrained_multimodal_dir not found: {pretrained_dir}")

    # Some checkpoints keep HF config/tokenizer under a nested `hf/` directory (e.g. data/epoch_267/hf/*)
    config_dir = pretrained_dir / "hf" if (pretrained_dir / "hf").is_dir() else pretrained_dir
    config = AutoConfig.from_pretrained(str(config_dir))
    tokenizer_path = args.tokenizer_path or str(config_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    print("[finetune] initializing multimodal model...")
    backbone = MultimodalRoberta(config)

    weights_bin = pretrained_dir / "pytorch_model.bin"
    weights_safe = pretrained_dir / "model.safetensors"
    if weights_bin.exists():
        print(f"[finetune] loading multimodal pretrained weights: {weights_bin}")
        state = torch.load(weights_bin, map_location="cpu")
    elif weights_safe.exists():
        print(f"[finetune] loading multimodal pretrained weights: {weights_safe}")
        try:
            from safetensors.torch import load_file  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Found model.safetensors but safetensors is not installed. "
                "Install it (e.g. `pip install safetensors`) or provide pytorch_model.bin instead."
            ) from exc
        state = load_file(str(weights_safe), device="cpu")
    else:
        raise FileNotFoundError(
            f"No weights found in {pretrained_dir}. Expected either pytorch_model.bin or model.safetensors."
        )

    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing:
        print(f"[finetune] warning: missing_keys={len(missing)}")
    if unexpected:
        print(f"[finetune] warning: unexpected_keys={len(unexpected)}")

    # Freeze all layers except last 3 encoder blocks (9, 10, 11)
    print("[finetune] freezing roberta layers (trainable: encoder.layer.9-11)...")
    for name, param in backbone.named_parameters():
        if "encoder.layer." in name:
            if any(f"encoder.layer.{i}" in name for i in [9, 10, 11]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif name.startswith("roberta."):
            param.requires_grad = False
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"[finetune] backbone trainable params: {trainable:,}/{total:,}")

    print("[finetune] building datasets/dataloaders...")
    label_dtype = torch.float32
    if args.task_type == "binary":
        label_dtype = torch.long
    train_ds = FinetuneDataset(
        selfies=train_df["selfies"].tolist(),
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=args.max_len,
        graph_emb=train_graph,
        text_emb=train_text,
        kg_emb=train_kg,
        label_dtype=label_dtype,
    )
    val_ds = FinetuneDataset(
        selfies=val_df["selfies"].tolist(),
        labels=val_labels,
        tokenizer=tokenizer,
        max_len=args.max_len,
        graph_emb=val_graph,
        text_emb=val_text,
        kg_emb=val_kg,
        label_dtype=label_dtype,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=MultimodalCollator()
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=MultimodalCollator()
    )
    test_loader = None
    if len(test_df):
        test_ds = FinetuneDataset(
            selfies=test_df["selfies"].tolist(),
            labels=test_labels,
            tokenizer=tokenizer,
            max_len=args.max_len,
            graph_emb=test_graph,
            text_emb=test_text,
            kg_emb=test_kg,
            label_dtype=label_dtype,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=MultimodalCollator()
        )
    print(
        f"[finetune] train_steps_per_epoch={len(train_loader):,} val_steps={len(val_loader):,} "
        f"test_steps={(len(test_loader) if test_loader is not None else 0):,}"
    )

    hidden_size = getattr(config, "hidden_size", 768)
    if args.task_type in {"binary", "multilabel"}:
        num_labels = args.num_labels
        if args.task_type == "binary":
            num_labels = 2
            if args.num_labels != 2:
                print(
                    f"[finetune] overriding --num_labels={args.num_labels} with num_labels=2 for binary"
                )
        if args.task_type == "multilabel":
            num_labels = int(train_labels.shape[1])
            if args.num_labels != num_labels:
                print(
                    f"[finetune] overriding --num_labels={args.num_labels} with num_labels_in_csv={num_labels}"
                )
        head = ClassificationHead(hidden_size * 4, num_labels)
    else:
        head = RegressionHead(hidden_size * 4, num_targets=args.num_labels)
    out_dim = num_labels if args.task_type in {"binary", "multilabel"} else args.num_labels
    print(f"[finetune] initialized head: in_dim={hidden_size*4} out={out_dim}")

    print("[finetune] initializing optimizer/scheduler/finetuner...")
    # HuggingFace Trainer parameter grouping (no weight decay for bias/LayerNorm weights)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    backbone_named = [(f"backbone.{n}", p) for n, p in backbone.named_parameters()]
    head_named = [(f"head.{n}", p) for n, p in head.named_parameters()]

    backbone_decay = [p for n, p in backbone_named if p.requires_grad and not any(nd in n for nd in no_decay)]
    backbone_nodecay = [p for n, p in backbone_named if p.requires_grad and any(nd in n for nd in no_decay)]
    head_decay = [p for n, p in head_named if p.requires_grad and not any(nd in n for nd in no_decay)]
    head_nodecay = [p for n, p in head_named if p.requires_grad and any(nd in n for nd in no_decay)]

    # sanity logs
    print(
        f"[finetune] optimizer groups: backbone_decay={len(backbone_decay)} backbone_nodecay={len(backbone_nodecay)} "
        f"head_decay={len(head_decay)} head_nodecay={len(head_nodecay)}"
    )
    param_groups = []
    if backbone_decay:
        param_groups.append({"params": backbone_decay, "weight_decay": args.weight_decay, "lr": args.backbone_lr})
    if backbone_nodecay:
        param_groups.append({"params": backbone_nodecay, "weight_decay": 0.0, "lr": args.backbone_lr})
    if head_decay:
        param_groups.append({"params": head_decay, "weight_decay": args.weight_decay, "lr": args.head_lr})
    if head_nodecay:
        param_groups.append({"params": head_nodecay, "weight_decay": 0.0, "lr": args.head_lr})
    if not param_groups:
        raise ValueError("No trainable parameters found (all parameter groups are empty).")
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = int(args.epochs * len(train_loader))
    try:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            # Match HuggingFace Trainer default: warmup_steps = 0
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )
    except Exception:
        scheduler = None
    finetuner = Finetuner(
        backbone=backbone,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        task_type=args.task_type,
        device=device,
    )

    print("[finetune] training loop...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Track best test metric for checkpoint saving
    best_test_metric: Optional[float] = None
    best_epoch: int = 0
    
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        train_loss = finetuner.train_epoch(train_loader)
        val_loss = finetuner.validate(val_loader)
        print(
            f"[finetune] epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={time.time()-epoch_t0:.1f}s tod={datetime.now().strftime('%H:%M')}"
        )
        # Periodic test evaluation and checkpoint saving
        should_eval_test = (
            test_loader is not None
            and (
                (args.test_eval_every > 0 and (epoch + 1) % args.test_eval_every == 0)
                or (args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0)
            )
        )
        
        if should_eval_test:
            metrics = _evaluate(backbone, head, test_loader, args.task_type, device)
            metrics_str = " ".join(
                [
                    f"{k}={v:.4f}"
                    for k, v in metrics.items()
                    if isinstance(v, float) and not np.isnan(v)
                ]
            )
            print(f"[finetune] test_metrics: epoch={epoch + 1} {metrics_str}")
            
            # Checkpoint saving: save if metric improves
            if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
                # Determine metric to track (higher is better for ROC-AUC, lower is better for RMSE)
                if args.task_type in {"binary", "multilabel"}:
                    current_metric = metrics.get("roc-auc")
                    is_better = current_metric is not None and (best_test_metric is None or current_metric > best_test_metric)
                else:  # regression
                    current_metric = metrics.get("rmse")
                    is_better = current_metric is not None and (best_test_metric is None or current_metric < best_test_metric)
                
                if is_better and current_metric is not None:
                    best_test_metric = current_metric
                    best_epoch = epoch + 1
                    ckpt_dir = save_dir / f"checkpoint_epoch_{epoch + 1:03d}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"backbone": backbone.state_dict(), "head": head.state_dict()},
                        ckpt_dir / "model.pt",
                    )
                    config.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    metric_name = "roc-auc" if args.task_type in {"binary", "multilabel"} else "rmse"
                    print(f"[finetune] saved checkpoint (best {metric_name}={best_test_metric:.4f}): {ckpt_dir}")
                else:
                    metric_name = "roc-auc" if args.task_type in {"binary", "multilabel"} else "rmse"
                    if best_test_metric is not None:
                        print(f"[finetune] checkpoint skipped (current {metric_name}={current_metric:.4f} vs best={best_test_metric:.4f} at epoch {best_epoch})")

    if test_loader is not None:
        print("[finetune] final evaluation on test split...")
        metrics = _evaluate(backbone, head, test_loader, args.task_type, device)
        metrics_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float) and not np.isnan(v)])
        print(f"[finetune] test_metrics: {metrics_str}")

    print("[finetune] saving model...")
    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict()}, save_dir / "model.pt")
    config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[finetune] saved finetuned model to: {save_dir}")
    print(f"[finetune] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
