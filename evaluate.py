"""Evaluate a fine-tuned SELFormerMM checkpoint and print its test metrics.

Reproduces the exact train/val/test split the checkpoint was trained with 
and scores the model on it, so a released checkpoint can be verified 
with a single command:

    python3 evaluate.py --config=data/models/finetuned/bbbp/finetuning_optimized_config.json

"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from SELFormerMM.models.downstream_heads import ClassificationHead, RegressionHead
from SELFormerMM.models.multimodal_roberta import MultimodalRoberta
from SELFormerMM.utils.datasets import (
    FinetuneDataset,
    MultimodalCollator,
    random_split,
    scaffold_split,
)
from train_finetuning import (
    _evaluate,
    _load_config,
    _load_embeddings_npz,
    _load_labels,
)


# Keys that belong to training only; ignored when a fine-tuning config is reused here.
_TRAIN_ONLY = {
    "model_path", "tokenizer_path", "pretrained_multimodal_dir", "epochs",
    "backbone_lr", "head_lr", "weight_decay", "checkpoint_every",
    "test_eval_every", "save_best_val", "save_split_csvs", "save_root", "save_dir",
    "freeze_backbone",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned SELFormerMM checkpoint.")
    parser.add_argument(
        "--config",
        default=None,
        help="Fine-tuning config (JSON/YAML) to take dataset and split settings from. "
        "Training-only keys are ignored. CLI flags override it.",
    )
    parser.add_argument("--model_dir", default=None, help="Directory holding the checkpoint (default: config's directory).")
    parser.add_argument("--model_file", default="model.pt", help="Weights file inside model_dir (model.pt or best_model.pt).")
    parser.add_argument("--tokenizer_path", default=None, help="Defaults to model_dir.")
    parser.add_argument("--dataset_meta_csv", default=None)
    parser.add_argument("--dataset_embs_npz", default=None)
    parser.add_argument("--task_type", choices=["binary", "multilabel", "regression"], default=None)
    parser.add_argument("--label_column", default=None)
    parser.add_argument("--num_labels", type=int, default=None)
    parser.add_argument("--use_scaffold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_frac", type=float, default=None)
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--test_frac", type=float, default=None)
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_json", default=None, help="Optional path to write the metrics to.")

    args = parser.parse_args()
    # Whether --seed came from the command line, as opposed to a config or the default.
    # Only an explicit flag may contradict a seed_<n> directory name.
    args.seed_explicit = any(a == "--seed" or a.startswith("--seed=") for a in sys.argv[1:])

    if args.config:
        cfg: dict[str, Any] = {k: v for k, v in _load_config(args.config).items() if k not in _TRAIN_ONLY}
        for key, value in cfg.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
        if args.model_dir is None:
            args.model_dir = str(Path(args.config).parent)

    # Defaults matching train_finetuning.py
    for key, default in (
        ("use_scaffold", 0), ("seed", 42), ("train_frac", 0.8),
        ("val_frac", 0.1), ("test_frac", 0.1),
    ):
        if getattr(args, key) is None:
            setattr(args, key, default)

    missing = [k for k in ("model_dir", "dataset_meta_csv", "dataset_embs_npz", "task_type")
               if getattr(args, k) is None]
    if missing:
        parser.error(f"Missing required setting(s): {', '.join(missing)} (pass --config or the flags).")
    return args


SEED_DIR_RE = re.compile(r"^seed_?(\d+)$")


def seed_from_dir(model_dir: Path) -> int | None:
    """Return the seed encoded in a `seed_42` / `seed42` directory name, if any."""
    m = SEED_DIR_RE.match(model_dir.name)
    return int(m.group(1)) if m else None


def evaluate_one(args: argparse.Namespace, model_dir: Path, seed: int, device: str) -> dict:
    weights = model_dir / args.model_file
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    full_df = pd.read_csv(args.dataset_meta_csv)
    if "selfies" not in full_df.columns or "smiles" not in full_df.columns:
        raise ValueError("dataset_meta_csv must include 'smiles' and 'selfies' columns.")
    embs = _load_embeddings_npz(args.dataset_embs_npz)
    graph, text, kg = embs["graph"], embs["text"], embs["kg"]
    if not (len(full_df) == graph.shape[0] == text.shape[0] == kg.shape[0]):
        raise ValueError("Row count mismatch between meta CSV and NPZ embeddings.")

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0.")

    print(
        f"[evaluate] model={weights} split={args.split} "
        f"use_scaffold={bool(args.use_scaffold)} seed={seed} "
        f"fracs=({args.train_frac},{args.val_frac},{args.test_frac})"
    )

    targets = _load_labels(full_df, args.task_type, args.label_column).tolist()
    if args.use_scaffold:
        train_idx, val_idx, test_idx = scaffold_split(
            smiles=full_df["smiles"].astype(str).tolist(), targets=targets,
            frac_train=args.train_frac, frac_valid=args.val_frac,
            frac_test=args.test_frac, seed=seed,
        )
    else:
        train_idx, val_idx, test_idx = random_split(
            len(full_df), frac_train=args.train_frac, frac_valid=args.val_frac,
            frac_test=args.test_frac, seed=seed,
        )
    idx = {"train": train_idx, "val": val_idx, "test": test_idx}[args.split]
    if not len(idx):
        raise ValueError(f"The '{args.split}' split is empty.")

    sub_df = full_df.iloc[idx].reset_index(drop=True)
    labels = _load_labels(sub_df, args.task_type, args.label_column)
    if args.task_type == "binary":
        num_labels = args.num_labels if args.num_labels else 2
    elif args.task_type == "multilabel":
        num_labels = labels.shape[1]
    else:
        num_labels = args.num_labels if args.num_labels else 1
    print(f"[evaluate] n={len(sub_df):,} num_labels={num_labels}")

    config = AutoConfig.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or str(model_dir), use_fast=True)
    checkpoint = torch.load(weights, map_location="cpu")

    backbone = MultimodalRoberta(config)
    backbone.load_state_dict(checkpoint["backbone"], strict=False)
    hidden_size = getattr(config, "hidden_size", 768)
    if args.task_type in {"binary", "multilabel"}:
        head = ClassificationHead(hidden_size * 4, num_labels)
    else:
        head = RegressionHead(hidden_size * 4, num_targets=num_labels)
    head.load_state_dict(checkpoint["head"], strict=False)
    backbone.to(device).eval()
    head.to(device).eval()

    dataset = FinetuneDataset(
        selfies=sub_df["selfies"].tolist(), labels=labels, tokenizer=tokenizer,
        max_len=args.max_len, graph_emb=graph[idx], text_emb=text[idx], kg_emb=kg[idx],
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=MultimodalCollator()
    )

    metrics = _evaluate(backbone, head, dataloader, args.task_type, device)
    headline = "roc-auc" if args.task_type in {"binary", "multilabel"} else "rmse"
    metrics_str = " ".join(
        f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float) and not np.isnan(v)
    )
    print(f"[evaluate] {args.split}_metrics: {metrics_str}")
    print(f"[evaluate] {headline} = {metrics.get(headline, float('nan')):.4f}")

    return {
        "model": str(weights), "split": args.split, "seed": seed,
        "use_scaffold": bool(args.use_scaffold), "n": len(sub_df),
        "headline_metric": headline,
        "headline_value": float(metrics.get(headline, float("nan"))),
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, float)},
    }


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.info")
    except Exception:
        pass

    model_dir = Path(args.model_dir)

    # A task directory holding seed_* subdirectories evaluates every seed and reports
    # mean +/- std.
    seed_dirs = sorted(
        (d for d in model_dir.iterdir() if d.is_dir() and SEED_DIR_RE.match(d.name)),
        key=lambda d: int(SEED_DIR_RE.match(d.name).group(1)),
    ) if model_dir.is_dir() and not (model_dir / args.model_file).exists() else []

    if seed_dirs:
        if args.seed_explicit:
            raise SystemExit(
                f"--seed cannot be combined with a multi-seed directory ({model_dir}); "
                "each seed's split is taken from its own directory name."
            )
        results = []
        for d in seed_dirs:
            seed = seed_from_dir(d)
            per_seed_cfg = d / "finetuning_optimized_config.json"
            sub_args = argparse.Namespace(**vars(args))
            if per_seed_cfg.exists():
                cfg = {k: v for k, v in _load_config(str(per_seed_cfg)).items()
                       if k not in _TRAIN_ONLY and k != "seed"}
                for key, value in cfg.items():
                    if hasattr(sub_args, key):
                        setattr(sub_args, key, value)
            print(f"\n[evaluate] === {d.name} ===")
            results.append(evaluate_one(sub_args, d, seed, device))

        values = [r["headline_value"] for r in results]
        headline = results[0]["headline_metric"]
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(f"\n[evaluate] {len(values)} seeds: " +
              "  ".join(f"{r['seed']}={r['headline_value']:.4f}" for r in results))
        print(f"[evaluate] {headline} = {mean:.4f} +/- {std:.4f}  (mean +/- std over seeds)")
        payload = {"model_dir": str(model_dir), "split": args.split,
                   "headline_metric": headline, "mean": mean, "std": std,
                   "per_seed": results}
    else:
        dir_seed = seed_from_dir(model_dir)
        if dir_seed is not None and args.seed_explicit and dir_seed != args.seed:
            raise SystemExit(
                f"--seed {args.seed} contradicts the checkpoint directory name "
                f"({model_dir.name} implies seed {dir_seed}). Evaluating a checkpoint under "
                "another seed's split leaks its training molecules into the test set."
            )
        seed = dir_seed if dir_seed is not None else args.seed
        payload = evaluate_one(args, model_dir, seed, device)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[evaluate] wrote {args.output_json}")


if __name__ == "__main__":
    main()
