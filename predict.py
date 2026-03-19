"""Run predictions with a finetuned SELFormerMM model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from SELFormerMM.models.downstream_heads import ClassificationHead, RegressionHead
from SELFormerMM.models.multimodal_roberta import MultimodalRoberta
from SELFormerMM.predictor import Predictor
from SELFormerMM.utils.datasets import FinetuneDataset, MultimodalCollator
from SELFormerMM.utils.embedders import save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with multimodal model.")
    parser.add_argument("--model_dir", required=True, help="Directory with model.pt")
    parser.add_argument("--tokenizer_path", default=None)
    # Prepared MoleculeNet bundle (meta CSV + NPZ embeddings)
    parser.add_argument("--input_meta_csv", required=True)
    parser.add_argument("--input_embs_npz", required=True)

    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--task_type", choices=["binary", "multilabel", "regression"], required=True
    )
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument(
        "--label_column", default=None, help="Optional label column for eval."
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output_id_column",
        default="smiles",
        help="Name of molecule ID in input_meta_csv to include in output CSV.",
    )
    return parser.parse_args()


def _load_embeddings_npz(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "graph": data["graph"].astype(np.float32),
        "text": data["text"].astype(np.float32),
        "kg": data["kg"].astype(np.float32),
    }


def _count_non_empty(arr: np.ndarray | None) -> int:
    if arr is None:
        return 0
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={arr.shape}")
    return int((np.linalg.norm(arr, axis=1) != 0).sum())


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(args.model_dir)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")

    config = AutoConfig.from_pretrained(model_dir)
    tokenizer_path = args.tokenizer_path or model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    backbone = MultimodalRoberta(config)
    backbone.load_state_dict(checkpoint["backbone"], strict=False)

    hidden_size = getattr(config, "hidden_size", 768)
    if args.task_type in {"binary", "multilabel"}:
        head = ClassificationHead(hidden_size * 4, args.num_labels)
    else:
        head = RegressionHead(hidden_size * 4, num_targets=args.num_labels)
    head.load_state_dict(checkpoint["head"], strict=False)

    df = pd.read_csv(args.input_meta_csv)
    if "selfies" not in df.columns:
        raise ValueError("input_meta_csv must include a 'selfies' column.")
    embs = _load_embeddings_npz(args.input_embs_npz)
    graph, text, kg = embs["graph"], embs["text"], embs["kg"]
    if len(df) != graph.shape[0] or len(df) != text.shape[0] or len(df) != kg.shape[0]:
        raise ValueError("Row count mismatch between input_meta_csv and input_embs_npz.")
    print(
        f"[predict] modality_non_empty: graph={_count_non_empty(graph):,}/{len(df):,} "
        f"text={_count_non_empty(text):,}/{len(df):,} kg={_count_non_empty(kg):,}/{len(df):,}"
    )

    if args.label_column and args.label_column in df.columns:
        labels = df[args.label_column].values
    else:
        labels = np.zeros(len(df), dtype=np.float32)

    dataset = FinetuneDataset(
        selfies=df["selfies"].tolist(),
        labels=labels,
        tokenizer=tokenizer,
        max_len=args.max_len,
        graph_emb=graph,
        text_emb=text,
        kg_emb=kg,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultimodalCollator(),
    )

    predictor = Predictor(
        backbone=backbone,
        head=head,
        task_type=args.task_type,
        device=device,
    )
    preds = predictor.predict(dataloader)

    # Require an ID column for all tasks and always include it in the output.
    if args.output_id_column not in df.columns:
        raise ValueError(
            f"Output ID column '{args.output_id_column}' not found in input_meta_csv."
        )
    output_ids = df[args.output_id_column].astype(str).tolist()

    # Shape/column handling:
    # - binary classification: keep only positive-class probability as 'prediction'
    # - regression: single 'prediction' column
    # - multilabel: keep all label probabilities with integer column names
    columns = None
    preds_array = preds
    if args.task_type == "binary":
        # preds shape: (N, num_labels). For num_labels=2 take column 1 (positive class).
        if preds_array.ndim == 2 and preds_array.shape[1] >= 2:
            preds_array = preds_array[:, 1:2]
        else:
            preds_array = preds_array.reshape(-1, 1)
        col_name = args.label_column or "prediction"
        columns = [col_name]
    elif args.task_type == "regression":
        if preds_array.ndim == 1:
            preds_array = preds_array.reshape(-1, 1)
        col_name = args.label_column or "prediction"
        columns = [col_name]
    else:  # multilabel
        if preds_array.ndim == 1:
            preds_array = preds_array.reshape(-1, 1)
        # Infer multilabel target column names.
        excluded = {
            "selfies",
            "smiles",
            "Description",
            "description",
            "chembl_id",
            "compound_id",
            "Compound ID",
            "id",
            "ID",
            "CID",
            "cid",
            "split",
            "fold",
        }
        candidates = [c for c in df.columns if c not in excluded]
        label_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        if not label_cols:
            raise ValueError(
                "Could not infer multilabel target columns for naming predictions "
                "(no numeric columns found after exclusions)."
            )
        num_cols = preds_array.shape[1]
        # Align number of prediction columns with inferred label columns.
        columns = label_cols[:num_cols]

    save_csv(args.output_csv, preds_array, ids=output_ids, columns=columns)
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
