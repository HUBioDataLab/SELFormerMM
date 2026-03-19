"""Generate text embeddings from a chemical descriptions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from SELFormerMM.utils.embedders import TextEmbedder, save_npy


def _normalize_embeddings(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Mean-center + L2 normalize non-zero rows; keep zero rows unchanged."""
    arr = arr.astype(np.float32, copy=True)
    non_zero_mask = np.linalg.norm(arr, axis=1) != 0
    zero_count = int((~non_zero_mask).sum())
    if non_zero_mask.any():
        mean_vec = arr[non_zero_mask].mean(axis=0, keepdims=True)
        centered = arr[non_zero_mask] - mean_vec
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr[non_zero_mask] = centered / norms
    return arr, zero_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text embeddings.")
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV containing text.",
    )
    parser.add_argument(
        "--output_npy",
        required=True,
        help="Path to output .npy file.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV output with id column and embeddings.",
    )
    parser.add_argument(
        "--text_column",
        default="Description",
        help="Name of the text column in the input CSV.",
    )
    parser.add_argument(
        "--id_column",
        default=None,
        help="Optional ID column to include in CSV output.",
    )
    parser.add_argument(
        "--model_name",
        default="allenai/scibert_scivocab_uncased",
        help="HF model name for text embeddings.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--normalize",
        type=int,
        default=1,
        help="Apply mean-center + L2 normalization (1) or not (0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_npy)

    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in {input_path}")

    texts = df[args.text_column].fillna("").astype(str).tolist()
    embedder = TextEmbedder(
        model_name=args.model_name, device=args.device, max_length=args.max_length
    )
    embeddings = embedder.embed_texts(texts)
    zero_mask = np.linalg.norm(embeddings, axis=1) == 0
    zero_rows = int(zero_mask.sum())
    non_zero_rows = int((~zero_mask).sum())

    if args.normalize:
        embeddings, zero_count = _normalize_embeddings(embeddings)
        print(
            f"Normalized text embeddings (zero_rows={zero_count}/{embeddings.shape[0]})"
        )

    save_npy(output_path, embeddings)
    print(f"Saved text embeddings to: {output_path}")
    print(
        "Text embedding summary: "
        f"total_rows={len(df)} non_zero_embeddings={non_zero_rows} zero_vectors={zero_rows}"
    )

    if args.output_csv:
        if not args.id_column or args.id_column not in df.columns:
            raise ValueError("Provide --id_column present in input CSV for CSV output.")
        ids = df[args.id_column].tolist()
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ID count mismatch with embeddings.")
        output_csv = Path(args.output_csv)
        columns = [f"emb_{i}" for i in range(embeddings.shape[1])]
        pd.DataFrame(embeddings, columns=columns).assign(**{args.id_column: ids})[
            [args.id_column] + columns
        ].to_csv(output_csv, index=False)
        print(f"Saved text embeddings CSV to: {output_csv}")


if __name__ == "__main__":
    main()
