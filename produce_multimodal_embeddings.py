"""Produce multimodal embeddings using SELFormerMM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from SELFormerMM.models.multimodal_roberta import MultimodalRoberta
from SELFormerMM.utils.datasets import PretrainDataset, MultimodalCollator
from SELFormerMM.utils.embedders import save_npy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce multimodal embeddings.")
    parser.add_argument(
        "--selfies_csv",
        required=True,
        help="CSV file containing a SELFIES column.",
    )
    parser.add_argument(
        "--selfies_column",
        default="selfies",
        help="SELFIES column name.",
    )
    parser.add_argument(
        "--pretrained_multimodal_dir",
        required=True,
        help=(
            "Directory from multimodal pretraining (train_pretraining.py): "
            "pytorch_model.bin or model.safetensors, plus config/tokenizer "
            "(or hf/ subfolder with tokenizer + config)."
        ),
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Tokenizer path (defaults to config dir next to weights).",
    )
    parser.add_argument("--graph_embs", default=None, help="Path to graph .npy.")
    parser.add_argument("--text_embs", default=None, help="Path to text .npy.")
    parser.add_argument("--kg_embs", default=None, help="Path to KG .npy.")
    parser.add_argument(
        "--output_npy",
        required=True,
        help="Output .npy file for embeddings.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV output with id column and embeddings.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--device", default=None, help="Device string, e.g. cpu/cuda.")
    parser.add_argument(
        "--output_mode",
        choices=["concat", "stacked"],
        default="concat",
        help="concat: (B, 4*D), stacked: (4B, D)",
    )
    parser.add_argument(
        "--id_column",
        default=None,
        help="Optional ID column for CSV output (e.g., chembl_id).",
    )
    return parser.parse_args()


def _load_optional_npy(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    return np.load(path).astype(np.float32)


def _load_multimodal_checkpoint(pretrained_dir: Path, tokenizer_path: str | None):
    if not pretrained_dir.is_dir():
        raise FileNotFoundError(f"pretrained_multimodal_dir not found: {pretrained_dir}")
    config_dir = pretrained_dir / "hf" if (pretrained_dir / "hf").is_dir() else pretrained_dir
    config = AutoConfig.from_pretrained(str(config_dir))
    tok = tokenizer_path or str(config_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok, use_fast=True)
    model = MultimodalRoberta(config)
    weights_bin = pretrained_dir / "pytorch_model.bin"
    weights_safe = pretrained_dir / "model.safetensors"
    if weights_bin.exists():
        state = torch.load(weights_bin, map_location="cpu")
    elif weights_safe.exists():
        try:
            from safetensors.torch import load_file  # type: ignore
        except Exception as exc:
            raise ImportError(
                "model.safetensors requires safetensors. pip install safetensors"
            ) from exc
        state = load_file(str(weights_safe), device="cpu")
    else:
        raise FileNotFoundError(
            f"No pytorch_model.bin or model.safetensors in {pretrained_dir}. "
            "Use the checkpoint directory saved by train_pretraining.py."
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[embed] warning: missing_keys={len(missing)}")
    if unexpected:
        print(f"[embed] warning: unexpected_keys={len(unexpected)}")
    return model, tokenizer


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.selfies_csv)
    if args.selfies_column not in df.columns:
        raise ValueError(f"Column '{args.selfies_column}' not found.")
    selfies = df[args.selfies_column].fillna("").astype(str).tolist()

    graph = _load_optional_npy(args.graph_embs)
    text = _load_optional_npy(args.text_embs)
    kg = _load_optional_npy(args.kg_embs)

    model, tokenizer = _load_multimodal_checkpoint(
        Path(args.pretrained_multimodal_dir), args.tokenizer_path
    )
    model.to(device)
    model.eval()

    dataset = PretrainDataset(
        selfies=selfies,
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

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                graph_emb=batch["graph_emb"].to(device),
                text_emb=batch["text_emb"].to(device),
                kg_emb=batch["kg_emb"].to(device),
            )
            if args.output_mode == "stacked":
                outputs.append(logits.detach().cpu().numpy())
            else:
                bsz = batch["input_ids"].size(0)
                chunks = [
                    logits[i * bsz : (i + 1) * bsz] for i in range(4)
                ]
                combined = torch.cat(chunks, dim=1)
                outputs.append(combined.detach().cpu().numpy())

    if outputs:
        embeddings = np.concatenate(outputs, axis=0)
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)

    save_npy(Path(args.output_npy), embeddings)
    print(f"Saved multimodal embeddings to: {args.output_npy}")

    if args.output_csv:
        if not args.id_column or args.id_column not in df.columns:
            raise ValueError("Provide --id_column present in selfies CSV for CSV output.")
        ids = df[args.id_column].tolist()
        if args.output_mode == "stacked":
            raise ValueError("CSV output requires concat mode to preserve row order.")
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ID count mismatch with embeddings.")
        columns = [f"emb_{i}" for i in range(embeddings.shape[1])]
        out_csv = Path(args.output_csv)
        pd.DataFrame(embeddings, columns=columns).assign(**{args.id_column: ids})[
            [args.id_column] + columns
        ].to_csv(out_csv, index=False)
        print(f"Saved multimodal embeddings CSV to: {out_csv}")


if __name__ == "__main__":
    main()
