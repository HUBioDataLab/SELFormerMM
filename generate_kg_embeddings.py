"""Generate KG embeddings using the pretrained DMGI model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from SELFormerMM.utils.embedders import KGEmbedder, save_npy


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


def _chembl_csv_to_kg_key(raw: object, prefix: str = "chembl:") -> str | None:
    """Map CSV chembl_id (e.g. CHEMBL6348) to KG key (e.g. chembl:CHEMBL6348)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    if not prefix.endswith(":"):
        prefix = prefix.rstrip(":") + ":"
    if s.lower().startswith("chembl:"):
        return "chembl:" + s.split(":", 1)[1].strip()
    return prefix + s


def _load_compound_mapping(data: object, node_type: str) -> dict[str, int]:
    """Load chembl:CHEMBL... -> node index from HeteroData Compound store."""
    store = data[node_type]
    m = getattr(store, "mapping", None)
    if m is None:
        try:
            m = store["mapping"]
        except (KeyError, TypeError):
            pass
    if m is None:
        raise ValueError(
            f"No 'mapping' on data['{node_type}']. Expected dict like "
            "'chembl:CHEMBL6206' -> 0."
        )
    out: dict[str, int] = {}
    for k, v in m.items():
        out[str(k)] = int(v)
    return out


def _align_to_meta_csv(
    node_embeddings: np.ndarray,
    mapping: dict[str, int],
    meta_csv: Path,
    id_column: str,
    kg_key_prefix: str,
) -> tuple[np.ndarray, int, int]:
    """
    One row per meta CSV row: embedding from KG node if chembl_id maps, else zeros.
    """
    df = pd.read_csv(meta_csv)
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in {meta_csv}. Columns: {list(df.columns)}")

    dim = node_embeddings.shape[1]
    n = len(df)
    out = np.zeros((n, dim), dtype=np.float32)
    matched = 0
    missing = 0
    for i, raw in enumerate(df[id_column].tolist()):
        key = _chembl_csv_to_kg_key(raw, prefix=kg_key_prefix)
        if key is None:
            missing += 1
            continue
        idx = mapping.get(key)
        if idx is None:
            missing += 1
            continue
        if idx < 0 or idx >= node_embeddings.shape[0]:
            missing += 1
            continue
        out[i] = node_embeddings[idx]
        matched += 1
    return out, matched, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KG embeddings.")
    parser.add_argument("--checkpoint_path", required=True, help="DMGI checkpoint path.")
    parser.add_argument("--output_npy", required=True, help="Output .npy path.")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV output with id column and embeddings (full KG nodes only).",
    )

    parser.add_argument(
        "--heterodata_path",
        required=True,
        help="Path to torch-saved HeteroData object.",
    )
    parser.add_argument(
        "--node_type",
        default="Compound",
        help="Node type to extract features from.",
    )
    parser.add_argument(
        "--node_feature_key",
        default="x",
        help="Feature key on node storage (default: x).",
    )
    parser.add_argument(
        "--id_column",
        default=None,
        help="Optional ID column on HeteroData for --output_csv (full-graph mode).",
    )

    parser.add_argument(
        "--align_meta_csv",
        default=None,
        help=(
            "If set, output .npy has one row per CSV row (same order as this file). "
            "chembl_id in CSV (CHEMBL…) is matched to KG mapping keys (chembl:CHEMBL…); "
            "missing compounds get zero vectors."
        ),
    )
    parser.add_argument(
        "--align_meta_column",
        default="chembl_id",
        help="Column in --align_meta_csv with ChEMBL ids (default: chembl_id).",
    )
    parser.add_argument(
        "--kg_id_prefix",
        default="chembl:",
        help="Prefix for KG mapping keys when CSV has bare CHEMBL ids (default: chembl:).",
    )

    parser.add_argument("--out_channels", type=int, default=128)
    parser.add_argument("--device", default=None, help="Device string, e.g. cpu/cuda.")
    parser.add_argument(
        "--normalize",
        type=int,
        default=1,
        help="Apply mean-center + L2 normalization (1) or not (0). Applies to full node embeddings before align.",
    )
    return parser.parse_args()


def _load_from_heterodata(path: Path, node_type: str, feature_key: str):
    data = torch.load(path, map_location="cpu")
    node_store = data[node_type]
    node_features = getattr(node_store, feature_key)
    edge_indices = list(data.edge_index_dict.values())
    return data, node_features, edge_indices


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_npy)

    data, node_features, edge_indices = _load_from_heterodata(
        Path(args.heterodata_path), args.node_type, args.node_feature_key
    )

    embedder = KGEmbedder(
        checkpoint_path=args.checkpoint_path,
        num_nodes=node_features.size(0),
        in_channels=node_features.size(-1),
        out_channels=args.out_channels,
        num_relations=len(edge_indices),
        device=args.device,
    )

    embeddings = embedder.embed(node_features, edge_indices)
    if args.normalize:
        embeddings, zero_count = _normalize_embeddings(embeddings)
        print(
            f"[kg] normalized full-graph embeddings (zero_rows={zero_count}/{embeddings.shape[0]})"
        )

    if args.align_meta_csv:
        meta_path = Path(args.align_meta_csv)
        mapping = _load_compound_mapping(data, args.node_type)
        aligned, matched, missing = _align_to_meta_csv(
            embeddings,
            mapping,
            meta_path,
            id_column=args.align_meta_column,
            kg_key_prefix=args.kg_id_prefix,
        )
        save_npy(output_path, aligned)
        print(
            f"[kg] aligned to {meta_path.name}: rows={len(aligned)}, dim={aligned.shape[1]}, "
            f"matched_in_kg={matched}, zero_or_unmapped={missing}"
        )
        print(f"Saved aligned KG embeddings to: {output_path}")
    else:
        save_npy(output_path, embeddings)
        print(f"Saved KG embeddings to: {output_path}")

    if args.output_csv and not args.align_meta_csv:
        if not args.id_column or args.id_column not in data[args.node_type]:
            raise ValueError("Provide --id_column present in HeteroData node store.")
        ids = getattr(data[args.node_type], args.id_column)
        if torch.is_tensor(ids):
            ids = ids.cpu().numpy().astype(str).tolist()
        else:
            ids = list(ids)
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ID count mismatch with embeddings.")
        emb_to_save = embeddings
        columns = [f"emb_{i}" for i in range(emb_to_save.shape[1])]
        out_csv = Path(args.output_csv)
        pd.DataFrame(emb_to_save, columns=columns).assign(**{args.id_column: ids})[
            [args.id_column] + columns
        ].to_csv(out_csv, index=False)
        print(f"Saved KG embeddings CSV to: {out_csv}")


if __name__ == "__main__":
    main()
