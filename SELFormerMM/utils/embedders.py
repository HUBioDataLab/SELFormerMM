from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class TextEmbedder:
    """Text embedding generator using a transformer encoder."""

    model_name: str = "allenai/scibert_scivocab_uncased"
    device: Optional[str] = None
    max_length: int = 512

    def __post_init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        embeddings: list[np.ndarray] = []
        for text in tqdm(texts, desc="Text embeddings", unit="text"):
            if not isinstance(text, str) or not text.strip():
                embeddings.append(np.zeros(768, dtype=np.float32))
                continue
            tokens = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**tokens)
            pooled = output.last_hidden_state[0].mean(dim=0)
            embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(embeddings)


@dataclass
class GraphEmbedder:
    """Graph embedding generator using UniMol representations."""

    use_gpu: bool = True

    def __post_init__(self) -> None:
        try:
            from unimol_tools import UniMolRepr
        except ImportError as exc:
            raise ImportError(
                "unimol_tools is required for GraphEmbedder."
            ) from exc
        self.model = UniMolRepr(
            data_type="molecule", remove_hs=False, use_gpu=self.use_gpu
        )

    def embed_smiles(
        self,
        smiles_list: Iterable[str],
        batch_size: int = 512,
        show_progress: bool = True,
    ) -> np.ndarray:
        smiles = list(smiles_list)
        all_embeddings: list[np.ndarray] = []
        batch_ranges = range(0, len(smiles), batch_size)
        if show_progress:
            batch_ranges = tqdm(
                batch_ranges,
                desc="Graph embeddings",
                unit="batch",
            )
        for i in batch_ranges:
            batch = smiles[i : i + batch_size]
            batch_repr = self.model.get_repr(batch, return_atomic_reprs=True)
            cls_reprs = np.array(batch_repr["cls_repr"], dtype=np.float32)
            all_embeddings.append(cls_reprs)
        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 512), dtype=np.float32)


@dataclass
class KGEmbedder:
    """Knowledge-graph embedding generator using the pretrained DMGI model."""

    checkpoint_path: str
    num_nodes: int
    in_channels: int
    out_channels: int
    num_relations: int
    device: Optional[str] = None

    def __post_init__(self) -> None:
        from SELFormerMM.models.dmgi import load_dmgi_model

        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        self.model = load_dmgi_model(
            checkpoint_path=self.checkpoint_path,
            num_nodes=self.num_nodes,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_relations=self.num_relations,
            map_location=self.device,
        ).to(self.device)

    def embed(
        self, node_features: torch.Tensor, edge_indices: Iterable[torch.Tensor]
    ) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            pos_hs, _, _ = self.model(node_features.to(self.device), edge_indices)
        embeddings = torch.stack(pos_hs, dim=0).mean(dim=0)
        return embeddings.detach().cpu().numpy().astype(np.float32)


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy(embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(embeddings, np.ndarray):
        return embeddings
    if torch.is_tensor(embeddings):
        return embeddings.detach().cpu().numpy()
    raise TypeError("embeddings must be a numpy array or torch tensor.")


def save_npy(path: str | Path, embeddings: np.ndarray | torch.Tensor) -> Path:
    """Save embeddings as a .npy file."""
    path = _ensure_parent(path)
    np.save(path, _to_numpy(embeddings))
    return path


def save_csv(
    path: str | Path,
    embeddings: np.ndarray | torch.Tensor,
    ids: Sequence[str] | None = None,
    columns: Iterable[str] | None = None,
) -> Path:
    """Save embeddings to CSV with optional ids/columns."""
    path = _ensure_parent(path)
    array = _to_numpy(embeddings)
    df = pd.DataFrame(array, columns=columns)
    if ids is not None:
        df.insert(0, "id", list(ids))
    df.to_csv(path, index=False)
    return path
