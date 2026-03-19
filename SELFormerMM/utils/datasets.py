"""Dataset utilities for pretraining and finetuning."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import selfies as sf


def _to_list(values: Sequence[str] | Iterable[str]) -> list[str]:
    if isinstance(values, list):
        return values
    return list(values)


def _infer_dim(embs: Optional[np.ndarray | torch.Tensor], default_dim: int) -> int:
    if embs is None:
        return default_dim
    return int(embs.shape[1])


def _get_embedding(
    embs: Optional[np.ndarray | torch.Tensor],
    idx: int,
    dim: int,
) -> torch.Tensor:
    if embs is None:
        return torch.zeros(dim, dtype=torch.float32)
    if isinstance(embs, np.ndarray):
        return torch.from_numpy(embs[idx]).float()
    return embs[idx].float()


def smiles_to_selfies(smiles: str, on_error: str = "keep") -> str:
    """
    Convert a SMILES string to SELFIES.

    on_error:
        - "keep": return the original SMILES
        - "empty": return empty string
        - "raise": re-raise the encoder error
    """
    try:
        return sf.encoder(smiles)
    except sf.EncoderError:
        if on_error == "empty":
            return ""
        if on_error == "raise":
            raise
        return smiles


def smiles_list_to_selfies(
    smiles_list: Iterable[str], on_error: str = "keep"
) -> list[str]:
    """Vectorized wrapper for list conversion."""
    return [smiles_to_selfies(smi, on_error=on_error) for smi in smiles_list]


def random_split(
    data: int | Iterable,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    if not np.isclose(frac_train + frac_valid + frac_test, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for random_split to match SELFormer."
        ) from exc

    n_samples = data if isinstance(data, int) else len(list(data))
    indices = np.arange(n_samples).tolist()
    train_idx, val_idx = train_test_split(
        indices, test_size=(1.0 - frac_train), random_state=seed
    )
    val_idx, test_idx = train_test_split(
        val_idx, test_size=(frac_test / (frac_valid + frac_test)), random_state=seed
    )
    return list(train_idx), list(val_idx), list(test_idx)


def scaffold_split(
    smiles: Iterable[str],
    targets: Iterable | None = None,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    if not np.isclose(frac_train + frac_valid + frac_test, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")
    try:
        import chemprop
    except ImportError as exc:
        raise ImportError(
            "chemprop is required for scaffold_split to match SELFormer."
        ) from exc

    molecule_list = []
    if targets is None:
        targets_iter = [None for _ in smiles]
    else:
        targets_iter = targets
    for smi, tgt in zip(smiles, targets_iter):
        molecule_list.append(
            chemprop.data.data.MoleculeDatapoint(smiles=[smi], targets=tgt)
        )
    molecule_dataset = chemprop.data.data.MoleculeDataset(molecule_list)

    train_ds, val_ds, test_ds = chemprop.data.scaffold.scaffold_split(
        data=molecule_dataset,
        sizes=(frac_train, frac_valid, frac_test),
        seed=seed,
        balanced=True,
    )

    id_to_index = {id(dp): idx for idx, dp in enumerate(molecule_dataset)}
    train_idx = [id_to_index[id(dp)] for dp in train_ds]
    valid_idx = [id_to_index[id(dp)] for dp in val_ds]
    test_idx = [id_to_index[id(dp)] for dp in test_ds]

    return train_idx, valid_idx, test_idx


class PretrainDataset(Dataset):
    """Pretraining dataset for multimodal inputs."""

    def __init__(
        self,
        selfies: Sequence[str] | Iterable[str],
        tokenizer,
        max_len: int = 512,
        graph_emb: Optional[np.ndarray | torch.Tensor] = None,
        text_emb: Optional[np.ndarray | torch.Tensor] = None,
        kg_emb: Optional[np.ndarray | torch.Tensor] = None,
        graph_dim: int = 512,
        text_dim: int = 768,
        kg_dim: int = 128,
    ) -> None:
        self.selfies = _to_list(selfies)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graph_emb = graph_emb
        self.text_emb = text_emb
        self.kg_emb = kg_emb

        self.graph_dim = _infer_dim(graph_emb, graph_dim)
        self.text_dim = _infer_dim(text_emb, text_dim)
        self.kg_dim = _infer_dim(kg_emb, kg_dim)

        if graph_emb is not None and len(self.selfies) != len(graph_emb):
            raise ValueError("selfies and graph embeddings length mismatch.")
        if text_emb is not None and len(self.selfies) != len(text_emb):
            raise ValueError("selfies and text embeddings length mismatch.")
        if kg_emb is not None and len(self.selfies) != len(kg_emb):
            raise ValueError("selfies and KG embeddings length mismatch.")

    def __len__(self) -> int:
        return len(self.selfies)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.selfies[idx]
        encoding = self.tokenizer.encode_plus(
            text, max_length=self.max_len, truncation=True, padding="max_length"
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "graph_emb": _get_embedding(self.graph_emb, idx, self.graph_dim),
            "text_emb": _get_embedding(self.text_emb, idx, self.text_dim),
            "kg_emb": _get_embedding(self.kg_emb, idx, self.kg_dim),
        }


class FinetuneDataset(Dataset):
    """Fine-tuning dataset for multimodal inputs and labels."""

    def __init__(
        self,
        selfies: Sequence[str] | Iterable[str],
        labels: Sequence | np.ndarray | torch.Tensor,
        tokenizer,
        max_len: int = 512,
        graph_emb: Optional[np.ndarray | torch.Tensor] = None,
        text_emb: Optional[np.ndarray | torch.Tensor] = None,
        kg_emb: Optional[np.ndarray | torch.Tensor] = None,
        graph_dim: int = 512,
        text_dim: int = 768,
        kg_dim: int = 128,
        label_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.selfies = _to_list(selfies)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graph_emb = graph_emb
        self.text_emb = text_emb
        self.kg_emb = kg_emb

        self.graph_dim = _infer_dim(graph_emb, graph_dim)
        self.text_dim = _infer_dim(text_emb, text_dim)
        self.kg_dim = _infer_dim(kg_emb, kg_dim)
        self.label_dtype = label_dtype

        if len(self.selfies) != len(labels):
            raise ValueError("selfies and labels length mismatch.")
        if graph_emb is not None and len(self.selfies) != len(graph_emb):
            raise ValueError("selfies and graph embeddings length mismatch.")
        if text_emb is not None and len(self.selfies) != len(text_emb):
            raise ValueError("selfies and text embeddings length mismatch.")
        if kg_emb is not None and len(self.selfies) != len(kg_emb):
            raise ValueError("selfies and KG embeddings length mismatch.")

    def __len__(self) -> int:
        return len(self.selfies)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.selfies[idx]
        encoding = self.tokenizer.encode_plus(
            text, max_length=self.max_len, truncation=True, padding="max_length"
        )
        label = self.labels[idx]
        if isinstance(label, np.ndarray):
            label_tensor = torch.from_numpy(label).to(self.label_dtype)
        elif torch.is_tensor(label):
            label_tensor = label.to(self.label_dtype)
        else:
            label_tensor = torch.tensor(label, dtype=self.label_dtype)

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "graph_emb": _get_embedding(self.graph_emb, idx, self.graph_dim),
            "text_emb": _get_embedding(self.text_emb, idx, self.text_dim),
            "kg_emb": _get_embedding(self.kg_emb, idx, self.kg_dim),
            "labels": label_tensor,
        }


class MultimodalCollator:
    """Simple collator that stacks tensors from dataset outputs."""

    def __init__(self, keys: Iterable[str] | None = None) -> None:
        self.keys = list(keys) if keys is not None else None

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if not batch:
            return {}
        keys = self.keys or list(batch[0].keys())
        collated: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [item[key] for item in batch if key in item]
            if not values:
                continue
            collated[key] = torch.stack(values, dim=0)
        return collated
