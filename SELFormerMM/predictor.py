from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn


@dataclass
class Predictor:
    """Run prediction for classification or regression heads."""

    backbone: nn.Module
    head: nn.Module
    task_type: str
    device: Optional[str] = None
    num_views: int = 4

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        self.head.to(self.device)
        self.task_type = self.task_type.lower()
        if self.task_type not in {"binary", "multilabel", "regression"}:
            raise ValueError("task_type must be binary, multilabel, or regression.")

    def _combine_views(self, embeddings: torch.Tensor, batch_size: int) -> torch.Tensor:
        chunks = [
            embeddings[i * batch_size : (i + 1) * batch_size]
            for i in range(self.num_views)
        ]
        return torch.cat(chunks, dim=1)

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        graph_emb = batch.get("graph_emb")
        text_emb = batch.get("text_emb")
        kg_emb = batch.get("kg_emb")
        if graph_emb is not None:
            graph_emb = graph_emb.to(self.device)
        if text_emb is not None:
            text_emb = text_emb.to(self.device)
        if kg_emb is not None:
            kg_emb = kg_emb.to(self.device)

        embeddings = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_emb=graph_emb,
            text_emb=text_emb,
            kg_emb=kg_emb,
        )
        batch_size = input_ids.size(0)
        combined = self._combine_views(embeddings, batch_size)
        return self.head(combined)

    def predict(self, dataloader: Iterable[dict]) -> np.ndarray:
        self.backbone.eval()
        self.head.eval()
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in dataloader:
                logits = self._forward_batch(batch)
                preds.append(logits.detach().cpu())
        if not preds:
            return np.zeros((0,), dtype=np.float32)
        logits = torch.cat(preds, dim=0)
        if self.task_type == "regression":
            return logits.numpy()
        probs = torch.sigmoid(logits)
        return probs.numpy()
