from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


@dataclass
class Finetuner:
    """Finetuning loop for classification/regression tasks."""

    backbone: nn.Module
    head: nn.Module
    optimizer: torch.optim.Optimizer
    task_type: str
    scheduler: Optional[object] = None
    device: Optional[str] = None
    num_views: int = 4
    pos_weight: Optional[torch.Tensor] = None
    max_grad_norm: float = 1.0

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

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.task_type == "binary":
            # SELFormer binary script uses CrossEntropyLoss with 2 logits
            if logits.ndim == 2 and logits.size(1) == 2:
                loss_fn = CrossEntropyLoss()
                return loss_fn(logits, labels.long())
            # fallback: allow 1-logit BCE if used elsewhere
            labels_f = labels.float().view(-1)
            logits_f = logits.view(-1)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            return loss_fn(logits_f, labels_f)
        if self.task_type == "multilabel":
            labels_f = labels.float()
            mask = ~torch.isnan(labels_f)
            labels_f = torch.nan_to_num(labels_f, nan=0.0)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")
            losses = loss_fn(logits, labels_f)
            if mask.any():
                return losses[mask].mean()
            return losses.mean() * 0.0
        labels_f = labels.float()
        mask = ~torch.isnan(labels_f)
        labels_f = torch.nan_to_num(labels_f, nan=0.0)
        loss_fn = nn.MSELoss(reduction="none")
        losses = loss_fn(logits, labels_f)
        if mask.any():
            return losses[mask].mean()
        return losses.mean() * 0.0

    def _forward_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
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
        logits = self.head(combined)
        labels = batch["labels"].to(self.device)
        return logits, labels

    def train_epoch(self, dataloader: Iterable[dict]) -> float:
        self.backbone.train()
        self.head.train()
        total_loss = 0.0
        for batch in dataloader:
            self.optimizer.zero_grad()
            logits, labels = self._forward_batch(batch)
            loss = self._loss(logits, labels)
            loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters()) + list(self.head.parameters()),
                    max_norm=self.max_grad_norm,
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)

    def validate(self, dataloader: Iterable[dict]) -> float:
        self.backbone.eval()
        self.head.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                logits, labels = self._forward_batch(batch)
                loss = self._loss(logits, labels)
                total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)
