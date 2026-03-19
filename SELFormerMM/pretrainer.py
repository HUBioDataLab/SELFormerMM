from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn


class SINCERELoss(nn.Module):
    """Supervised InfoNCE REvisited loss with cosine distance."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = embeds @ embeds.T
        logits /= self.temperature
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        denom_activations = torch.full_like(logits, float("-inf"))
        denom_activations[~same_label] = logits[~same_label]
        base_denom_row = torch.logsumexp(denom_activations, dim=0)
        base_denom = base_denom_row.unsqueeze(1).repeat((1, len(base_denom_row)))

        in_numer = same_label
        in_numer[torch.eye(in_numer.shape[0], dtype=bool)] = False
        del same_label

        numer_count = in_numer.sum(dim=0)
        numer_logits = torch.zeros_like(logits)
        numer_logits[in_numer] = logits[in_numer]

        log_denom = torch.zeros_like(logits)
        log_denom[in_numer] = torch.stack(
            (numer_logits[in_numer], base_denom[in_numer]), dim=0
        ).logsumexp(dim=0)

        ce = -1 * (numer_logits - log_denom)
        loss = torch.sum(ce / numer_count) / ce.shape[0]
        return loss


@dataclass
class Pretrainer:
    """Pretraining loop for multimodal contrastive training."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    device: Optional[str] = None
    num_views: int = 4

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_labels(self, batch_size: int) -> torch.Tensor:
        return torch.arange(batch_size, dtype=torch.long, device=self.device).repeat(
            self.num_views
        )

    def train_epoch(self, dataloader: Iterable[dict]) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
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

            batch_size = input_ids.size(0)
            labels = self._build_labels(batch_size)

            self.optimizer.zero_grad()
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_emb=graph_emb,
                text_emb=text_emb,
                kg_emb=kg_emb,
            )
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)

    def validate(self, dataloader: Iterable[dict]) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
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

                batch_size = input_ids.size(0)
                labels = self._build_labels(batch_size)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    graph_emb=graph_emb,
                    text_emb=text_emb,
                    kg_emb=kg_emb,
                )
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)
