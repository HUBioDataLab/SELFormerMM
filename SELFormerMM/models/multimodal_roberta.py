from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import RobertaModel


def _mlp_projection(
    input_dim: int,
    hidden_size: int,
    expansion_factors: Iterable[int] = (4, 6, 6, 4),
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for factor in expansion_factors:
        dim = hidden_size * factor
        layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ReLU()])
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, hidden_size))
    return nn.Sequential(*layers)


class MultimodalRoberta(nn.Module):
    """RoBERTa + projection heads for graph/text/KG embeddings."""

    def __init__(
        self,
        config,
        graph_dim: int = 512,
        text_dim: int = 768,
        kg_dim: int = 128,
        expansion_factors: Iterable[int] = (4, 6, 6, 4),
    ) -> None:
        super().__init__()
        self.hidden_size = getattr(config, "hidden_size", 768)
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.graph_proj = _mlp_projection(
            graph_dim, self.hidden_size, expansion_factors
        )
        self.text_proj = _mlp_projection(
            text_dim, self.hidden_size, expansion_factors
        )
        self.kg_proj = _mlp_projection(kg_dim, self.hidden_size, expansion_factors)

    def _project(
        self,
        proj: nn.Module,
        emb: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if emb is None:
            return torch.zeros(batch_size, self.hidden_size, device=device)
        if emb.device != device:
            emb = emb.to(device)
        return proj(emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_emb: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        kg_emb: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_roberta_emb = outputs.last_hidden_state[:, 0, :]
        text_roberta_emb = F.normalize(text_roberta_emb, p=2, dim=1)

        batch_size = text_roberta_emb.size(0)
        device = text_roberta_emb.device

        graph_hidden = self._project(
            self.graph_proj, graph_emb, batch_size, device
        )
        text_hidden = self._project(
            self.text_proj, text_emb, batch_size, device
        )
        kg_hidden = self._project(self.kg_proj, kg_emb, batch_size, device)

        combined = torch.cat(
            [text_roberta_emb, graph_hidden, text_hidden, kg_hidden], dim=0
        )

        if return_dict:
            return {
                "combined": combined,
                "selfies": text_roberta_emb,
                "graph": graph_hidden,
                "text": text_hidden,
                "kg": kg_hidden,
            }
        return combined
