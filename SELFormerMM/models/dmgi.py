from __future__ import annotations

import re
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class DMGI(nn.Module):
    """Deep Multimodal Graph Infomax (DMGI) encoder."""

    def __init__(
        self, num_nodes: int, in_channels: int, out_channels: int, num_relations: int
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_channels, out_channels) for _ in range(num_relations)]
        )
        self.M = nn.Bilinear(out_channels, out_channels, 1)
        self.Z = nn.Parameter(torch.empty(num_nodes, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        nn.init.xavier_uniform_(self.Z)

    def forward(
        self, x: torch.Tensor, edge_indices: Iterable[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        pos_hs, neg_hs, summaries = [], [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            pos_hs.append(pos_h)

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(neg_h, edge_index).relu()
            neg_hs.append(neg_h)

            summaries.append(pos_h.mean(dim=0, keepdim=True))

        return pos_hs, neg_hs, summaries

    def loss(
        self,
        pos_hs: list[torch.Tensor],
        neg_hs: list[torch.Tensor],
        summaries: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0.0
        for pos_h, neg_h, s in zip(pos_hs, neg_hs, summaries):
            s = s.expand_as(pos_h)
            loss += -torch.log(self.M(pos_h, s).sigmoid() + 1e-15).mean()
            loss += -torch.log(1 - self.M(neg_h, s).sigmoid() + 1e-15).mean()

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)
        pos_reg_loss = (self.Z - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum()
        loss += 0.001 * (pos_reg_loss - neg_reg_loss)
        return loss


def _strip_module_prefix(state: dict) -> dict:
    if not state or not any(str(k).startswith("module.") for k in state):
        return state
    return {str(k).replace("module.", "", 1): v for k, v in state.items()}


def infer_num_relations_from_state(state: dict) -> int:
    """Number of GCN relation heads in a saved DMGI state_dict."""
    state = _strip_module_prefix(state)
    ids: set[int] = set()
    for k in state:
        m = re.match(r"^convs\.(\d+)\.", k)
        if m:
            ids.add(int(m.group(1)))
    if not ids:
        raise ValueError("Could not infer num_relations: no convs.N.* keys in checkpoint.")
    return max(ids) + 1


def load_dmgi_model(
    checkpoint_path: str,
    num_nodes: int,
    in_channels: int,
    out_channels: int,
    num_relations: Optional[int] = None,
    map_location: Optional[str] = "cpu",
) -> DMGI:
    """
    Load a pretrained DMGI checkpoint.

    If ``num_relations`` is None, it is inferred from the checkpoint (must match
    how the model was trained). HeteroData may list more edge types; only the
    first ``num_relations`` edge tensors should be passed to ``forward`` (same
    ordering as at training time).
    """
    raw = torch.load(checkpoint_path, map_location=map_location)
    state = _strip_module_prefix(raw)
    inferred = infer_num_relations_from_state(state)
    if num_relations is not None and num_relations != inferred:
        import warnings

        warnings.warn(
            f"HeteroData suggests {num_relations} edge types but checkpoint has "
            f"{inferred} DMGI relation heads; using {inferred} from checkpoint.",
            stacklevel=2,
        )
    num_relations = inferred
    model = DMGI(num_nodes, in_channels, out_channels, num_relations)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
