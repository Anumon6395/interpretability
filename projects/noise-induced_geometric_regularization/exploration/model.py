"""Toy model: Encoder (input_dim -> 2D embedding) + linear Readout to two binary heads (A, B)."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP: input_dim -> hidden_dim -> embed_dim (2)."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Readout(nn.Module):
    """Two linear heads: embed_dim (2) -> 2 logits each for feature A and feature B."""

    def __init__(self, embed_dim: int = 2):
        super().__init__()
        self.head_a = nn.Linear(embed_dim, 2)  # binary classification
        self.head_b = nn.Linear(embed_dim, 2)

    def forward(
        self, embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_a = self.head_a(embed)   # (N, 2)
        logits_b = self.head_b(embed)   # (N, 2)
        return logits_a, logits_b


class ToySuperpositionModel(nn.Module):
    """Encoder -> 2D embedding -> Readout (pred_A, pred_B)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 2,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, embed_dim)
        self.readout = Readout(embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.encoder(x)
        logits_a, logits_b = self.readout(embed)
        return embed, logits_a, logits_b

    def get_feature_directions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return unit vectors w_A, w_B in embedding space (direction of increasing P(A=1), P(B=1)).
        Uses the class-1 row of each head's weight matrix.
        """
        w_a = self.readout.head_a.weight[1].detach()   # (embed_dim,)
        w_b = self.readout.head_b.weight[1].detach()
        w_a = w_a / (w_a.norm() + 1e-8)
        w_b = w_b / (w_b.norm() + 1e-8)
        return w_a, w_b
