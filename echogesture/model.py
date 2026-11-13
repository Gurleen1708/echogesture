"""Small temporal conv net over landmark features. Conv/BN/ReLU only -> quantises cleanly to INT8."""
from __future__ import annotations

import torch
from torch import nn

from . import config


class Block(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class SignTCN(nn.Module):
    """Input (B, SEQ_LEN, FEAT_DIM) -> logits (B, num_classes)."""

    def __init__(self, num_classes: int, feat_dim: int = config.FEAT_DIM,
                 channels: int = config.CHANNELS, dilations=config.BLOCK_DILATIONS,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(feat_dim, channels, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[Block(channels, d, dropout * 0.5) for d in dilations])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x.transpose(1, 2)))
        return self.head(self.drop(x.mean(dim=2)))
