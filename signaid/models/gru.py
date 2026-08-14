from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


class GRUBaseline(nn.Module if torch else object):
    def __init__(self, joints: int = 47, channels: int = 3, hidden: int = 128, num_classes: int = 32) -> None:
        if torch is None:
            raise ImportError("Install SignAid[ml] to use GRUBaseline")
        super().__init__()
        self.gru = nn.GRU(joints * channels, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, keypoints, mask=None):
        batch, time = keypoints.shape[:2]
        output, _ = self.gru(keypoints.reshape(batch, time, -1))
        if mask is None:
            pooled = output.mean(dim=1)
        else:
            weights = mask.to(output.dtype).unsqueeze(-1)
            pooled = (output * weights).sum(1) / weights.sum(1).clamp_min(1)
        return self.head(pooled)

