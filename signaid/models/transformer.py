from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


class TransformerBaseline(nn.Module if torch else object):
    def __init__(self, joints: int = 47, channels: int = 3, width: int = 192, layers: int = 3, num_classes: int = 32) -> None:
        if torch is None:
            raise ImportError("Install SignAid[ml] to use TransformerBaseline")
        super().__init__()
        self.project = nn.Linear(joints * channels, width)
        encoder = nn.TransformerEncoderLayer(width, 6, width * 4, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder, layers)
        self.head = nn.Linear(width, num_classes)

    def forward(self, keypoints, mask=None):
        x = self.project(keypoints.flatten(2))
        x = self.encoder(x, src_key_padding_mask=(~mask.bool()) if mask is not None else None)
        if mask is None:
            return self.head(x.mean(1))
        weights = mask.to(x.dtype).unsqueeze(-1)
        return self.head((x * weights).sum(1) / weights.sum(1).clamp_min(1))

