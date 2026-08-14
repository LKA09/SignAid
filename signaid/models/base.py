from __future__ import annotations

from typing import Protocol


class SignRecognitionModel(Protocol):
    def forward(self, keypoints, mask=None): ...

