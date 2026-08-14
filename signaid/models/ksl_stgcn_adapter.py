from __future__ import annotations

import importlib
import sys
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


def _load_ksl_models():
    repo = Path(__file__).resolve().parents[3] / "third_party" / "KSL-main"
    if not repo.exists():
        raise RuntimeError(f"KSL-main not found at {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return importlib.import_module("Models")


class KSLSTGCNAdapter(nn.Module if torch else object):
    """Thin adapter around third_party/KSL-main's actual 47-joint ST-GCN."""

    def __init__(self, num_classes: int = 32, two_stream: bool = False) -> None:
        if torch is None:
            raise ImportError("Install SignAid[ml] to use KSLSTGCNAdapter")
        super().__init__()
        models = _load_ksl_models()
        graph_args = {"layout": "mediapipe_KSL", "strategy": "spatial"}
        self.two_stream = two_stream
        self.model = (models.TwoStreamSpatialTemporalGraph(graph_args, num_classes)
                      if two_stream else models.StreamSpatialTemporalGraph(3, graph_args, num_classes))

    @staticmethod
    def _select_47(keypoints):
        if keypoints.shape[-2] == 47:
            return keypoints
        if keypoints.shape[-2] >= 75:  # MediaPipe: pose 33 + two hands 21 each
            indices = [0, 11, 12, 13, 14, *range(33, 54), *range(54, 75)]
            return keypoints[..., indices, :]
        raise ValueError("KSL ST-GCN expects 47 joints or MediaPipe's 75 pose/hand joints")

    def forward(self, keypoints, mask=None):
        points = self._select_47(keypoints)
        points = points.permute(0, 3, 1, 2).contiguous()  # N,T,V,C -> N,C,T,V
        if self.two_stream:
            motion = torch.zeros_like(points)
            motion[:, :, 1:] = points[:, :, 1:] - points[:, :, :-1]
            return self.model((points, motion))
        return self.model(points)

