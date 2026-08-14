from __future__ import annotations

import numpy as np

from signaid.recognition.keypoints import normalize_keypoints


class RecognitionPipeline:
    """Stable recognition boundary; deterministic mock until a checkpoint is supplied."""

    def __init__(self, model=None, labels: list[str] | None = None) -> None:
        self.model = model
        self.labels = labels or ["HELP", "CHEST_PAIN", "EVACUATE"]

    def predict(self, keypoints: np.ndarray) -> dict:
        normalized, mask = normalize_keypoints(keypoints)
        if self.model is None:
            energy = float(np.abs(np.diff(normalized, axis=0)).mean()) if len(normalized) > 1 else 0.0
            index = int(energy * 1000) % len(self.labels)
            return {"intent": self.labels[index], "text": "도움이 필요합니다" if index == 0 else "", "confidence": 0.51, "mock": True, "frames": int(mask.sum())}
        try:
            import torch
            tensor = torch.from_numpy(normalized).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor, torch.from_numpy(mask).unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)[0]
            index = int(probs.argmax())
            return {"intent": self.labels[index], "confidence": float(probs[index]), "mock": False, "frames": int(mask.sum())}
        except Exception as exc:
            return {"intent": "UNKNOWN", "confidence": 0.0, "mock": True, "error": str(exc), "frames": int(mask.sum())}

