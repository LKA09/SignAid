from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SignSample:
    sample_id: str
    text: str
    gloss: list[str]
    category: str = "unknown"
    signer_id: str = ""
    source: str = "AIHUB"
    keypoint_path: Path | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "sample_id": self.sample_id,
            "text": self.text,
            "gloss": self.gloss,
            "category": self.category,
            "signer_id": self.signer_id,
            "source": self.source,
        }
        if self.keypoint_path:
            data["keypoint_path"] = str(self.keypoint_path)
        return data

