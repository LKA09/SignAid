from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AvatarRenderer(Protocol):
    def render(self, motion_path: Path, output_path: Path) -> Path: ...

