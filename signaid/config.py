from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = Path(os.getenv("SIGNAID_DATA_DIR", PROJECT_ROOT / "data"))
    transition_frames: int = int(os.getenv("SIGNAID_TRANSITION_FRAMES", "4"))
    fps: int = int(os.getenv("SIGNAID_FPS", "20"))
    signavatars_python: str | None = os.getenv("SIGNAVATARS_PYTHON")

    @property
    def motion_dir(self) -> Path:
        return self.data_dir / "processed" / "motions"

    @property
    def render_dir(self) -> Path:
        return self.data_dir / "processed" / "renders"


settings = Settings()

