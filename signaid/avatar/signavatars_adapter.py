from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from signaid.avatar.skeleton_renderer import SkeletonRenderer
from signaid.config import settings


class SignAvatarsAdapter:
    """Isolated boundary for the legacy SignAvatars SMPL-X renderer.

    SignAvatars consumes its native 182-parameter SMPL-X pickle, not SignAid's
    lightweight joint NPZ. Skeleton input therefore falls back by design.
    """

    def __init__(self, python_executable: str | None = None, fallback: SkeletonRenderer | None = None) -> None:
        self.python_executable = python_executable or settings.signavatars_python
        self.fallback = fallback or SkeletonRenderer(settings.fps)
        self.repo = Path(__file__).resolve().parents[3] / "third_party" / "SignAvatars-main"
        self.last_error: str | None = None

    def availability(self) -> dict:
        if not self.repo.exists():
            return {"available": False, "reason": "SignAvatars repository not found"}
        if not self.python_executable:
            return {"available": False, "reason": "SIGNAVATARS_PYTHON is not configured"}
        try:
            check = subprocess.run(
                [self.python_executable, "-c", "import torch,pyrender,smplx; print(torch.cuda.is_available())"],
                capture_output=True, text=True, timeout=15,
            )
            if check.returncode != 0:
                return {"available": False, "reason": check.stderr.strip()[-300:]}
            if check.stdout.strip().lower() != "true":
                return {"available": False, "reason": "SignAvatars vis.py requires CUDA"}
            return {"available": True, "reason": "legacy CUDA environment detected"}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"available": False, "reason": str(exc)}

    def render(self, motion_path: Path, output_path: Path) -> Path:
        motion_path, output_path = Path(motion_path), Path(output_path)
        status = self.availability()
        if status["available"] and motion_path.suffix.lower() == ".pkl":
            try:
                result = subprocess.run(
                    [self.python_executable, str(self.repo / "vis.py"), "--pkl_file_path", str(motion_path)],
                    cwd=self.repo,
                    capture_output=True,
                    text=True,
                    timeout=900,
                    env={**os.environ, "PYOPENGL_PLATFORM": "egl"},
                )
                generated = self.repo / "render_results" / f"{motion_path.stem}.mp4"
                if result.returncode == 0 and generated.exists():
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(generated, output_path)
                    return output_path
                self.last_error = (result.stderr or result.stdout)[-1000:]
            except (OSError, subprocess.SubprocessError) as exc:
                self.last_error = str(exc)
        else:
            self.last_error = status["reason"] if motion_path.suffix.lower() == ".pkl" else "Skeleton NPZ has no SMPL-X parameters"
        return self.fallback.render(motion_path, output_path)

