from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from signaid.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Index NPZ motion clips")
    parser.add_argument("--motion-dir", type=Path, default=settings.motion_dir)
    parser.add_argument("--output", type=Path, default=settings.data_dir / "index" / "motions.json")
    args = parser.parse_args()
    records = []
    for path in sorted(args.motion_dir.glob("*.npz")):
        with np.load(path) as data:
            motion = data["motion"] if "motion" in data else data[data.files[0]]
            records.append({"id": path.stem, "path": str(path), "frames": len(motion), "joints": motion.shape[1], "fps": int(data.get("fps", 20))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(records)} motions")


if __name__ == "__main__":
    main()

