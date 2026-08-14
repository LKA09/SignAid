from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from signaid.config import PROJECT_ROOT


EXCLUDED_PARTS = {
    ".codex_tmp", ".git", ".pytest_cache", ".venv", "__pycache__",
    "checkpoints", "node_modules", "release",
}
EXCLUDED_SUFFIXES = {".7z", ".gif", ".mp4", ".pyc", ".tgz", ".zip"}


def _include(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    posix = relative.as_posix()
    if posix.startswith("data/01.") or posix.startswith("data/raw/") or posix.startswith("data/filtered/"):
        return False
    if posix.startswith("data/processed/") and not posix.startswith("data/processed/aihub_motions/"):
        return False
    return True


def _build_web() -> None:
    web_root = PROJECT_ROOT / "app" / "web"
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        raise RuntimeError("Node.js/npm is required to refresh the submission web build")
    subprocess.run([npm, "run", "build"], cwd=web_root, check=True)


def main() -> None:
    cli = argparse.ArgumentParser(description="Build a compact SignAid submission ZIP without raw AI Hub archives")
    cli.add_argument("--output", type=Path, default=PROJECT_ROOT / "release" / "SignAid-submission.zip")
    cli.add_argument("--skip-web-build", action="store_true")
    args = cli.parse_args()
    if not args.skip_web_build:
        _build_web()

    files = [path for path in PROJECT_ROOT.rglob("*") if path.is_file() and _include(path)]
    raw_archives = [path for path in files if path.suffix.lower() in {".7z", ".zip"}]
    if raw_archives:
        raise RuntimeError(f"Raw archives would be included: {raw_archives}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".zip", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        with ZipFile(temp_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(files):
                archive.write(path, Path("SignAid") / path.relative_to(PROJECT_ROOT))
        os.replace(temp_path, args.output)
    finally:
        temp_path.unlink(missing_ok=True)

    derived = list((PROJECT_ROOT / "data" / "processed" / "aihub_motions").glob("*.npz"))
    print(f"Created: {args.output}")
    print(f"Package size: {args.output.stat().st_size / 1024 / 1024:.1f} MiB")
    print(f"Included compact motion assets: {len(derived)} files / {sum(p.stat().st_size for p in derived) / 1024:.1f} KiB")
    print("Raw AI Hub ZIP/7z archives included: 0")


if __name__ == "__main__":
    main()
