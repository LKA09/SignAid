from __future__ import annotations

import argparse
from pathlib import Path

from signaid.datasets.mock import build_mock_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic mock emergency motions")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    written = build_mock_dataset(args.output)
    print(f"Created {len(written)} mock motion clips in {written[0].parent if written else args.output}")


if __name__ == "__main__":
    main()

