from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="SignAid recognition training entry point")
    parser.add_argument("--config", type=Path, default=Path("configs/train_transformer.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.dry_run:
        print(f"Configuration valid: {config}")
        return
    index = Path(config["data"]["index"])
    if not index.exists():
        raise SystemExit(f"Dataset index is not ready: {index}. Run filter_emergency_dataset.py first.")
    raise SystemExit("Training is intentionally gated until real AI Hub labels are inspected. Use --dry-run to validate setup.")


if __name__ == "__main__":
    main()

