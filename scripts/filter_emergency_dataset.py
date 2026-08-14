from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from signaid.config import PROJECT_ROOT
from signaid.datasets.aihub import iter_aihub_archive_records, iter_zip_records, parse_record


def load_keywords(path: Path) -> list[str]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [str(item).lower() for item in value.get("keywords", value if isinstance(value, list) else [])]


def load_archive_categories(path: Path) -> set[str] | None:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    categories = value.get("archive_categories", []) if isinstance(value, dict) else []
    return {str(item).upper() for item in categories} or None


def filter_zip(zip_path: Path, keywords: list[str]):
    for filename, record in iter_zip_records(zip_path):
        sample = parse_record(record, Path(filename).stem)
        haystack = " ".join((sample.text, *sample.gloss, sample.category)).lower()
        if any(keyword in haystack for keyword in keywords):
            yield sample


def filter_archive(archive_path: Path, keywords: list[str], categories: set[str] | None = None):
    for sample in iter_aihub_archive_records(archive_path, categories=categories):
        haystack = " ".join((sample.text, *sample.gloss, sample.category)).lower()
        if any(keyword in haystack for keyword in keywords):
            yield sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an emergency-only AI Hub JSONL index without extracting ZIPs")
    parser.add_argument("--zip", type=Path, action="append", required=True, dest="zips")
    parser.add_argument("--keywords", type=Path, default=PROJECT_ROOT / "configs" / "emergency_keywords.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "index" / "emergency_samples.jsonl")
    args = parser.parse_args()
    keywords = load_keywords(args.keywords)
    categories = load_archive_categories(args.keywords)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with args.output.open("w", encoding="utf-8") as target:
        for zip_path in args.zips:
            for sample in filter_archive(zip_path, keywords, categories):
                if sample.sample_id in seen:
                    continue
                seen.add(sample.sample_id)
                target.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                count += 1
    print(f"Wrote {count} emergency samples to {args.output}")


if __name__ == "__main__":
    main()
