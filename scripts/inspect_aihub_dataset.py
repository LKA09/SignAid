from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import islice

from signaid.datasets.aihub import archive_format, iter_aihub_archive_records, list_archive_members


def inspect(zip_path: Path, samples: int = 3, categories: set[str] | None = None) -> dict:
    members = list_archive_members(zip_path)
    names = [item["name"] for item in members if not item["is_directory"]]
    parsed = []
    iterator = iter_aihub_archive_records(zip_path, categories=categories, workbook_limit=1 if archive_format(zip_path) == "7z" else None)
    for sample in islice(iterator, samples):
        parsed.append({
            "filename": sample.metadata.get("archive_member", sample.sample_id),
            "sample": sample.to_dict(),
            "raw_gloss": sample.metadata.get("raw_gloss", []),
        })
    return {
        "zip": str(zip_path),
        "archive_format": archive_format(zip_path),
        "total_files": len(names),
        "structured_files": sum(Path(name).suffix.lower() in {".json", ".xml", ".xlsx"} for name in names),
        "sample_file_names": names[:20],
        "samples": parsed,
        "detected_schema": {
            "sample_id": "Information / File name",
            "text": "Information / Korean sentence",
            "gloss": "sign_gestures_both|strong|weak ordered by start(s)",
            "category": "archive member parent directory",
            "signer_id": "CROWD identifier in file name",
        },
        "signer_ids": sorted({item["sample"]["signer_id"] for item in parsed if item["sample"]["signer_id"]}),
        "categories": sorted({item["sample"]["category"] for item in parsed if item["sample"]["category"] != "unknown"}),
        "sample_ids": [item["sample"]["sample_id"] for item in parsed],
    }


def main() -> None:
    cli = argparse.ArgumentParser(description="Stream-inspect AI Hub JSON/XML ZIP files")
    cli.add_argument("--zip", required=True, type=Path)
    cli.add_argument("--samples", type=int, default=3)
    cli.add_argument("--category", action="append", help="Optional archive category such as FIRE or FIRSTAID")
    args = cli.parse_args()
    if not args.zip.exists():
        cli.error(f"ZIP not found: {args.zip}")
    categories = {item.upper() for item in args.category} if args.category else None
    print(json.dumps(inspect(args.zip, args.samples, categories), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
