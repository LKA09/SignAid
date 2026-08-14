from __future__ import annotations

import importlib
import json
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from zipfile import ZipFile

import numpy as np
import py7zr

from signaid.datasets.schema import SignSample


TEXT_KEYS = ("korean_text", "text", "sentence", "script", "ko", "한국어", "문장")
GLOSS_KEYS = ("gloss", "gloss_sequence", "glosses", "sign_gloss", "수어", "수어단어")
ID_KEYS = ("sample_id", "id", "video_id", "file_id")
SIGNER_KEYS = ("signer_id", "signer", "speaker_id", "person_id")
CATEGORY_KEYS = ("category", "topic", "class", "domain")
SEVEN_ZIP_MAGIC = b"7z\xbc\xaf'\x1c"


def _walk(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from _walk(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value[:100]):
            yield from _walk(child, f"{prefix}[{index}]")


def _first(record: dict, keys: tuple[str, ...], default: Any = "") -> Any:
    lowered = {path.rsplit(".", 1)[-1].split("[")[0].lower(): value for path, value in _walk(record)}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, "", []):
            return value
    return default


def _third_party_gloss(record: dict) -> list[str] | None:
    """Reuse the benchmark repository's gloss ordering when its deps are available."""
    repo = Path(__file__).resolve().parents[3] / "third_party" / "Korean-Disaster-Safety-Information-Sign-Language-Translation-Benchmark-Dataset-main"
    if not repo.exists():
        return None
    path = str(repo)
    try:
        if path not in sys.path:
            sys.path.insert(0, path)
        module = importlib.import_module("src.language_processor")
        processor = module.LanguageProcessor.__new__(module.LanguageProcessor)
        value = processor._create_gloss_sequence(record)  # intentional direct reuse
        return value.split() if value else None
    except Exception:
        return None


def parse_record(record: dict, fallback_id: str = "unknown") -> SignSample:
    gloss = _third_party_gloss(record)
    if not gloss:
        raw = _first(record, GLOSS_KEYS, [])
        if isinstance(raw, str):
            gloss = [item for item in raw.replace(",", " ").split() if item]
        elif isinstance(raw, list):
            gloss = [str(item.get("gloss_id", item.get("gloss", item))) if isinstance(item, dict) else str(item) for item in raw]
        else:
            gloss = []
    return SignSample(
        sample_id=str(_first(record, ID_KEYS, fallback_id)),
        text=str(_first(record, TEXT_KEYS, "")),
        gloss=gloss,
        category=str(_first(record, CATEGORY_KEYS, "unknown")),
        signer_id=str(_first(record, SIGNER_KEYS, "")),
        metadata=record,
    )


def xml_to_dict(content: bytes) -> dict:
    root = ET.fromstring(content)

    def convert(node: ET.Element) -> Any:
        children = list(node)
        if not children:
            return (node.text or "").strip()
        result: dict[str, Any] = dict(node.attrib)
        for child in children:
            value = convert(child)
            if child.tag in result:
                result[child.tag] = result[child.tag] if isinstance(result[child.tag], list) else [result[child.tag]]
                result[child.tag].append(value)
            else:
                result[child.tag] = value
        return result
    return {root.tag: convert(root)}


def iter_zip_records(zip_path: Path, limit: int | None = None) -> Iterator[tuple[str, dict]]:
    """Stream JSON/XML members without extracting the archive."""
    count = 0
    with ZipFile(zip_path) as archive:
        for info in archive.infolist():
            suffix = Path(info.filename).suffix.lower()
            if info.is_dir() or suffix not in {".json", ".xml"}:
                continue
            try:
                with archive.open(info) as stream:
                    content = stream.read()
                record = json.loads(content.decode("utf-8-sig")) if suffix == ".json" else xml_to_dict(content)
                if isinstance(record, list):
                    for index, child in enumerate(record):
                        if isinstance(child, dict):
                            yield f"{info.filename}#{index}", child
                elif isinstance(record, dict):
                    yield info.filename, record
                count += 1
                if limit is not None and count >= limit:
                    return
            except (UnicodeDecodeError, json.JSONDecodeError, ET.ParseError):
                continue


def archive_format(path: Path) -> str:
    with Path(path).open("rb") as stream:
        signature = stream.read(6)
    if signature.startswith(SEVEN_ZIP_MAGIC):
        return "7z"
    if signature.startswith(b"PK"):
        return "zip"
    return "unknown"


def list_archive_members(path: Path) -> list[dict]:
    kind = archive_format(path)
    if kind == "zip":
        with ZipFile(path) as archive:
            return [
                {"name": item.filename, "size": item.file_size, "is_directory": item.is_dir()}
                for item in archive.infolist()
            ]
    if kind == "7z":
        with py7zr.SevenZipFile(path, mode="r") as archive:
            return [
                {"name": item.filename, "size": item.uncompressed or 0, "is_directory": item.is_directory}
                for item in archive.list()
            ]
    raise ValueError(f"Unsupported archive signature: {path}")


def _excel_column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    result = 0
    for char in letters.group(0) if letters else "A":
        result = result * 26 + ord(char) - 64
    return result - 1


def iter_xlsx_rows(path: Path) -> Iterator[list[Any]]:
    """Read cell values from AI Hub XLSX using ZIP/XML streaming.

    This intentionally targets data ingestion rather than workbook authoring,
    and avoids loading 100k-row source sheets into memory.
    """
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(path) as workbook:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{namespace}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{namespace}t")))
        sheet_names = sorted(name for name in workbook.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
        if not sheet_names:
            return
        with workbook.open(sheet_names[0]) as sheet:
            for _, row in ET.iterparse(sheet, events=("end",)):
                if row.tag != f"{namespace}row":
                    continue
                values: list[Any] = []
                for cell in row.findall(f"{namespace}c"):
                    index = _excel_column_index(cell.attrib.get("r", "A1"))
                    while len(values) <= index:
                        values.append(None)
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(f"{namespace}v")
                    if cell_type == "inlineStr":
                        inline = cell.find(f"{namespace}is")
                        value: Any = "".join(node.text or "" for node in inline.iter(f"{namespace}t")) if inline is not None else ""
                    elif value_node is None:
                        value = None
                    elif cell_type == "s":
                        value = shared[int(value_node.text or 0)]
                    elif cell_type in {"str", "e"}:
                        value = value_node.text or ""
                    else:
                        raw = value_node.text or ""
                        try:
                            number = float(raw)
                            value = int(number) if number.is_integer() else number
                        except ValueError:
                            value = raw
                    values[index] = value
                while values and values[-1] is None:
                    values.pop()
                yield values
                row.clear()


def _clean_gloss(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value in {"gloss_id :", "descriptor :"}:
        return None
    return re.sub(r"\d+(?:#.*)?$", "", value).strip() or None


def _workbook_block_to_sample(rows: list[list[Any]], member_name: str) -> SignSample | None:
    if len(rows) < 2:
        return None
    file_name = str(rows[0][2]) if len(rows[0]) > 2 else Path(member_name).stem
    text = str(rows[1][2]) if len(rows[1]) > 2 else ""
    events: list[tuple[float, int, str, str]] = []
    sequence = 0
    for index, row in enumerate(rows):
        kind = str(row[0]) if row and row[0] is not None else ""
        if not kind.startswith("sign_gestures"):
            continue
        starts = rows[index + 1] if index + 1 < len(rows) else []
        for column in range(1, len(row)):
            gloss = _clean_gloss(row[column])
            if not gloss:
                continue
            start = starts[column] if column < len(starts) and isinstance(starts[column], (int, float)) else float("inf")
            events.append((float(start), sequence, gloss, str(row[column])))
            sequence += 1
    events.sort(key=lambda item: (item[0], item[1]))
    parts = Path(member_name.replace("\\", "/")).parts
    category = parts[-2] if len(parts) > 1 else "unknown"
    signer = re.search(r"_([A-Z]+\d+)\.json$", file_name, re.IGNORECASE)
    return SignSample(
        sample_id=Path(file_name).stem,
        text=text,
        gloss=[event[2] for event in events],
        category=category,
        signer_id=signer.group(1) if signer else "",
        source="AIHUB",
        metadata={"archive_member": member_name, "source_file": file_name, "raw_gloss": [event[3] for event in events]},
    )


def iter_aihub_workbook(path: Path, member_name: str = "") -> Iterator[SignSample]:
    block: list[list[Any]] = []
    for row in iter_xlsx_rows(path):
        if row and row[0] == "Information":
            if block:
                sample = _workbook_block_to_sample(block, member_name or path.name)
                if sample:
                    yield sample
            block = [row]
        elif block:
            block.append(row)
    if block:
        sample = _workbook_block_to_sample(block, member_name or path.name)
        if sample:
            yield sample


def iter_aihub_archive_records(
    archive_path: Path,
    *,
    categories: set[str] | None = None,
    workbook_limit: int | None = None,
) -> Iterator[SignSample]:
    """Read JSON/XML ZIPs or selected XLSX members from disguised 7-Zip files."""
    kind = archive_format(archive_path)
    if kind == "zip":
        members = list_archive_members(archive_path)
        xlsx_members = [item["name"] for item in members if item["name"].lower().endswith(".xlsx")]
        if not xlsx_members:
            for filename, record in iter_zip_records(archive_path):
                yield parse_record(record, Path(filename).stem)
            return
        if categories:
            xlsx_members = [name for name in xlsx_members if Path(name).parent.name.upper() in categories]
        if workbook_limit is not None:
            xlsx_members = xlsx_members[:workbook_limit]
        with tempfile.TemporaryDirectory(prefix="signaid_aihub_") as temp:
            root = Path(temp)
            with ZipFile(archive_path) as archive:
                for member in xlsx_members:
                    archive.extract(member, root)
                    yield from iter_aihub_workbook(root / member, member)
        return
    if kind != "7z":
        raise ValueError(f"Unsupported archive: {archive_path}")
    members = [item for item in list_archive_members(archive_path) if not item["is_directory"] and item["name"].lower().endswith(".xlsx")]
    if categories:
        members = [item for item in members if Path(item["name"]).parent.name.upper() in categories]
    else:
        members.sort(key=lambda item: item["size"])
    if workbook_limit is not None:
        members = members[:workbook_limit]
    targets = [item["name"] for item in members]
    if not targets:
        return
    with tempfile.TemporaryDirectory(prefix="signaid_aihub_") as temp:
        root = Path(temp)
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extract(path=root, targets=targets)
        for member in targets:
            yield from iter_aihub_workbook(root / member, member)


def detect_schema(record: dict) -> dict:
    paths = [path for path, value in _walk(record) if not isinstance(value, (dict, list))]
    return {
        "root_type": type(record).__name__,
        "field_paths": paths[:80],
        "text_fields": [p for p in paths if any(k in p.lower() for k in TEXT_KEYS)],
        "gloss_fields": [p for p in paths if any(k in p.lower() for k in GLOSS_KEYS)],
    }


def standardize_keypoints(raw: np.ndarray, label: str = "", signer_id: str = "") -> dict[str, Any]:
    points = np.asarray(raw, dtype=np.float32)
    if points.ndim == 4 and points.shape[0] == 3 and points.shape[2:] == (33, 3):
        # Benchmark output is (coordinate, time, padded_joint, left/right/body).
        benchmark = np.transpose(points, (1, 3, 2, 0))
        pose = benchmark[:, 2, :33]
        left = benchmark[:, 0, :21]
        right = benchmark[:, 1, :21]
        all_points = np.concatenate((pose, left, right), axis=1)
        return {
            "pose": pose,
            "left_hand": left,
            "right_hand": right,
            "face": np.empty((len(pose), 0, 3), dtype=np.float32),
            "mask": np.any(all_points[..., :2] != 0, axis=(1, 2)),
            "length": np.int32(len(pose)),
            "label": np.array(label),
            "signer_id": np.array(signer_id),
        }
    if points.ndim == 4 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 3 or points.shape[-1] < 2:
        raise ValueError("keypoints must resolve to shape (T, J, C)")
    if points.shape[-1] == 2:
        points = np.concatenate([points, np.ones((*points.shape[:2], 1), dtype=np.float32)], axis=-1)
    joints = points.shape[1]
    pose_end = min(33, joints)
    left_end = min(pose_end + 21, joints)
    right_end = min(left_end + 21, joints)
    return {
        "pose": points[:, :pose_end],
        "left_hand": points[:, pose_end:left_end],
        "right_hand": points[:, left_end:right_end],
        "face": points[:, right_end:],
        "mask": np.any(points[..., :2] != 0, axis=(1, 2)),
        "length": np.int32(len(points)),
        "label": np.array(label),
        "signer_id": np.array(signer_id),
    }


def save_standardized_npz(raw: np.ndarray, output_path: Path, **metadata: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **standardize_keypoints(raw, **metadata))
    return output_path
