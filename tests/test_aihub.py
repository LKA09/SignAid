import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import py7zr

from signaid.datasets.aihub import archive_format, detect_schema, iter_aihub_archive_records, iter_aihub_workbook, iter_zip_records, parse_record, standardize_keypoints, xml_to_dict
from signaid.datasets.aihub_motion import (
    landmark_json_to_avatar_motion,
    timed_glosses_from_landmark_json,
    timed_samples_from_rows,
    xml_keypoints_to_motion,
)
from scripts.filter_emergency_dataset import filter_zip
from scripts.inspect_aihub_dataset import inspect


def make_zip(path: Path) -> Path:
    record = {
        "metadata": {"id": "sample-1", "signer_id": "signer-7", "category": "medical"},
        "korean_text": "가슴 통증으로 응급 도움이 필요합니다",
        "gloss_sequence": "가슴 아프다",
    }
    with ZipFile(path, "w") as archive:
        archive.writestr("nested/sample.json", json.dumps(record, ensure_ascii=False))
        archive.writestr("nested/sample.xml", "<record><id>sample-2</id><text>화재 대피</text><gloss>불 대피</gloss></record>")
        archive.writestr("video/sample.mp4", b"not-a-real-video")
    return path


def test_zip_streaming_and_schema(tmp_path: Path):
    archive = make_zip(tmp_path / "sample.zip")
    records = list(iter_zip_records(archive))
    assert len(records) == 2
    first = parse_record(records[0][1])
    assert first.sample_id == "sample-1"
    assert first.gloss == ["가슴", "아프다"]
    assert detect_schema(records[0][1])["text_fields"]


def test_inspector_does_not_extract(tmp_path: Path):
    archive = make_zip(tmp_path / "sample.zip")
    report = inspect(archive, samples=2)
    assert report["total_files"] == 3
    assert report["structured_files"] == 2
    assert report["sample_ids"] == ["sample-1", "sample-2"]
    assert not (tmp_path / "nested").exists()


def test_emergency_filter(tmp_path: Path):
    archive = make_zip(tmp_path / "sample.zip")
    selected = list(filter_zip(archive, ["응급", "화재"]))
    assert {sample.sample_id for sample in selected} == {"sample-1", "sample-2"}


def test_defensive_xml_parser():
    parsed = xml_to_dict(b"<root><item id='1'>hello</item><item id='2'>world</item></root>")
    assert parsed["root"]["item"] == ["hello", "world"]


def test_benchmark_keypoint_shape_is_standardized():
    raw = np.zeros((3, 5, 33, 3), dtype=np.float32)
    raw[0, :, :, :] = 1.0
    result = standardize_keypoints(raw, label="HELP", signer_id="s1")
    assert result["pose"].shape == (5, 33, 3)
    assert result["left_hand"].shape == (5, 21, 3)
    assert result["right_hand"].shape == (5, 21, 3)
    assert result["length"] == 5


def make_aihub_xlsx(path: Path) -> Path:
    sheet = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
      <row r="1"><c r="A1" t="inlineStr"><is><t>Information</t></is></c><c r="B1" t="inlineStr"><is><t>File name : </t></is></c><c r="C1" t="inlineStr"><is><t>NIA_SL_G2_FIRE000001_1_CROWD7.json</t></is></c></row>
      <row r="2"><c r="B2" t="inlineStr"><is><t>Korean sentence : </t></is></c><c r="C2" t="inlineStr"><is><t>화재가 발생하여 계단으로 대피하세요</t></is></c></row>
      <row r="3"><c r="A3" t="inlineStr"><is><t>sign_gestures_both</t></is></c><c r="B3" t="inlineStr"><is><t>gloss_id :</t></is></c><c r="C3" t="inlineStr"><is><t>불1</t></is></c><c r="D3" t="inlineStr"><is><t>대피1</t></is></c></row>
      <row r="4"><c r="B4" t="inlineStr"><is><t>start(s) :</t></is></c><c r="C4"><v>1.5</v></c><c r="D4"><v>2.5</v></c></row>
    </sheetData></worksheet>"""
    with ZipFile(path, "w") as workbook:
        workbook.writestr("xl/worksheets/sheet1.xml", sheet)
    return path


def test_actual_aihub_workbook_block_parser(tmp_path: Path):
    workbook = make_aihub_xlsx(tmp_path / "FIRE.xlsx")
    sample = list(iter_aihub_workbook(workbook, "root/FIRE/FIRE.xlsx"))[0]
    assert sample.sample_id == "NIA_SL_G2_FIRE000001_1_CROWD7"
    assert sample.gloss == ["불", "대피"]
    assert sample.signer_id == "CROWD7"
    assert sample.category == "FIRE"


def test_disguised_7z_archive_is_supported(tmp_path: Path):
    workbook = make_aihub_xlsx(tmp_path / "FIRE.xlsx")
    archive_path = tmp_path / "script_TL.zip"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(workbook, arcname="script/2.untact_morpheme/FIRE/FIRE.xlsx")
    assert archive_format(archive_path) == "7z"
    samples = list(iter_aihub_archive_records(archive_path, categories={"FIRE"}))
    assert len(samples) == 1
    assert samples[0].text.startswith("화재")


def test_timed_gloss_rows_are_parsed():
    rows = [
        ["Information", "File name : ", "sample-1.json"],
        ["sign_gestures_both", "gloss_id : ", "지진1"],
        [None, "start(s) : ", 1.25],
        [None, "end(s) : ", 1.8],
    ]
    event = timed_samples_from_rows(rows)["sample-1"][0]
    assert event.gloss == "지진"
    assert event.start == 1.25
    assert event.end == 1.8


def test_aihub_xml_converts_to_59_joint_motion(tmp_path: Path):
    def points(count: int, offset: float) -> str:
        return ";".join(f"{offset + index * 3},{100 + index * 2},0.9" for index in range(count)) + ";"

    tracks = []
    track_id = 0
    for frame in range(2):
        for label, tag, count in (
            ("pose_keypoints_2d", "body", 25),
            ("hand_left_keypoints_2d", "leftHand", 21),
            ("hand_right_keypoints_2d", "rightHand", 21),
        ):
            tracks.append(
                f'<track id="{track_id}" label="{label}"><{tag} frame="{frame}" outside="0" '
                f'points="{points(count, frame + track_id)}"/></track>'
            )
            track_id += 1
    path = tmp_path / "sample.xml"
    path.write_text(f"<annotations>{''.join(tracks)}</annotations>", encoding="utf-8")
    motion, fps = xml_keypoints_to_motion(path)
    assert motion.shape == (2, 59, 3)
    assert fps == 30
    assert np.isfinite(motion).all()
    assert np.allclose(motion[..., 2], 0)


def test_aihub_3d_json_preserves_depth_hands_face_and_timing():
    frames = 8
    pose = np.zeros((frames, 25, 3), dtype=np.float32)
    pose[:] = np.array([0, 0, 2500], dtype=np.float32)
    pose[:, 2] = [-150, 0, 2500]
    pose[:, 5] = [150, 0, 2500]
    pose[:, 4] = [-260, 180, 2350]
    pose[:, 7] = [260, 180, 2650]
    pose[:, 8] = [0, 400, 2500]
    pose[:, 1] = [0, 0, 2500]
    pose[:, 0] = [0, -160, 2480]
    pose[:, 4, 2] -= np.linspace(0, 180, frames)

    hand_template = np.stack([
        np.array([(joint % 5) * 22, -(joint // 5) * 24, joint * 3], dtype=np.float32)
        for joint in range(21)
    ])
    right = np.stack([hand_template + pose[index, 4] for index in range(frames)])
    left = np.stack([hand_template * np.array([-1, 1, 1]) + pose[index, 7] for index in range(frames)])
    face_template = np.stack([
        np.array([(joint % 10) * 12 - 54, (joint // 10) * 11 - 140, 2520 + joint % 3], dtype=np.float32)
        for joint in range(70)
    ])
    face = np.stack([face_template.copy() for _ in range(frames)])
    record = {
        "metadata": {"video_fps": 30},
        "landmarks": {
            "pose_keypoints_3d": pose.reshape(frames, -1).tolist(),
            "hand_left_keypoints_3d": left.reshape(frames, -1).tolist(),
            "hand_right_keypoints_3d": right.reshape(frames, -1).tolist(),
            "face_keypoints_3d": face.reshape(frames, -1).tolist(),
        },
        "sign_script": {
            "sign_gestures_both": [{"gloss_id": "불1", "start": 0.1, "end": 0.2}],
            "sign_gestures_strong": [],
            "sign_gestures_weak": [],
        },
    }
    converted = landmark_json_to_avatar_motion(record)
    assert converted.motion.shape == (frames, 59, 3)
    assert np.ptp(converted.motion[..., 2]) > 0.1
    assert converted.palm_normals.shape == (frames, 2, 3)
    assert np.allclose(np.linalg.norm(converted.palm_normals, axis=-1), 1, atol=1e-4)
    assert converted.facial_expressions.shape == (frames, 4)
    assert converted.head_rotations.shape == (frames, 4)
    assert np.isfinite(converted.motion).all()
    assert timed_glosses_from_landmark_json(record)[0].gloss == "불"
