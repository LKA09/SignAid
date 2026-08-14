from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class TimedGloss:
    gloss: str
    start: float
    end: float
    raw: str


@dataclass(frozen=True, slots=True)
class AvatarMotion3D:
    """Compact avatar-ready representation derived from AI Hub 3D landmarks."""

    motion: np.ndarray
    palm_normals: np.ndarray
    facial_expressions: np.ndarray
    head_rotations: np.ndarray
    fps: int
    tracking_quality: float


FACIAL_EXPRESSION_NAMES = ("mouth_open", "blink_left", "blink_right", "brow_raise")


def _clean_gloss(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value in {"gloss_id :", "descriptor :"}:
        return None
    return re.sub(r"\d+(?:#.*)?$", "", value).strip() or None


def timed_samples_from_rows(rows: Iterable[list[Any]]) -> dict[str, list[TimedGloss]]:
    """Read sample IDs and annotated gloss time spans from one AI Hub workbook."""
    materialized = list(rows)
    starts = [index for index, row in enumerate(materialized) if row and row[0] == "Information"]
    result: dict[str, list[TimedGloss]] = {}
    for block_index, start in enumerate(starts):
        stop = starts[block_index + 1] if block_index + 1 < len(starts) else len(materialized)
        block = materialized[start:stop]
        if not block or len(block[0]) < 3:
            continue
        sample_id = Path(str(block[0][2])).stem
        events: list[TimedGloss] = []
        for row_index, row in enumerate(block):
            if not row or not str(row[0]).startswith("sign_gestures"):
                continue
            start_row = block[row_index + 1] if row_index + 1 < len(block) else []
            end_row = block[row_index + 2] if row_index + 2 < len(block) else []
            for column in range(2, len(row)):
                gloss = _clean_gloss(row[column])
                if not gloss or column >= len(start_row) or not isinstance(start_row[column], (int, float)):
                    continue
                start_second = float(start_row[column])
                end_second = None
                for candidate_column in (column, column + 1):
                    if candidate_column < len(end_row) and isinstance(end_row[candidate_column], (int, float)):
                        end_second = float(end_row[candidate_column])
                        break
                if end_second is None or end_second <= start_second:
                    continue
                events.append(TimedGloss(gloss, start_second, end_second, str(row[column])))
        if events:
            result[sample_id] = events
    return result


def timed_glosses_from_landmark_json(record: dict) -> list[TimedGloss]:
    """Read aligned gloss spans directly from an AI Hub landmark JSON record."""
    events: list[TimedGloss] = []
    seen: set[tuple[str, float, float]] = set()
    sign_script = record.get("sign_script", {})
    if not isinstance(sign_script, dict):
        return events
    for group in ("sign_gestures_both", "sign_gestures_strong", "sign_gestures_weak"):
        for item in sign_script.get(group, []) or []:
            if not isinstance(item, dict):
                continue
            gloss = _clean_gloss(item.get("gloss_id"))
            try:
                start, end = float(item.get("start")), float(item.get("end"))
            except (TypeError, ValueError):
                continue
            key = (gloss or "", start, end)
            if not gloss or end <= start or key in seen:
                continue
            seen.add(key)
            events.append(TimedGloss(gloss, start, end, str(item.get("gloss_id", ""))))
    return sorted(events, key=lambda item: (item.start, item.end))


def _landmark_array(record: dict, key: str, joints: int) -> np.ndarray:
    raw = record.get("landmarks", {}).get(key, [])
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != joints * 3:
        raise ValueError(f"{key} must have shape (T, {joints * 3})")
    return values.reshape(len(values), joints, 3)


def _repair_landmarks(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Reject triangulation spikes, interpolate gaps, and lightly smooth tracks."""
    values = np.asarray(points, dtype=np.float32).copy()
    finite = np.isfinite(values).all(axis=-1)
    nonzero = np.any(values != 0, axis=-1)
    safe = np.where((finite & nonzero)[..., None], values, np.nan)
    median = np.nanmedian(safe, axis=0)
    deviation = np.abs(safe - median[None, ...])
    mad = np.nanmedian(deviation, axis=0)
    # Millimetre-space floors retain intentional signing motion while removing
    # the occasional multi-metre triangulation spike found in the source data.
    floor = np.array([250.0, 250.0, 500.0], dtype=np.float32)
    threshold = np.maximum(10.0 * 1.4826 * np.nan_to_num(mad, nan=0.0), floor)
    plausible = np.all(deviation <= threshold[None, ...], axis=-1)
    valid = finite & nonzero & plausible
    quality = float(valid.mean())

    timeline = np.arange(len(values), dtype=np.float32)
    for joint in range(values.shape[1]):
        joint_valid = valid[:, joint]
        for axis in range(3):
            if joint_valid.any():
                values[:, joint, axis] = np.interp(
                    timeline,
                    timeline[joint_valid],
                    values[joint_valid, joint, axis],
                )
            else:
                values[:, joint, axis] = float(np.nan_to_num(median[joint, axis]))

    if len(values) >= 3:
        padded = np.pad(values, ((2, 2), (0, 0), (0, 0)), mode="edge")
        weights = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        values = sum(padded[offset:offset + len(values)] * weight for offset, weight in enumerate(weights)) / weights.sum()
    return values.astype(np.float32), quality


def _unit(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-6)


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrices to xyzw quaternions without SciPy."""
    output = np.empty((len(matrix), 4), dtype=np.float32)
    for index, value in enumerate(matrix):
        trace = float(np.trace(value))
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2
            quat = np.array([
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
                0.25 * scale,
            ])
        else:
            axis = int(np.argmax(np.diag(value)))
            if axis == 0:
                scale = np.sqrt(max(1e-8, 1.0 + value[0, 0] - value[1, 1] - value[2, 2])) * 2
                quat = np.array([0.25 * scale, (value[0, 1] + value[1, 0]) / scale, (value[0, 2] + value[2, 0]) / scale, (value[2, 1] - value[1, 2]) / scale])
            elif axis == 1:
                scale = np.sqrt(max(1e-8, 1.0 + value[1, 1] - value[0, 0] - value[2, 2])) * 2
                quat = np.array([(value[0, 1] + value[1, 0]) / scale, 0.25 * scale, (value[1, 2] + value[2, 1]) / scale, (value[0, 2] - value[2, 0]) / scale])
            else:
                scale = np.sqrt(max(1e-8, 1.0 + value[2, 2] - value[0, 0] - value[1, 1])) * 2
                quat = np.array([(value[0, 2] + value[2, 0]) / scale, (value[1, 2] + value[2, 1]) / scale, 0.25 * scale, (value[1, 0] - value[0, 1]) / scale])
        output[index] = quat / max(np.linalg.norm(quat), 1e-6)
    for index in range(1, len(output)):
        if np.dot(output[index - 1], output[index]) < 0:
            output[index] *= -1
    return output


def _facial_features(face: np.ndarray) -> np.ndarray:
    def distance(first: int, second: int) -> np.ndarray:
        return np.linalg.norm(face[:, first] - face[:, second], axis=-1)

    face_width = np.maximum(distance(0, 16), 1e-6)
    mouth_width = np.maximum(distance(48, 54), 1e-6)
    mouth_ratio = (distance(62, 66) + distance(63, 65)) * 0.5 / mouth_width
    mouth_baseline = float(np.quantile(mouth_ratio, 0.15))
    mouth_open = np.clip((mouth_ratio - mouth_baseline - 0.015) / 0.24, 0, 1)

    right_eye = (distance(37, 41) + distance(38, 40)) * 0.5 / np.maximum(distance(36, 39), 1e-6)
    left_eye = (distance(43, 47) + distance(44, 46)) * 0.5 / np.maximum(distance(42, 45), 1e-6)

    def blink(openness: np.ndarray) -> np.ndarray:
        open_level = max(float(np.quantile(openness, 0.8)), 1e-4)
        return np.clip((open_level * 0.72 - openness) / (open_level * 0.42), 0, 1)

    brow_distance = (
        np.linalg.norm(face[:, 19] - face[:, 37], axis=-1)
        + np.linalg.norm(face[:, 24] - face[:, 44], axis=-1)
    ) * 0.5 / face_width
    brow_baseline = max(float(np.quantile(brow_distance, 0.4)), 1e-4)
    brow_raise = np.clip((brow_distance / brow_baseline - 1.03) * 3.5, 0, 1)
    return np.stack((mouth_open, blink(left_eye), blink(right_eye), brow_raise), axis=1).astype(np.float32)


def landmark_json_to_avatar_motion(record: dict) -> AvatarMotion3D:
    """Convert AI Hub 3D body, hand, and face landmarks into avatar space."""
    pose, pose_quality = _repair_landmarks(_landmark_array(record, "pose_keypoints_3d", 25))
    left_hand, left_quality = _repair_landmarks(_landmark_array(record, "hand_left_keypoints_3d", 21))
    right_hand, right_quality = _repair_landmarks(_landmark_array(record, "hand_right_keypoints_3d", 21))
    face, face_quality = _repair_landmarks(_landmark_array(record, "face_keypoints_3d", 70))
    frame_count = min(len(pose), len(left_hand), len(right_hand), len(face))
    pose, left_hand, right_hand, face = (value[:frame_count] for value in (pose, left_hand, right_hand, face))

    # Join hand roots to the body wrists to avoid visible gaps caused by the
    # independent hand/body triangulators.
    right_hand += pose[:, 4:5] - right_hand[:, :1]
    left_hand += pose[:, 7:8] - left_hand[:, :1]

    shoulder_width = np.linalg.norm(pose[:, 2] - pose[:, 5], axis=-1)
    valid_width = shoulder_width[(shoulder_width > 80) & (shoulder_width < 800)]
    source_width = float(np.median(valid_width)) if len(valid_width) else 300.0
    scale = 0.48 / source_width
    center = pose[:, 8:9]

    def avatar_space(points: np.ndarray) -> np.ndarray:
        result = (points - center) * scale
        result[..., 1] *= -1
        result[..., 1] += 0.95
        result[..., 2] *= -1
        return result.astype(np.float32)

    body_avatar = avatar_space(pose)
    right_avatar = avatar_space(right_hand)
    left_avatar = avatar_space(left_hand)
    face_avatar = avatar_space(face)
    motion = np.zeros((frame_count, 59, 3), dtype=np.float32)
    motion[:, :16] = body_avatar[:, :16]
    motion[:, 16] = (body_avatar[:, 1] + body_avatar[:, 8]) / 2
    motion[:, 17:38] = right_avatar
    motion[:, 38:59] = left_avatar

    def palm_normal(hand: np.ndarray) -> np.ndarray:
        primary = _unit(hand[:, 9] - hand[:, 0])
        across = hand[:, 17] - hand[:, 5]
        across -= primary * np.sum(across * primary, axis=-1, keepdims=True)
        return _unit(np.cross(_unit(across), primary))

    palm_normals = np.stack((palm_normal(right_avatar), palm_normal(left_avatar)), axis=1).astype(np.float32)

    right_axis = _unit(face_avatar[:, 45] - face_avatar[:, 36])
    up_axis = _unit(face_avatar[:, 27] - face_avatar[:, 8])
    forward_axis = _unit(np.cross(right_axis, up_axis))
    up_axis = _unit(np.cross(forward_axis, right_axis))
    face_basis = np.stack((right_axis, up_axis, forward_axis), axis=-1)
    reference = np.median(face_basis, axis=0)
    u, _, vh = np.linalg.svd(reference)
    reference = u @ vh
    relative_basis = face_basis @ reference.T
    head_rotations = _rotation_matrix_to_quaternion(relative_basis)

    fps = int(record.get("metadata", {}).get("video_fps", 30) or 30)
    quality = float(np.mean((pose_quality, left_quality, right_quality, face_quality)))
    return AvatarMotion3D(
        motion=motion,
        palm_normals=palm_normals,
        facial_expressions=_facial_features(face),
        head_rotations=head_rotations,
        fps=fps,
        tracking_quality=quality,
    )


def _parse_points(value: str, expected: int) -> np.ndarray:
    points = []
    for item in value.rstrip(";").split(";"):
        parts = item.split(",")
        if len(parts) >= 2:
            points.append((float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 else 1.0))
    array = np.asarray(points, dtype=np.float32)
    if array.shape != (expected, 3):
        raise ValueError(f"Expected {expected} keypoints, received {array.shape}")
    return array


def xml_keypoints_to_motion(path: Path) -> tuple[np.ndarray, int]:
    """Convert AI Hub OpenPose-style 2D XML into SignAid's 59-joint layout."""
    root = ET.parse(path).getroot()
    frame_data: dict[str, dict[int, np.ndarray]] = {"body": {}, "leftHand": {}, "rightHand": {}}
    expected = {"body": 25, "leftHand": 21, "rightHand": 21}
    for tag, target in frame_data.items():
        for element in root.iter(tag):
            if element.attrib.get("outside", "0") != "0":
                continue
            frame = int(element.attrib["frame"])
            target[frame] = _parse_points(element.attrib.get("points", ""), expected[tag])

    frames = sorted(set(frame_data["body"]) & set(frame_data["leftHand"]) & set(frame_data["rightHand"]))
    if len(frames) < 2:
        raise ValueError(f"No complete motion frames in {path.name}")
    body = np.stack([frame_data["body"][frame] for frame in frames])
    left_hand = np.stack([frame_data["leftHand"][frame] for frame in frames])
    right_hand = np.stack([frame_data["rightHand"][frame] for frame in frames])

    shoulder_widths = np.linalg.norm(body[:, 2, :2] - body[:, 5, :2], axis=1)
    valid_widths = shoulder_widths[shoulder_widths > 1]
    source_width = float(np.median(valid_widths)) if len(valid_widths) else 300.0
    scale = 0.48 / source_width
    center = body[:, 8, :2]

    def normalize(points: np.ndarray) -> np.ndarray:
        output = np.zeros((*points.shape[:2], 3), dtype=np.float32)
        output[..., 0] = (points[..., 0] - center[:, None, 0]) * scale
        output[..., 1] = 0.95 - (points[..., 1] - center[:, None, 1]) * scale
        return output

    normalized_body = normalize(body)
    normalized_left = normalize(left_hand)
    normalized_right = normalize(right_hand)
    motion = np.zeros((len(frames), 59, 3), dtype=np.float32)
    motion[:, :16] = normalized_body[:, :16]
    motion[:, 16] = (normalized_body[:, 1] + normalized_body[:, 8]) / 2
    # OpenPose right hand belongs to body wrist 4; left hand belongs to wrist 7.
    motion[:, 17:38] = normalized_right
    motion[:, 38:59] = normalized_left
    return motion, 30
