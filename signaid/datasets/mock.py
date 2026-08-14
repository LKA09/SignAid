from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from signaid.config import settings
from signaid.emergency.taxonomy import CONCEPTS


# 17 body joints plus MediaPipe-compatible 21 joints per hand.
BASE_SKELETON = np.array([
    [0.0, 1.80, 0.0], [0.0, 1.58, 0.0],
    [-0.24, 1.52, 0.0], [-0.48, 1.30, 0.0], [-0.62, 1.08, 0.0],
    [0.24, 1.52, 0.0], [0.48, 1.30, 0.0], [0.62, 1.08, 0.0],
    [0.0, 1.18, 0.0], [0.0, 0.95, 0.0],
    [-0.18, 0.92, 0.0], [-0.20, 0.50, 0.0], [-0.21, 0.05, 0.0],
    [0.18, 0.92, 0.0], [0.20, 0.50, 0.0], [0.21, 0.05, 0.0],
    [0.0, 1.38, 0.0],
], dtype=np.float32)

SKELETON_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 16), (16, 8), (8, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15),
    (4, 17), (7, 38),
    # Left hand: wrist, thumb, index, middle, ring, little finger.
    (17, 18), (18, 19), (19, 20), (20, 21),
    (17, 22), (22, 23), (23, 24), (24, 25),
    (17, 26), (26, 27), (27, 28), (28, 29),
    (17, 30), (30, 31), (31, 32), (32, 33),
    (17, 34), (34, 35), (35, 36), (36, 37),
    # Right hand.
    (38, 39), (39, 40), (40, 41), (41, 42),
    (38, 43), (43, 44), (44, 45), (45, 46),
    (38, 47), (47, 48), (48, 49), (49, 50),
    (38, 51), (51, 52), (52, 53), (53, 54),
    (38, 55), (55, 56), (56, 57), (57, 58),
)


def _hand_points(wrist: np.ndarray, mirror: float, mode: str, spread: float = 1.0) -> np.ndarray:
    """Build a readable 21-joint hand shape around a wrist point."""
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = wrist
    if mode == "open":
        finger_open = [1.0, 1.0, 1.0, 1.0]
        thumb_open = 1.0
    elif mode == "point":
        finger_open = [1.0, 0.12, 0.10, 0.08]
        thumb_open = 0.35
    elif mode == "phone":
        finger_open = [0.10, 0.08, 0.08, 1.0]
        thumb_open = 1.0
    elif mode == "curved":
        finger_open = [0.52, 0.52, 0.48, 0.42]
        thumb_open = 0.55
    else:  # fist
        finger_open = [0.08, 0.07, 0.06, 0.05]
        thumb_open = 0.18

    # Thumb (MediaPipe 1..4), visibly separated from the palm.
    thumb_direction = np.array([mirror * 0.78 * spread, 0.55 * thumb_open, 0.34 * (1 - thumb_open)], dtype=np.float32)
    thumb_direction /= np.linalg.norm(thumb_direction)
    points[1] = wrist + np.array([mirror * 0.035, 0.028, 0.0], dtype=np.float32)
    for joint, length in zip(range(2, 5), (0.045, 0.04, 0.035)):
        points[joint] = points[joint - 1] + thumb_direction * length

    # Four fingers (5..20). Closed fingers curl toward the camera so their
    # shape remains visible instead of collapsing to a single line.
    bases = (-0.052, -0.017, 0.020, 0.052)
    lengths = (0.060, 0.052, 0.043)
    for finger, (base_x, amount) in enumerate(zip(bases, finger_open)):
        start = 5 + finger * 4
        points[start] = wrist + np.array([mirror * base_x * spread, 0.052, 0.0], dtype=np.float32)
        direction = np.array([0.0, 0.18 + 0.82 * amount, 0.92 * (1 - amount)], dtype=np.float32)
        direction /= np.linalg.norm(direction)
        for offset, length in enumerate(lengths, start=1):
            # Curl successive phalanges more strongly for closed fingers.
            curl = np.array([0.0, -0.10 * offset * (1 - amount), 0.06 * offset * (1 - amount)], dtype=np.float32)
            step = direction + curl
            step /= np.linalg.norm(step)
            points[start + offset] = points[start + offset - 1] + step * length
    return points


def _hand_modes(concept_id: str, frame: int, frames: int) -> tuple[str, str, float]:
    phase = frame / max(1, frames - 1)
    if concept_id in {"FIRE", "BURN", "SMOKE", "EXPLOSION"}:
        return "open", "open", 1.25 + 0.15 * np.sin(phase * np.pi * 4)
    if concept_id in {"CHEST_PAIN", "PAIN", "BREATHING_DIFFICULTY"}:
        return "curved", "curved", 0.9
    if concept_id in {"EVACUATE", "STAIRS", "LEFT", "RIGHT"}:
        return "point", "open", 1.05
    if concept_id == "CALL_119":
        return "phone", "fist", 1.1
    if concept_id in {"HELP", "WAIT", "DANGER"}:
        return "open", "fist", 1.1
    return ("open", "curved", 1.0) if frame < frames // 2 else ("curved", "open", 1.0)


def generate_dummy_motion(concept_id: str, frames: int = 32, fps: int = 20) -> np.ndarray:
    """Create deterministic, visibly distinct emergency sign demo motion."""
    digest = hashlib.sha256(concept_id.encode()).digest()
    phase = digest[0] / 255 * np.pi * 2
    amplitude = 0.12 + digest[1] / 255 * 0.16
    body = np.repeat(BASE_SKELETON[None, :, :], frames, axis=0)
    t = np.linspace(0, np.pi * 2, frames, dtype=np.float32)

    # Both wrists and elbows perform concept-specific gestures in front of torso.
    body[:, 4, 0] += amplitude * np.sin(t + phase)
    body[:, 4, 1] += amplitude * np.cos(t * (1 + digest[2] % 2) + phase)
    body[:, 4, 2] += 0.18 + amplitude * np.sin(t * 0.5)
    body[:, 7, 0] -= amplitude * np.sin(t + phase)
    body[:, 7, 1] += amplitude * np.cos(t * (1 + digest[3] % 2) + phase)
    body[:, 7, 2] += 0.18 - amplitude * np.sin(t * 0.5)
    body[:, 3, :] = (body[:, 2, :] + body[:, 4, :]) / 2
    body[:, 6, :] = (body[:, 5, :] + body[:, 7, :]) / 2

    if concept_id == "CHEST_PAIN":
        chest = body[:, 16, :] + np.array([-0.08, 0.0, 0.18], dtype=np.float32)
        body[:, 4, :] = chest
        body[:, 7, :] = chest + np.array([0.16, 0.0, 0.0], dtype=np.float32)
    elif concept_id in {"EVACUATE", "STAIRS", "EVACUATION_INSTRUCTION"}:
        body[:, 4, 1] += np.linspace(0, 0.4, frames)
        body[:, 7, 1] += np.linspace(0, 0.4, frames)

    motion = np.zeros((frames, 59, 3), dtype=np.float32)
    motion[:, :17] = body
    for frame in range(frames):
        left_mode, right_mode, spread = _hand_modes(concept_id, frame, frames)
        motion[frame, 17:38] = _hand_points(body[frame, 4], -1.0, left_mode, spread)
        motion[frame, 38:59] = _hand_points(body[frame, 7], 1.0, right_mode, spread)
    return motion.astype(np.float32)


def build_mock_dataset(output_dir: Path | None = None) -> list[Path]:
    output_dir = output_dir or settings.motion_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for concept in CONCEPTS.values():
        path = output_dir / f"{concept.id}.npz"
        np.savez_compressed(
            path,
            motion=generate_dummy_motion(concept.id),
            fps=np.int32(settings.fps),
            mock=np.bool_(True),
            motion_source=np.array("procedural_hand_demo"),
            linguistically_validated=np.bool_(False),
        )
        written.append(path)
    index = output_dir.parent.parent / "index" / "emergency_signs.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps([c.to_dict() for c in CONCEPTS.values()], ensure_ascii=False, indent=2), encoding="utf-8")
    return written
