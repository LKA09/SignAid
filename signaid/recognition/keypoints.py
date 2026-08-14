from __future__ import annotations

import numpy as np


def normalize_keypoints(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim == 2:
        points = points[None, ...]
    if points.ndim != 3 or points.shape[-1] not in (2, 3):
        raise ValueError("keypoints must have shape (T, J, 2|3)")
    mask = np.any(points[..., :2] != 0, axis=(1, 2))
    root = points[:, :1, :2]
    xy = points[..., :2] - root
    scale = np.linalg.norm(xy, axis=-1).max(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    xy /= scale[..., None]
    if points.shape[-1] == 3:
        output = np.concatenate((xy, points[..., 2:3]), axis=-1)
    else:
        output = xy
    return output.astype(np.float32), mask


class WebcamKeypointExtractor:
    def __init__(self, mock_when_unavailable: bool = True) -> None:
        self.mock_when_unavailable = mock_when_unavailable
        try:
            import mediapipe as mp
            if not hasattr(mp, "solutions"):
                raise ImportError("MediaPipe solutions API unavailable")
            self.available = True
        except ImportError:
            self.available = False

    def extract(self, frames: list[np.ndarray]) -> np.ndarray:
        if not self.available:
            if not self.mock_when_unavailable:
                raise RuntimeError("MediaPipe is not installed")
            mock = np.zeros((len(frames), 75, 3), dtype=np.float32)
            if frames:
                mock[..., 2] = 1.0
            return mock
        import mediapipe as mp
        output = np.zeros((len(frames), 75, 3), dtype=np.float32)
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            for frame_index, frame in enumerate(frames):
                result = holistic.process(np.asarray(frame))
                groups = (
                    (result.pose_landmarks, 0, 33),
                    (result.left_hand_landmarks, 33, 21),
                    (result.right_hand_landmarks, 54, 21),
                )
                for landmarks, offset, count in groups:
                    if landmarks is None:
                        continue
                    for joint, point in enumerate(landmarks.landmark[:count]):
                        output[frame_index, offset + joint] = (
                            point.x, point.y, getattr(point, "visibility", 1.0)
                        )
        return output
