from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def resample_motion(motion: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 3 or motion.shape[2] != 3:
        raise ValueError("motion must have shape (T, J, 3)")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS must be positive")
    target_frames = max(1, round(len(motion) * target_fps / source_fps))
    if target_frames == len(motion):
        return motion.copy()
    old_t = np.linspace(0.0, 1.0, len(motion))
    new_t = np.linspace(0.0, 1.0, target_frames)
    output = np.empty((target_frames, motion.shape[1], 3), dtype=np.float32)
    for joint in range(motion.shape[1]):
        for axis in range(3):
            output[:, joint, axis] = np.interp(new_t, old_t, motion[:, joint, axis])
    return output


def blend_motions(
    clips: Sequence[np.ndarray],
    transition_frames: int = 4,
    source_fps: Sequence[float] | None = None,
    target_fps: float = 20,
) -> np.ndarray:
    if not clips:
        return np.empty((0, 0, 3), dtype=np.float32)
    source_fps = source_fps or [target_fps] * len(clips)
    if len(source_fps) != len(clips):
        raise ValueError("source_fps must match clips")
    normalized = [resample_motion(clip, fps, target_fps) for clip, fps in zip(clips, source_fps)]
    joints = {clip.shape[1] for clip in normalized}
    if len(joints) != 1:
        raise ValueError("all clips must preserve the same joint count")
    output = normalized[0]
    for clip in normalized[1:]:
        count = min(max(0, transition_frames), len(output), len(clip))
        if count:
            alpha = np.linspace(0, 1, count + 2, dtype=np.float32)[1:-1, None, None]
            bridge = output[-1][None, :, :] * (1 - alpha) + clip[0][None, :, :] * alpha
            output = np.concatenate((output, bridge, clip), axis=0)
        else:
            output = np.concatenate((output, clip), axis=0)
    return output.astype(np.float32)

