from __future__ import annotations

import numpy as np


def add_avatar_hand_clearance(motion: np.ndarray) -> np.ndarray:
    """Add conservative camera-depth to 2D signing keypoints.

    AI Hub's XML coordinates contain no depth. Leaving every joint at z=0 makes
    the VRM forearms rotate through its torso. The small staged offsets below
    keep observed x/y trajectories intact while placing signing hands in front
    of the body.
    """
    points = np.asarray(motion, dtype=np.float32)
    if points.ndim != 3 or points.shape[1] < 59 or points.shape[2] != 3:
        raise ValueError("motion must have shape (T, 59+, 3)")
    output = points.copy()
    output[:, [3, 6], 2] = np.maximum(output[:, [3, 6], 2], 0.07)  # elbows
    output[:, [4, 7], 2] = np.maximum(output[:, [4, 7], 2], 0.19)  # wrists
    for hand_start in (17, 38):
        output[:, hand_start, 2] = np.maximum(output[:, hand_start, 2], 0.19)
        for finger_start in (1, 5, 9, 13, 17):
            for offset, depth in enumerate((0.21, 0.225, 0.235, 0.24)):
                joint = hand_start + finger_start + offset
                output[:, joint, 2] = np.maximum(output[:, joint, 2], depth)
    return output

