from pathlib import Path

import numpy as np

from signaid.avatar.skeleton_renderer import SkeletonRenderer
from signaid.datasets.mock import build_mock_dataset, generate_dummy_motion
from signaid.motion.blender import blend_motions, resample_motion
from signaid.motion.constraints import add_avatar_hand_clearance
from signaid.motion.retriever import MotionRetriever


def test_dummy_motion_generation():
    motion = generate_dummy_motion("CHEST_PAIN", frames=12)
    assert motion.shape == (12, 59, 3)
    assert np.isfinite(motion).all()
    assert not np.allclose(motion[0], motion[5])
    assert np.ptp(motion[0, 17:38, 0]) > 0.08  # left-hand fingers are separated
    assert np.ptp(motion[0, 38:59, 0]) > 0.08  # right-hand fingers are separated


def test_motion_retrieval_exact_alias_fuzzy_and_missing(tmp_path: Path):
    build_mock_dataset(tmp_path)
    result = MotionRetriever(tmp_path).retrieve(["가슴", "흉통", "아푸다", "존재하지않는동작"])
    assert result.matches[0].concept_id == "CHEST_PAIN"
    assert result.matches[1].concept_id == "CHEST_PAIN"
    assert result.matches[2].path is not None
    assert "존재하지않는동작" in result.missing
    assert 0 < result.confidence < 1


def test_real_aihub_motion_is_preferred_over_demo(tmp_path: Path):
    mock_dir = tmp_path / "motions"
    real_dir = tmp_path / "aihub_motions"
    build_mock_dataset(mock_dir)
    real_dir.mkdir()
    real_motion = np.full((6, 59, 3), 7, dtype=np.float32)
    np.savez_compressed(real_dir / "FIRE.npz", motion=real_motion, fps=30, mock=False)
    match = MotionRetriever(mock_dir).retrieve(["불"]).matches[0]
    assert match.path == real_dir / "FIRE.npz"


def test_2d_aihub_hands_are_kept_in_front_of_avatar():
    motion = np.zeros((4, 59, 3), dtype=np.float32)
    constrained = add_avatar_hand_clearance(motion)
    assert np.all(constrained[:, [3, 6], 2] == 0.07)
    assert np.all(constrained[:, [4, 7, 17, 38], 2] == 0.19)
    assert np.all(constrained[:, [20, 41], 2] >= 0.235)
    assert np.allclose(constrained[..., :2], motion[..., :2])


def test_blending_and_fps_normalization():
    first = np.zeros((10, 4, 3), dtype=np.float32)
    second = np.ones((10, 4, 3), dtype=np.float32)
    blended = blend_motions([first, second], transition_frames=3)
    assert blended.shape == (23, 4, 3)
    assert np.allclose(blended[10], 0.25)
    assert np.allclose(blended[12], 0.75)
    assert resample_motion(first, 10, 20).shape[0] == 20


def test_blending_rejects_joint_mismatch():
    try:
        blend_motions([np.zeros((2, 4, 3)), np.zeros((2, 5, 3))])
    except ValueError as exc:
        assert "joint count" in str(exc)
    else:
        raise AssertionError("joint mismatch must fail")


def test_skeleton_gif_render(tmp_path: Path):
    motion = generate_dummy_motion("HELP", frames=3)
    output = SkeletonRenderer(fps=5).render_array(motion, tmp_path / "preview.gif", fps=5)
    assert output.exists()
    assert output.stat().st_size > 100
