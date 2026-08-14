from pathlib import Path

from signaid.avatar.signavatars_adapter import SignAvatarsAdapter
from signaid.datasets.mock import build_mock_dataset


def test_missing_signavatars_dependency_falls_back(tmp_path: Path):
    motion = build_mock_dataset(tmp_path / "motions")[0]
    adapter = SignAvatarsAdapter(python_executable="definitely-not-a-python-command")
    status = adapter.availability()
    assert status["available"] is False
    output = adapter.render(motion, tmp_path / "fallback.mp4")
    assert output.exists()
    assert output.suffix in {".mp4", ".gif"}
    assert adapter.last_error
