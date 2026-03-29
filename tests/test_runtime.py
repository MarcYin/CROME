from pathlib import Path

from crome.runtime import ensure_proj_data_env


def test_ensure_proj_data_env_sets_proj_vars(monkeypatch, tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "proj.db").write_text("", encoding="utf-8")

    monkeypatch.delenv("PROJ_DATA", raising=False)
    monkeypatch.delenv("PROJ_LIB", raising=False)

    class FakeDataDir:
        @staticmethod
        def get_data_dir() -> str:
            return str(proj_dir)

        @staticmethod
        def set_data_dir(_path: str) -> None:
            return None

    monkeypatch.setattr("pyproj.datadir", FakeDataDir())

    resolved = ensure_proj_data_env()
    assert resolved == proj_dir
