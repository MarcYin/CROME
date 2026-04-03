import sqlite3
from pathlib import Path

from crome.runtime import ensure_proj_data_env


def _create_proj_db(proj_dir: Path, minor_version: int = 6) -> None:
    """Create a minimal proj.db with the required metadata table."""
    db_path = proj_dir / "proj.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
    conn.execute(
        "INSERT INTO metadata VALUES ('DATABASE.LAYOUT.VERSION.MINOR', ?)",
        (str(minor_version),),
    )
    conn.commit()
    conn.close()


def test_ensure_proj_data_env_sets_proj_vars(monkeypatch, tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    _create_proj_db(proj_dir, minor_version=6)

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


def test_ensure_proj_data_env_skips_old_proj_db(monkeypatch, tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    _create_proj_db(proj_dir, minor_version=4)

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
    assert resolved is None
