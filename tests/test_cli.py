import json
from types import SimpleNamespace

from crome import cli


def test_cli_download_alphaearth_dry_run(capsys) -> None:
    exit_code = cli.main(
        [
            "download-alphaearth",
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--bbox",
            "-1.0",
            "51.0",
            "0.0",
            "52.0",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["aoi_label"] == "east-anglia"
    assert payload["bands"][0] == "A00"
    assert payload["bands"][-1] == "A63"


def test_cli_download_alphaearth_forwards_request(monkeypatch, capsys) -> None:
    captured = {}

    def fake_download(request):
        captured["request"] = request
        return SimpleNamespace(
            aoi_label=request.aoi_label,
            bands=request.bands,
            collection_id=request.collection_id,
            conditional_year=request.conditional_year,
            manifest_path=None,
            output_root=request.dataset_output_root,
            source_image_ids=(),
            year=request.year,
        )

    monkeypatch.setattr(cli.alphaearth, "download_alphaearth_images", fake_download)

    exit_code = cli.main(
        [
            "download-alphaearth",
            "--year",
            "2024",
            "--aoi-label",
            "east-anglia",
            "--bbox",
            "-1.0",
            "51.0",
            "0.0",
            "52.0",
        ]
    )

    assert exit_code == 0
    assert captured["request"].aoi_label == "east-anglia"
    assert json.loads(capsys.readouterr().out)["year"] == 2024
