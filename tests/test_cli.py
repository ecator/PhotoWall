import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from photowall import assets, tags
from photowall.cli import build_parser, main


def test_build_parser():
    parser = build_parser()
    assert parser.prog == "photowall"


def test_cli_no_args(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage: photowall" in captured.out
    assert "assets" in captured.out
    assert "tags" in captured.out


def test_cli_assets_args():
    parser = build_parser()
    args = parser.parse_args(["assets", "--title", "测试画廊", "--force", "-d", "."])
    assert args.command == "assets"
    assert args.title == "测试画廊"
    assert args.force is True
    assert args.dir == Path(".")


def test_cli_tags_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tags",
            "--force",
            "--filter",
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "-v",
            "--dir",
            ".",
        ]
    )
    assert args.command == "tags"
    assert args.force is True
    assert args.filter is True
    assert args.device == "cpu"
    assert args.batch_size == 8
    assert args.verbose is True
    assert args.dir == Path(".")


def test_cli_dispatch_assets(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr(assets, "run", mock_run)

    main(["assets", "--title", "Hello"])
    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args.command == "assets"
    assert args.title == "Hello"


def test_cli_dispatch_tags(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr(tags, "run", mock_run)

    main(["tags", "--filter"])
    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args.command == "tags"
    assert args.filter is True


def test_assets_run_missing_assets_dir(tmp_path):
    args = argparse.Namespace(
        command="assets",
        dir=tmp_path,
        title="Test",
        force=False,
    )
    with pytest.raises(SystemExit) as exc_info:
        assets.run(args)
    assert exc_info.value.code == 1


def test_tags_run_missing_assets_dir(tmp_path):
    args = argparse.Namespace(
        command="tags",
        dir=tmp_path,
        force=False,
        filter=False,
        device="cpu",
        batch_size=0,
        verbose=False,
    )
    with pytest.raises(SystemExit) as exc_info:
        tags.run(args)
    assert exc_info.value.code == 1


def test_assets_run_success(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    args = argparse.Namespace(
        command="assets",
        dir=tmp_path,
        title="临时测试相册",
        force=False,
    )
    assets.run(args)

    output_file = tmp_path / "assets.js"
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert content.startswith(
        "// 由 photowall assets 自动生成\nwindow.PHOTO_WALL_DATA = "
    )

    prefix = "// 由 photowall assets 自动生成\nwindow.PHOTO_WALL_DATA = "
    suffix = ";\n"
    assert content.endswith(suffix)
    json_str = content[len(prefix) : -len(suffix)]
    data = json.loads(json_str)

    assert data["meta"]["title"] == "临时测试相册"
    assert data["meta"]["totalCount"] == 0
    assert data["items"] == []
    assert len(data["folders"]) == 1
    assert data["folders"][0]["name"] == "根目录"
