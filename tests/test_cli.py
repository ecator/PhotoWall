import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from photowall import assets, index, tags
from photowall.cli import build_parser, main
from photowall.index import copy_index_html, get_template_index_html_path
from photowall.version import __version__, get_version


def test_build_parser():
    parser = build_parser()
    assert parser.prog == "photowall"


def test_cli_no_args(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage: photowall" in captured.out
    assert "index" in captured.out
    assert "assets" in captured.out
    assert "tags" in captured.out
    assert "version" in captured.out
    assert get_version() in captured.out


def test_cli_index_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "index",
            "--home",
            ".",
            "--title",
            "测试画廊",
            "--force",
            "--filter",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--skip-tags",
            "-v",
        ]
    )
    assert args.command == "index"
    assert args.home == Path(".")
    assert args.title == "测试画廊"
    assert args.force is True
    assert args.filter is True
    assert args.device == "cpu"
    assert args.batch_size == 4
    assert args.skip_tags is True
    assert args.verbose is True


def test_cli_index_dir_alias():
    parser = build_parser()
    args = parser.parse_args(["index", "-d", "my_album"])
    assert args.command == "index"
    assert args.home == Path("my_album")


def test_cli_assets_args():
    parser = build_parser()
    args = parser.parse_args(["assets", "--title", "测试画廊", "--force", "-d", "."])
    assert args.command == "assets"
    assert args.title == "测试画廊"
    assert args.force is True
    assert args.home == Path(".")


def test_cli_assets_home_arg():
    parser = build_parser()
    args = parser.parse_args(["assets", "--home", "test_path"])
    assert args.command == "assets"
    assert args.home == Path("test_path")


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
    assert args.home == Path(".")


def test_cli_tags_home_arg():
    parser = build_parser()
    args = parser.parse_args(["tags", "--home", "test_path"])
    assert args.command == "tags"
    assert args.home == Path("test_path")


def test_cli_dispatch_index(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr(index, "run", mock_run)

    main(["index", "--title", "HelloIndex"])
    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args.command == "index"
    assert args.title == "HelloIndex"


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


def test_index_run_missing_assets_dir(tmp_path):
    args = argparse.Namespace(
        command="index",
        home=tmp_path,
        dir=None,
        title="Test",
        force=False,
    )
    with pytest.raises(SystemExit) as exc_info:
        index.run(args)
    assert exc_info.value.code == 1


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


def test_copy_index_html(tmp_path):
    out_file = copy_index_html(tmp_path)
    assert out_file.exists()
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content or "<html" in content
    assert 'src="assets.js"' in content

    # Test template lookup
    tpl_path = get_template_index_html_path()
    assert tpl_path is not None
    assert tpl_path.is_file()


def test_index_run_success(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    mock_tags_run = MagicMock()
    monkeypatch.setattr(tags, "run", mock_tags_run)

    args = argparse.Namespace(
        command="index",
        home=tmp_path,
        dir=None,
        title="测试全量相册",
        force=False,
        filter=False,
        device="cpu",
        batch_size=1,
        verbose=False,
        skip_tags=False,
    )
    index.run(args)

    assert mock_tags_run.called

    output_js = tmp_path / "assets.js"
    assert output_js.exists()

    output_html = tmp_path / "index.html"
    assert output_html.exists()
    assert 'src="assets.js"' in output_html.read_text(encoding="utf-8")


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


def test_assets_run_default_title_from_home(tmp_path):
    album_dir = tmp_path / "MySpecialAlbum"
    album_dir.mkdir()
    (album_dir / "assets").mkdir()

    args = argparse.Namespace(
        command="assets",
        home=album_dir,
        dir=album_dir,
        title=None,
        force=False,
    )
    assets.run(args)

    content = (album_dir / "assets.js").read_text(encoding="utf-8")
    assert '"title": "MySpecialAlbum"' in content


def test_index_run_default_title_from_home(tmp_path, monkeypatch):
    album_dir = tmp_path / "SummerVacation"
    album_dir.mkdir()
    (album_dir / "assets").mkdir()

    mock_tags_run = MagicMock()
    monkeypatch.setattr(tags, "run", mock_tags_run)

    args = argparse.Namespace(
        command="index",
        home=album_dir,
        dir=None,
        title=None,
        force=False,
        filter=False,
        device="cpu",
        batch_size=1,
        verbose=False,
        skip_tags=True,
    )
    index.run(args)

    content = (album_dir / "assets.js").read_text(encoding="utf-8")
    assert '"title": "SummerVacation"' in content


def test_cli_version_subcommand(capsys):
    main(["version"])
    captured = capsys.readouterr()
    expected = f"PhotoWall v{get_version()}"
    assert expected in captured.out


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert f"PhotoWall v{get_version()}" in captured.out

    with pytest.raises(SystemExit) as exc_info2:
        main(["-V"])
    assert exc_info2.value.code == 0
    captured2 = capsys.readouterr()
    assert f"PhotoWall v{get_version()}" in captured2.out


def test_get_version_fallback(monkeypatch):
    import importlib.metadata

    from photowall import version as v_mod

    def mock_version_fail(pkg_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(v_mod, "version", mock_version_fail)
    assert v_mod.get_version() == "unknown"


def test_assets_run_banner_includes_version(tmp_path, capsys):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    assets_args = argparse.Namespace(
        command="assets",
        dir=tmp_path,
        title="测试版本",
        force=False,
    )
    assets.run(assets_args)
    captured = capsys.readouterr()
    assert f"PhotoWall v{__version__}" in captured.out


def test_index_run_banner_includes_version(tmp_path, capsys, monkeypatch):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    mock_tags_run = MagicMock()
    monkeypatch.setattr(tags, "run", mock_tags_run)
    index_args = argparse.Namespace(
        command="index",
        home=tmp_path,
        dir=None,
        title="测试版本",
        force=False,
        filter=False,
        device="cpu",
        batch_size=1,
        verbose=False,
        skip_tags=True,
    )
    index.run(index_args)
    captured = capsys.readouterr()
    assert f"PhotoWall v{__version__}" in captured.out


def test_tags_run_banner_includes_version(tmp_path, capsys, monkeypatch):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    monkeypatch.setattr(tags, "load_translations", lambda *args, **kwargs: {})

    tags_args = argparse.Namespace(
        command="tags",
        dir=tmp_path,
        force=False,
        filter=True,
        device="cpu",
        batch_size=1,
        verbose=False,
        checkpoint_interval=100,
    )
    tags.run(tags_args)
    captured = capsys.readouterr()
    assert f"PhotoWall v{__version__}" in captured.out
