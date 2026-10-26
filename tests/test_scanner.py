import sys
from pathlib import Path

import pytest

from photowall.scanner import (
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    VIDEO_EXTENSIONS,
    is_image_file,
    is_media_file,
    is_video_file,
    natural_sort_key,
    scan_image_files,
    scan_media_files,
)


def test_extensions_definition():
    """验证图片与视频扩展名集合的定义与关系。"""
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".png" in IMAGE_EXTENSIONS
    assert ".webp" in IMAGE_EXTENSIONS
    assert ".gif" in IMAGE_EXTENSIONS
    assert ".mp4" in VIDEO_EXTENSIONS
    assert ".webm" in VIDEO_EXTENSIONS

    # MEDIA_EXTENSIONS 应为 IMAGE 与 VIDEO 的并集
    assert MEDIA_EXTENSIONS == (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    # 两者无交集
    assert len(IMAGE_EXTENSIONS & VIDEO_EXTENSIONS) == 0


def test_predicate_helpers():
    """验证媒体格式判断辅助函数。"""
    assert is_image_file("pic.JPG")
    assert is_image_file(Path("sub/pic.png"))
    assert not is_image_file("movie.mp4")
    assert not is_image_file("notes.txt")

    assert is_video_file("movie.mp4")
    assert is_video_file(Path("sub/movie.WEBM"))
    assert not is_video_file("pic.png")

    assert is_media_file("pic.jpg")
    assert is_media_file("movie.mp4")
    assert not is_media_file("doc.pdf")


def test_natural_sort_key():
    """验证自然排序函数（数字按大小而非字符排，支持中英文）。"""
    items = ["2.jpg", "10.jpg", "1.jpg", "图2.png", "图10.png", "图1.png"]
    sorted_items = sorted(items, key=natural_sort_key)
    assert sorted_items == [
        "1.jpg",
        "2.jpg",
        "10.jpg",
        "图1.png",
        "图2.png",
        "图10.png",
    ]


def test_scan_media_files_basic(tmp_path):
    """验证基本媒体扫描与自然排序。"""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    # 创建若干文件
    (assets_dir / "10.jpg").write_text("a")
    (assets_dir / "2.png").write_text("b")
    (assets_dir / "1.mp4").write_text("c")
    (assets_dir / "readme.txt").write_text("d")

    sub = assets_dir / "subfolder"
    sub.mkdir()
    (sub / "photo.webp").write_text("e")

    # 扫描所有媒体文件
    all_media = scan_media_files(assets_dir)
    rel_paths = [p.relative_to(assets_dir).as_posix() for p in all_media]
    assert rel_paths == ["1.mp4", "2.png", "10.jpg", "subfolder/photo.webp"]

    # 扫描仅图片
    images = scan_image_files(assets_dir)
    image_rel_paths = [p.relative_to(assets_dir).as_posix() for p in images]
    assert image_rel_paths == ["2.png", "10.jpg", "subfolder/photo.webp"]


def test_scan_non_existent_directory(tmp_path):
    """验证扫描不存在的目录时安全返回空列表。"""
    missing_dir = tmp_path / "does_not_exist"
    assert scan_media_files(missing_dir) == []
    assert scan_image_files(missing_dir) == []


def test_scan_symlink_and_junction(tmp_path):
    """验证扫描能够递归进入软链接与 Windows Junction 目录。"""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()

    (external_dir / "ext_pic.jpg").write_text("img")
    (external_dir / "ext_vid.mp4").write_text("vid")

    linked_dir = assets_dir / "linked"
    link_created = False
    try:
        linked_dir.symlink_to(external_dir, target_is_directory=True)
        link_created = True
    except OSError:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(external_dir), str(linked_dir))
            link_created = True

    if not link_created:
        pytest.skip("当前环境不支持创建符号链接或 Junction")

    media_files = scan_media_files(assets_dir)
    rel_paths = [p.relative_to(assets_dir).as_posix() for p in media_files]
    assert "linked/ext_pic.jpg" in rel_paths
    assert "linked/ext_vid.mp4" in rel_paths

    image_files = scan_image_files(assets_dir)
    img_rel_paths = [p.relative_to(assets_dir).as_posix() for p in image_files]
    assert "linked/ext_pic.jpg" in img_rel_paths
    assert "linked/ext_vid.mp4" not in img_rel_paths


def test_scan_broken_symlink_ignored(tmp_path):
    """验证扫描时失效链接会被安全忽略。"""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "valid.jpg").write_text("img")

    target = tmp_path / "temp_target"
    target.mkdir()
    broken_link = assets_dir / "broken"

    link_created = False
    try:
        broken_link.symlink_to(target, target_is_directory=True)
        link_created = True
    except OSError:
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(target), str(broken_link))
            link_created = True

    target.rmdir()

    if link_created:
        media_files = scan_media_files(assets_dir)
        rel_paths = [p.relative_to(assets_dir).as_posix() for p in media_files]
        assert rel_paths == ["valid.jpg"]
