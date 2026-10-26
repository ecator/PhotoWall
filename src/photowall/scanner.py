"""
scanner.py - 媒体资源扫描与文件格式定义模块

功能说明:
    1. 统一维护相册支持的图片 (IMAGE_EXTENSIONS) 与视频 (VIDEO_EXTENSIONS) 扩展名事实来源。
    2. 提供自然排序键生成函数 natural_sort_key。
    3. 提供支持递归遍历普通目录、软链接 (Linux symlink) 及 Windows Junction 的媒体文件扫描逻辑。
"""

import re
from pathlib import Path
from typing import Any

# 支持的媒体文件后缀定义 (统一事实来源)
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTENSIONS: set[str] = {".mp4", ".webm"}
MEDIA_EXTENSIONS: set[str] = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def natural_sort_key(s: str) -> list[Any]:
    """自然排序键函数 (使 2.jpg 排在 10.jpg 前面，对中文与英文路径均生效)。"""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def is_image_file(path: Path | str) -> bool:
    """判断给定路径是否为支持的图片格式。"""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path | str) -> bool:
    """判断给定路径是否为支持的视频格式。"""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_media_file(path: Path | str) -> bool:
    """判断给定路径是否为支持的媒体（图片或视频）格式。"""
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def scan_media_files(
    assets_dir: Path,
    *,
    allowed_extensions: set[str] | None = None,
    sort: bool = True,
) -> list[Path]:
    """
    递归扫描 assets 目录下的所有媒体文件。
    支持普通目录、Windows Junction 以及 Linux/Windows 符号链接 (recurse_symlinks=True)。
    遇到断链、无权限或异常项时安全忽略。

    :param assets_dir: assets 根目录路径
    :param allowed_extensions: 过滤的扩展名集合，默认使用 MEDIA_EXTENSIONS (所有图片与视频)
    :param sort: 是否按相对于 assets_dir 的 POSIX 路径进行自然排序
    :return: 匹配的文件 Path 列表
    """
    valid_exts = (
        allowed_extensions if allowed_extensions is not None else MEDIA_EXTENSIONS
    )
    raw_files: list[Path] = []
    if not assets_dir.is_dir():
        return raw_files

    for p in assets_dir.rglob("*", recurse_symlinks=True):
        try:
            if p.is_file() and p.suffix.lower() in valid_exts:
                raw_files.append(p)
        except OSError:
            continue

    if sort:
        raw_files.sort(
            key=lambda p: natural_sort_key(p.relative_to(assets_dir).as_posix())
        )

    return raw_files


def scan_image_files(assets_dir: Path, *, sort: bool = True) -> list[Path]:
    """扫描 assets 目录下的所有图片文件。"""
    return scan_media_files(
        assets_dir,
        allowed_extensions=IMAGE_EXTENSIONS,
        sort=sort,
    )
