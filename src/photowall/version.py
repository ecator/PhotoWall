"""
version.py - PhotoWall 版本管理模块
"""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """动态获取 PhotoWall 的包版本号。"""
    try:
        return version("PhotoWall")
    except PackageNotFoundError:
        try:
            return version("photowall")
        except PackageNotFoundError:
            return "unknown"


__version__ = get_version()
