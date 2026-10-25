"""
cli.py - 照片墙 CLI 统一入口
"""

import argparse
import sys
from collections.abc import Sequence

from . import assets, index, tags
from .version import get_version


def build_parser() -> argparse.ArgumentParser:
    """构建 photowall 根命令行解析器及子命令。"""
    version_str = get_version()
    parser = argparse.ArgumentParser(
        prog="photowall",
        description=f"PhotoWall 照片墙命令行工具 (v{version_str})",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"PhotoWall v{version_str}",
        help="显示版本信息并退出",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="子命令",
        description="可用子命令",
        metavar="<command>",
    )

    parser_index = subparsers.add_parser(
        "index",
        help="全量索引相册：生成 tags.json、assets.js 并输出 index.html",
        description=f"PhotoWall v{version_str} - 从指定的 --home 中查找 assets 目录，并输出 .photowall、tags.json、assets.js 与 index.html",
    )
    index.build_parser(parser_index)

    parser_assets = subparsers.add_parser(
        "assets",
        help="扫描 assets 目录并生成 assets.js 元数据",
        description=f"PhotoWall v{version_str} - 扫描 assets/ 目录下的所有媒体文件，融合标签并生成 assets.js",
    )
    assets.build_parser(parser_assets)

    parser_tags = subparsers.add_parser(
        "tags",
        help="使用 WD14 Tagger 自动为图片打标并生成 tags.json",
        description=f"PhotoWall v{version_str} - 使用 WD14 Tagger AI 模型增量扫描并为 assets/ 图片生成 tags.json 标签缓存",
    )
    tags.build_parser(parser_tags)

    subparsers.add_parser(
        "version",
        help="显示版本信息并退出",
        description=f"PhotoWall v{version_str} - 显示版本信息并退出",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI 主入口函数。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "version":
        print(f"PhotoWall v{get_version()}")
        return

    if args.command == "index":
        index.run(args)
    elif args.command == "assets":
        assets.run(args)
    elif args.command == "tags":
        tags.run(args)


if __name__ == "__main__":
    main()
