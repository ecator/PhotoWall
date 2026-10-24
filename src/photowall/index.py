"""
index.py - 照片墙全量索引与构建脚本

使用方法:
    uv run photowall index                          # 索引当前目录相册 (默认标题为当前目录名)
    uv run photowall index --home /path/to/gallery  # 索引指定相册目录
    uv run photowall index --title "我的相册"        # 指定相册标题
    uv run photowall index --force                  # 强制重新推理 AI 标签并重新解析所有媒体
    uv run photowall index --filter                 # 仅重算过滤规则与翻译 (不调用 AI 模型)
    uv run photowall index --skip-tags              # 跳过 AI 视觉打标步骤

功能说明:
    1. 自动从 --home 指定目录中查找 assets/ 目录；
    2. 执行 AI 视觉打标并生成/更新 tags.json 以及 .cache/；
    3. 解析媒体文件尺寸与时长，融合标签并生成 assets.js；
    4. 将内置 index.html 网页入口文件输出到 --home 目录。
"""

import argparse
import importlib.resources
import sys
from collections.abc import Sequence
from pathlib import Path

from photowall import assets, tags


def get_template_index_html_path() -> Path | None:
    """获取内置 index.html 模板文件的绝对路径。"""
    try:
        res = importlib.resources.files("photowall").joinpath("index.html")
        with importlib.resources.as_file(res) as p:
            if p.is_file():
                return Path(p)
    except Exception:
        pass

    local_in_pkg = Path(__file__).resolve().parent / "index.html"
    if local_in_pkg.is_file():
        return local_in_pkg

    repo_root_file = Path(__file__).resolve().parent.parent.parent / "index.html"
    if repo_root_file.is_file():
        return repo_root_file

    return None


def copy_index_html(target_dir: Path) -> Path:
    """复制内置 index.html 到指定的目标目录中。"""
    target_dir = target_dir.resolve()
    target_file = target_dir / "index.html"

    content: bytes | None = None
    try:
        res = importlib.resources.files("photowall").joinpath("index.html")
        content = res.read_bytes()
    except Exception:
        pass

    if content is None:
        src_path = get_template_index_html_path()
        if src_path and src_path.is_file():
            if src_path.resolve() == target_file.resolve():
                return target_file
            content = src_path.read_bytes()

    if content is None:
        print("[错误] 未找到 index.html 模板文件，无法复制到目标目录。")
        sys.exit(1)

    # 若目标文件就是源模板文件本身，跳过写入，避免自覆盖
    src_path = get_template_index_html_path()
    if src_path and src_path.resolve() == target_file.resolve():
        return target_file

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_target_file = target_dir / "index.html.tmp"
    temp_target_file.write_bytes(content)
    temp_target_file.replace(target_file)
    return target_file


def build_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """构建 index 命令的命令行参数解析器。"""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="从相册目录中查找 assets 并输出对应的 .cache、tags.json、assets.js 与 index.html"
        )
    parser.add_argument(
        "--home",
        "--dir",
        "-d",
        type=Path,
        default=None,
        help="相册根目录路径 (默认: 当前工作目录)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="相册标题 (写入 meta.title，默认从 --home 目录名称提取)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新推理所有图片 AI 标签并强制重新读取所有媒体元数据",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="仅重算过滤规则与翻译 (不调用 AI 模型推理)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "dml", "cpu"],
        default="auto",
        help="指定 AI 打标推理设备 (auto: 优先 GPU/DirectML/CUDA, 自动回退 CPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="AI 打标推理 Batch 大小 (默认: 0 自动选择)",
    )
    parser.add_argument(
        "--skip-tags",
        action="store_true",
        help="跳过 AI 标签生成步骤 (仅更新 assets.js 与 index.html)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出每个文件的完整路径与缓存状态",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """执行照片墙全量索引主流程。"""
    home_dir = (
        (getattr(args, "home", None) or getattr(args, "dir", None)) or Path.cwd()
    ).resolve()
    assets_dir = home_dir / "assets"

    if not assets_dir.is_dir():
        print(f"[错误] 未找到 assets 目录: {assets_dir.resolve()}")
        sys.exit(1)

    print("=" * 60)
    print(f"[*] 开始照片墙相册索引: {home_dir.resolve()}")
    print("=" * 60)

    # 1. 执行 AI 打标 (tags.json)
    if not getattr(args, "skip_tags", False):
        print("\n" + "-" * 50)
        print("[步骤 1/3] 执行 AI 视觉打标并生成/更新 tags.json...")
        print("-" * 50)
        tags_args = argparse.Namespace(
            home=home_dir,
            dir=home_dir,
            force=getattr(args, "force", False),
            filter=getattr(args, "filter", False),
            device=getattr(args, "device", "auto"),
            batch_size=getattr(args, "batch_size", 0),
            verbose=getattr(args, "verbose", False),
        )
        tags.run(tags_args)
    else:
        print("\n[*] 跳过 AI 打标步骤 (--skip-tags)。")

    # 2. 执行媒体数据提取与标签融合 (assets.js)
    print("\n" + "-" * 50)
    print("[步骤 2/3] 扫描媒体文件并生成 assets.js...")
    print("-" * 50)
    album_title = getattr(args, "title", None)
    if not album_title:
        album_title = home_dir.name if home_dir.name else "PhotoWall"

    assets_args = argparse.Namespace(
        home=home_dir,
        dir=home_dir,
        title=album_title,
        force=getattr(args, "force", False),
    )
    assets.run(assets_args)

    # 3. 输出 index.html 到相册根目录
    print("\n" + "-" * 50)
    print("[步骤 3/3] 输出 index.html 网页入口文件...")
    print("-" * 50)
    out_html = copy_index_html(home_dir)
    print(f"[√] 已输出 index.html 到: {out_html.resolve()}")

    print("\n" + "=" * 60)
    print("[√] 全量索引已完成！")
    print(f"    - 相册目录: {home_dir.resolve()}")
    print(f"    - 标签文件: {(home_dir / 'tags.json').resolve()}")
    print(f"    - 数据文件: {(home_dir / 'assets.js').resolve()}")
    print(f"    - 网页文件: {out_html.resolve()}")
    print(f"    - 缓存目录: {(home_dir / '.cache').resolve()}")
    print("=" * 60)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
