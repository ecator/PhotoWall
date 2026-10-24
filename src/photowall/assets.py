"""
assets.py - 照片墙元数据生成脚本

使用方法:
    uv run assets.py                         # 使用默认标题 "PhotoWall"
    uv run assets.py --title "我的二次元画廊"   # 指定自定义相册标题

功能说明:
    1. 自动扫描 assets/ 目录下的所有图片与视频，无条件全量生成 assets.js。
    2. 校验 tags.json：
       - 若 tags.json 不存在，输出显眼警告提示用户先运行 uv run tags.py；
       - 若某张图片的文件大小或修改时间与 tags.json 不一致，输出该文件的 FullName 警告。
    3. 解析每张图片的真实尺寸，并自动根据 EXIF Orientation 校正宽高旋转。
    4. 自动提取 MP4 视频时长。
    5. 全路径 URL 安全编码 (urllib.parse.quote)，避免特殊字符与空格导致裂图。
    6. 融合目录分词标签、画幅形态标签与 AI 视觉标签，并聚合生成全局顶层 tags 索引。
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image
from tqdm import tqdm

# 支持的文件后缀定义
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTENSIONS: set[str] = {".mp4", ".webm"}


def natural_sort_key(s: str) -> list[Any]:
    """自然排序键函数 (使 2.jpg 排在 10.jpg 前面，对中文与英文路径均生效)。"""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def encode_url_path(rel_path: str) -> str:
    """对相对路径的每一段进行 URL 编码并保留 '/'，防止 #、?、% 和空格导致图片 404。"""
    parts = rel_path.replace("\\", "/").split("/")
    encoded_parts = [urllib.parse.quote(part) for part in parts]
    return "/".join(encoded_parts)


def get_image_info(path: Path) -> tuple[int, int, int | None]:
    """
    使用 Pillow 读取图片信息：
    返回: (实际显示宽度, 实际显示高度, EXIF 拍摄时间戳)
    自动校正 EXIF Orientation 旋转 (5, 6, 7, 8 交换宽高)。
    """
    width, height = 0, 0
    exif_timestamp: int | None = None

    try:
        with Image.open(path) as img:
            width, height = img.size

            # 1. 处理 EXIF Orientation 旋转
            exif = img.getexif()
            if exif:
                orientation = exif.get(ExifTags.Base.Orientation)
                # 方向 5, 6, 7, 8 需要对调宽高
                if orientation in (5, 6, 7, 8):
                    width, height = height, width

            # 2. 尝试读取真实拍摄时间 (DateTimeOriginal / DateTime)
            dt_str: str | None = None
            if hasattr(img, "_getexif"):
                raw_exif = img._getexif()
                if raw_exif:
                    # 0x9003: DateTimeOriginal, 0x0132: DateTime
                    dt_str = raw_exif.get(0x9003) or raw_exif.get(0x0132)

            if dt_str and isinstance(dt_str, str):
                try:
                    # 常见格式: "YYYY:MM:DD HH:MM:SS"
                    dt_clean = dt_str.strip()[:19]
                    dt = datetime.strptime(dt_clean, "%Y:%m:%d %H:%M:%S")
                    exif_timestamp = int(dt.timestamp())
                except Exception:
                    pass
    except Exception:
        pass

    return width, height, exif_timestamp


def get_video_duration(path: Path) -> float:
    """纯 Python 解析 MP4 文件头部/尾部 moov-mvhd box 获取视频时长 (秒)，避免引入庞大外部依赖。"""
    try:
        with open(path, "rb") as f:
            file_size = path.stat().st_size
            while f.tell() < file_size:
                header = f.read(8)
                if len(header) < 8:
                    break
                size = int.from_bytes(header[:4], "big")
                atom_type = header[4:8]

                # 64位超大尺寸 (size == 1) 或直到文件末尾 (size == 0)
                if size == 1:
                    large_size_bytes = f.read(8)
                    if len(large_size_bytes) < 8:
                        break
                    payload_size = int.from_bytes(large_size_bytes, "big") - 16
                elif size >= 8:
                    payload_size = size - 8
                elif size == 0:
                    payload_size = file_size - f.tell()
                else:
                    break

                if atom_type == b"moov":
                    moov_end = f.tell() + payload_size
                    while f.tell() < moov_end:
                        sub_header = f.read(8)
                        if len(sub_header) < 8:
                            break
                        sub_size = int.from_bytes(sub_header[:4], "big")
                        sub_type = sub_header[4:8]

                        if sub_size == 1:
                            sub_large = f.read(8)
                            if len(sub_large) < 8:
                                break
                            sub_payload = int.from_bytes(sub_large, "big") - 16
                        elif sub_size >= 8:
                            sub_payload = sub_size - 8
                        else:
                            break

                        if sub_type == b"mvhd":
                            mvhd_data = f.read(min(sub_payload, 32))
                            if len(mvhd_data) >= 20:
                                version = mvhd_data[0]
                                if version == 0:
                                    timescale = int.from_bytes(mvhd_data[12:16], "big")
                                    duration = int.from_bytes(mvhd_data[16:20], "big")
                                else:
                                    timescale = int.from_bytes(mvhd_data[20:24], "big")
                                    duration = int.from_bytes(mvhd_data[24:32], "big")
                                if timescale > 0:
                                    return round(duration / timescale, 1)
                            return 0.0
                        else:
                            if sub_payload > 0:
                                f.seek(sub_payload, os.SEEK_CUR)
                    break
                else:
                    if payload_size > 0:
                        f.seek(payload_size, os.SEEK_CUR)
    except Exception:
        pass
    return 0.0


def extract_directory_tags(folder_path: str) -> list[str]:
    """从文件夹路径中提取有意义的分类标签 (过滤掉 [60P 1GB]、NO.172、纯数字等噪点)。"""
    tags: list[str] = []
    if not folder_path:
        return tags

    for part in folder_path.split("/"):
        if not part:
            continue
        # 清理常见图集括号与序号噪点
        cleaned = re.sub(
            r"\[.*?\]|\(.*?\)|NO\.\d+|^\d+_", " ", part, flags=re.IGNORECASE
        )
        for token in cleaned.split():
            token = token.strip()
            if len(token) > 1 and not token.isdigit():
                tags.append(token)
    return tags


def compute_aspect_tags(
    width: int, height: int, ext: str, media_type: str
) -> list[str]:
    """根据画面尺寸与文件格式生成形态属性标签。"""
    tags: list[str] = []
    if media_type == "image":
        if width > 0 and height > 0:
            ratio = width / height
            if ratio >= 1.2:
                tags.append("横屏")
            elif ratio <= 0.83:
                tags.append("竖屏")
            else:
                tags.append("方图")

            if max(width, height) >= 3840 or min(width, height) >= 1440:
                tags.append("4K")

        if ext == "gif":
            tags.append("动图")
    elif media_type == "video":
        tags.append("视频")

    return tags


def parse_media_file(
    item: tuple[Path, str, str, int, int],
) -> tuple[str, int, int, int, int, float]:
    """
    单文件解析任务 (供多线程池并发调用)。
    返回: (rel_path, size, mtime, width, height, final_timestamp, duration)
    """
    full_path, rel_path, media_type, file_size, file_mtime = item
    width, height = 0, 0
    duration = 0.0
    final_timestamp = file_mtime

    if media_type == "image":
        w, h, exif_time = get_image_info(full_path)
        width, height = w, h
        if exif_time:
            final_timestamp = exif_time
    else:
        duration = get_video_duration(full_path)

    return rel_path, file_size, file_mtime, width, height, final_timestamp, duration


def main():
    parser = argparse.ArgumentParser(description="照片墙 assets.js 全量生成脚本")
    parser.add_argument(
        "--title",
        type=str,
        default="PhotoWall",
        help="相册标题 (写入 meta.title，默认 'PhotoWall')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略本地元数据缓存，强制重新使用多线程读取并解析所有媒体文件的尺寸与时长",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    assets_dir = project_dir / "assets"
    tags_file = project_dir / "tags.json"
    output_file = project_dir / "assets.js"
    media_cache_dir = project_dir / ".cache"
    media_cache_dir.mkdir(parents=True, exist_ok=True)
    media_cache_file = media_cache_dir / "assets_cache.json"

    if not assets_dir.is_dir():
        print(f"[错误] 未找到 assets 目录: {assets_dir.resolve()}")
        sys.exit(1)

    # 1. 检查并读取 tags.json
    tags_data: dict[str, Any] | None = None
    tags_cache: dict[str, dict[str, Any]] = {}

    if tags_file.is_file():
        try:
            with open(tags_file, "r", encoding="utf-8") as f:
                tags_data = json.load(f)
            tags_cache = tags_data.get("files", {})
            print(
                f"[*] 已加载标签缓存文件: {tags_file.resolve()} (条目数: {len(tags_cache)})"
            )
        except Exception as e:
            print(f"[警告] tags.json 解析失败: {e}")
    else:
        print("\n" + "=" * 70)
        print("[警告] 未检测到 tags.json 标签缓存文件！")
        print(f"       期望路径: {tags_file.resolve()}")
        print("       建议先运行: uv run tags.py 生成 AI 标签。")
        print("       当前将以无 AI 标签模式继续生成 assets.js...")
        print("=" * 70 + "\n")

    # 2. 读取媒体元数据缓存 (assets_cache.json)
    media_cache: dict[str, dict[str, Any]] = {}
    if not args.force and media_cache_file.is_file():
        try:
            with open(media_cache_file, "r", encoding="utf-8") as f:
                media_cache = json.load(f)
            print(
                f"[*] 已载入媒体元数据缓存: {media_cache_file.resolve()} (历史条目: {len(media_cache)})"
            )
        except Exception as e:
            print(f"[警告] 读取媒体缓存失败: {e}")

    # 3. 扫描 assets 目录下的所有媒体文件
    print(f"[*] 扫描资源目录: {assets_dir.resolve()}")
    raw_files: list[Path] = []
    for p in assets_dir.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                raw_files.append(p)

    # 自然排序
    raw_files.sort(key=lambda p: natural_sort_key(p.relative_to(assets_dir).as_posix()))
    print(f"[*] 扫描到媒体文件总数: {len(raw_files)}")

    # 4. 统计文件属性并比对缓存 (检测新增、过期及已删除)
    file_info_list: list[tuple[Path, str, str, int, int]] = []
    to_process: list[tuple[Path, str, str, int, int]] = []
    current_disk_set: set[str] = set()

    for p in raw_files:
        rel_to_assets = p.relative_to(assets_dir).as_posix()
        current_disk_set.add(rel_to_assets)
        file_stat = p.stat()
        file_size = file_stat.st_size
        file_mtime = int(file_stat.st_mtime)
        ext = p.suffix.lower()
        media_type = "video" if ext in VIDEO_EXTENSIONS else "image"

        file_tuple = (p, rel_to_assets, media_type, file_size, file_mtime)
        file_info_list.append(file_tuple)

        # 检查缓存是否有效 (文件大小和修改时间均匹配)
        cached_entry = media_cache.get(rel_to_assets)
        if (
            not cached_entry
            or cached_entry.get("size") != file_size
            or cached_entry.get("mtime") != file_mtime
        ):
            to_process.append(file_tuple)

    cached_count = len(raw_files) - len(to_process)
    print(f"[*] 媒体缓存比对: 命中缓存 {cached_count} 个，需解析 {len(to_process)} 个")

    # 5. 多线程并发解析新增或变动的媒体文件
    if to_process:
        max_workers = min(16, (os.cpu_count() or 4) * 2)
        print(f"[*] 启动多线程并发解析 (线程数: {max_workers})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(parse_media_file, to_process),
                    total=len(to_process),
                    desc="多线程解析媒体",
                    unit="文件",
                )
            )
        # pyrefly: ignore[bad-unpacking]
        for (
            rel_path,
            file_size,
            file_mtime,
            width,
            height,
            final_timestamp,
            duration,
        ) in results:
            media_cache[rel_path] = {
                "size": file_size,
                "mtime": file_mtime,
                "width": width,
                "height": height,
                "timestamp": final_timestamp,
                "duration": duration,
            }
    else:
        print("[*] 所有媒体文件均已命中元数据缓存，跳过耗时解码！")

    # 6. 维护缓存一致性：自动移除磁盘上已删除文件的陈旧条目
    cleaned_cache = {
        rel: entry for rel, entry in media_cache.items() if rel in current_disk_set
    }
    purged_count = len(media_cache) - len(cleaned_cache)
    if purged_count > 0:
        print(f"[*] 缓存维护: 已从缓存中剔除 {purged_count} 个已删除文件的陈旧条目。")

    # 原子写入保存最新的缓存文件
    temp_cache_file = media_cache_dir / "assets_cache.json.tmp"
    with open(temp_cache_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_cache, f, ensure_ascii=False)
    temp_cache_file.replace(media_cache_file)

    # 7. 组装 items 媒体列表并融合标签
    items: list[dict[str, Any]] = []
    all_folder_paths: set[str] = {""}  # 根目录为 ""
    mismatched_tags_count = 0
    missing_tags_count = 0
    total_size = 0
    image_count = 0
    video_count = 0

    print("[*] 正在组装媒体数据并融合标签...")
    for idx, (full_path, rel_to_assets, media_type, file_size, file_mtime) in enumerate(
        file_info_list, start=1
    ):
        total_size += file_size
        ext = full_path.suffix.lower().lstrip(".")
        if media_type == "image":
            image_count += 1
        else:
            video_count += 1

        # 收集目录路径
        parent_rel = full_path.parent.relative_to(assets_dir).as_posix()
        folder_path = "" if parent_rel == "." else parent_rel
        if folder_path:
            curr = ""
            for part in folder_path.split("/"):
                curr = f"{curr}/{part}" if curr else part
                all_folder_paths.add(curr)

        # 直接从已维护的缓存中瞬时获取宽高、时长与拍摄时间
        cached_meta = cleaned_cache.get(rel_to_assets, {})
        width = cached_meta.get("width", 0)
        height = cached_meta.get("height", 0)
        final_timestamp = cached_meta.get("timestamp", file_mtime)
        duration = cached_meta.get("duration", 0.0)

        # 匹配 tags.json
        ai_tags: list[str] = []
        if tags_data is not None and media_type == "image":
            cached_entry = tags_cache.get(rel_to_assets)
            if cached_entry:
                c_size = cached_entry.get("size")
                c_mtime = cached_entry.get("mtime")
                if c_size != file_size or c_mtime != file_mtime:
                    mismatched_tags_count += 1
                    tqdm.write(
                        f"[警告] 图片信息不匹配: {full_path.resolve()}\n"
                        f"       磁盘: (size={file_size}, mtime={file_mtime}) vs 缓存: (size={c_size}, mtime={c_mtime})\n"
                        f"       可能 tags.json 不是最新的，请考虑重新运行 uv run tags.py"
                    )
                ai_tags = cached_entry.get("tags", [])
            else:
                missing_tags_count += 1
                tqdm.write(
                    f"[警告] 图片未在 tags.json 中找到: {full_path.resolve()}\n"
                    f"       该图片可能为新增文件，请考虑重新运行 uv run tags.py 进行打标"
                )

        # 标签融合：目录标签 + 形态属性标签 + AI标签
        dir_tags = extract_directory_tags(folder_path)
        aspect_tags = compute_aspect_tags(width, height, ext, media_type)

        merged_tags_list: list[str] = []
        seen_tags: set[str] = set()
        for t in dir_tags + ai_tags + aspect_tags:
            if t and t not in seen_tags:
                seen_tags.add(t)
                merged_tags_list.append(t)

        # URL 编码的访问安全路径
        safe_src = f"assets/{encode_url_path(rel_to_assets)}"

        item_obj = {
            "id": idx,
            "src": safe_src,
            "name": full_path.name,
            "folder": folder_path,
            "type": media_type,
            "ext": ext,
            "size": file_size,
            "mtime": final_timestamp,
            "width": width,
            "height": height,
            "duration": duration,
            "tags": merged_tags_list,
        }
        items.append(item_obj)

    if mismatched_tags_count > 0 or missing_tags_count > 0:
        print("\n" + "-" * 60)
        if missing_tags_count > 0:
            print(f"[提示] 共有 {missing_tags_count} 个图片文件未在 tags.json 中找到。")
        if mismatched_tags_count > 0:
            print(
                f"[提示] 共有 {mismatched_tags_count} 个图片文件属性与 tags.json 不匹配。"
            )
        print("       建议运行: uv run tags.py 更新标签缓存。")
        print("-" * 60 + "\n")

    # 5. 构建 folders 列表 (计算 directCount, totalCount, subFolderCount, cover)
    # 排序目录：根目录 "" 排第一，其余自然排序
    sorted_folder_paths = sorted(
        all_folder_paths,
        key=lambda p: (0 if p == "" else 1, natural_sort_key(p)),
    )

    folders: list[dict[str, Any]] = []
    for f_path in sorted_folder_paths:
        if f_path == "":
            f_name = "根目录"
            parent: str | None = None
            depth = 0
        else:
            parts = f_path.split("/")
            f_name = parts[-1]
            parent = "/".join(parts[:-1]) if len(parts) > 1 else ""
            depth = len(parts)

        # 直属子目录数量
        sub_folders = [
            p
            for p in all_folder_paths
            if p != f_path
            and (
                (f_path == "" and "/" not in p)
                or (p.startswith(f"{f_path}/") and len(p.split("/")) == depth + 1)
            )
        ]
        sub_folder_count = len(sub_folders)

        # 统计媒体数量
        direct_items = [it for it in items if it["folder"] == f_path]
        direct_count = len(direct_items)

        if f_path == "":
            total_items = items
        else:
            total_items = [
                it
                for it in items
                if it["folder"] == f_path or it["folder"].startswith(f"{f_path}/")
            ]
        folder_total_count = len(total_items)

        # 封面图：优先直属第一张图片，次选子目录第一张图片，无图则为 null
        cover: str | None = None
        for it in direct_items:
            if it["type"] == "image":
                cover = it["src"]
                break

        if cover is None:
            for it in total_items:
                if it["type"] == "image":
                    cover = it["src"]
                    break

        folder_obj = {
            "path": f_path,
            "name": f_name,
            "parent": parent,
            "depth": depth,
            "subFolderCount": sub_folder_count,
            "directCount": direct_count,
            "totalCount": folder_total_count,
            "cover": cover,
        }
        folders.append(folder_obj)

    # 6. 聚合全局 tags 索引 (按出现次数降序排列)
    tag_counts: dict[str, int] = {}
    for it in items:
        for t in it["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    sorted_tags = [
        {"name": t, "count": count}
        for t, count in sorted(
            tag_counts.items(), key=lambda x: (-x[1], natural_sort_key(x[0]))
        )
    ]

    # 7. 组装最终 PHOTO_WALL_DATA 结构
    now_iso = datetime.now(UTC).astimezone().isoformat()
    wall_data = {
        "meta": {
            "title": args.title,
            "generatedAt": now_iso,
            "basePath": "assets",
            "totalCount": len(items),
            "imageCount": image_count,
            "videoCount": video_count,
            "folderCount": len(folders),
            "tagCount": len(sorted_tags),
            "totalSize": total_size,
        },
        "folders": folders,
        "tags": sorted_tags,
        "items": items,
    }

    # 8. 无条件生成写入 assets.js (原子写入)
    temp_output_file = project_dir / "assets.js.tmp"
    js_content = (
        "// 由 assets.py 自动生成\nwindow.PHOTO_WALL_DATA = "
        + json.dumps(wall_data, ensure_ascii=False, indent=2)
        + ";\n"
    )

    with open(temp_output_file, "w", encoding="utf-8") as f:
        f.write(js_content)
    temp_output_file.replace(output_file)

    print("\n" + "=" * 60)
    print("[√] assets.js 生成成功！")
    print(f"    - 文件路径: {output_file.resolve()}")
    print(f"    - 相册标题: {args.title}")
    print(f"    - 媒体总数: {len(items)} (图片: {image_count}, 视频: {video_count})")
    print(f"    - 目录数量: {len(folders)}")
    print(f"    - 唯一标签: {len(sorted_tags)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
