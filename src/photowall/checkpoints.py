"""
checkpoints.py - 统一的断点缓存 (Checkpoints) 管理模块
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .scanner import natural_sort_key

DEFAULT_CHECKPOINT_INTERVAL = 100
CHECKPOINT_RETENTION_DAYS = 7


def get_checkpoint_dir(home_dir: Path, category: str) -> Path:
    """获取指定类别 (assets / tags) 的断点缓存目录 (.photowall/checkpoints/<category>)。"""
    return home_dir / ".photowall" / "checkpoints" / category


def load_checkpoints(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    """
    扫描并加载指定断点缓存目录下的所有 checkpoint JSON 文件。
    当多个 checkpoints 遇到重复的文件名时，按文件修改时间（及文件名自然序）以时间最新的为准合并。
    """
    merged: dict[str, dict[str, Any]] = {}
    if not checkpoint_dir.is_dir():
        return merged

    json_files: list[tuple[int, list[Any], Path]] = []
    for jf in checkpoint_dir.glob("*.json"):
        try:
            mtime_ns = jf.stat().st_mtime_ns
        except OSError:
            continue
        json_files.append((mtime_ns, natural_sort_key(jf.name), jf))

    # 按修改时间从旧到新排序，相同时间按文件名自然排序，后续 update 让最新时间覆盖旧条目
    json_files.sort(key=lambda item: (item[0], item[1]))

    for _, _, jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    merged.update(data)
        except Exception as e:
            print(f"[警告] 读取断点缓存文件 {jf.name} 失败: {e}")
    return merged


def save_checkpoint(
    checkpoint_dir: Path,
    chunk_data: dict[str, dict[str, Any]],
    chunk_id: str,
    prefix: str = "checkpoint",
) -> Path | None:
    """原子化写入单个断点缓存 JSON 文件 (如 assets_xxx.json / tags_xxx.json)。"""
    if not chunk_data:
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = chunk_id if chunk_id.startswith(f"{prefix}_") else f"{prefix}_{chunk_id}"
    target_file = checkpoint_dir / f"{stem}.json"
    temp_file = checkpoint_dir / f"{stem}.json.tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        temp_file.replace(target_file)
        return target_file
    except Exception as e:
        print(f"\n[警告] 保存断点缓存失败 ({target_file.name}): {e}")
        return None


def cleanup_expired_checkpoints(
    checkpoint_dir: Path,
    max_age_days: int = CHECKPOINT_RETENTION_DAYS,
) -> int:
    """删除超过指定天数（默认 7 天）的过期断点缓存文件及残留临时文件。"""
    if not checkpoint_dir.is_dir():
        return 0
    now_ts = datetime.now().timestamp()
    max_age_seconds = max_age_days * 86400
    count = 0
    for f in checkpoint_dir.iterdir():
        if not f.is_file():
            continue
        if f.name.endswith(".tmp"):
            try:
                f.unlink()
                count += 1
            except Exception as e:
                print(f"[警告] 清理断点临时文件 {f.name} 失败: {e}")
        elif f.suffix == ".json":
            try:
                if now_ts - f.stat().st_mtime > max_age_seconds:
                    f.unlink()
                    count += 1
            except Exception as e:
                print(f"[警告] 清理过期断点缓存文件 {f.name} 失败: {e}")
    return count
