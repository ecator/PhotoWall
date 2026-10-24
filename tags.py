# /// script
# requires-python = "~=3.14.0"
# dependencies = [
#     "huggingface-hub>=0.20.0",
#     "numpy>=1.24.0",
#     "onnxruntime>=1.16.0",
#     "pillow>=10.0.0",
#     "tqdm>=4.66.0",
# ]
# ///
"""
tags.py - 照片墙 WD14 Tagger 自动增量打标脚本

使用方法:
    uv run tags.py            # 增量扫描 assets/ 并生成/更新 tags.json
    uv run tags.py --force    # 忽略缓存，强制重新对所有图片进行 AI 推理
    uv run tags.py --filter   # 不调用 AI 模型，仅根据最新的黑名单/同义词/阈值重算 tags

说明:
    1. 首次运行会自动从 HuggingFace 下载 WD14 ONNX 模型与词表 (约 300MB，支持 HF_ENDPOINT 镜像)。
    2. 具备智能增量缓存：仅对新增或修改过的图片调用 AI 推理；已扫描图片秒级跳过。
    3. 若仅修改 BLACKLIST / ALIASES / GENERAL_THRESHOLD，只需几毫秒即可从 raw 缓存重算。
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# 1. 内嵌配置区域 (可根据个人喜好随意调整，修改后无需重跑 AI 推理)
# ==============================================================================

# 模型仓库与文件 (默认推荐 SwinV2 v3，二次元立绘与人像识别效果极佳)
MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
MODEL_FILENAME = "model.onnx"
TAGS_FILENAME = "selected_tags.csv"

# 采样与过滤阈值 (高精简模式：大幅削减体积，只保留绝对高确信度特征)
BASE_THRESHOLD = (
    0.50  # 存入 raw 缓存的底线阈值 (由 0.35 调高至 0.50，直接剔除所有低频与微弱推测)
)
GENERAL_THRESHOLD = 0.60  # 最终进入 tags 的常规标签阈值 (由 0.48 调高至 0.60，只保留画面最显著的核心特征)
CHARACTER_THRESHOLD = 0.85  # 最终进入 tags 的角色名阈值 (设高防止误认)
MAX_TAGS_PER_IMAGE = 6  # 单张图片最终保留的最大标签数量 (照片墙展示最佳体验: 3~6 个)
MAX_RAW_TAGS_PER_IMAGE = (
    10  # 单张图片 raw 缓存最多保留的条目数 (双重防线，防止单图臃肿)
)

# 全局黑名单 (中英文均可匹配，命中的标签不会出现在 tags 中)
BLACKLIST: Set[str] = {
    # Danbooru 常见无意义构图/人数词
    "1girl",
    "solo",
    "monochrome",
    "greyscale",
    "comic",
    "photo",
    "looking_at_viewer",
    "smile",
    "open_mouth",
    "closed_mouth",
    "simple_background",
    "white_background",
    "grey_background",
    "blurry_background",
    "blurry",
    "depth_of_field",
    "indoors",
    # 中文对应词
    "单人",
    "单人女性",
    "黑白",
    "灰度",
    "看向观众",
    "微笑",
    "张嘴",
    "闭嘴",
    "简单背景",
    "白色背景",
    "模糊背景",
    "景深",
}

# 同义词归并映射 (原始词/译名 -> 规范词)
ALIASES: Dict[str, str] = {
    "水手服": "JK制服",
    "校服": "JK制服",
    "西装校服": "JK制服",
    "百褶裙": "百褶裙",
    "兔耳": "兔女郎",
    "泳装": "泳衣",
    "比基尼": "泳衣",
    "猫耳": "兽耳",
    "狐狸耳": "兽耳",
}

# 支持的图片格式 (包含 GIF 动图，Pillow 会自动抽取 GIF 第一帧用于打标)
IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# 基础内置常用 Danbooru 标签汉化字典 (无需额外下载，开箱即用)
BUILTIN_TRANSLATIONS: Dict[str, str] = {
    # 头发发色与发型
    "black_hair": "黑发",
    "blonde_hair": "金发",
    "brown_hair": "棕发",
    "blue_hair": "蓝发",
    "green_hair": "绿发",
    "pink_hair": "粉发",
    "purple_hair": "紫发",
    "red_hair": "红发",
    "white_hair": "白发",
    "silver_hair": "银发",
    "grey_hair": "灰发",
    "orange_hair": "橙发",
    "short_hair": "短发",
    "long_hair": "长发",
    "medium_hair": "中长发",
    "twintails": "双马尾",
    "ponytail": "单马尾",
    "side_ponytail": "侧马尾",
    "braid": "麻花辫",
    "twin_braids": "双麻花辫",
    "ahoge": "呆毛",
    "bob_cut": "波波头",
    "straight_hair": "黑长直",
    "wavy_hair": "卷发",
    # 眼睛颜色
    "blue_eyes": "蓝瞳",
    "red_eyes": "红瞳",
    "brown_eyes": "棕瞳",
    "green_eyes": "绿瞳",
    "purple_eyes": "紫瞳",
    "yellow_eyes": "金瞳",
    "amber_eyes": "琥珀瞳",
    "pink_eyes": "粉瞳",
    "black_eyes": "黑瞳",
    "heterochromia": "异色瞳",
    # 服装穿搭
    "school_uniform": "校服",
    "serafuku": "水手服",
    "sailor_collar": "水手领",
    "dress": "连衣裙",
    "skirt": "半身裙",
    "pleated_skirt": "百褶裙",
    "shirt": "衬衫",
    "white_shirt": "白衬衫",
    "t-shirt": "T恤",
    "sweater": "毛衣",
    "hoodie": "连帽卫衣",
    "jacket": "夹克外套",
    "coat": "大衣",
    "suit": "西装",
    "maid": "女仆装",
    "kimono": "和服",
    "yukata": "浴衣",
    "cheongsam": "旗袍",
    "swimsuit": "泳装",
    "bikini": "比基尼",
    "one-piece_swimsuit": "连体泳衣",
    "bunny_suit": "兔女郎",
    "cosplay": "Cosplay",
    "apron": "围裙",
    # 腿部与鞋袜
    "thighhighs": "过膝袜",
    "black_thighhighs": "黑丝过膝袜",
    "white_thighhighs": "白丝过膝袜",
    "pantyhose": "连裤袜",
    "black_pantyhose": "黑丝",
    "white_pantyhose": "白丝",
    "fishnet_pantyhose": "渔网袜",
    "socks": "短袜",
    "white_socks": "白袜",
    "kneehighs": "及膝袜",
    "barefoot": "赤脚",
    "boots": "靴子",
    "high_heels": "高跟鞋",
    "sneakers": "运动鞋",
    "loafers": "乐福鞋",
    # 饰品与特征
    "glasses": "眼镜",
    "sunglasses": "墨镜",
    "hair_ribbon": "发带",
    "hair_bow": "蝴蝶结发饰",
    "hairband": "发箍",
    "hair_ornament": "发饰",
    "choker": "项圈",
    "necklace": "项链",
    "earrings": "耳环",
    "gloves": "手套",
    "hat": "帽子",
    "beret": "贝雷帽",
    "cat_ears": "猫耳",
    "animal_ears": "兽耳",
    "tail": "尾巴",
    "wings": "翅膀",
    "halo": "光环",
    "bandage": "绷带",
    "bandaid": "创可贴",
    "mask": "口罩",
    # 动作与状态
    "standing": "站姿",
    "sitting": "坐姿",
    "lying": "躺卧",
    "kneeling": "跪姿",
    "squatting": "蹲姿",
    "peace_sign": "剪刀手",
    "waving": "招手",
    "holding_hands": "牵手",
    "hug": "拥抱",
    "sleeping": "熟睡",
    "eating": "进食",
    "drinking": "喝水",
    "pointing": "指向",
    "hands_on_hips": "叉腰",
    # 场景与氛围
    "outdoors": "户外",
    "sky": "天空",
    "blue_sky": "蓝天",
    "clouds": "白云",
    "sunset": "日落",
    "night": "夜晚",
    "stars": "星空",
    "beach": "海滩",
    "ocean": "海洋",
    "water": "水面",
    "forest": "森林",
    "tree": "树木",
    "flower": "花卉",
    "cherry_blossoms": "樱花",
    "city": "城市街景",
    "street": "街道",
    "room": "室内房间",
    "bed": "床",
    "classroom": "教室",
    "sunlight": "阳光",
}

# ==============================================================================
# 2. 图像预处理与模型推理辅助函数
# ==============================================================================

# 社区开源全量 Danbooru 汉化词库下载源 (收录 10 万 ~ 15 万+ 标签)
ONLINE_TRANSLATION_URLS = [
    # 1. Yellow-Rush/zh_CN-Tags (Stable Diffusion / Danbooru 社区主流全量汉化表)
    "https://raw.githubusercontent.com/Yellow-Rush/zh_CN-Tags/main/danbooru.csv",
    "https://ghproxy.net/https://raw.githubusercontent.com/Yellow-Rush/zh_CN-Tags/main/danbooru.csv",
    "https://fastly.jsdelivr.net/gh/Yellow-Rush/zh_CN-Tags@main/danbooru.csv",
    # 2. wangjue520/danbooru-zh-dict (12 万+ 标签精校中英字典)
    "https://raw.githubusercontent.com/wangjue520/danbooru-zh-dict/main/dist/danbooru_zh_dict.csv",
    "https://ghproxy.net/https://raw.githubusercontent.com/wangjue520/danbooru-zh-dict/main/dist/danbooru_zh_dict.csv",
]


def parse_translation_csv(file_path: Path) -> Dict[str, str]:
    """解析 tags_zh_cn.csv 汉化文件 (自动适配标准 2列、3列及带表头的字典格式)。"""
    translations: Dict[str, str] = {}
    try:
        # 使用 utf-8-sig 自动处理 Windows/Excel 可能带来的 UTF-8 BOM
        with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            header_checked = False
            tag_col = 0
            zh_col = 1

            for row in reader:
                if not row:
                    continue
                # 检查首行是否为表头 (如: 标签,提示词写法,分类,图数,中文名...)
                if not header_checked:
                    header_checked = True
                    first_cell = row[0].strip().lower()
                    if first_cell in ("标签", "tag", "name", "english"):
                        for idx, cell in enumerate(row):
                            c = cell.strip().lower()
                            if c in ("标签", "tag", "name", "english"):
                                tag_col = idx
                            elif c in (
                                "中文名",
                                "translation",
                                "chinese",
                                "cn_name",
                                "zh",
                            ):
                                zh_col = idx
                        continue

                if len(row) <= max(tag_col, zh_col):
                    continue

                tag = row[tag_col].strip().lower()
                if not tag or tag.startswith("#"):
                    continue

                # 动态识别提取中文列
                if len(row) == 2:
                    zh = row[1].strip()
                elif len(row) >= 3 and zh_col == 1 and row[1].isdigit():
                    # 第二列为数字分类时，取最后一列翻译
                    zh = row[-1].strip()
                else:
                    zh = row[zh_col].strip()

                if zh:
                    # 部分词表翻译含有逗号或竖线分隔的别名，取首选中文
                    primary_zh = zh.replace("|", ",").split(",")[0].strip()
                    if primary_zh:
                        translations[tag] = primary_zh
    except Exception as e:
        print(f"[警告] 解析翻译文件 {file_path.name} 失败: {e}")
    return translations


def load_translations(base_dir: Path, cache_dir: Path) -> Dict[str, str]:
    """
    加载中英翻译字典：
    1. 优先读取项目根目录下的本地文件: tags_zh_cn.csv / translations.json
    2. 其次读取缓存目录中的 .cache/tags_zh_cn.csv
    3. 若都不存在，尝试自动从网络下载 10 万+ 社区标准词库并保存为 tags_zh_cn.csv
    4. 若离线，回退到内置的常用词典
    """
    translations = dict(BUILTIN_TRANSLATIONS)

    # 1. 检查根目录用户自定义文件 (优先 tags_zh_cn.csv)
    local_csv_candidates = [
        base_dir / "tags_zh_cn.csv",
        base_dir / "zh_cn.csv",
        base_dir / "danbooru.csv",
    ]
    for csv_file in local_csv_candidates:
        if csv_file.is_file():
            print(f"[*] 读取本地汉化词表: {csv_file.resolve()}")
            translations.update(parse_translation_csv(csv_file))
            return translations

    custom_json = base_dir / "translations.json"
    if custom_json.is_file():
        try:
            with open(custom_json, "r", encoding="utf-8") as f:
                translations.update(json.load(f))
            print(f"[*] 读取本地自定义翻译: {custom_json.resolve()}")
            return translations
        except Exception as e:
            print(f"[警告] 读取 translations.json 失败: {e}")

    # 2. 检查缓存目录中的词库
    cached_csv = cache_dir / "tags_zh_cn.csv"
    if cached_csv.is_file():
        community_dict = parse_translation_csv(cached_csv)
        if community_dict:
            print(f"[*] 读取缓存汉化词表: {cached_csv.resolve()}")
            translations.update(community_dict)
            return translations

    # 3. 尝试从网络下载社区全量词库 (10万+ 标签，约 2MB)
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[*] 正在下载社区全量 Danbooru 汉化词库 (10万+ 标签)...")
    for url in ONLINE_TRANSLATION_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (PhotoWall-Tags-Downloader)"},
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                content = response.read()
                if len(content) > 10000:
                    with open(cached_csv, "wb") as f:
                        f.write(content)
                    print(f"[√] 汉化词库下载完成，已缓存至: {cached_csv.resolve()}")
                    translations.update(parse_translation_csv(cached_csv))
                    return translations
        except Exception:
            continue

    print("[*] 暂无法连接到在线词库，已使用内置常用高频词典兜底。")
    print(
        f"    (提示: 你也可以手动下载 tags_zh_cn.csv 放到: {(base_dir / 'tags_zh_cn.csv').resolve()} 生效)"
    )
    return translations


def get_model_and_tags(cache_dir: Path) -> Tuple[str, str]:
    """下载并获取 WD14 模型文件与词表路径 (自动缓存至本地)。"""
    print(f"[*] 检查模型文件: {MODEL_REPO}")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        cache_dir=cache_dir,
    )
    tags_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=TAGS_FILENAME,
        cache_dir=cache_dir,
    )
    print(f"    - 模型权重: {Path(model_path).resolve()}")
    print(f"    - 标签映射: {Path(tags_path).resolve()}")
    return model_path, tags_path


def load_tags_csv(tags_csv_path: str) -> Tuple[List[str], List[int]]:
    """读取 selected_tags.csv，获取标签名称列表及类别 (category)。"""
    tag_names: List[str] = []
    tag_categories: List[int] = []

    with open(tags_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_names.append(row["name"])
            tag_categories.append(int(row["category"]))

    return tag_names, tag_categories


def preprocess_image(image_path: Path, target_size: int, is_nchw: bool) -> np.ndarray:
    """使用 Pillow 对图片进行 BGR 转换、白底填充与缩放。"""
    with Image.open(image_path) as img:
        # 处理透明通道
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            img = img.convert("RGBA")
            canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
            canvas.alpha_composite(img)
            img_rgb = canvas.convert("RGB")
        else:
            img_rgb = img.convert("RGB")

        # 保持比例缩放长边到 target_size
        w, h = img_rgb.size
        scale = target_size / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img_resized = img_rgb.resize((nw, nh), Image.Resampling.BICUBIC)

        # 居中粘贴到 448x448 白色画布
        canvas_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        pad_x = (target_size - nw) // 2
        pad_y = (target_size - nh) // 2
        canvas_img.paste(img_resized, (pad_x, pad_y))

        # 转换为 BGR numpy 数组
        arr = np.array(canvas_img, dtype=np.float32)[:, :, ::-1]

        if is_nchw:
            arr = np.transpose(arr, (2, 0, 1))

        return np.expand_dims(arr, axis=0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 概率映射函数。"""
    return 1.0 / (1.0 + np.exp(-x))


# ==============================================================================
# 3. 标签清洗、过滤、翻译流水线 (纯 CPU 毫秒级)
# ==============================================================================


def filter_tags(
    raw_dict: Dict[str, float],
    translations: Dict[str, str],
) -> List[str]:
    """
    根据阈值、黑名单、翻译映射与同义词，将 raw 原始分数字典转换为最终的中文 tags 数组。
    """
    char_candidates: List[Tuple[str, float]] = []
    general_candidates: List[Tuple[str, float]] = []

    for raw_tag, score in raw_dict.items():
        is_char = raw_tag.startswith("char:")
        clean_name = raw_tag[5:] if is_char else raw_tag

        # 1. 阈值检查
        if is_char:
            if score < CHARACTER_THRESHOLD:
                continue
        else:
            if score < GENERAL_THRESHOLD:
                continue

        # 2. 英文黑名单过滤
        if clean_name in BLACKLIST or clean_name.replace("_", " ") in BLACKLIST:
            continue

        # 3. 翻译为中文 (未命中的保持单词或替换下划线)
        zh_name = translations.get(clean_name, clean_name.replace("_", " "))

        # 4. 中文黑名单过滤
        if zh_name in BLACKLIST:
            continue

        # 5. 同义词别名替换
        final_name = ALIASES.get(zh_name, zh_name)
        if final_name in BLACKLIST:
            continue

        if is_char:
            char_candidates.append((final_name, score))
        else:
            general_candidates.append((final_name, score))

    # 按置信度排序
    char_candidates.sort(key=lambda x: x[1], reverse=True)
    general_candidates.sort(key=lambda x: x[1], reverse=True)

    # 角色词优先排在最前，合并去重
    seen: Set[str] = set()
    result: List[str] = []

    for name, _ in char_candidates + general_candidates:
        if name not in seen:
            seen.add(name)
            result.append(name)
            if len(result) >= MAX_TAGS_PER_IMAGE:
                break

    return result


# ==============================================================================
# 4. 主流程逻辑
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="照片墙 WD14 Tagger 增量打标工具")
    parser.add_argument(
        "--force", action="store_true", help="强制对所有图片重新运行 AI 模型推理"
    )
    parser.add_argument(
        "--filter", action="store_true", help="仅重算过滤规则与翻译 (不调用 AI 模型)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出每个文件的完整路径 (FullName) 与缓存状态",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    assets_dir = project_dir / "assets"
    tags_file = project_dir / "tags.json"
    cache_dir = project_dir / ".cache" / "huggingface"

    if not assets_dir.is_dir():
        print(f"[错误] 未找到 assets 目录: {assets_dir.resolve()}")
        sys.exit(1)

    translations_cache_dir = project_dir / ".cache"
    translations = load_translations(project_dir, translations_cache_dir)

    # 1. 加载既有的 tags.json 缓存
    cached_data: Dict[str, Any] = {"meta": {}, "files": {}}
    if tags_file.exists():
        try:
            with open(tags_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            print(f"[*] 读取标签缓存文件: {tags_file.resolve()}")
        except Exception as e:
            print(f"[警告] tags.json 解析失败，将作为全新缓存处理: {e}")
    else:
        print(f"[*] 标签缓存文件尚未生成: {tags_file.resolve()}")

    prev_meta = cached_data.get("meta", {})
    files_cache: Dict[str, Dict[str, Any]] = cached_data.get("files", {})

    # 检查缓存是否整体失效 (模型变动 或 底线阈值下调)
    prev_model = prev_meta.get("model")
    prev_base = prev_meta.get("baseThreshold", 1.0)
    need_full_rebuild = args.force

    if not need_full_rebuild and prev_model and prev_model != MODEL_REPO:
        print(f"[*] 检测到模型已更换 ({prev_model} -> {MODEL_REPO})，需重新推理！")
        need_full_rebuild = True
    elif not need_full_rebuild and prev_base and BASE_THRESHOLD < prev_base:
        print(
            f"[*] 底线阈值下调 ({prev_base} -> {BASE_THRESHOLD})，旧 raw 数据不足，需重新推理！"
        )
        need_full_rebuild = True
    elif prev_base and BASE_THRESHOLD > prev_base:
        print(
            f"[*] 检测到底线阈值上调 ({prev_base} -> {BASE_THRESHOLD})，将自动裁剪 raw 中低分标签为文件瘦身..."
        )

    if need_full_rebuild:
        files_cache.clear()

    # 2. 扫描 assets 目录下的所有图片
    print(f"[*] 扫描目录: {assets_dir.resolve()}")
    disk_images: Dict[str, Tuple[Path, int, int]] = {}
    for p in assets_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            rel_path = p.relative_to(assets_dir).as_posix()
            stat = p.stat()
            disk_images[rel_path] = (p, stat.st_size, int(stat.st_mtime))

    # 3. 统计需要 AI 推理的文件与缓存命中文件
    cached_items: List[Tuple[str, Path]] = []
    to_infer: List[Tuple[str, Path, int, int]] = []
    for rel_path, (full_path, size, mtime) in disk_images.items():
        if args.filter:
            cached_items.append((rel_path, full_path))
            continue
        entry = files_cache.get(rel_path)
        if (
            not entry
            or entry.get("size") != size
            or entry.get("mtime") != mtime
            or "raw" not in entry
        ):
            to_infer.append((rel_path, full_path, size, mtime))
        else:
            cached_items.append((rel_path, full_path))

    print(
        f"[*] 发现图片总数: {len(disk_images)} 张 (命中缓存: {len(cached_items)} 张，待推理: {len(to_infer)} 张)"
    )

    if args.verbose:
        if cached_items:
            print(f"[*] 已命中缓存的文件列表 ({len(cached_items)} 个):")
            for _, fp in cached_items:
                print(f"    [缓存命中] {fp.resolve()}")
        if to_infer:
            print(f"[*] 待进行 AI 推理的文件列表 ({len(to_infer)} 个):")
            for _, fp, _, _ in to_infer:
                print(f"    [待推理]   {fp.resolve()}")

    # 4. 执行 AI 模型推理 (如果有需要推理的图片)
    if to_infer and not args.filter:
        print(f"[*] 需要进行 AI 打标的图片数量: {len(to_infer)}")
        model_path, tags_path = get_model_and_tags(cache_dir)
        tag_names, tag_categories = load_tags_csv(tags_path)

        # 启动 ONNX Runtime
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available_providers = ort.get_available_providers()
        active_providers = [p for p in providers if p in available_providers]
        print(f"[*] 启动 ONNX Runtime 推理会话 (执行器: {active_providers})...")
        session = ort.InferenceSession(model_path, providers=active_providers)

        input_node = session.get_inputs()[0]
        output_name = session.get_outputs()[0].name
        input_shape = input_node.shape
        # 判断 NCHW 还是 NHWC
        is_nchw = len(input_shape) == 4 and input_shape[1] == 3
        target_size = input_shape[2] if is_nchw else input_shape[1]
        if not isinstance(target_size, int) or target_size <= 0:
            target_size = 448

        print(
            f"[*] 开始批量打标 (输入尺寸: {target_size}x{target_size}, NCHW={is_nchw})..."
        )
        for rel_path, full_path, size, mtime in tqdm(
            to_infer, desc="AI 打标中", unit="图"
        ):
            try:
                img_tensor = preprocess_image(full_path, target_size, is_nchw)
                logits = session.run([output_name], {input_node.name: img_tensor})[0]
                probs = sigmoid(logits[0])

                # 提取 >= BASE_THRESHOLD 的标签
                raw_candidates: List[Tuple[str, float]] = []
                for idx, prob in enumerate(probs):
                    if prob >= BASE_THRESHOLD:
                        category = tag_categories[idx]
                        # category 9 为 rating 分级，不作为视觉标签
                        if category == 9:
                            continue
                        tag_name = tag_names[idx]
                        tag_key = f"char:{tag_name}" if category == 4 else tag_name
                        raw_candidates.append((tag_key, round(float(prob), 3)))

                # 仅保留最高分的前 MAX_RAW_TAGS_PER_IMAGE 项
                raw_candidates.sort(key=lambda x: x[1], reverse=True)
                raw_dict = dict(raw_candidates[:MAX_RAW_TAGS_PER_IMAGE])

                files_cache[rel_path] = {
                    "size": size,
                    "mtime": mtime,
                    "raw": raw_dict,
                    "tags": [],  # 下一步统一计算
                }
            except Exception as e:
                print(f"\n[警告] 图片处理失败 {full_path.resolve()}: {e}")
                files_cache[rel_path] = {
                    "size": size,
                    "mtime": mtime,
                    "raw": {},
                    "tags": [],
                }
    else:
        if args.filter:
            print("[*] 处于 --filter 模式，跳过 AI 模型推理，仅应用最新规则。")
        else:
            print("[*] 所有图片均已命中缓存，无需重新推理！")

    # 5. 清理磁盘上已删除的图片缓存，并统一步长重算 tags
    print("[*] 应用黑名单/同义词规则生成最终 tags 数组...")
    updated_files: Dict[str, Dict[str, Any]] = {}
    for rel_path, (full_path, size, mtime) in disk_images.items():
        entry = files_cache.get(rel_path)
        if not entry:
            continue
        raw_dict = entry.get("raw", {})
        # 裁剪掉低于当前 BASE_THRESHOLD 的条目，并仅保留最高分的前 MAX_RAW_TAGS_PER_IMAGE 个，极限瘦身
        sorted_raw = sorted(
            [(k, v) for k, v in raw_dict.items() if v >= BASE_THRESHOLD],
            key=lambda x: x[1],
            reverse=True,
        )[:MAX_RAW_TAGS_PER_IMAGE]
        pruned_raw = {k: round(v, 3) for k, v in sorted_raw}
        # 从 raw 重算 tags
        final_tags = filter_tags(pruned_raw, translations)
        updated_files[rel_path] = {
            "size": size,
            "mtime": mtime,
            "tags": final_tags,
            "raw": pruned_raw,
        }

    # 6. 写入 tags.json
    result_data = {
        "meta": {
            "model": MODEL_REPO,
            "generatedAt": datetime.now(timezone.utc).astimezone().isoformat(),
            "baseThreshold": BASE_THRESHOLD,
            "generalThreshold": GENERAL_THRESHOLD,
            "characterThreshold": CHARACTER_THRESHOLD,
            "maxTagsPerImage": MAX_TAGS_PER_IMAGE,
            "totalCount": len(updated_files),
        },
        "files": updated_files,
    }

    temp_tags_file = project_dir / "tags.json.tmp"
    with open(temp_tags_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    temp_tags_file.replace(tags_file)

    print(
        f"[√] 标签缓存已写入: {tags_file.resolve()} (共计索引 {len(updated_files)} 张图片)"
    )


if __name__ == "__main__":
    main()
