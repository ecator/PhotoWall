"""
tags.py - 照片墙 WD14 Tagger 自动增量打标脚本

使用方法:
    photowall tags            # 增量扫描 assets/ 并生成/更新 tags.json
    photowall tags --force    # 忽略缓存，强制重新对所有图片进行 AI 推理
    photowall tags --filter   # 不调用 AI 模型，仅根据最新的黑名单/同义词/阈值重算 tags

说明:
    1. 首次运行会自动从 HuggingFace 下载 WD14 ONNX 模型与词表 (约 300MB，支持 HF_ENDPOINT 镜像)。
    2. 具备智能增量缓存：仅对新增或修改过的图片调用 AI 推理；已扫描图片秒级跳过。
    3. 若仅修改 BLACKLIST / ALIASES / GENERAL_THRESHOLD，只需几毫秒即可从 raw 缓存重算。
"""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from photowall.version import __version__

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
DEFAULT_CHECKPOINT_INTERVAL = 100  # 默认每处理 100 张图片保存一个断点缓存 JSON

# 全局黑名单 (中英文均可匹配，命中的标签不会出现在 tags 中)
BLACKLIST: set[str] = {
    # Danbooru 常见无意义构图/人数词
    "1girl",
    "2girls",
    "multiple_girls",
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
    "1boy",
    "2boys",
    "multiple_boys",
}

# 同义词归并映射 (原始词/译名 -> 规范词)
ALIASES: dict[str, str] = {
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
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# 基础内置常用 Danbooru 标签汉化字典 (无需额外下载，开箱即用)
BUILTIN_TRANSLATIONS: dict[str, str] = {
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


def parse_translation_csv(file_path: Path) -> dict[str, str]:
    """解析 tags_zh_cn.csv 汉化文件 (自动适配标准 2列、3列及带表头的字典格式)。"""
    translations: dict[str, str] = {}
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


def load_translations(base_dir: Path, cache_dir: Path) -> dict[str, str]:
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


def get_model_and_tags(cache_dir: Path) -> tuple[str, str]:
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


def load_tags_csv(tags_csv_path: str) -> tuple[list[str], list[int]]:
    """读取 selected_tags.csv，获取标签名称列表及类别 (category)。"""
    tag_names: list[str] = []
    tag_categories: list[int] = []

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
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
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


def create_inference_session(
    model_path: str, device: str = "auto"
) -> tuple[ort.InferenceSession, str]:
    """
    创建 ONNX Runtime 会话，严格贯彻 GPU 优先策略：
    - auto: 依次尝试 CUDA -> TensorRT -> DirectML -> CPU
    - cuda: 优先 CUDA
    - dml:  优先 DirectML (基于 DirectX 12，Windows 原生硬件加速，免安装 CUDA Toolkit)
    - cpu:  使用 CPU

    返回: (session, active_provider_name)
    """
    available_providers = ort.get_available_providers()

    gpu_provider_candidates: list[tuple[str, dict[str, Any]]] = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            },
        ),
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
            },
        ),
        (
            "DmlExecutionProvider",
            {
                "device_id": 0,
            },
        ),
    ]

    # 根据指定的 device 过滤候选列表
    targets: list[tuple[str, dict[str, Any]]] = []
    if device == "cuda":
        targets = [
            c for c in gpu_provider_candidates if c[0] == "CUDAExecutionProvider"
        ]
    elif device == "dml":
        targets = [c for c in gpu_provider_candidates if c[0] == "DmlExecutionProvider"]
    elif device == "cpu":
        targets = []
    else:  # "auto"
        targets = gpu_provider_candidates

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 1. 尝试 GPU 执行器
    for provider_name, provider_opts in targets:
        if provider_name in available_providers:
            try:
                # 仅传入单个 GPU provider + CPU 兜底
                sess = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=[(provider_name, provider_opts), "CPUExecutionProvider"],
                )
                active = sess.get_providers()
                if active and active[0] == provider_name:
                    return sess, provider_name
            except Exception:
                pass

    # 2. 回退到 CPU
    sess = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    return sess, "CPUExecutionProvider"


# ==============================================================================
# 3. 标签清洗、过滤、翻译流水线 (纯 CPU 毫秒级)
# ==============================================================================


def filter_tags(
    raw_dict: dict[str, float],
    translations: dict[str, str],
) -> list[str]:
    """
    根据阈值、黑名单、翻译映射与同义词，将 raw 原始分数字典转换为最终的中文 tags 数组。
    """
    char_candidates: list[tuple[str, float]] = []
    general_candidates: list[tuple[str, float]] = []

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
    seen: set[str] = set()
    result: list[str] = []

    for name, _ in char_candidates + general_candidates:
        if name not in seen:
            seen.add(name)
            result.append(name)
            if len(result) >= MAX_TAGS_PER_IMAGE:
                break

    return result


# ==============================================================================
# 4. 临时断点缓存管理 (.cache/tags/*.json)
# ==============================================================================


def get_checkpoint_dir(project_dir: Path) -> Path:
    """获取断点缓存目录 (.cache/tags)。"""
    return project_dir / ".cache" / "tags"


def load_checkpoints(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    """
    扫描并加载 .cache/tags 目录下的所有临时 checkpoint JSON 文件。
    返回合并后的图片元数据字典。
    """
    merged: dict[str, dict[str, Any]] = {}
    if not checkpoint_dir.is_dir():
        return merged

    json_files = sorted(checkpoint_dir.glob("*.json"))
    for jf in json_files:
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
) -> Path | None:
    """原子化写入单个断点缓存 JSON 文件。"""
    if not chunk_data:
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_file = checkpoint_dir / f"checkpoint_{chunk_id}.json"
    temp_file = checkpoint_dir / f"checkpoint_{chunk_id}.json.tmp"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        temp_file.replace(target_file)
        return target_file
    except Exception as e:
        print(f"\n[警告] 保存断点缓存失败 ({target_file.name}): {e}")
        return None


def clear_checkpoints(checkpoint_dir: Path) -> int:
    """清空断点缓存目录下的所有临时 JSON 及 TMP 文件。"""
    if not checkpoint_dir.is_dir():
        return 0
    count = 0
    for f in checkpoint_dir.iterdir():
        if f.is_file() and (f.suffix == ".json" or f.name.endswith(".tmp")):
            try:
                f.unlink()
                count += 1
            except Exception as e:
                print(f"[警告] 清理断点临时文件 {f.name} 失败: {e}")
    return count


# ==============================================================================
# 5. 主流程逻辑
# ==============================================================================


def build_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """构建 tags 命令的命令行参数解析器。"""
    if parser is None:
        parser = argparse.ArgumentParser(
            description=f"PhotoWall v{__version__} - 照片墙 WD14 Tagger 增量打标工具"
        )
    parser.add_argument(
        "--dir",
        "-d",
        "--home",
        type=Path,
        default=None,
        help="项目根目录路径 (默认: 当前工作目录)",
    )
    parser.add_argument(
        "--force", action="store_true", help="强制对所有图片重新运行 AI 模型推理"
    )
    parser.add_argument(
        "--filter", action="store_true", help="仅重算过滤规则与翻译 (不调用 AI 模型)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "dml", "cpu"],
        default="auto",
        help="指定推理设备 (auto: 优先 GPU/CUDA/DirectML, 自动回退 CPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="推理 Batch 大小 (默认: 0 自动选择，GPU 为 4，CPU 为 1)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"断点续传缓存步长 (默认: 每 {DEFAULT_CHECKPOINT_INTERVAL} 张图片保存一个 checkpoint)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出每个文件的完整路径 (FullName) 与缓存状态",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """执行 WD14 Tagger 打标主流程。"""
    project_dir = (
        (getattr(args, "home", None) or getattr(args, "dir", None)) or Path.cwd()
    ).resolve()
    assets_dir = project_dir / "assets"
    tags_file = project_dir / "tags.json"
    cache_dir = project_dir / ".cache" / "huggingface"
    checkpoint_dir = get_checkpoint_dir(project_dir)
    checkpoint_interval = getattr(
        args, "checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL
    )
    if checkpoint_interval <= 0:
        checkpoint_interval = DEFAULT_CHECKPOINT_INTERVAL

    if not assets_dir.is_dir():
        print(f"[错误] 未找到 assets 目录: {assets_dir.resolve()}")
        sys.exit(1)

    print(f"[*] PhotoWall v{__version__} - 开始标签扫描与处理")

    translations_cache_dir = project_dir / ".cache"
    translations = load_translations(project_dir, translations_cache_dir)

    # 1. 加载既有的 tags.json 缓存
    cached_data: dict[str, Any] = {"meta": {}, "files": {}}
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
    files_cache: dict[str, dict[str, Any]] = cached_data.get("files", {})

    # 检查缓存是否整体失效 (模型变动 或 底线阈值下调)
    prev_model = prev_meta.get("model")
    prev_base = prev_meta.get("baseThreshold")
    need_full_rebuild = args.force

    if not need_full_rebuild and prev_model and prev_model != MODEL_REPO:
        print(f"[*] 检测到模型已更换 ({prev_model} -> {MODEL_REPO})，需重新推理！")
        need_full_rebuild = True
    elif not need_full_rebuild and prev_base is not None and BASE_THRESHOLD < prev_base:
        print(
            f"[*] 底线阈值下调 ({prev_base} -> {BASE_THRESHOLD})，旧 raw 数据不足，需重新推理！"
        )
        need_full_rebuild = True
    elif prev_base is not None and BASE_THRESHOLD > prev_base:
        print(
            f"[*] 检测到底线阈值上调 ({prev_base} -> {BASE_THRESHOLD})，将自动裁剪 raw 中低分标签为文件瘦身..."
        )

    if need_full_rebuild:
        files_cache.clear()
        cleaned = clear_checkpoints(checkpoint_dir)
        if cleaned > 0:
            print(f"[*] 强制重建模式，已清空历史断点缓存 ({cleaned} 个分块文件)")
    else:
        # 加载断点缓存 (.cache/tags/*.json)
        checkpoint_data = load_checkpoints(checkpoint_dir)
        if checkpoint_data:
            print(
                f"[*] 从断点缓存 ({checkpoint_dir.resolve()}) 恢复了 {len(checkpoint_data)} 张图片的临时打标数据"
            )
            files_cache.update(checkpoint_data)

    # 2. 扫描 assets 目录下的所有图片
    print(f"[*] 扫描目录: {assets_dir.resolve()}")
    disk_images: dict[str, tuple[Path, int, int]] = {}
    for p in assets_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            rel_path = p.relative_to(assets_dir).as_posix()
            stat = p.stat()
            disk_images[rel_path] = (p, stat.st_size, int(stat.st_mtime))

    # 3. 统计需要 AI 推理的文件与缓存命中文件
    cached_items: list[tuple[str, Path]] = []
    to_infer: list[tuple[str, Path, int, int]] = []
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

        # 启动 ONNX Runtime (GPU 优先策略)
        print(f"[*] 启动 ONNX Runtime 推理会话 (请求模式: {args.device})...")
        session, active_provider = create_inference_session(
            model_path, device=args.device
        )
        is_gpu = active_provider in (
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "DmlExecutionProvider",
        )

        if is_gpu:
            print(f"[√] 成功启用 GPU 加速执行器: {active_provider}")
        else:
            print(f"[*] 当前使用执行器: {active_provider}")
            if args.device != "cpu":
                print("    [提示] GPU 执行环境未能载入。若您使用的是 NVIDIA 显卡:")
                print(
                    "           - 方案1 (推荐，免配直接跑 GPU): 运行 `uv pip install onnxruntime-directml`"
                )
                print(
                    "           - 方案2 (CUDA 原生加速): 安装 NVIDIA CUDA 13.x Toolkit 与 cuDNN 9.x 并配置 PATH"
                )

        batch_size = args.batch_size
        if batch_size <= 0:
            batch_size = 4 if is_gpu else 1
        print(f"[*] 推理批次大小 (Batch Size): {batch_size}")

        input_node = session.get_inputs()[0]
        output_name = session.get_outputs()[0].name
        input_shape = input_node.shape
        # 判断 NCHW 还是 NHWC
        is_nchw = len(input_shape) == 4 and input_shape[1] == 3
        target_size = input_shape[2] if is_nchw else input_shape[1]
        if not isinstance(target_size, int) or target_size <= 0:
            target_size = 448

        print(
            f"[*] 开始打标流程 (输入尺寸: {target_size}x{target_size}, NCHW={is_nchw}, 硬件: {active_provider})..."
        )

        checkpoint_buffer: dict[str, dict[str, Any]] = {}
        checkpoint_counter = 0

        def flush_checkpoint(counter: int) -> None:
            nonlocal checkpoint_buffer
            if not checkpoint_buffer:
                return
            ts = int(datetime.now().timestamp())
            cid = f"{ts}_{counter:04d}"
            save_checkpoint(checkpoint_dir, checkpoint_buffer, cid)
            checkpoint_buffer = {}

        def record_result(rel_path: str, item_data: dict[str, Any]) -> None:
            nonlocal checkpoint_counter
            files_cache[rel_path] = item_data
            checkpoint_buffer[rel_path] = item_data
            if len(checkpoint_buffer) >= checkpoint_interval:
                checkpoint_counter += 1
                flush_checkpoint(checkpoint_counter)

        try:
            with tqdm(
                total=len(to_infer),
                desc=f"AI 打标中 ({active_provider})",
                unit="图",
            ) as pbar:
                for i in range(0, len(to_infer), batch_size):
                    chunk = to_infer[i : i + batch_size]
                    batch_tensors: list[np.ndarray] = []
                    valid_items: list[tuple[str, Path, int, int]] = []

                    for rel_path, full_path, size, mtime in chunk:
                        try:
                            tensor = preprocess_image(full_path, target_size, is_nchw)
                            batch_tensors.append(tensor)
                            valid_items.append((rel_path, full_path, size, mtime))
                        except Exception as e:
                            print(f"\n[警告] 图片预处理失败 {full_path.resolve()}: {e}")
                            item_data = {
                                "size": size,
                                "mtime": mtime,
                                "raw": {},
                                "tags": [],
                            }
                            record_result(rel_path, item_data)
                            pbar.update(1)

                    if not batch_tensors:
                        continue

                    try:
                        batched_input = (
                            np.concatenate(batch_tensors, axis=0)
                            if len(batch_tensors) > 1
                            else batch_tensors[0]
                        )
                        logits = session.run(
                            [output_name], {input_node.name: batched_input}
                        )[0]

                        for idx_in_batch, (
                            rel_path,
                            full_path,
                            size,
                            mtime,
                        ) in enumerate(valid_items):
                            probs = sigmoid(logits[idx_in_batch])

                            raw_candidates: list[tuple[str, float]] = []
                            for idx, prob in enumerate(probs):
                                if prob >= BASE_THRESHOLD:
                                    category = tag_categories[idx]
                                    if category == 9:
                                        continue
                                    tag_name = tag_names[idx]
                                    tag_key = (
                                        f"char:{tag_name}"
                                        if category == 4
                                        else tag_name
                                    )
                                    raw_candidates.append(
                                        (tag_key, round(float(prob), 3))
                                    )

                            raw_candidates.sort(key=lambda x: x[1], reverse=True)
                            raw_dict = dict(raw_candidates[:MAX_RAW_TAGS_PER_IMAGE])

                            item_data = {
                                "size": size,
                                "mtime": mtime,
                                "raw": raw_dict,
                                "tags": [],
                            }
                            record_result(rel_path, item_data)
                            pbar.update(1)
                    except Exception as e:
                        print(f"\n[警告] 批次推理异常，自动降级为单图模式: {e}")
                        for t, (rel_path, full_path, size, mtime) in zip(
                            batch_tensors, valid_items
                        ):
                            try:
                                l = session.run([output_name], {input_node.name: t})[0]
                                probs = sigmoid(l[0])
                                raw_candidates = []
                                for idx, prob in enumerate(probs):
                                    if prob >= BASE_THRESHOLD:
                                        category = tag_categories[idx]
                                        if category == 9:
                                            continue
                                        tag_name = tag_names[idx]
                                        tag_key = (
                                            f"char:{tag_name}"
                                            if category == 4
                                            else tag_name
                                        )
                                        raw_candidates.append(
                                            (tag_key, round(float(prob), 3))
                                        )
                                raw_candidates.sort(key=lambda x: x[1], reverse=True)
                                raw_dict = dict(raw_candidates[:MAX_RAW_TAGS_PER_IMAGE])
                                item_data = {
                                    "size": size,
                                    "mtime": mtime,
                                    "raw": raw_dict,
                                    "tags": [],
                                }
                                record_result(rel_path, item_data)
                            except Exception as sub_e:
                                print(
                                    f"\n[错误] 图片处理失败 {full_path.resolve()}: {sub_e}"
                                )
                                item_data = {
                                    "size": size,
                                    "mtime": mtime,
                                    "raw": {},
                                    "tags": [],
                                }
                                record_result(rel_path, item_data)
                            finally:
                                pbar.update(1)

            # 正常推理完成：将未满分块的剩余数据存盘
            if checkpoint_buffer:
                checkpoint_counter += 1
                flush_checkpoint(checkpoint_counter)

        except KeyboardInterrupt:
            print("\n[!] 检测到用户中断操作 (Ctrl+C)。")
            if checkpoint_buffer:
                checkpoint_counter += 1
                flush_checkpoint(checkpoint_counter)
                print(
                    f"[*] 已将未满分块的临时进度 ({len(checkpoint_buffer)} 张图片) 保存至断点缓存: {checkpoint_dir.resolve()}"
                )
            print("[*] 下次启动打标时将自动从断点继续。")
            sys.exit(130)
    else:
        if args.filter:
            print("[*] 处于 --filter 模式，跳过 AI 模型推理，仅应用最新规则。")
        else:
            print("[*] 所有图片均已命中缓存，无需重新推理！")

    # 5. 清理磁盘上已删除的图片缓存，并统一步长重算 tags
    print("[*] 应用黑名单/同义词规则生成最终 tags 数组...")
    updated_files: dict[str, dict[str, Any]] = {}
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
            "generatedAt": datetime.now(UTC).astimezone().isoformat(),
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

    # 7. 清理临时断点缓存
    cleaned_count = clear_checkpoints(checkpoint_dir)
    if cleaned_count > 0:
        print(
            f"[*] 标签生成完毕，已清理 {cleaned_count} 个断点分块缓存文件: {checkpoint_dir.resolve()}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
