# AGENTS.md

基于预生成元数据的轻量级纯本地单页照片墙，无需服务端，浏览器打开 `index.html` 即可浏览。

## 架构与核心模块

- **`photowall.cli`**：CLI 统一入口，提供 `index`、`assets`、`tags`、`version` 子命令。
- **`photowall.index`**：全流程构建入口，协调 `tags` 与 `assets` 并部署内置 `index.html`。
- **`photowall.tags`**：基于 WD14 Tagger (ONNX) 增量生成 AI 视觉标签，支持断点缓存，输出 `tags.json`。
- **`photowall.assets`**：扫描 `assets/` 媒体文件，解析 Exif/尺寸/时长，融合标签输出 `assets.js`。
- **`photowall.scanner`**：统一媒体扫描器与格式定义（支持软链接/Junction 递归遍历与自然排序）。
- **`photowall.version`**：动态版本号管理（支持 VCS 与包元数据）。
- **`photowall/index.html`**：内置的前端单页应用模板。

## 数据流与核心产物

```
assets/ (媒体文件)
   ├─► tags.py   ──► tags.json (.photowall/ 增量与断点缓存)
   └─► assets.py ──► assets.js (融合标签与媒体元数据)
index.py ─────────► 复制 index.html 至相册根目录
```

## 开发与测试

- **环境管理**：`uv`（Python 限制 `~=3.14.0`，部分依赖不支持 3.15+）
- **单元测试**：`uv run pytest`
- **代码检查**：`uv run ruff check` / `uv run ruff format`
- **本地调试**：`uv run photowall index --home <dir>`

## 编码规范

- **模块导入**：
  - 包内模块之间必须使用**相对导入**（如 `from . import assets`、`from .version import __version__`）。
  - 外部引用（如 `tests/` 目录）必须使用**绝对导入**（如 `from photowall.cli import build_parser`）。
- **路径与编码**：媒体相对路径输出至前端时统一转为 POSIX 格式并进行 URL 安全编码，避免特殊字符导致资源 404。
- **增量缓存**：优先复用 `.photowall/` 缓存，避免重复进行 AI 推理或全量扫描。
