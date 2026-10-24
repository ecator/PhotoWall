# 照片墙

这是一个纯本地单页`index.html`+`assets.js`的照片墙应用，通过`photowall index`子命令扫描相册目录生成`tags.json`、`assets.js`并输出`index.html`，无需服务端即可在浏览器中展示照片。

## 常用命令

- `photowall index`：一键全量索引。从指定相册根目录（`--home`，默认为当前工作目录）查找 `assets/` 媒体文件，自动进行 AI 打标生成 `tags.json`、提取元数据生成 `assets.js`，并输出 `index.html` 网页文件。
  - `--home`：相册根目录路径（可选，默认当前目录）。
  - `--title`：相册标题（可选，默认使用相册文件夹名）。
  - `--force`：强制重新推理 AI 标签并重新解析所有媒体文件。
  - `--skip-tags`：跳过 AI 标签生成步骤。
- `photowall tags`：仅使用 WD14 Tagger 自动为图片打标并生成 `tags.json`。
- `photowall assets`：仅扫描 `assets/` 目录并融合标签生成 `assets.js`。

## 快速开始

1. 安装 `photowall`：
   ```bash
   uv tool install photowall
   ```
2. 把图片和视频放进 `assets` 文件夹，不需要特别分门别类，直接扔进去就行。
   > 最开始没有 `assets` 文件夹需要手动创建。
3. 运行 `photowall index` 生成相册索引。
4. 打开生成的 `index.html` 即可。
   > 如果更新了 `assets` 的内容则再次运行 `photowall index` 即可。
5. 如果想启动本地服务器也可以直接运行 `start.bat`，会自动启动本地服务器并使用浏览器打开。
