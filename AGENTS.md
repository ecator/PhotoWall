# AGENTS.md

一个生成照片元数据让后显示的照片墙工具。

## 开发环境

使用`uv`管理，`ruff`进行代码规范检查，`pytest`进行单元测试。

## 编码规范

- **模块导入**：同一个 package 内部必须使用相对导入方式（例如 `from . import xxx`、`from .version import __version__`）；外部引用（如 `tests` 目录）使用绝对导入。
