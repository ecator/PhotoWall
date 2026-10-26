import ctypes
import os
import site
import sys
from pathlib import Path

_IS_LOADED = False
_DLL_HANDLES = []  # 保存 Windows add_dll_directory 的句柄，防止被垃圾回收失效


def setup_cuda_env() -> None:
    """
    跨平台动态挂载虚拟环境中的 CUDA / cuDNN 动态链接库（支持 Windows & Linux）。
    同时兼容 pip 安装的 `nvidia-*` 独立运行库以及 `torch/lib` 内置库。
    必须在创建 onnxruntime.InferenceSession 之前调用。
    """
    global _IS_LOADED
    if _IS_LOADED:
        return

    # 1. 收集当前虚拟环境中所有可能的 site-packages 根目录
    search_roots: list[Path] = []
    if hasattr(site, "getsitepackages"):
        search_roots.extend(Path(p) for p in site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        user_sp = site.getusersitepackages()
        if isinstance(user_sp, str):
            search_roots.append(Path(user_sp))
    # 兜底：加入 sys.path 中的 site-packages（兼容特殊嵌入式环境或 uv 独立环境）
    for p in sys.path:
        if "site-packages" in p or "dist-packages" in p:
            search_roots.append(Path(p))

    # 去重并保留真实存在的目录
    unique_roots = list({p.resolve() for p in search_roots if p.is_dir()})

    # 2. 扫描所有包含 CUDA/cuDNN 动态库的目录
    lib_dirs: list[Path] = []
    subdir_name = "bin" if sys.platform == "win32" else "lib"

    for root in unique_roots:
        # 来源 A: pip 安装的 nvidia-* 独立包 (如 nvidia/cudnn/bin 或 nvidia/cudnn/lib)
        nvidia_dir = root / "nvidia"
        if nvidia_dir.is_dir():
            for d in nvidia_dir.glob(f"*/{subdir_name}"):
                if d.is_dir() and d not in lib_dirs:
                    lib_dirs.append(d)

        # 来源 B: 如果环境里装了 PyTorch，顺便把 torch/lib 兜底加进来
        torch_lib = root / "torch" / "lib"
        if torch_lib.is_dir() and torch_lib not in lib_dirs:
            lib_dirs.append(torch_lib)

    if not lib_dirs:
        return

    # 3. 按操作系统执行对应的动态库挂载策略
    if sys.platform == "win32":
        _setup_windows(lib_dirs)
    elif sys.platform.startswith("linux"):
        _setup_linux(lib_dirs)

    _IS_LOADED = True


def _setup_windows(lib_dirs: list[Path]) -> None:
    """Windows: 注入 PATH 环境变量（解决 cuDNN 子引擎二级 LoadLibrary）并注册 DLL 目录"""
    dir_strs: list[str] = []
    for d in lib_dirs:
        d_str = str(d)
        dir_strs.append(d_str)
        if hasattr(os, "add_dll_directory"):
            try:
                # 必须将返回值保存在全局变量中，否则句柄被 GC 回收后目录注册会失效
                _DLL_HANDLES.append(os.add_dll_directory(d_str))
            except OSError:
                pass

    # 去重后一次性拼入 PATH 最前面
    current_path = os.environ.get("PATH", "")
    current_parts = current_path.split(os.pathsep)
    new_parts = [p for p in dir_strs if p not in current_parts]
    if new_parts:
        os.environ["PATH"] = os.pathsep.join(new_parts + [current_path])


def _setup_linux(lib_dirs: list[Path]) -> None:
    """
    Linux: 进程启动后修改 LD_LIBRARY_PATH 无效，
    必须按底层依赖顺序通过 ctypes.CDLL(..., RTLD_GLOBAL) 将 .so 载入进程全局内存空间。
    """
    # 严格按照依赖拓扑顺序排列（被依赖的基础库在前，高层库和子引擎在后）
    so_load_order = [
        "libcudart.so*",  # 1. CUDA Runtime 基础库
        "libnvrtc*.so*",  # 2. 运行时编译库
        "libcublasLt.so*",  # 3. cuBLAS Lt (被 libcublas 依赖，必须在前)
        "libcublas.so*",  # 4. cuBLAS 主库
        "libcufft.so*",  # 5. 傅里叶变换库
        "libcurand.so*",  # 6. 随机数库
        "libcusparse.so*",  # 7. 稀疏矩阵库
        "libcusolver.so*",  # 8. 线性求解器
        "libcudnn.so*",  # 9. cuDNN 主入口
        "libcudnn_graph.so*",  # 10. cuDNN 9 子模块（被 ops 和 engines 依赖）
        "libcudnn_ops.so*",  # 11. cuDNN 9 算子库
        "libcudnn_cnn.so*",  # 12. cuDNN 9 CNN 库
        "libcudnn_adv.so*",  # 13. cuDNN 9 高级算子库
        "libcudnn_heuristic.so*",  # 14. cuDNN 9 启发式搜索库
        "libcudnn_engines_precompiled.so*",  # 15. cuDNN 9 预编译引擎
        "libcudnn_engines_runtime_compiled.so*",  # 16. cuDNN 9 运行时编译引擎
    ]

    for pattern in so_load_order:
        for d in lib_dirs:
            # 排序以确保优先加载带具体主版本号的 .so（如 libcudnn.so.9）
            for so_path in sorted(d.glob(pattern)):
                if so_path.is_file():
                    try:
                        ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
