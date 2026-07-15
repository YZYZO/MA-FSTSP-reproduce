"""为多个 H2H 原生测试文件提供一次性编译和产物路径夹具。"""

import platform
import subprocess
import sys
import threading
from pathlib import Path

from config import H2H_NATIVE_BUILD_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / 'scripts' / 'build_h2h_native.py'
_BUILD_LOCK = threading.Lock()
_BUILT_MODE: str | None = None


def native_artifacts() -> tuple[Path, Path]:
    """
    返回当前平台的 builder 与查询动态库路径。

    输入：无。
    输出：`(builder_path, library_path)`。
    """
    if platform.system() == 'Windows':
        return H2H_NATIVE_BUILD_DIR / 'h2h_builder.exe', H2H_NATIVE_BUILD_DIR / 'h2h_query.dll'
    return H2H_NATIVE_BUILD_DIR / 'h2h_builder', H2H_NATIVE_BUILD_DIR / 'libh2h_query.so'


def ensure_native_built(mode: str = 'release') -> tuple[Path, Path]:
    """
    在当前测试进程中按模式至多编译一次原生后端。

    输入：`release` 或 `debug`。
    输出：构建完成的 builder 与动态库路径。
    """
    global _BUILT_MODE
    normalized_mode = mode.lower()
    if normalized_mode not in {'release', 'debug'}:
        raise ValueError('mode 只能是 release 或 debug。')
    with _BUILD_LOCK:
        if _BUILT_MODE != normalized_mode:
            subprocess.run(
                [sys.executable, str(BUILD_SCRIPT), f'--{normalized_mode}'],
                cwd=PROJECT_ROOT,
                check=True,
            )
            _BUILT_MODE = normalized_mode
    builder_path, library_path = native_artifacts()
    if not builder_path.is_file() or not library_path.is_file():
        raise FileNotFoundError('H2H 原生测试产物未生成。')
    return builder_path, library_path
