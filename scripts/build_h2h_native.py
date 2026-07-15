"""使用标准库直接调用 g++ 构建 H2H builder 与查询动态库。"""

from __future__ import annotations

import argparse
import ctypes
import math
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import H2H_CXX, H2H_NATIVE_BUILD_DIR  # noqa: E402


SOURCE_DIR = PROJECT_ROOT / 'native' / 'h2h' / 'src'
INCLUDE_DIR = PROJECT_ROOT / 'native' / 'h2h' / 'include'
SOURCE_NAMES = (
    'h2h_graph.cpp',
    'h2h_index.cpp',
    'h2h_builder.cpp',
    'h2h_query.cpp',
    'h2h_c_api.cpp',
)


def _parse_arguments() -> argparse.Namespace:
    """
    解析一键构建脚本参数。

    输入：当前进程命令行。
    输出：包含编译器、构建目录、模式和日志选项的命名空间。
    """
    parser = argparse.ArgumentParser(description='构建 MA-FSTSP 的 H2H 原生后端。')
    parser.add_argument('--compiler', type=Path, help='显式指定 g++ 路径。')
    parser.add_argument('--build-dir', type=Path, default=H2H_NATIVE_BUILD_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--debug', action='store_true', help='使用 -O0 -g。')
    mode.add_argument('--release', action='store_true', help='使用 -O3 -DNDEBUG（默认）。')
    parser.add_argument(
        '--sanitize',
        action='store_true',
        help='增加 AddressSanitizer/UndefinedBehaviorSanitizer，用于阶段 3 内存检查。',
    )
    parser.add_argument('--clean', action='store_true', help='只删除本脚本生成的已知产物。')
    parser.add_argument('--verbose', action='store_true', help='打印完整编译与链接参数。')
    return parser.parse_args()


def _resolve_compiler(explicit: Path | None) -> Path:
    """
    按命令行、配置、Windows 默认路径和 PATH 顺序定位 g++。

    输入：可选显式路径。
    输出：已存在的编译器绝对路径。
    """
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend((H2H_CXX, Path(r'D:\dev\mingw64\bin\g++.exe')))
    path_compiler = shutil.which('g++')
    if path_compiler:
        candidates.append(Path(path_compiler))

    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.is_file():
            return expanded.resolve()
    rendered = ', '.join(str(candidate) for candidate in candidates) or '<none>'
    raise FileNotFoundError(f'没有找到 g++；已检查：{rendered}')


def _artifact_paths(build_dir: Path) -> tuple[Path, Path, Path]:
    """
    根据当前平台返回对象目录、builder 和查询库路径。

    输入：构建根目录。
    输出：`(obj_dir, builder_path, library_path)`。
    """
    obj_dir = build_dir / 'obj'
    if platform.system() == 'Windows':
        return obj_dir, build_dir / 'h2h_builder.exe', build_dir / 'h2h_query.dll'
    return obj_dir, build_dir / 'h2h_builder', build_dir / 'libh2h_query.so'


def _clean_known_outputs(build_dir: Path) -> None:
    """
    只删除脚本自身生成且名称固定的目标文件。

    输入：经过 `resolve()` 的构建目录。
    输出：无；不会递归删除构建目录中的未知文件。
    """
    obj_dir, builder_path, library_path = _artifact_paths(build_dir)
    known_files = [builder_path, library_path]
    known_files.extend(obj_dir / f'{Path(name).stem}.o' for name in SOURCE_NAMES)
    for path in known_files:
        if path.is_file():
            path.unlink()
    if obj_dir.is_dir() and not any(obj_dir.iterdir()):
        obj_dir.rmdir()
    if build_dir.is_dir() and not any(build_dir.iterdir()):
        build_dir.rmdir()


def _run(command: list[str], verbose: bool) -> None:
    """
    以 shell-free 参数列表运行编译或链接命令。

    输入：命令数组和是否打印完整参数。
    输出：无；子进程非零时保留原输出并抛出异常。
    """
    if verbose:
        print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def _write_smoke_graph(path: Path) -> None:
    """
    写入不接触正式数据的两节点非对称 graph.bin。

    输入：临时文件路径。
    输出：无；图中 0->1 为 1.25，1->0 为 2.5。
    """
    header = struct.pack('<8sIIIQ', b'H2HGRPH1', 1, 0x01020304, 2, 2)
    edges = b''.join((
        struct.pack('<IId', 0, 1, 1.25),
        struct.pack('<IId', 1, 0, 2.5),
    ))
    path.write_bytes(header + edges)


def _run_smoke_test(builder_path: Path, library_path: Path, verbose: bool) -> None:
    """
    构建临时两节点索引并通过 C ABI 查询两个方向。

    输入：builder、动态库路径与日志开关。
    输出：无；任一版本、打开或距离断言失败即使构建命令失败。
    """
    with tempfile.TemporaryDirectory(prefix='h2h-native-smoke-') as temporary_directory:
        temporary = Path(temporary_directory)
        graph_path = temporary / 'graph.bin'
        index_path = temporary / 'index.bin'
        _write_smoke_graph(graph_path)
        _run([
            str(builder_path), '--graph', str(graph_path), '--output', str(index_path),
            '--max-nodes', '10', '--progress-interval', '0',
        ], verbose)

        library = ctypes.CDLL(str(library_path))
        library.h2h_api_version.argtypes = []
        library.h2h_api_version.restype = ctypes.c_uint32
        library.h2h_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        library.h2h_open.restype = ctypes.c_void_p
        library.h2h_query.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        library.h2h_query.restype = ctypes.c_double
        library.h2h_close.argtypes = [ctypes.c_void_p]
        library.h2h_close.restype = None
        if library.h2h_api_version() != 1:
            raise RuntimeError('H2H C ABI 版本不是 1。')

        error_buffer = ctypes.create_string_buffer(1024)
        handle = library.h2h_open(
            str(index_path).encode('utf-8'), error_buffer, len(error_buffer)
        )
        if not handle:
            raise RuntimeError(f'H2H smoke 索引打开失败：{error_buffer.value.decode("utf-8")}')
        try:
            forward = library.h2h_query(handle, 0, 1)
            backward = library.h2h_query(handle, 1, 0)
            if not math.isclose(forward, 1.25, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f'H2H smoke 0->1 错误：{forward}')
            if not math.isclose(backward, 2.5, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f'H2H smoke 1->0 错误：{backward}')
        finally:
            library.h2h_close(handle)


def main() -> int:
    """
    编译对象文件、链接 builder/DLL，并执行临时索引烟雾测试。

    输入：命令行参数。
    输出：进程退出码；成功为 0，异常由 Python 打印并返回非零。
    """
    arguments = _parse_arguments()
    build_dir = arguments.build_dir.expanduser()
    if not build_dir.is_absolute():
        build_dir = (PROJECT_ROOT / build_dir).resolve()
    else:
        build_dir = build_dir.resolve()

    if arguments.clean:
        _clean_known_outputs(build_dir)
        print(f'已清理 H2H 已知构建产物：{build_dir}')
        return 0

    compiler = _resolve_compiler(arguments.compiler)
    subprocess.run([str(compiler), '--version'], check=True)
    obj_dir, builder_path, library_path = _artifact_paths(build_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)

    common_flags = [
        '-std=c++17', '-Wall', '-Wextra', '-Wpedantic', f'-I{INCLUDE_DIR}',
    ]
    if arguments.debug:
        # 当前 MinGW 发行版未附带 libasan；Debug 模式启用 libstdc++ 边界断言、
        # debug iterator 和栈保护，作为 Windows 本机的等效运行期检查。
        common_flags.extend((
            '-O0', '-g', '-D_GLIBCXX_ASSERTIONS', '-D_GLIBCXX_DEBUG', '-fstack-protector-all',
        ))
    else:
        common_flags.extend(('-O3', '-DNDEBUG'))
    if platform.system() != 'Windows':
        common_flags.append('-fPIC')
    if arguments.sanitize:
        common_flags.extend(('-fsanitize=address,undefined', '-fno-omit-frame-pointer'))

    objects = {}
    for source_name in SOURCE_NAMES:
        source_path = SOURCE_DIR / source_name
        object_path = obj_dir / f'{source_path.stem}.o'
        _run([
            str(compiler), *common_flags, '-c', str(source_path), '-o', str(object_path),
        ], arguments.verbose)
        objects[source_name] = object_path

    link_flags = []
    if platform.system() == 'Windows':
        link_flags.extend(('-static-libgcc', '-static-libstdc++'))
    if arguments.sanitize:
        link_flags.extend(('-fsanitize=address,undefined', '-fno-omit-frame-pointer'))

    _run([
        str(compiler),
        str(objects['h2h_graph.cpp']),
        str(objects['h2h_index.cpp']),
        str(objects['h2h_builder.cpp']),
        *link_flags,
        '-o', str(builder_path),
    ], arguments.verbose)
    _run([
        str(compiler), '-shared',
        str(objects['h2h_query.cpp']),
        str(objects['h2h_c_api.cpp']),
        *link_flags,
        '-o', str(library_path),
    ], arguments.verbose)

    if not builder_path.is_file() or not library_path.is_file():
        raise RuntimeError('链接命令成功但 H2H 原生产物不存在。')
    _run_smoke_test(builder_path, library_path, arguments.verbose)
    print(f'H2H 原生构建完成：{builder_path}')
    print(f'H2H 查询动态库：{library_path}')
    print('Smoke test: OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
