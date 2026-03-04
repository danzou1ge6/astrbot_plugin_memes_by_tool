"""原子写入工具

提供原子文件写入功能，防止写入过程中断导致文件损坏。

原子写入流程：
1. 写入临时文件（与目标文件同目录）
2. 刷新并同步到磁盘（flush + fsync）
3. 原子重命名临时文件为目标文件
"""

import gzip
import json
import os
import tempfile
from pathlib import Path

from .errors import MemesFileError


def atomic_write_json(file_path: Path, data: dict | list, indent: int = 2) -> None:
    """原子写入 JSON 文件

    Args:
        file_path: 目标文件路径
        data: 要写入的数据
        indent: JSON 缩进空格数

    Raises:
        MemesFileError: 写入失败
    """
    file_path = Path(file_path)
    parent_dir = file_path.parent

    # 确保父目录存在
    parent_dir.mkdir(parents=True, exist_ok=True)

    # 创建临时文件（与目标文件同目录，确保同一文件系统）
    fd, temp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=file_path.name,
        dir=parent_dir,
    )

    temp_path = Path(temp_path)

    try:
        # 写入临时文件
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())

        # 原子重命名
        os.replace(temp_path, file_path)

    except OSError as e:
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise MemesFileError("原子写入 JSON 文件失败", file_path) from e

    except Exception as e:
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise MemesFileError(f"写入 JSON 文件失败: {e}", file_path) from e


def atomic_write_gzip_json(file_path: Path, data: dict | list) -> None:
    """原子写入 GZIP 压缩的 JSON 文件

    Args:
        file_path: 目标文件路径（.json.gz）
        data: 要写入的数据

    Raises:
        MemesFileError: 写入失败
    """
    file_path = Path(file_path)
    parent_dir = file_path.parent

    # 确保父目录存在
    parent_dir.mkdir(parents=True, exist_ok=True)

    # 创建临时文件（与目标文件同目录，确保同一文件系统）
    fd, temp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=file_path.name,
        dir=parent_dir,
    )

    temp_path = Path(temp_path)

    try:
        # 写入临时文件（GZIP 压缩）
        with os.fdopen(fd, "wb") as raw_f:
            with gzip.GzipFile(fileobj=raw_f, mode="wb") as gz_f:
                gz_f.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            raw_f.flush()
            os.fsync(raw_f.fileno())

        # 原子重命名
        os.replace(temp_path, file_path)

    except OSError as e:
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise MemesFileError("原子写入 GZIP 文件失败", file_path) from e

    except Exception as e:
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise MemesFileError(f"写入 GZIP 文件失败: {e}", file_path) from e


def atomic_write_text(file_path: Path, content: str, encoding: str = "utf-8") -> None:
    """原子写入文本文件

    Args:
        file_path: 目标文件路径
        content: 文本内容
        encoding: 文本编码

    Raises:
        MemesFileError: 写入失败
    """
    file_path = Path(file_path)
    parent_dir = file_path.parent

    # 确保父目录存在
    parent_dir.mkdir(parents=True, exist_ok=True)

    # 创建临时文件（与目标文件同目录，确保同一文件系统）
    fd, temp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=file_path.name,
        dir=parent_dir,
    )

    temp_path = Path(temp_path)

    try:
        # 写入临时文件
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # 原子重命名
        os.replace(temp_path, file_path)

    except OSError as e:
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise MemesFileError("原子写入文本文件失败", file_path) from e

    except Exception as e:
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise MemesFileError(f"写入文本文件失败: {e}", file_path) from e
