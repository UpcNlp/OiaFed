# fedcl/utils/file_utils.py
"""
文件系统操作工具类

该模块提供了FedCL框架中所有文件操作的基础工具函数，包括：
- 目录管理和创建
- JSON和Pickle数据序列化
- 文件压缩和解压缩
- 原子写入操作
- 安全文件删除

所有操作都具备异常安全性和大文件支持能力。
"""

import gzip
import json
import os
import pickle
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Union, Optional
import stat


class FileOperationError(Exception):
    """文件操作异常基类"""
    pass


class FilePermissionError(FileOperationError):
    """文件权限异常"""
    pass


class FileSizeError(FileOperationError):
    """文件大小相关异常"""
    pass


class FileCorruptionError(FileOperationError):
    """文件损坏异常"""
    pass


class FileUtils:
    """
    文件系统操作工具类
    
    提供静态方法进行各种文件系统操作，包括目录管理、数据序列化、
    文件压缩等功能。所有操作都具备异常安全性。
    """
    
    # 默认配置
    DEFAULT_COMPRESSION_LEVEL = 6
    DEFAULT_FILE_MODE = 0o644
    DEFAULT_DIR_MODE = 0o755
    CHUNK_SIZE = 8192  # 8KB chunks for large file operations
    
    @staticmethod
    def ensure_dir_exists(path: Union[str, Path]) -> Path:
        """
        确保目录存在，如果不存在则创建
        
        Args:
            path: 目录路径
            
        Returns:
            Path: 标准化的Path对象
            
        Raises:
            FilePermissionError: 当没有创建权限时
            FileOperationError: 当创建失败时
        """
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True, mode=FileUtils.DEFAULT_DIR_MODE)
            elif not dir_path.is_dir():
                raise FileOperationError(f"Path {path} exists but is not a directory")
            return dir_path
        except PermissionError as e:
            raise FilePermissionError(f"Permission denied creating directory {path}: {e}")
        except OSError as e:
            raise FileOperationError(f"Failed to create directory {path}: {e}")
    
    @staticmethod
    def save_json(data: Dict, path: Union[str, Path], indent: int = 2) -> None:
        """
        保存数据为JSON文件
        
        Args:
            data: 要保存的数据
            path: 文件路径
            indent: JSON缩进空格数
            
        Raises:
            FileOperationError: 当保存失败时
            FilePermissionError: 当没有写入权限时
        """
        try:
            file_path = Path(path)
            FileUtils.ensure_dir_exists(file_path.parent)
            
            # 使用原子写入确保数据一致性
            json_content = json.dumps(data, indent=indent, ensure_ascii=False)
            FileUtils.atomic_write(json_content.encode('utf-8'), file_path)
            
        except (TypeError, ValueError) as e:
            raise FileOperationError(f"Failed to serialize data to JSON: {e}")
        except Exception as e:
            raise FileOperationError(f"Failed to save JSON file {path}: {e}")
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict:
        """
        从JSON文件加载数据
        
        Args:
            path: 文件路径
            
        Returns:
            Dict: 加载的数据
            
        Raises:
            FileNotFoundError: 当文件不存在时
            FileCorruptionError: 当JSON文件损坏时
            FileOperationError: 当读取失败时
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"JSON file not found: {path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except FileNotFoundError:
            # 重新抛出FileNotFoundError，不要转换
            raise
        except json.JSONDecodeError as e:
            raise FileCorruptionError(f"Corrupted JSON file {path}: {e}")
        except PermissionError as e:
            raise FilePermissionError(f"Permission denied reading {path}: {e}")
        except Exception as e:
            raise FileOperationError(f"Failed to load JSON file {path}: {e}")
    
    @staticmethod
    def save_pickle(obj: Any, path: Union[str, Path]) -> None:
        """
        保存对象为Pickle文件
        
        Args:
            obj: 要保存的对象
            path: 文件路径
            
        Raises:
            FileOperationError: 当保存失败时
            FilePermissionError: 当没有写入权限时
        """
        try:
            file_path = Path(path)
            FileUtils.ensure_dir_exists(file_path.parent)
            
            # 序列化对象
            pickle_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            FileUtils.atomic_write(pickle_data, file_path)
            
        except (pickle.PickleError, AttributeError, TypeError) as e:
            # 处理各种序列化错误
            raise FileOperationError(f"Failed to serialize object: {e}")
        except Exception as e:
            raise FileOperationError(f"Failed to save pickle file {path}: {e}")
    
    @staticmethod
    def load_pickle(path: Union[str, Path]) -> Any:
        """
        从Pickle文件加载对象
        
        Args:
            path: 文件路径
            
        Returns:
            Any: 加载的对象
            
        Raises:
            FileNotFoundError: 当文件不存在时
            FileCorruptionError: 当Pickle文件损坏时
            FileOperationError: 当读取失败时
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"Pickle file not found: {path}")
            
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        except FileNotFoundError:
            # 重新抛出FileNotFoundError，不要转换
            raise
        except pickle.PickleError as e:
            raise FileCorruptionError(f"Corrupted pickle file {path}: {e}")
        except PermissionError as e:
            raise FilePermissionError(f"Permission denied reading {path}: {e}")
        except Exception as e:
            raise FileOperationError(f"Failed to load pickle file {path}: {e}")
    
    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int:
        """
        获取文件大小（字节）
        
        Args:
            path: 文件路径
            
        Returns:
            int: 文件大小（字节）
            
        Raises:
            FileNotFoundError: 当文件不存在时
            FileOperationError: 当获取大小失败时
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            return file_path.stat().st_size
            
        except FileNotFoundError:
            # 重新抛出FileNotFoundError，不要转换
            raise
        except PermissionError as e:
            raise FilePermissionError(f"Permission denied accessing {path}: {e}")
        except OSError as e:
            raise FileOperationError(f"Failed to get file size for {path}: {e}")
    
    @staticmethod
    def compress_file(source: Path, target: Path, compression_level: Optional[int] = None) -> None:
        """
        压缩文件
        
        Args:
            source: 源文件路径
            target: 目标压缩文件路径
            compression_level: 压缩级别 (0-9)，None使用默认值
            
        Raises:
            FileNotFoundError: 当源文件不存在时
            FileOperationError: 当压缩失败时
        """
        if compression_level is None:
            compression_level = FileUtils.DEFAULT_COMPRESSION_LEVEL
            
        try:
            source_path = Path(source)
            target_path = Path(target)
            
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source}")
            
            FileUtils.ensure_dir_exists(target_path.parent)
            
            # 使用流式压缩处理大文件
            with open(source_path, 'rb') as f_in:
                with gzip.open(target_path, 'wb', compresslevel=compression_level) as f_out:
                    while True:
                        chunk = f_in.read(FileUtils.CHUNK_SIZE)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        
        except FileNotFoundError:
            # 重新抛出FileNotFoundError，不要转换
            raise
        except PermissionError as e:
            raise FilePermissionError(f"Permission denied during compression: {e}")
        except OSError as e:
            raise FileOperationError(f"Failed to compress file {source} to {target}: {e}")
    
    @staticmethod
    def decompress_file(source: Path, target: Path) -> None:
        """
        解压缩文件
        
        Args:
            source: 源压缩文件路径
            target: 目标解压文件路径
            
        Raises:
            FileNotFoundError: 当源文件不存在时
            FileCorruptionError: 当压缩文件损坏时
            FileOperationError: 当解压失败时
        """
        try:
            source_path = Path(source)
            target_path = Path(target)
            
            if not source_path.exists():
                raise FileNotFoundError(f"Source compressed file not found: {source}")
            
            FileUtils.ensure_dir_exists(target_path.parent)
            
            # 使用流式解压处理大文件
            with gzip.open(source_path, 'rb') as f_in:
                with open(target_path, 'wb') as f_out:
                    while True:
                        chunk = f_in.read(FileUtils.CHUNK_SIZE)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        
        except FileNotFoundError:
            # 重新抛出FileNotFoundError，不要转换
            raise
        except gzip.BadGzipFile as e:
            raise FileCorruptionError(f"Corrupted compressed file {source}: {e}")
        except PermissionError as e:
            raise FilePermissionError(f"Permission denied during decompression: {e}")
        except OSError as e:
            raise FileOperationError(f"Failed to decompress file {source} to {target}: {e}")
    
    @staticmethod
    def safe_remove(path: Union[str, Path]) -> bool:
        """
        安全删除文件或目录
        
        Args:
            path: 要删除的路径
            
        Returns:
            bool: 删除是否成功
            
        Raises:
            FilePermissionError: 当没有删除权限时
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                return True  # 文件不存在视为删除成功
            
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                # 处理符号链接等特殊文件
                file_path.unlink()
                
            return True
            
        except PermissionError as e:
            raise FilePermissionError(f"Permission denied deleting {path}: {e}")
        except OSError:
            # 某些情况下删除可能会失败，但不抛出异常
            return False
    
    @staticmethod
    def atomic_write(content: Union[str, bytes], path: Path) -> None:
        """
        原子写入文件内容
        
        使用临时文件和重命名操作确保写入的原子性，
        避免写入过程中系统崩溃导致的文件损坏。
        
        Args:
            content: 要写入的内容
            path: 目标文件路径
            
        Raises:
            FileOperationError: 当写入失败时
            FilePermissionError: 当没有写入权限时
        """
        try:
            target_path = Path(path)
            FileUtils.ensure_dir_exists(target_path.parent)
            
            # 在同一目录下创建临时文件确保原子性
            temp_dir = target_path.parent
            with tempfile.NamedTemporaryFile(
                mode='wb' if isinstance(content, bytes) else 'w',
                dir=temp_dir,
                delete=False,
                encoding='utf-8' if isinstance(content, str) else None
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # 确保数据写入磁盘
            
            # 设置适当的文件权限
            temp_path.chmod(FileUtils.DEFAULT_FILE_MODE)
            
            # 原子重命名操作
            temp_path.replace(target_path)
            
        except PermissionError as e:
            # 清理临时文件
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise FilePermissionError(f"Permission denied writing to {path}: {e}")
        except OSError as e:
            # 清理临时文件
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise FileOperationError(f"Failed to write file {path}: {e}")
    
    @staticmethod
    def copy_file_safe(source: Union[str, Path], target: Union[str, Path], 
                      preserve_metadata: bool = True) -> None:
        """
        安全复制文件，支持大文件和元数据保持
        
        Args:
            source: 源文件路径
            target: 目标文件路径
            preserve_metadata: 是否保持文件元数据
            
        Raises:
            FileNotFoundError: 当源文件不存在时
            FileOperationError: 当复制失败时
        """
        try:
            source_path = Path(source)
            target_path = Path(target)
            
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source}")
            
            FileUtils.ensure_dir_exists(target_path.parent)
            
            if preserve_metadata:
                shutil.copy2(source_path, target_path)
            else:
                shutil.copy(source_path, target_path)
                
        except FileNotFoundError:
            # 重新抛出FileNotFoundError，不要转换
            raise
        except PermissionError as e:
            raise FilePermissionError(f"Permission denied copying {source} to {target}: {e}")
        except OSError as e:
            raise FileOperationError(f"Failed to copy file {source} to {target}: {e}")
    
    @staticmethod
    def get_disk_usage(path: Union[str, Path]) -> Dict[str, int]:
        """
        获取路径所在磁盘的使用情况
        
        Args:
            path: 检查路径
            
        Returns:
            Dict[str, int]: 包含total, used, free字段的字典（字节）
            
        Raises:
            FileOperationError: 当获取信息失败时
        """
        try:
            check_path = Path(path)
            
            # 递归向上查找存在的路径
            while not check_path.exists() and check_path != check_path.parent:
                check_path = check_path.parent
            
            # 如果找到了根路径但仍不存在，使用当前工作目录
            if not check_path.exists():
                check_path = Path.cwd()
                
            usage = shutil.disk_usage(check_path)
            return {
                'total': usage.total,
                'used': usage.used, 
                'free': usage.free
            }
            
        except OSError as e:
            raise FileOperationError(f"Failed to get disk usage for {path}: {e}")