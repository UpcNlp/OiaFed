"""
MOE-FedCL 序列化工具
moe_fedcl/utils/serialization.py
"""

import json
import pickle
from typing import Any, Dict, Union
from datetime import datetime
from enum import Enum

from ..exceptions import SerializationError


class SerializationFormat(Enum):
    """序列化格式"""
    JSON = "json"
    PICKLE = "pickle"
    AUTO = "auto"  # 自动选择格式


class DataSerializer:
    """统一数据序列化器"""
    
    @staticmethod
    def serialize(data: Any, format: Union[str, SerializationFormat] = SerializationFormat.AUTO) -> bytes:
        """序列化数据
        
        Args:
            data: 要序列化的数据
            format: 序列化格式
            
        Returns:
            bytes: 序列化后的字节数据
            
        Raises:
            SerializationError: 序列化失败
        """
        try:
            if isinstance(format, str):
                format = SerializationFormat(format)
            
            # 自动选择格式
            if format == SerializationFormat.AUTO:
                format = DataSerializer._choose_format(data)
            
            if format == SerializationFormat.JSON:
                return DataSerializer._serialize_json(data)
            elif format == SerializationFormat.PICKLE:
                return DataSerializer._serialize_pickle(data)
            else:
                raise SerializationError(f"Unsupported serialization format: {format}")
                
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            else:
                raise SerializationError(f"Serialization failed: {str(e)}")
    
    @staticmethod
    def deserialize(data: bytes, format: Union[str, SerializationFormat] = SerializationFormat.AUTO) -> Any:
        """反序列化数据
        
        Args:
            data: 序列化的字节数据
            format: 反序列化格式
            
        Returns:
            Any: 反序列化后的数据
            
        Raises:
            SerializationError: 反序列化失败
        """
        try:
            if isinstance(format, str):
                format = SerializationFormat(format)
            
            # 自动检测格式
            if format == SerializationFormat.AUTO:
                format = DataSerializer._detect_format(data)
            
            if format == SerializationFormat.JSON:
                return DataSerializer._deserialize_json(data)
            elif format == SerializationFormat.PICKLE:
                return DataSerializer._deserialize_pickle(data)
            else:
                raise SerializationError(f"Unsupported deserialization format: {format}")
                
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            else:
                raise SerializationError(f"Deserialization failed: {str(e)}")
    
    @staticmethod
    def _choose_format(data: Any) -> SerializationFormat:
        """根据数据类型选择最适合的序列化格式"""
        # 检查是否可以用JSON序列化
        if DataSerializer._is_json_serializable(data):
            return SerializationFormat.JSON
        else:
            return SerializationFormat.PICKLE
    
    @staticmethod
    def _is_json_serializable(data: Any) -> bool:
        """检查数据是否可以用JSON序列化"""
        try:
            json.dumps(data, cls=CustomJSONEncoder)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def _detect_format(data: bytes) -> SerializationFormat:
        """检测序列化数据的格式"""
        # 尝试检测JSON格式
        try:
            # JSON数据通常以 { 或 [ 开头
            text = data.decode('utf-8')
            text = text.strip()
            if text.startswith(('{', '[')):
                json.loads(text)
                return SerializationFormat.JSON
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        
        # 尝试检测Pickle格式
        try:
            # Pickle数据有特定的魔数开头
            if data.startswith(b'\x80'):  # Pickle protocol 2+
                return SerializationFormat.PICKLE
        except Exception:
            pass
        
        # 默认使用Pickle
        return SerializationFormat.PICKLE
    
    @staticmethod
    def _serialize_json(data: Any) -> bytes:
        """JSON序列化"""
        try:
            json_str = json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False)
            return json_str.encode('utf-8')
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {str(e)}")
    
    @staticmethod
    def _deserialize_json(data: bytes) -> Any:
        """JSON反序列化"""
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str, cls=CustomJSONDecoder)
        except Exception as e:
            raise SerializationError(f"JSON deserialization failed: {str(e)}")
    
    @staticmethod
    def _serialize_pickle(data: Any) -> bytes:
        """Pickle序列化"""
        try:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise SerializationError(f"Pickle serialization failed: {str(e)}")
    
    @staticmethod
    def _deserialize_pickle(data: bytes) -> Any:
        """Pickle反序列化"""
        try:
            return pickle.loads(data)
        except Exception as e:
            raise SerializationError(f"Pickle deserialization failed: {str(e)}")


class CustomJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理特殊类型"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return {
                '__type__': 'datetime',
                'value': obj.isoformat()
            }
        elif isinstance(obj, Enum):
            return {
                '__type__': 'enum',
                'class': obj.__class__.__module__ + '.' + obj.__class__.__qualname__,
                'value': obj.value
            }
        elif hasattr(obj, '__dict__'):
            # 处理自定义对象
            return {
                '__type__': 'object',
                'class': obj.__class__.__module__ + '.' + obj.__class__.__qualname__,
                'data': obj.__dict__
            }
        elif isinstance(obj, bytes):
            return {
                '__type__': 'bytes',
                'value': obj.hex()
            }
        elif isinstance(obj, set):
            return {
                '__type__': 'set',
                'value': list(obj)
            }
        else:
            return super().default(obj)


class CustomJSONDecoder(json.JSONDecoder):
    """自定义JSON解码器，处理特殊类型"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
    
    def object_hook(self, obj):
        if isinstance(obj, dict) and '__type__' in obj:
            obj_type = obj['__type__']
            
            if obj_type == 'datetime':
                return datetime.fromisoformat(obj['value'])
            
            elif obj_type == 'enum':
                try:
                    # 动态导入枚举类
                    class_path = obj['class']
                    module_name, class_name = class_path.rsplit('.', 1)
                    module = __import__(module_name, fromlist=[class_name])
                    enum_class = getattr(module, class_name)
                    return enum_class(obj['value'])
                except Exception:
                    # 如果无法重构枚举，返回原始值
                    return obj['value']
            
            elif obj_type == 'object':
                try:
                    # 动态重构对象
                    class_path = obj['class']
                    module_name, class_name = class_path.rsplit('.', 1)
                    module = __import__(module_name, fromlist=[class_name])
                    obj_class = getattr(module, class_name)
                    
                    # 创建对象实例
                    instance = obj_class.__new__(obj_class)
                    instance.__dict__.update(obj['data'])
                    return instance
                except Exception:
                    # 如果无法重构对象，返回数据字典
                    return obj['data']
            
            elif obj_type == 'bytes':
                return bytes.fromhex(obj['value'])
            
            elif obj_type == 'set':
                return set(obj['value'])
        
        return obj


# ==================== 便捷函数 ====================

def serialize_data(data: Any, format: str = "auto") -> bytes:
    """便捷的数据序列化函数
    
    Args:
        data: 要序列化的数据
        format: 序列化格式 ("json", "pickle", "auto")
        
    Returns:
        bytes: 序列化后的数据
    """
    return DataSerializer.serialize(data, format)


def deserialize_data(data: bytes, format: str = "auto") -> Any:
    """便捷的数据反序列化函数
    
    Args:
        data: 序列化的字节数据
        format: 反序列化格式 ("json", "pickle", "auto")
        
    Returns:
        Any: 反序列化后的数据
    """
    return DataSerializer.deserialize(data, format)


def safe_serialize(data: Any, fallback_format: str = "pickle") -> tuple[bytes, str]:
    """安全序列化，如果主格式失败则使用备用格式
    
    Args:
        data: 要序列化的数据
        fallback_format: 备用序列化格式
        
    Returns:
        tuple[bytes, str]: (序列化数据, 使用的格式)
    """
    try:
        # 首先尝试JSON
        serialized_data = DataSerializer.serialize(data, SerializationFormat.JSON)
        return serialized_data, "json"
    except SerializationError:
        try:
            # 备用格式
            serialized_data = DataSerializer.serialize(data, fallback_format)
            return serialized_data, fallback_format
        except SerializationError as e:
            raise SerializationError(f"All serialization attempts failed: {str(e)}")


def estimate_serialized_size(data: Any, format: str = "auto") -> int:
    """估算序列化后的数据大小
    
    Args:
        data: 数据对象
        format: 序列化格式
        
    Returns:
        int: 估算的字节大小
    """
    try:
        serialized_data = serialize_data(data, format)
        return len(serialized_data)
    except Exception:
        return -1  # 表示无法估算


def compare_serialization_formats(data: Any) -> Dict[str, Dict[str, Any]]:
    """比较不同序列化格式的效果
    
    Args:
        data: 要测试的数据
        
    Returns:
        Dict[str, Dict[str, Any]]: 各格式的比较结果
    """
    results = {}
    
    formats_to_test = ["json", "pickle"]
    
    for format_name in formats_to_test:
        result = {
            "supported": False,
            "size": -1,
            "serialize_time": -1,
            "deserialize_time": -1,
            "error": None
        }
        
        try:
            import time
            
            # 测试序列化
            start_time = time.time()
            serialized_data = serialize_data(data, format_name)
            serialize_time = time.time() - start_time
            
            # 测试反序列化
            start_time = time.time()
            deserialized_data = deserialize_data(serialized_data, format_name)
            deserialize_time = time.time() - start_time
            
            result.update({
                "supported": True,
                "size": len(serialized_data),
                "serialize_time": serialize_time,
                "deserialize_time": deserialize_time
            })
            
        except Exception as e:
            result["error"] = str(e)
        
        results[format_name] = result
    
    return results


# ==================== 高级序列化类 ====================

class StreamingSerializer:
    """流式序列化器，适用于大数据"""
    
    def __init__(self, format: str = "pickle", chunk_size: int = 8192):
        self.format = SerializationFormat(format)
        self.chunk_size = chunk_size
    
    def serialize_to_stream(self, data: Any, stream):
        """序列化数据到流
        
        Args:
            data: 要序列化的数据
            stream: 输出流对象（需要有write方法）
        """
        try:
            serialized_data = DataSerializer.serialize(data, self.format)
            
            # 分块写入
            for i in range(0, len(serialized_data), self.chunk_size):
                chunk = serialized_data[i:i + self.chunk_size]
                stream.write(chunk)
                
        except Exception as e:
            raise SerializationError(f"Stream serialization failed: {str(e)}")
    
    def deserialize_from_stream(self, stream) -> Any:
        """从流反序列化数据
        
        Args:
            stream: 输入流对象（需要有read方法）
            
        Returns:
            Any: 反序列化后的数据
        """
        try:
            # 读取所有数据
            data_chunks = []
            while True:
                chunk = stream.read(self.chunk_size)
                if not chunk:
                    break
                data_chunks.append(chunk)
            
            serialized_data = b''.join(data_chunks)
            return DataSerializer.deserialize(serialized_data, self.format)
            
        except Exception as e:
            raise SerializationError(f"Stream deserialization failed: {str(e)}")


class CompressedSerializer:
    """压缩序列化器"""
    
    def __init__(self, format: str = "pickle", compression: str = "zlib"):
        self.format = SerializationFormat(format)
        self.compression = compression
    
    def serialize(self, data: Any) -> bytes:
        """压缩序列化
        
        Args:
            data: 要序列化的数据
            
        Returns:
            bytes: 压缩后的序列化数据
        """
        try:
            # 先序列化
            serialized_data = DataSerializer.serialize(data, self.format)
            
            # 再压缩
            if self.compression == "zlib":
                import zlib
                compressed_data = zlib.compress(serialized_data)
            elif self.compression == "gzip":
                import gzip
                compressed_data = gzip.compress(serialized_data)
            else:
                raise SerializationError(f"Unsupported compression: {self.compression}")
            
            return compressed_data
            
        except Exception as e:
            raise SerializationError(f"Compressed serialization failed: {str(e)}")
    
    def deserialize(self, compressed_data: bytes) -> Any:
        """压缩反序列化
        
        Args:
            compressed_data: 压缩的序列化数据
            
        Returns:
            Any: 反序列化后的数据
        """
        try:
            # 先解压
            if self.compression == "zlib":
                import zlib
                serialized_data = zlib.decompress(compressed_data)
            elif self.compression == "gzip":
                import gzip
                serialized_data = gzip.decompress(compressed_data)
            else:
                raise SerializationError(f"Unsupported compression: {self.compression}")
            
            # 再反序列化
            return DataSerializer.deserialize(serialized_data, self.format)
            
        except Exception as e:
            raise SerializationError(f"Compressed deserialization failed: {str(e)}")