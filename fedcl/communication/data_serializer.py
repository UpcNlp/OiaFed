# fedcl/communication/data_serializer.py
"""
数据序列化模块

实现PyTorch模型、张量和其他数据的序列化与反序列化功能。
支持数据压缩以优化网络传输效率。
"""

import pickle
import gzip
import io
from typing import Type, Dict, Any, Union
import torch
import torch.nn as nn
from loguru import logger

from .exceptions import SerializationError


class DataSerializer:
    """
    数据序列化器
    
    负责PyTorch模型、张量等数据的序列化和反序列化，
    支持数据压缩以减少网络传输开销。
    """
    
    def __init__(self, compression_level: int = 6) -> None:
        """
        初始化数据序列化器
        
        Args:
            compression_level: 压缩级别 (0-9)，0为不压缩，9为最大压缩
        """
        if not 0 <= compression_level <= 9:
            raise ValueError("Compression level must be between 0 and 9")
        
        self.compression_level = compression_level
        logger.debug(f"DataSerializer initialized with compression level {compression_level}")
    
    def serialize_model(self, model: torch.nn.Module) -> bytes:
        """
        序列化PyTorch模型
        
        Args:
            model: PyTorch模型
            
        Returns:
            bytes: 序列化后的模型数据
        """
        try:
            # 使用内存缓冲区
            buffer = io.BytesIO()
            
            # 保存模型状态字典
            torch.save(model.state_dict(), buffer)
            
            # 获取序列化数据
            model_data = buffer.getvalue()
            buffer.close()
            
            # 如果启用压缩则压缩数据
            if self.compression_level > 0:
                model_data = self.compress_data(model_data)
            
            logger.debug(f"Model serialized, size: {len(model_data)} bytes")
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to serialize model: {e}")
            raise SerializationError(f"Failed to serialize model: {e}")
    
    def deserialize_model(self, data: bytes, model_class: Type[torch.nn.Module], 
                         model_args: tuple = (), model_kwargs: dict = None) -> torch.nn.Module:
        """
        反序列化PyTorch模型
        
        Args:
            data: 序列化的模型数据
            model_class: 模型类
            model_args: 模型构造参数
            model_kwargs: 模型构造关键字参数
            
        Returns:
            torch.nn.Module: 反序列化的模型
        """
        try:
            # 如果数据被压缩则先解压
            if self.compression_level > 0:
                try:
                    data = self.decompress_data(data)
                except:
                    # 如果解压失败，假设数据未压缩
                    pass
            
            # 创建模型实例
            if model_kwargs is None:
                model_kwargs = {}
            model = model_class(*model_args, **model_kwargs)
            
            # 从字节数据加载状态字典
            buffer = io.BytesIO(data)
            state_dict = torch.load(buffer, map_location='cpu')
            buffer.close()
            
            # 加载状态字典到模型
            model.load_state_dict(state_dict)
            
            logger.debug("Model deserialized successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to deserialize model: {e}")
            raise SerializationError(f"Failed to deserialize model: {e}")
    
    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """
        序列化PyTorch张量
        
        Args:
            tensor: PyTorch张量
            
        Returns:
            bytes: 序列化后的张量数据
        """
        try:
            # 使用内存缓冲区
            buffer = io.BytesIO()
            
            # 保存张量
            torch.save(tensor, buffer)
            
            # 获取序列化数据
            tensor_data = buffer.getvalue()
            buffer.close()
            
            # 如果启用压缩则压缩数据
            if self.compression_level > 0:
                tensor_data = self.compress_data(tensor_data)
            
            logger.debug(f"Tensor serialized, shape: {tensor.shape}, size: {len(tensor_data)} bytes")
            return tensor_data
            
        except Exception as e:
            logger.error(f"Failed to serialize tensor: {e}")
            raise SerializationError(f"Failed to serialize tensor: {e}")
    
    def deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """
        反序列化PyTorch张量
        
        Args:
            data: 序列化的张量数据
            
        Returns:
            torch.Tensor: 反序列化的张量
        """
        try:
            # 如果数据被压缩则先解压
            if self.compression_level > 0:
                try:
                    data = self.decompress_data(data)
                except:
                    # 如果解压失败，假设数据未压缩
                    pass
            
            # 从字节数据加载张量
            buffer = io.BytesIO(data)
            tensor = torch.load(buffer, map_location='cpu')
            buffer.close()
            
            logger.debug(f"Tensor deserialized, shape: {tensor.shape}")
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to deserialize tensor: {e}")
            raise SerializationError(f"Failed to deserialize tensor: {e}")
    
    def serialize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> bytes:
        """
        序列化状态字典
        
        Args:
            state_dict: PyTorch状态字典
            
        Returns:
            bytes: 序列化后的状态字典数据
        """
        try:
            # 使用内存缓冲区
            buffer = io.BytesIO()
            
            # 保存状态字典
            torch.save(state_dict, buffer)
            
            # 获取序列化数据
            state_dict_data = buffer.getvalue()
            buffer.close()
            
            # 如果启用压缩则压缩数据
            if self.compression_level > 0:
                state_dict_data = self.compress_data(state_dict_data)
            
            logger.debug(f"State dict serialized, keys: {len(state_dict)}, size: {len(state_dict_data)} bytes")
            return state_dict_data
            
        except Exception as e:
            logger.error(f"Failed to serialize state dict: {e}")
            raise SerializationError(f"Failed to serialize state dict: {e}")
    
    def deserialize_state_dict(self, data: bytes) -> Dict[str, torch.Tensor]:
        """
        反序列化状态字典
        
        Args:
            data: 序列化的状态字典数据
            
        Returns:
            Dict[str, torch.Tensor]: 反序列化的状态字典
        """
        try:
            # 如果数据被压缩则先解压
            if self.compression_level > 0:
                try:
                    data = self.decompress_data(data)
                except:
                    # 如果解压失败，假设数据未压缩
                    pass
            
            # 从字节数据加载状态字典
            buffer = io.BytesIO(data)
            state_dict = torch.load(buffer, map_location='cpu')
            buffer.close()
            
            logger.debug(f"State dict deserialized, keys: {len(state_dict)}")
            return state_dict
            
        except Exception as e:
            logger.error(f"Failed to deserialize state dict: {e}")
            raise SerializationError(f"Failed to deserialize state dict: {e}")
    
    def serialize_object(self, obj: Any) -> bytes:
        """
        序列化任意Python对象
        
        Args:
            obj: 要序列化的对象
            
        Returns:
            bytes: 序列化后的对象数据
        """
        try:
            # 使用pickle序列化
            obj_data = pickle.dumps(obj)
            
            # 如果启用压缩则压缩数据
            if self.compression_level > 0:
                obj_data = self.compress_data(obj_data)
            
            logger.debug(f"Object serialized, type: {type(obj).__name__}, size: {len(obj_data)} bytes")
            return obj_data
            
        except Exception as e:
            logger.error(f"Failed to serialize object: {e}")
            raise SerializationError(f"Failed to serialize object: {e}")
    
    def deserialize_object(self, data: bytes) -> Any:
        """
        反序列化Python对象
        
        Args:
            data: 序列化的对象数据
            
        Returns:
            Any: 反序列化的对象
        """
        try:
            # 如果数据被压缩则先解压
            if self.compression_level > 0:
                try:
                    data = self.decompress_data(data)
                except:
                    # 如果解压失败，假设数据未压缩
                    pass
            
            # 使用pickle反序列化
            obj = pickle.loads(data)
            
            logger.debug(f"Object deserialized, type: {type(obj).__name__}")
            return obj
            
        except Exception as e:
            logger.error(f"Failed to deserialize object: {e}")
            raise SerializationError(f"Failed to deserialize object: {e}")
    
    def compress_data(self, data: bytes) -> bytes:
        """
        压缩数据
        
        Args:
            data: 原始数据
            
        Returns:
            bytes: 压缩后的数据
        """
        try:
            if self.compression_level == 0:
                return data
            
            compressed_data = gzip.compress(data, compresslevel=self.compression_level)
            
            compression_ratio = self.get_compression_ratio(len(data), len(compressed_data))
            logger.debug(f"Data compressed: {len(data)} -> {len(compressed_data)} bytes "
                        f"(ratio: {compression_ratio:.2f})")
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Failed to compress data: {e}")
            raise SerializationError(f"Failed to compress data: {e}")
    
    def decompress_data(self, compressed_data: bytes) -> bytes:
        """
        解压数据
        
        Args:
            compressed_data: 压缩的数据
            
        Returns:
            bytes: 解压后的数据
        """
        try:
            decompressed_data = gzip.decompress(compressed_data)
            
            logger.debug(f"Data decompressed: {len(compressed_data)} -> {len(decompressed_data)} bytes")
            return decompressed_data
            
        except Exception as e:
            logger.error(f"Failed to decompress data: {e}")
            raise SerializationError(f"Failed to decompress data: {e}")
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """
        获取压缩比
        
        Args:
            original_size: 原始数据大小
            compressed_size: 压缩后数据大小
            
        Returns:
            float: 压缩比（压缩后大小/原始大小）
        """
        if original_size == 0:
            return 1.0
        
        return compressed_size / original_size
    
    def estimate_memory_usage(self, data_size: int) -> Dict[str, int]:
        """
        估算内存使用量
        
        Args:
            data_size: 数据大小（字节）
            
        Returns:
            Dict[str, int]: 内存使用估算
        """
        # 简单估算：序列化过程中大约需要2-3倍的内存
        estimated_peak = data_size * 3
        estimated_steady = data_size * 2
        
        return {
            'peak_memory': estimated_peak,
            'steady_memory': estimated_steady,
            'input_size': data_size
        }
    
    def set_compression_level(self, compression_level: int) -> None:
        """
        设置压缩级别
        
        Args:
            compression_level: 新的压缩级别 (0-9)
        """
        if not 0 <= compression_level <= 9:
            raise ValueError("Compression level must be between 0 and 9")
        
        old_level = self.compression_level
        self.compression_level = compression_level
        
        logger.debug(f"Compression level changed from {old_level} to {compression_level}")
    
    def get_serializer_info(self) -> Dict[str, Any]:
        """
        获取序列化器信息
        
        Returns:
            Dict[str, Any]: 序列化器配置信息
        """
        return {
            'compression_level': self.compression_level,
            'torch_version': torch.__version__,
            'supports_cuda': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
