# fedcl/federation/model_manager.py
"""
模型管理器实现

负责联邦学习中的全局模型管理，包括模型聚合、版本控制、检查点保存、
模型压缩、差异计算等功能。支持真联邦和伪联邦两种模式。
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from omegaconf import DictConfig
from loguru import logger
import pickle
import json
from datetime import datetime
import copy

from ...core.base_aggregator import BaseAggregator
from ...exceptions import ConfigurationError, ModelStateError


class ModelManager:
    """
    模型管理器
    
    负责联邦学习系统中的全局模型管理，包括客户端更新聚合、模型版本控制、
    检查点管理、模型压缩和历史记录等功能。
    
    Attributes:
        config: 模型管理配置
        aggregator: 聚合器实例
        device: 计算设备
        global_model: 当前全局模型
        model_history: 模型历史记录
        checkpoint_dir: 检查点保存目录
    """
    
    def __init__(self, config: DictConfig, aggregator: BaseAggregator) -> None:
        """
        初始化模型管理器
        
        Args:
            config: 配置参数
            aggregator: 聚合器实例
            
        Raises:
            ConfigurationError: 配置参数无效时抛出
        """
        if not isinstance(config, DictConfig):
            raise ConfigurationError("Invalid configuration provided")
            
        if not isinstance(aggregator, BaseAggregator):
            raise ConfigurationError("Invalid aggregator provided")
            
        self.config = config
        self.aggregator = aggregator
        self.device = torch.device(config.get("device", "cpu"))
        
        # 模型相关
        self.global_model: Optional[nn.Module] = None
        self.model_history: List[Dict[str, Any]] = []
        
        # 检查点配置
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = config.get("save_frequency", 10)  # 每10轮保存一次
        self.max_checkpoints = config.get("max_checkpoints", 20)  # 最多保存20个检查点
        
        # 压缩配置
        self.enable_compression = config.get("enable_compression", False)
        self.compression_ratio = config.get("compression_ratio", 0.1)
        
        logger.debug(f"Initialized ModelManager with device: {self.device}")
        logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def initialize_global_model(self, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """
        初始化全局模型
        
        Args:
            model: 可选的初始模型，如果未提供则创建默认模型
            
        Returns:
            初始化的全局模型
            
        Raises:
            ModelStateError: 模型初始化失败时抛出
        """
        try:
            if model is not None:
                self.global_model = model.to(self.device)
            else:
                # 如果没有提供模型，返回现有的全局模型或者None
                # 这里保持兼容性，让外部代码提供模型
                if self.global_model is not None:
                    return self.global_model
                else:
                    logger.warning("No global model provided and no existing model found")
                    return None
                
            self._record_model_initialization()
            logger.debug("Global model initialized successfully")
            return self.global_model
            
        except Exception as e:
            logger.error(f"Failed to initialize global model: {e}")
            raise ModelStateError(f"Failed to initialize global model: {e}")
    
    def update_global_model(self, client_updates: List[Dict[str, Any]]) -> torch.nn.Module:
        """
        更新全局模型
        
        Args:
            client_updates: 客户端更新列表，每个更新包含完整的客户端数据
            
        Returns:
            更新后的全局模型
            
        Raises:
            ModelStateError: 模型状态异常时抛出
        """
        try:
            if self.global_model is None:
                raise ModelStateError("Global model not initialized")
                
            if not client_updates:
                logger.warning("No client updates provided")
                return self.global_model
                
            # 提取模型参数更新
            logger.debug(f"Extracting model parameters from {len(client_updates)} client updates")
            parameter_updates = []
            for update in client_updates:
                # 从客户端更新中提取模型参数
                if 'aggregated_model_update' in update:
                    parameter_updates.append(update['aggregated_model_update'])
                elif 'model_state' in update:
                    parameter_updates.append(update['model_state'])
                else:
                    logger.warning(f"No model parameters found in client update: {list(update.keys())}")
                    
            if not parameter_updates:
                logger.warning("No valid parameter updates found")
                return self.global_model
                
            logger.debug(f"Extracted {len(parameter_updates)} parameter updates")
            
            # 使用聚合器聚合客户端更新
            aggregated_update = self.aggregator.aggregate(
                parameter_updates
            )
            
            # 应用聚合后的更新到全局模型
            with torch.no_grad():
                for name, param in self.global_model.named_parameters():
                    if name in aggregated_update:
                        param.data += aggregated_update[name].to(param.device)
                        
            # 记录模型历史
            self._record_model_update(len(client_updates), aggregated_update)
            
            logger.debug(f"Global model updated with {len(client_updates)} client updates")
            return self.global_model
            
        except Exception as e:
            logger.error(f"Failed to update global model: {e}")
            raise ModelStateError(f"Failed to update global model: {e}")
    
    def get_current_model(self) -> torch.nn.Module:
        """
        获取当前全局模型
        
        Returns:
            当前全局模型的副本
            
        Raises:
            ModelStateError: 模型未初始化时抛出
        """
        if self.global_model is None:
            raise ModelStateError("Global model not initialized")
            
        # 返回模型的深拷贝以避免意外修改
        return copy.deepcopy(self.global_model)
    
    def set_global_model(self, model: torch.nn.Module) -> None:
        """
        设置全局模型
        
        Args:
            model: 新的全局模型
        """
        try:
            self.global_model = copy.deepcopy(model).to(self.device)
            
            # 记录模型设置
            self._record_model_initialization()
            
            logger.debug("Global model set successfully")
            
        except Exception as e:
            logger.error(f"Failed to set global model: {e}")
            raise ModelStateError(f"Failed to set global model: {e}")
    
    def save_checkpoint(self, round_id: int, additional_info: Dict[str, Any] = None) -> Path:
        """
        保存检查点
        
        Args:
            round_id: 轮次ID
            additional_info: 附加信息
            
        Returns:
            检查点文件路径
        """
        try:
            if self.global_model is None:
                raise ModelStateError("No global model to save")
                
            checkpoint_filename = f"checkpoint_round_{round_id}.pt"
            checkpoint_path = self.checkpoint_dir / checkpoint_filename
            
            checkpoint_data = {
                "round_id": round_id,
                "model_state_dict": self.global_model.state_dict(),
                "model_history": self.model_history[-10:],  # 只保存最近10次历史
                "timestamp": datetime.now().isoformat(),
                "config": dict(self.config),
                "additional_info": additional_info or {}
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
            logger.debug(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise ModelStateError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> torch.nn.Module:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            加载的模型
        """
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            
            if self.global_model is None:
                raise ModelStateError("Global model structure not initialized")
                
            # 加载模型状态
            self.global_model.load_state_dict(checkpoint_data["model_state_dict"])
            
            # 恢复历史记录
            if "model_history" in checkpoint_data:
                self.model_history = checkpoint_data["model_history"]
                
            logger.debug(f"Checkpoint loaded: {checkpoint_path}")
            logger.debug(f"Restored from round {checkpoint_data.get('round_id', 'unknown')}")
            
            return self.global_model
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise ModelStateError(f"Failed to load checkpoint: {e}")
    
    def get_model_diff(self, old_model: torch.nn.Module, 
                      new_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        获取模型差异
        
        Args:
            old_model: 旧模型
            new_model: 新模型
            
        Returns:
            模型参数差异
        """
        try:
            old_params = dict(old_model.named_parameters())
            new_params = dict(new_model.named_parameters())
            
            model_diff = {}
            for name in old_params:
                if name in new_params:
                    model_diff[name] = new_params[name].data - old_params[name].data
                    
            logger.debug(f"Computed model diff with {len(model_diff)} parameters")
            return model_diff
            
        except Exception as e:
            logger.error(f"Failed to compute model diff: {e}")
            raise ModelStateError(f"Failed to compute model diff: {e}")
    
    def apply_model_diff(self, model: torch.nn.Module, 
                        diff: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """
        应用模型差异
        
        Args:
            model: 基础模型
            diff: 模型差异
            
        Returns:
            应用差异后的模型
        """
        try:
            updated_model = copy.deepcopy(model)
            
            with torch.no_grad():
                for name, param in updated_model.named_parameters():
                    if name in diff:
                        param.data += diff[name].to(param.device)
                        
            logger.debug(f"Applied model diff with {len(diff)} parameters")
            return updated_model
            
        except Exception as e:
            logger.error(f"Failed to apply model diff: {e}")
            raise ModelStateError(f"Failed to apply model diff: {e}")
    
    def get_model_size(self, model: torch.nn.Module) -> int:
        """
        获取模型大小（字节）
        
        Args:
            model: 模型
            
        Returns:
            模型大小（字节）
        """
        try:
            # 序列化模型以获取准确大小
            import io
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            size = buffer.tell()
            
            logger.debug(f"Model size: {size} bytes ({size / 1024 / 1024:.2f} MB)")
            return size
            
        except Exception as e:
            logger.error(f"Failed to compute model size: {e}")
            return 0
    
    def compress_model(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        压缩模型
        
        Args:
            model: 要压缩的模型
            
        Returns:
            压缩后的模型参数
        """
        try:
            if not self.enable_compression:
                return {name: param.data for name, param in model.named_parameters()}
                
            compressed_params = {}
            
            for name, param in model.named_parameters():
                # 简单的Top-K压缩
                flat_param = param.data.flatten()
                k = max(1, int(len(flat_param) * self.compression_ratio))
                
                # 找到Top-K个最大的绝对值
                _, indices = torch.topk(torch.abs(flat_param), k)
                
                # 创建稀疏表示
                compressed_param = torch.zeros_like(flat_param)
                compressed_param[indices] = flat_param[indices]
                
                compressed_params[name] = compressed_param.reshape(param.shape)
                
            logger.debug(f"Compressed model with ratio {self.compression_ratio}")
            return compressed_params
            
        except Exception as e:
            logger.error(f"Failed to compress model: {e}")
            return {name: param.data for name, param in model.named_parameters()}
    
    def decompress_model(self, compressed_model: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """
        解压模型
        
        Args:
            compressed_model: 压缩的模型参数
            
        Returns:
            解压后的模型
        """
        try:
            if self.global_model is None:
                raise ModelStateError("Global model structure not available")
                
            decompressed_model = copy.deepcopy(self.global_model)
            
            with torch.no_grad():
                for name, param in decompressed_model.named_parameters():
                    if name in compressed_model:
                        param.data.copy_(compressed_model[name].to(param.device))
                        
            logger.debug("Model decompressed successfully")
            return decompressed_model
            
        except Exception as e:
            logger.error(f"Failed to decompress model: {e}")
            raise ModelStateError(f"Failed to decompress model: {e}")
    
    def get_model_history(self) -> List[Dict[str, Any]]:
        """
        获取模型历史记录
        
        Returns:
            模型历史记录列表
        """
        return self.model_history.copy()
    
    def validate_model_update(self, update: Dict[str, torch.Tensor]) -> bool:
        """
        验证模型更新
        
        Args:
            update: 模型更新
            
        Returns:
            更新是否有效
        """
        try:
            if self.global_model is None:
                return False
                
            model_params = dict(self.global_model.named_parameters())
            
            # 检查参数名称匹配
            for name in update:
                if name not in model_params:
                    logger.warning(f"Unknown parameter in update: {name}")
                    return False
                    
                # 检查形状匹配
                if update[name].shape != model_params[name].shape:
                    logger.warning(f"Shape mismatch for parameter {name}: "
                                 f"期望 {model_params[name].shape}, "
                                 f"实际 {update[name].shape}")
                    return False
                    
                # 检查数值有效性
                if torch.isnan(update[name]).any() or torch.isinf(update[name]).any():
                    logger.warning(f"Invalid values in parameter {name}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Model update validation failed: {e}")
            return False
    
    def _record_model_update(self, num_clients: int, update: Dict[str, torch.Tensor]) -> None:
        """记录模型更新历史"""
        # 计算更新统计
        update_norm = 0.0
        for param_update in update.values():
            update_norm += param_update.norm().item() ** 2
        update_norm = update_norm ** 0.5
        
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "num_客户端": num_clients,
            "update_norm": update_norm,
            "num_parameters": len(update)
        }
        
        self.model_history.append(history_entry)
        
        # 限制历史记录长度
        if len(self.model_history) > 100:
            self.model_history = self.model_history[-100:]
    
    def _record_model_initialization(self) -> None:
        """记录模型初始化"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "model_initialization",
            "model_size": self.get_model_size(self.global_model),
            "num_parameters": sum(p.numel() for p in self.global_model.parameters())
        }
        
        self.model_history.append(history_entry)
    
    def _cleanup_old_checkpoints(self) -> None:
        """清理旧检查点"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_round_*.pt"))
            
            if len(checkpoint_files) > self.max_checkpoints:
                # 按修改时间排序，删除最旧的
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
                files_to_delete = checkpoint_files[:-self.max_checkpoints]
                
                for file_path in files_to_delete:
                    file_path.unlink()
                    logger.debug(f"Deleted old checkpoint: {file_path}")
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        获取模型管理器汇总统计
        
        Returns:
            汇总统计信息
        """
        stats = {
            "has_global_model": self.global_model is not None,
            "model_history_length": len(self.model_history),
            "checkpoint_dir": str(self.checkpoint_dir),
            "compression_enabled": self.enable_compression,
            "device": str(self.device)
        }
        
        if self.global_model is not None:
            stats.update({
                "model_size_bytes": self.get_model_size(self.global_model),
                "num_parameters": sum(p.numel() for p in self.global_model.parameters()),
                "num_trainable_parameters": sum(
                    p.numel() for p in self.global_model.parameters() if p.requires_grad
                )
            })
            
        # 检查点统计
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_round_*.pt"))
        stats["num_checkpoints"] = len(checkpoint_files)
        
        return stats
