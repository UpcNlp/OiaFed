"""
Checkpoint - 检查点管理器

支持模型保存、加载和自动清理
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import os
import json
import time
import glob
from .logging import get_module_logger

if TYPE_CHECKING:
    from ..core.model import Model

logger = get_module_logger(__name__)


class Checkpoint:
    """
    检查点管理器
    
    功能：
    - 保存/加载模型检查点
    - 自动清理旧检查点
    - 支持额外元数据
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检查点管理器
        
        Args:
            config: 配置字典
                - dir: 检查点目录
                - max_to_keep: 最多保留数量
                - filename_format: 文件名格式
        """
        self._dir = config.get("dir", "./checkpoints")
        self._max_to_keep = config.get("max_to_keep", 5)
        self._filename_format = config.get("filename_format", "checkpoint_{round:04d}")
        
        # 确保目录存在
        os.makedirs(self._dir, exist_ok=True)
    
    def save(
        self,
        model: "Model",
        round: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        保存检查点
        
        Args:
            model: 模型
            round: 轮次号
            extra: 额外信息
            
        Returns:
            检查点路径
        """
        # 生成文件名
        filename = self._filename_format.format(round=round)
        checkpoint_path = os.path.join(self._dir, filename)
        
        # 保存模型权重
        weights_path = checkpoint_path + ".weights"
        with open(weights_path, "wb") as f:
            f.write(model.serialize())
        
        # 保存元数据
        meta = {
            "round": round,
            "timestamp": time.time(),
            "extra": extra or {},
        }
        meta_path = checkpoint_path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # 清理旧检查点
        self._cleanup()
        
        return checkpoint_path
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            path: 检查点路径（不带后缀）
            
        Returns:
            包含 weights 和 meta 的字典
        """
        weights_path = path + ".weights" if not path.endswith(".weights") else path
        meta_path = path.replace(".weights", "") + ".meta.json"
        
        # 加载权重
        with open(weights_path, "rb") as f:
            weights_data = f.read()
        
        # 加载元数据
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
        
        return {
            "weights_data": weights_data,
            "round": meta.get("round"),
            "extra": meta.get("extra", {}),
        }
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        加载最新的检查点
        
        Returns:
            检查点数据，如果没有则返回 None
        """
        checkpoints = self.list()
        if not checkpoints:
            return None
        
        # 按轮次排序，取最新
        latest = max(checkpoints, key=lambda x: x.get("round", 0))
        return self.load(latest["path"])
    
    def list(self) -> List[Dict[str, Any]]:
        """
        列出所有检查点
        
        Returns:
            检查点信息列表
        """
        pattern = os.path.join(self._dir, "*.meta.json")
        meta_files = glob.glob(pattern)
        
        checkpoints = []
        for meta_file in meta_files:
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                
                path = meta_file.replace(".meta.json", "")
                checkpoints.append({
                    "path": path,
                    "round": meta.get("round"),
                    "timestamp": meta.get("timestamp"),
                })
            except Exception as e:
                logger.warning(f"Error reading checkpoint meta: {e}")
        
        return sorted(checkpoints, key=lambda x: x.get("round", 0))
    
    def delete(self, path: str) -> bool:
        """
        删除检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            是否成功删除
        """
        try:
            weights_path = path + ".weights" if not path.endswith(".weights") else path
            meta_path = path.replace(".weights", "") + ".meta.json"
            
            if os.path.exists(weights_path):
                os.remove(weights_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            
            logger.info(f"Deleted checkpoint: {path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting checkpoint: {e}")
            return False
    
    def _cleanup(self) -> None:
        """清理旧检查点"""
        if self._max_to_keep <= 0:
            return
        
        checkpoints = self.list()
        if len(checkpoints) > self._max_to_keep:
            # 删除最旧的
            to_delete = checkpoints[:-self._max_to_keep]
            for ckpt in to_delete:
                self.delete(ckpt["path"])
