"""
MOE-FedCL 组件自动发现
fedcl/api/discovery.py

提供自动发现和注册用户自定义组件的功能。
"""

import os
import sys
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from loguru import logger

from ..registry import registry


def auto_discover_components(search_paths: Optional[List[Union[str, Path]]] = None,
                           recursive: bool = True,
                           component_types: Optional[List[str]] = None,
                           ignore_patterns: Optional[List[str]] = None) -> Dict[str, int]:
    """
    自动发现并注册组件
    
    Args:
        search_paths: 搜索路径列表，默认为当前工作目录
        recursive: 是否递归搜索子目录
        component_types: 要发现的组件类型列表，None表示所有类型
        ignore_patterns: 忽略的文件/目录模式列表
        
    Returns:
        Dict[str, int]: 每种组件类型的发现数量
    """
    if search_paths is None:
        search_paths = [Path.cwd()]
    
    if component_types is None:
        component_types = ['learner', 'trainer', 'aggregator', 'evaluator']
    
    if ignore_patterns is None:
        ignore_patterns = ['__pycache__', '.git', '.DS_Store', '*.pyc', 'test_*', '*_test.py']
    
    discovered = {comp_type: 0 for comp_type in component_types}
    
    for search_path in search_paths:
        search_path = Path(search_path)
        if not search_path.exists():
            logger.warning(f"搜索路径不存在: {search_path}")
            continue
            
        logger.info(f"开始搜索组件: {search_path}")
        
        # 搜索Python文件
        pattern = "**/*.py" if recursive else "*.py"
        
        for py_file in search_path.glob(pattern):
            # 跳过忽略的文件
            if any(_should_ignore(py_file, pattern) for pattern in ignore_patterns):
                continue
            
            try:
                # 导入模块并检查组件
                discovered_in_file = _discover_from_file(py_file, component_types)
                
                for comp_type, count in discovered_in_file.items():
                    discovered[comp_type] += count
                    
            except Exception as e:
                logger.debug(f"跳过文件 {py_file}: {e}")
    
    # 记录发现结果
    total_discovered = sum(discovered.values())
    if total_discovered > 0:
        logger.info(f"自动发现完成，共发现 {total_discovered} 个组件:")
        for comp_type, count in discovered.items():
            if count > 0:
                logger.info(f"  - {comp_type}: {count} 个")
    else:
        logger.info("未发现任何组件")
    
    return discovered


def register_from_module(module_name: str, 
                        component_types: Optional[List[str]] = None) -> Dict[str, int]:
    """
    从指定模块注册组件
    
    Args:
        module_name: 模块名称
        component_types: 要注册的组件类型列表
        
    Returns:
        Dict[str, int]: 每种组件类型的注册数量
    """
    if component_types is None:
        component_types = ['learner', 'trainer', 'aggregator', 'evaluator']
    
    discovered = {comp_type: 0 for comp_type in component_types}
    
    try:
        # 导入模块
        module = importlib.import_module(module_name)
        
        # 检查模块中的所有类
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, '_component_metadata'):
                metadata = obj._component_metadata
                comp_type = metadata.get('type')
                
                if comp_type in component_types:
                    discovered[comp_type] += 1
                    logger.info(f"从模块 {module_name} 发现组件: {metadata.get('name', name)} ({comp_type})")
        
    except ImportError as e:
        logger.error(f"无法导入模块 {module_name}: {e}")
    except Exception as e:
        logger.error(f"处理模块 {module_name} 时出错: {e}")
    
    return discovered


def register_from_package(package_name: str,
                         recursive: bool = True,
                         component_types: Optional[List[str]] = None) -> Dict[str, int]:
    """
    从指定包注册组件
    
    Args:
        package_name: 包名称
        recursive: 是否递归搜索子包
        component_types: 要注册的组件类型列表
        
    Returns:
        Dict[str, int]: 每种组件类型的注册数量
    """
    if component_types is None:
        component_types = ['learner', 'trainer', 'aggregator', 'evaluator']
    
    discovered = {comp_type: 0 for comp_type in component_types}
    
    try:
        # 导入包
        package = importlib.import_module(package_name)
        
        # 获取包路径
        if hasattr(package, '__path__'):
            package_path = package.__path__
        else:
            logger.warning(f"{package_name} 不是一个包")
            return discovered
        
        # 遍历包中的模块
        for importer, modname, ispkg in pkgutil.walk_packages(
            package_path, 
            prefix=package_name + '.',
            onerror=lambda x: None
        ):
            if not recursive and '.' in modname[len(package_name)+1:]:
                continue
                
            try:
                # 导入并处理模块
                module_discovered = register_from_module(modname, component_types)
                
                for comp_type, count in module_discovered.items():
                    discovered[comp_type] += count
                    
            except Exception as e:
                logger.debug(f"跳过模块 {modname}: {e}")
        
    except ImportError as e:
        logger.error(f"无法导入包 {package_name}: {e}")
    except Exception as e:
        logger.error(f"处理包 {package_name} 时出错: {e}")
    
    return discovered


def list_registered_components(component_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    列出已注册的组件
    
    Args:
        component_type: 组件类型，None表示所有类型
        
    Returns:
        Dict[str, Dict[str, Any]]: 组件信息
    """
    result = {}
    
    if component_type is None or component_type == 'learner':
        result['learners'] = {
            name: _get_component_info(cls) 
            for name, cls in registry.learners.items()
        }
    
    if component_type is None or component_type == 'trainer':
        result['trainers'] = {
            name: _get_component_info(cls) 
            for name, cls in registry.trainers.items()
        }
    
    if component_type is None or component_type == 'aggregator':
        result['aggregators'] = {
            name: _get_component_info(cls) 
            for name, cls in registry.aggregators.items()
        }
    
    if component_type is None or component_type == 'evaluator':
        result['evaluators'] = {
            name: _get_component_info(cls) 
            for name, cls in registry.evaluators.items()
        }
    
    return result


def clear_registry(component_type: Optional[str] = None):
    """
    清空注册表
    
    Args:
        component_type: 要清空的组件类型，None表示所有类型
    """
    if component_type is None or component_type == 'learner':
        registry.learners.clear()
        logger.info("已清空学习器注册表")
    
    if component_type is None or component_type == 'trainer':
        registry.trainers.clear()
        logger.info("已清空训练器注册表")
    
    if component_type is None or component_type == 'aggregator':
        registry.aggregators.clear()
        logger.info("已清空聚合器注册表")
    
    if component_type is None or component_type == 'evaluator':
        registry.evaluators.clear()
        logger.info("已清空评估器注册表")


# ==================== 私有辅助函数 ====================

def _should_ignore(file_path: Path, pattern: str) -> bool:
    """检查文件是否应该被忽略"""
    import fnmatch
    return fnmatch.fnmatch(str(file_path), pattern) or fnmatch.fnmatch(file_path.name, pattern)


def _discover_from_file(file_path: Path, component_types: List[str]) -> Dict[str, int]:
    """从单个文件发现组件"""
    discovered = {comp_type: 0 for comp_type in component_types}
    
    try:
        # 构造模块名
        module_name = _path_to_module_name(file_path)
        if not module_name:
            return discovered
        
        # 导入模块
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return discovered
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # 检查模块中的类
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, '_component_metadata'):
                metadata = obj._component_metadata
                comp_type = metadata.get('type')
                
                if comp_type in component_types:
                    discovered[comp_type] += 1
                    logger.debug(f"发现组件: {metadata.get('name', name)} ({comp_type}) 在 {file_path}")
        
    except Exception as e:
        logger.debug(f"处理文件 {file_path} 时出错: {e}")
    
    return discovered


def _path_to_module_name(file_path: Path) -> Optional[str]:
    """将文件路径转换为模块名"""
    try:
        # 移除.py扩展名
        if file_path.suffix != '.py':
            return None
        
        # 构造模块名
        module_name = str(file_path.with_suffix(''))
        module_name = module_name.replace(os.sep, '.')
        
        # 移除前导点
        module_name = module_name.lstrip('.')
        
        return module_name
    except Exception:
        return None


def _get_component_info(cls: Type) -> Dict[str, Any]:
    """获取组件信息"""
    if hasattr(cls, '_component_metadata'):
        return cls._component_metadata.copy()
    else:
        return {
            'type': 'unknown',
            'name': cls.__name__,
            'description': cls.__doc__ or '',
            'version': 'unknown',
            'class': str(cls)
        }
