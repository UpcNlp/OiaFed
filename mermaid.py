#!/usr/bin/env python3
"""
Python目录扫描生成Mermaid类图工具
支持递归扫描目录，解析Python文件并生成Mermaid格式的类图
"""

import os
import ast
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional


class ClassInfo:
    """存储类信息"""
    def __init__(self, name: str, file_path: str):
        self.name = name
        self.file_path = file_path
        self.methods: List[str] = []
        self.attributes: List[str] = []
        self.parent_classes: List[str] = []
        self.imports: Set[str] = set()


class MermaidClassDiagramGenerator:
    """Mermaid类图生成器"""
    
    def __init__(self, target_dir: str, exclude_dirs: List[str] = None):
        self.target_dir = Path(target_dir)
        self.exclude_dirs = exclude_dirs or ['__pycache__', '.git', 'venv', 'env', '.pytest_cache']
        self.classes: Dict[str, ClassInfo] = {}
        self.relationships: List[tuple] = []
    
    def should_skip_directory(self, dir_path: Path) -> bool:
        """检查是否应该跳过目录"""
        return any(exclude in str(dir_path) for exclude in self.exclude_dirs)
    
    def extract_class_info(self, file_path: Path) -> List[ClassInfo]:
        """从Python文件中提取类信息"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = ClassInfo(node.name, str(file_path))
                    
                    # 提取父类
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            class_info.parent_classes.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            # 处理 module.ClassName 形式
                            parent_name = self._get_full_name(base)
                            if parent_name:
                                class_info.parent_classes.append(parent_name)
                    
                    # 提取方法和属性
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = item.name
                            if not method_name.startswith('_') or method_name in ['__init__', '__str__', '__repr__']:
                                visibility = '+' if not method_name.startswith('_') else '-'
                                class_info.methods.append(f"{visibility}{method_name}()")
                        
                        elif isinstance(item, ast.Assign):
                            # 提取类属性
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    attr_name = target.id
                                    if not attr_name.startswith('_'):
                                        class_info.attributes.append(f"+{attr_name}")
                    
                    classes.append(class_info)
            
            return classes
            
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
            return []
    
    def _get_full_name(self, node: ast.Attribute) -> Optional[str]:
        """获取属性的完整名称"""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            parent = self._get_full_name(node.value)
            return f"{parent}.{node.attr}" if parent else None
        return None
    
    def scan_directory(self):
        """扫描目录并提取所有类信息"""
        print(f"扫描目录: {self.target_dir}")
        
        for root, dirs, files in os.walk(self.target_dir):
            root_path = Path(root)
            
            # 跳过排除的目录
            if self.should_skip_directory(root_path):
                continue
            
            # 移除排除的子目录
            dirs[:] = [d for d in dirs if not any(exclude in d for exclude in self.exclude_dirs)]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('_'):
                    file_path = root_path / file
                    classes = self.extract_class_info(file_path)
                    
                    for class_info in classes:
                        self.classes[class_info.name] = class_info
                        print(f"发现类: {class_info.name} in {file_path}")
    
    def generate_relationships(self):
        """生成类之间的关系"""
        for class_name, class_info in self.classes.items():
            for parent in class_info.parent_classes:
                # 简单的父类名称匹配
                parent_simple = parent.split('.')[-1]
                if parent_simple in self.classes:
                    self.relationships.append((parent_simple, class_name, "inheritance"))
                elif parent in self.classes:
                    self.relationships.append((parent, class_name, "inheritance"))
    
    def generate_mermaid(self) -> str:
        """生成Mermaid类图代码"""
        lines = ["classDiagram"]
        
        # 生成类定义
        for class_name, class_info in self.classes.items():
            lines.append(f"    class {class_name} {{")
            
            # 添加属性
            for attr in class_info.attributes:
                lines.append(f"        {attr}")
            
            # 添加方法
            for method in class_info.methods:
                lines.append(f"        {method}")
            
            lines.append("    }")
            lines.append("")
        
        # 生成关系
        for parent, child, rel_type in self.relationships:
            if rel_type == "inheritance":
                lines.append(f"    {parent} <|-- {child}")
        
        return "\n".join(lines)
    
    def save_diagram(self, output_file: str):
        """保存Mermaid图表到文件"""
        mermaid_code = self.generate_mermaid()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        
        print(f"Mermaid类图已保存到: {output_file}")
        print(f"共发现 {len(self.classes)} 个类")
        print(f"共发现 {len(self.relationships)} 个关系")


def main():
    parser = argparse.ArgumentParser(description='从Python目录生成Mermaid类图')
    parser.add_argument('directory', nargs='?', 
                       default=str(Path(__file__).parent / 'fedcl'), 
                       help='要扫描的目录路径 (默认: ./fedcl)')
    parser.add_argument('-o', '--output', default='class_diagram.mmd', help='输出文件名')
    parser.add_argument('-e', '--exclude', nargs='*', default=[], help='要排除的目录名')
    parser.add_argument('--show-private', action='store_true', help='显示私有方法和属性')
    
    args = parser.parse_args()
    
    # 默认排除目录
    default_excludes = ['__pycache__', '.git', 'venv', 'env', '.pytest_cache', 'node_modules']
    exclude_dirs = default_excludes + args.exclude
    
    try:
        generator = MermaidClassDiagramGenerator(args.directory, exclude_dirs)
        generator.scan_directory()
        generator.generate_relationships()
        generator.save_diagram(args.output)
        
        print("\n生成的类:")
        for class_name, class_info in generator.classes.items():
            print(f"  - {class_name} ({class_info.file_path})")
            
    except Exception as e:
        print(f"生成类图时出错: {e}")


if __name__ == '__main__':
    main()