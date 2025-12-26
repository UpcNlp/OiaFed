#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OiaFed - One Framework for All Federation

统一的联邦学习框架，支持所有联邦场景
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# 读取版本号
def get_version():
    version_file = this_directory / "src" / "__init__.py"
    for line in version_file.read_text().splitlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    return "0.1.0"

# 核心依赖
INSTALL_REQUIRES = [
    # 深度学习框架
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "torchaudio>=0.12.0",
    
    # gRPC 通信（必需）
    "grpcio>=1.50.0",
    "grpcio-tools>=1.50.0",
    "protobuf>=3.20.0",
    
    # 配置和日志
    "pyyaml>=6.0",
    "omegaconf>=2.2.0",
    "loguru>=0.6.0",
    "toml>=0.10.0",
    
    # 数据处理
    "numpy>=1.21.0",
    "pandas>=1.4.0",
    "scikit-learn>=1.0.0",
    
    # 可视化
    "matplotlib>=3.5.0",
    "tqdm>=4.60.0",
    
    # 网络和异步
    "aiohttp>=3.8.0",
    
    # 工具
    "psutil>=5.9.0",
    "openpyxl>=3.0.0",
]

# 可选依赖
EXTRAS_REQUIRE = {
    # MLflow 实验追踪
    "mlflow": [
        "mlflow>=2.0.0",
    ],
    # 开发依赖
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.20.0",
        "pytest-html>=3.2.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "mypy>=0.990",
        "flake8>=5.0.0",
    ],
    # 文档
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "myst-parser>=0.18.0",
    ],
}

# all = mlflow + dev
EXTRAS_REQUIRE["all"] = (
    EXTRAS_REQUIRE["mlflow"] + 
    EXTRAS_REQUIRE["dev"] + 
    EXTRAS_REQUIRE["docs"]
)

setup(
    name="oiafed",
    version=get_version(),
    author="OiaFed Team",
    author_email="contact@oiafed.cn",
    description="OiaFed: One Framework for All Federation - A unified federated learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oiafed/oiafed",
    project_urls={
        "Homepage": "https://oiafed.cn",
        "Documentation": "https://docs.oiafed.cn",
        "Repository": "https://github.com/oiafed/oiafed",
        "Issues": "https://github.com/oiafed/oiafed/issues",
        "Changelog": "https://github.com/oiafed/oiafed/blob/main/CHANGELOG.md",
    },
    license="MIT",
    
    # 包配置
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    package_dir={"oiafed": "src"},
    py_modules=[],
    include_package_data=True,
    
    # 依赖
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # 入口点
    entry_points={
        "console_scripts": [
            "oiafed=oiafed.cli:main",
        ],
    },
    
    # 分类
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    keywords=[
        "federated-learning",
        "machine-learning", 
        "deep-learning",
        "distributed-computing",
        "continual-learning",
        "personalized-federated-learning",
        "privacy-preserving",
    ],
)