"""
OiaFed 命令行接口

提供专业的命令行工具，支持多种操作：
- init: 初始化新实验（生成配置模板）
- generate: 从基础配置生成多客户端配置
- run: 运行联邦学习实验
- validate: 验证配置文件
- list: 列出可用组件
- version: 显示版本信息

使用方式:
    # 初始化实验
    oiafed init my_experiment --algorithm fedavg --dataset cifar10
    oiafed init fcl_exp --scenario fcl --num-clients 5
    
    # 生成配置
    oiafed generate --base base.yaml --num-clients 10
    
    # 运行实验
    oiafed run --config config.yaml
    oiafed run --config configs/ --mode parallel
    
    # 其他
    oiafed validate --config config.yaml
    oiafed list aggregators
    oiafed version
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # 避免 grpcio 与 protobuf 版本冲突问题
import argparse
import sys
import os
import re
from pathlib import Path
from typing import List, Optional


def _load_yaml_with_env(file_path: Path) -> dict:
    """
    加载 YAML 文件并替换环境变量
    
    支持语法: ${VAR_NAME:default_value} 或 ${VAR_NAME}
    
    Args:
        file_path: YAML 文件路径
        
    Returns:
        解析后的字典
    """
    import yaml
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 替换环境变量 ${VAR:default}
    def replace_env_var(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else ""
        return os.environ.get(var_name, default_value)
    
    # 匹配 ${VAR:default} 或 ${VAR}
    content = re.sub(r'\$\{([^}:]+)(?::([^}]*))?\}', replace_env_var, content)
    
    return yaml.safe_load(content) or {}


def get_version() -> str:
    """获取版本号"""
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "0.1.0"


def get_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="oiafed",
        description="OiaFed: One Framework for All Federation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 初始化新实验
  oiafed init my_exp --algorithm fedavg --dataset cifar10 --num-clients 10
  oiafed init fcl_exp --scenario fcl --dataset cifar10
  
  # 从基础配置生成多客户端配置
  oiafed generate --base base.yaml --num-clients 10
  oiafed generate --base base.yaml -n 5 --partition dirichlet --alpha 0.3
  
  # 运行实验
  oiafed run --config configs/fedavg.yaml
  oiafed run --config configs/experiment/ --mode parallel
  
  # 验证配置
  oiafed validate --config config.yaml
  
  # 列出组件
  oiafed list aggregators
  oiafed list learners
  
  # 显示版本
  oiafed version

Documentation: https://docs.oiafed.cn
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ========== run 命令 ==========
    run_parser = subparsers.add_parser(
        "run",
        help="运行联邦学习实验",
        description="从配置文件或论文定义运行联邦学习实验"
    )
    
    # 配置来源
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件/目录路径。如果是目录则直接运行；如果是文件则作为补充配置"
    )
    run_parser.add_argument(
        "--paper", "-p",
        type=str,
        default=None,
        help="论文ID (如 fedavg, fot)，使用论文默认配置"
    )
    
    # 客户端数量（--paper 模式必需）
    run_parser.add_argument(
        "--num-clients", "-n",
        type=int,
        default=None,
        help="客户端数量 (--paper 模式必需)"
    )
    
    # 运行模式
    run_parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["auto", "serial", "parallel"],
        default="parallel",
        help="运行模式 (default: parallel)"
    )
    
    # 便捷参数 - 覆盖配置
    run_parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="训练轮数"
    )
    run_parser.add_argument(
        "--local-epochs",
        type=int,
        default=None,
        help="本地训练轮数"
    )
    run_parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="学习率"
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小"
    )
    run_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据目录"
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录"
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )
    
    # 其他参数
    run_parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (default: INFO)"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示配置，不实际运行"
    )
    run_parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="保存生成的配置到指定目录"
    )
    
    # ========== validate 命令 ==========
    validate_parser = subparsers.add_parser(
        "validate",
        help="验证配置文件",
        description="验证配置文件的正确性"
    )
    validate_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="配置文件路径"
    )
    validate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细信息"
    )
    
    # ========== list 命令 ==========
    list_parser = subparsers.add_parser(
        "list",
        help="列出可用组件",
        description="列出框架中注册的组件"
    )
    list_parser.add_argument(
        "component_type",
        type=str,
        nargs="?",
        choices=["all", "aggregators", "learners", "models", "datasets", "trainers"],
        default="all",
        help="组件类型 (default: all)"
    )
    
    # ========== show-config 命令 ==========
    show_parser = subparsers.add_parser(
        "show-config",
        help="显示解析后的配置",
        description="显示完整的解析后配置（包括继承和默认值）"
    )
    show_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="配置文件路径"
    )
    
    # ========== version 命令 ==========
    subparsers.add_parser(
        "version",
        help="显示版本信息"
    )
    
    # ========== info 命令 ==========
    subparsers.add_parser(
        "info",
        help="显示框架信息"
    )
    
    # ========== init 命令 ==========
    init_parser = subparsers.add_parser(
        "init",
        help="初始化新实验",
        description="生成实验配置模板"
    )
    init_parser.add_argument(
        "name",
        type=str,
        help="实验名称"
    )
    init_parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="fedavg",
        help="聚合算法 (default: fedavg)"
    )
    init_parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="cifar10",
        choices=["mnist", "fmnist", "cifar10", "cifar100"],
        help="数据集 (default: cifar10)"
    )
    init_parser.add_argument(
        "--num-clients", "-n",
        type=int,
        default=10,
        help="客户端数量 (default: 10)"
    )
    init_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./configs",
        help="输出目录 (default: ./configs)"
    )
    init_parser.add_argument(
        "--scenario",
        type=str,
        default="hfl",
        choices=["hfl", "pfl", "fcl", "vfl", "fu"],
        help="联邦场景 (default: hfl)"
    )
    init_parser.add_argument(
        "--paper", "-p",
        type=str,
        default=None,
        help="从论文生成配置 (如 --paper fedavg)，会覆盖 --algorithm 和 --scenario"
    )
    
    # ========== generate 命令 ==========
    gen_parser = subparsers.add_parser(
        "generate",
        help="生成多客户端配置",
        description="从基础配置生成多个 learner 配置文件"
    )
    gen_parser.add_argument(
        "--base", "-b",
        type=str,
        required=True,
        help="基础配置文件路径"
    )
    gen_parser.add_argument(
        "--num-clients", "-n",
        type=int,
        required=True,
        help="客户端数量"
    )
    gen_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录 (default: 与base同目录)"
    )
    gen_parser.add_argument(
        "--partition",
        type=str,
        default="dirichlet",
        choices=["iid", "dirichlet", "label_skew", "quantity_skew"],
        help="数据划分策略 (default: dirichlet)"
    )
    gen_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha参数 (default: 0.5)"
    )
    
    # ========== papers 命令 ==========
    papers_parser = subparsers.add_parser(
        "papers",
        help="论文库管理",
        description="浏览、查询论文实现。运行论文请使用: oiafed run --paper <paper_id> -n <num_clients>"
    )
    papers_subparsers = papers_parser.add_subparsers(dest="papers_command", help="论文操作")
    
    # papers list
    papers_list_parser = papers_subparsers.add_parser(
        "list",
        help="列出所有论文"
    )
    papers_list_parser.add_argument(
        "--category", "-c",
        type=str,
        choices=["HFL", "VFL", "FCL", "FU"],
        help="按类别筛选"
    )
    
    # papers show
    papers_show_parser = papers_subparsers.add_parser(
        "show",
        help="显示论文详情"
    )
    papers_show_parser.add_argument(
        "paper_id",
        type=str,
        help="论文ID"
    )
    papers_show_parser.add_argument(
        "--params", "-p",
        action="store_true",
        help="显示可调参数"
    )
    
    # papers init
    papers_init_parser = papers_subparsers.add_parser(
        "init",
        help="生成论文配置模板"
    )
    papers_init_parser.add_argument(
        "paper_id",
        type=str,
        help="论文ID"
    )
    papers_init_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./configs",
        help="输出目录 (default: ./configs)"
    )
    papers_init_parser.add_argument(
        "--num-clients", "-n",
        type=int,
        help="客户端数量 (覆盖默认值)"
    )
    
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """执行 run 命令"""
    from .infra.logging import setup_logging
    
    # 设置日志
    setup_logging(node_id="cli", level=args.log_level, console=True)
    
    # 判断运行模式
    config_path = Path(args.config) if args.config else None
    
    # 情况1: --config 是目录，直接运行（原有逻辑）
    if config_path and config_path.is_dir():
        return _run_from_config_dir(args, config_path)
    
    # 情况2: --paper 模式，从论文生成配置运行
    if args.paper:
        return _run_from_paper(args, config_path)
    
    # 情况3: 只有 --config 文件，没有 --paper
    if config_path and config_path.is_file():
        print("错误: 单个配置文件需要配合 --paper 使用，或者指定配置目录", file=sys.stderr)
        print("示例:")
        print("  oiafed run --config ./configs/my_exp/     # 运行配置目录")
        print("  oiafed run --paper fedavg -n 10           # 使用论文默认")
        print("  oiafed run --paper fedavg -n 10 --config base.yaml  # 论文+配置文件")
        return 1
    
    # 情况4: 什么都没指定
    print("错误: 必须指定 --config 或 --paper", file=sys.stderr)
    print("示例:")
    print("  oiafed run --config ./configs/my_exp/     # 运行配置目录")
    print("  oiafed run --paper fedavg -n 10           # 使用论文默认")
    return 1


def _run_from_config_dir(args: argparse.Namespace, config_path: Path) -> int:
    """从配置目录运行（原有逻辑）"""
    from .runner import FederationRunner
    
    if not config_path.exists():
        print(f"错误: 配置目录不存在: {config_path}", file=sys.stderr)
        return 1
    
    try:
        runner = FederationRunner(str(config_path))
        
        if args.dry_run:
            print("配置解析成功！")
            print(f"  节点数: {len(runner.configs)}")
            print(f"  节点ID: {runner.get_node_ids()}")
            print(f"  运行模式: {runner._get_execution_mode()}")
            return 0
        
        # 运行实验
        result = runner.run_sync()
        
        print("\n实验完成！")
        if result:
            print(f"  结果: {result}")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1


def _run_from_paper(args: argparse.Namespace, config_file: Optional[Path]) -> int:
    """从论文定义运行"""
    import yaml
    import tempfile
    import shutil
    import socket
    
    try:
        from .papers import get_registry
    except ImportError:
        from oiafed.papers import get_registry
    
    # 验证参数
    if args.num_clients is None:
        print("错误: --paper 模式必须指定 -n/--num-clients", file=sys.stderr)
        return 1
    
    registry = get_registry()
    paper = registry.get(args.paper)
    
    if not paper:
        print(f"错误: 未找到论文 '{args.paper}'", file=sys.stderr)
        print("使用 'oiafed papers list' 查看所有可用论文")
        return 1
    
    # 1. 加载论文默认配置
    paper_defaults = registry.get_defaults(args.paper)
    
    # 2. 加载配置文件（如果指定）- 支持环境变量替换
    file_config = {}
    if config_file and config_file.exists():
        file_config = _load_yaml_with_env(config_file)
    elif config_file is None:
        # 尝试加载默认 base.yaml
        default_base = Path("configs/base.yaml")
        if default_base.exists():
            file_config = _load_yaml_with_env(default_base)
            print(f"使用默认配置: {default_base}")
    
    # 3. 构建命令行覆盖参数
    cli_override = {}
    
    if args.rounds is not None:
        cli_override.setdefault("trainer", {})["num_rounds"] = args.rounds
    if args.local_epochs is not None:
        cli_override.setdefault("trainer", {})["local_epochs"] = args.local_epochs
    if args.learning_rate is not None:
        cli_override.setdefault("learner", {})["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        cli_override.setdefault("learner", {})["batch_size"] = args.batch_size
    if args.data_dir is not None:
        cli_override.setdefault("dataset", {})["data_dir"] = args.data_dir
    if args.seed is not None:
        cli_override["seed"] = args.seed
    
    # 4. 合并配置：paper_defaults < file_config < cli_override
    merged = _deep_merge(paper_defaults, file_config)
    merged = _deep_merge(merged, cli_override)
    
    # 5. 获取运行模式
    mode = args.mode
    if mode == "auto":
        mode = file_config.get("mode", "parallel")
    
    # 6. 生成临时配置目录或保存到指定目录
    if args.save_config:
        temp_dir = args.save_config
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.mkdtemp(prefix=f"oiafed_{args.paper}_")
        cleanup = True
    
    try:
        # 生成配置文件
        _generate_paper_configs(
            paper=paper,
            registry=registry,
            merged_config=merged,
            num_clients=args.num_clients,
            output_dir=temp_dir,
            mode=mode,
            file_config=file_config,
        )
        
        if args.dry_run:
            # 计算 exp_name 和划分信息用于显示
            dataset_type = paper.get_component("dataset") or "cifar10"
            dataset_config = merged.get("dataset", {})
            partition_config = dataset_config.get("partition", {})
            partition_strategy = partition_config.get("strategy", "dirichlet")
            
            if "exp_name" in file_config:
                exp_name = file_config["exp_name"]
            else:
                exp_name = _generate_exp_name(
                    algorithm=paper.id,
                    dataset=dataset_type,
                    partition_strategy=partition_strategy,
                    partition_config=partition_config,
                )
            
            print(f"\n[Dry Run] 论文: {paper.name}")
            print(f"{'=' * 50}")
            print(f"  实验名称: {exp_name}")
            print(f"  客户端数: {args.num_clients}")
            print(f"  运行模式: {mode}")
            print(f"  配置目录: {temp_dir}")
            print(f"")
            print(f"  数据集配置:")
            print(f"    - 数据集: {dataset_type}")
            print(f"    - 划分策略: {partition_strategy}")
            if partition_strategy == "dirichlet":
                print(f"    - alpha: {partition_config.get('alpha', 0.5)}")
            elif partition_strategy == "label_skew":
                print(f"    - num_labels_per_client: {partition_config.get('num_labels_per_client', 2)}")
            print(f"")
            print(f"  组件:")
            for comp_type, comp_name in paper.components.items():
                print(f"    - {comp_type}: {comp_name}")
            print(f"")
            print(f"  训练参数:")
            print(f"    - rounds: {merged.get('trainer', {}).get('num_rounds', 'N/A')}")
            print(f"    - local_epochs: {merged.get('trainer', {}).get('local_epochs', 'N/A')}")
            print(f"    - learning_rate: {merged.get('learner', {}).get('learning_rate', 'N/A')}")
            print(f"    - batch_size: {merged.get('learner', {}).get('batch_size', 'N/A')}")
            print(f"")
            # 显示 tracker 配置
            tracker_config = merged.get("tracker", {})
            if tracker_config.get("enabled", False) or tracker_config.get("backends"):
                print(f"  实验追踪:")
                for backend in tracker_config.get("backends", []):
                    backend_type = backend.get("type", "unknown")
                    print(f"    - {backend_type}")
                    if backend_type == "mlflow":
                        args_dict = backend.get("args", {})
                        if args_dict.get("tracking_uri"):
                            print(f"      tracking_uri: {args_dict['tracking_uri']}")
                        if args_dict.get("experiment_name"):
                            print(f"      experiment_name: {args_dict['experiment_name']}")
                print(f"")
            print(f"{'=' * 50}")
            
            if args.save_config:
                print(f"\n配置已保存到: {temp_dir}")
            return 0
        
        # 运行实验
        # 计算 exp_name 用于显示
        dataset_type = paper.get_component("dataset") or "cifar10"
        dataset_config = merged.get("dataset", {})
        partition_config = dataset_config.get("partition", {})
        partition_strategy = partition_config.get("strategy", "dirichlet")
        
        if "exp_name" in file_config:
            exp_name = file_config["exp_name"]
        else:
            exp_name = _generate_exp_name(
                algorithm=paper.id,
                dataset=dataset_type,
                partition_strategy=partition_strategy,
                partition_config=partition_config,
            )
        
        print(f"\n运行论文实验: {paper.name}")
        print(f"  实验名称: {exp_name}")
        print(f"  客户端数: {args.num_clients}")
        print(f"  运行模式: {mode}")
        print(f"{'=' * 50}")
        
        from .runner import FederationRunner
        runner = FederationRunner(temp_dir)
        result = runner.run_sync()
        
        print(f"\n实验完成！")
        return 0
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # 清理临时目录
        if cleanup and not args.dry_run:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _generate_paper_configs(
    paper,
    registry,
    merged_config: dict,
    num_clients: int,
    output_dir: str,
    mode: str,
    file_config: dict,
) -> None:
    """生成论文配置文件"""
    import yaml
    import socket
    
    output_path = Path(output_dir)
    
    # 获取网络配置
    network_config = file_config.get("network", {})
    trainer_port = network_config.get("trainer_port", 50051)
    learner_base_port = network_config.get("learner_base_port", 50052)
    auto_find_port = network_config.get("auto_find_port", True)
    
    # 获取数据集和划分配置
    dataset_type = paper.get_component("dataset") or "cifar10"
    dataset_config = merged_config.get("dataset", {})
    partition_config = dataset_config.get("partition", {})
    partition_strategy = partition_config.get("strategy", "dirichlet")
    
    # 生成 exp_name: {algorithm}_{dataset}_{partition}
    # 如果用户显式指定了 exp_name，则使用用户指定的
    if "exp_name" in file_config:
        exp_name = file_config["exp_name"]
    else:
        exp_name = _generate_exp_name(
            algorithm=paper.id,
            dataset=dataset_type,
            partition_strategy=partition_strategy,
            partition_config=partition_config,
        )
    
    data_dir = dataset_config.get("data_dir", "./data")
    seed = merged_config.get("seed", 42)
    
    # 生成统一的 run_name（时间戳），确保所有节点使用相同的值
    from datetime import datetime
    run_name = file_config.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 日志和追踪配置（深度合并）
    # 确保 exp_name 和 run_name 传递到 logging 配置
    logging_config = merged_config.get("logging", {})
    logging_config = {
        "level": logging_config.get("level", "INFO"),
        "console": logging_config.get("console", True),
        "exp_name": exp_name,      # 使用生成的实验名称
        "run_name": run_name,      # 使用统一的运行名称
        **{k: v for k, v in logging_config.items() if k not in ["level", "console", "exp_name", "run_name"]},
    }
    tracker_config = merged_config.get("tracker", {})
    
    # 如果需要，查找可用端口
    if mode == "parallel" and auto_find_port:
        trainer_port = _find_available_port(trainer_port)
        learner_base_port = _find_available_port(learner_base_port)
    
    # ===== 生成 Trainer 配置 =====
    trainer_config = {
        "node_id": "trainer",
        "role": "trainer",
        "exp_name": exp_name,
        
        "listen": {
            "host": "localhost",
            "port": trainer_port,
        },
        "min_peers": num_clients,
        
        "trainer": {
            "type": paper.get_component("trainer") or "default",
            "args": merged_config.get("trainer", {}),
        },
        
        "aggregator": {
            "type": paper.get_component("aggregator") or "fedavg",
            "args": merged_config.get("aggregator", {}),
        },
        
        "model": {
            "type": paper.get_component("model") or "cnn",
            "args": merged_config.get("model", {}),
        },
        
        "logging": logging_config,
        "transport": {"mode": "grpc" if mode == "parallel" else "memory"},
    }
    
    if tracker_config.get("enabled", True) and tracker_config.get("backends"):
        trainer_config["tracker"] = tracker_config
    
    trainer_path = output_path / "trainer.yaml"
    with open(trainer_path, "w", encoding="utf-8") as f:
        yaml.dump(trainer_config, f, default_flow_style=False, allow_unicode=True)
    
    # ===== 生成 Learner 配置 =====
    # 注意：dataset_type 和 partition_config 已在上面获取
    # 这里需要复制 dataset_config 以避免修改原始配置
    dataset_config_copy = {k: v for k, v in dataset_config.items() if k != "partition"}
    
    for i in range(num_clients):
        learner_port = learner_base_port + i
        if mode == "parallel" and auto_find_port and i > 0:
            learner_port = _find_available_port(learner_port)
        
        learner_config = {
            "node_id": f"learner_{i}",
            "role": "learner",
            "exp_name": exp_name,
            
            "listen": {
                "host": "localhost",
                "port": learner_port,
            },
            "connect_to": [f"trainer@localhost:{trainer_port}"],
            
            "learner": {
                "type": paper.get_component("learner") or "default",
                "args": merged_config.get("learner", {}),
            },
            
            "model": {
                "type": paper.get_component("model") or "cnn",
                "args": merged_config.get("model", {}),
            },
            
            "datasets": [
                {
                    "type": dataset_type,
                    "split": "train",
                    "args": {
                        "data_dir": data_dir,
                        "download": True,
                        **{k: v for k, v in dataset_config_copy.items() if k not in ["data_dir", "download"]},
                    },
                    "partition": {
                        "strategy": partition_strategy,
                        "num_partitions": num_clients,
                        "partition_id": i,
                        "seed": seed,
                        **{k: v for k, v in partition_config.items() if k not in ["strategy", "num_partitions", "partition_id", "seed"]},
                    },
                },
                {
                    "type": dataset_type,
                    "split": "test",
                    "args": {
                        "data_dir": data_dir,
                    },
                },
            ],
            
            "logging": logging_config,
            "transport": {"mode": "grpc" if mode == "parallel" else "memory"},
        }
        
        # Learner 也需要 tracker 配置（用于同步后记录指标）
        if tracker_config.get("enabled", True) and tracker_config.get("backends"):
            learner_config["tracker"] = tracker_config
        
        learner_path = output_path / f"learner_{i}.yaml"
        with open(learner_path, "w", encoding="utf-8") as f:
            yaml.dump(learner_config, f, default_flow_style=False, allow_unicode=True)


def _find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """查找可用端口"""
    import socket
    
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    
    # 如果找不到，返回原端口（让后续程序报错）
    return start_port


def _generate_exp_name(
    algorithm: str,
    dataset: str,
    partition_strategy: str,
    partition_config: dict,
) -> str:
    """
    自动生成实验名称
    
    格式: {algorithm}_{dataset}_{partition}
    
    示例:
    - fedavg_cifar10_iid
    - fedavg_cifar10_dir_0.5
    - moon_mnist_label_skew_3
    - scaffold_cifar100_quantity_skew_0.8
    
    Args:
        algorithm: 算法名称 (paper.id)
        dataset: 数据集名称
        partition_strategy: 划分策略 (iid, dirichlet, label_skew, quantity_skew)
        partition_config: 划分配置字典
        
    Returns:
        生成的实验名称
    """
    # 基础名称
    parts = [algorithm, dataset]
    
    # 根据划分策略添加后缀
    if partition_strategy == "iid":
        parts.append("iid")
    elif partition_strategy == "dirichlet":
        alpha = partition_config.get("alpha", 0.5)
        parts.append(f"dir_{alpha}")
    elif partition_strategy == "label_skew":
        num_labels = partition_config.get("num_labels_per_client", 2)
        parts.append(f"label_skew_{num_labels}")
    elif partition_strategy == "quantity_skew":
        imbalance = partition_config.get("imbalance_ratio", 0.5)
        parts.append(f"quantity_skew_{imbalance}")
    else:
        # 未知策略，直接使用策略名
        parts.append(partition_strategy)
    
    return "_".join(parts)


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并字典"""
    import copy
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def cmd_validate(args: argparse.Namespace) -> int:
    """执行 validate 命令"""
    from .config import ConfigManager
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}", file=sys.stderr)
        return 1
    
    try:
        manager = ConfigManager()
        configs = manager.load(str(config_path))
        
        print(f"✓ 配置验证通过: {config_path}")
        
        if args.verbose:
            print(f"\n解析结果:")
            for config in configs:
                print(f"  - {config.node_id} ({config.role})")
                print(f"    exp_name: {config.exp_name}")
                if hasattr(config, 'trainer') and config.trainer:
                    print(f"    trainer: {config.trainer.get('name', 'default')}")
                if hasattr(config, 'aggregator') and config.aggregator:
                    print(f"    aggregator: {config.aggregator.get('name', 'fedavg')}")
        
        return 0
        
    except Exception as e:
        print(f"✗ 配置验证失败: {e}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """执行 list 命令"""
    from .registry import Registry
    
    registry = Registry()
    
    component_map = {
        "aggregators": "aggregator",
        "learners": "learner",
        "models": "model",
        "datasets": "dataset",
        "trainers": "trainer",
    }
    
    if args.component_type == "all":
        types_to_show = list(component_map.keys())
    else:
        types_to_show = [args.component_type]
    
    for comp_type in types_to_show:
        prefix = component_map.get(comp_type, comp_type)
        components = registry.list(prefix)
        
        print(f"\n{comp_type.upper()}:")
        print("-" * 40)
        
        if components:
            for name, info in sorted(components.items()):
                desc = info.get("description", "")
                if desc:
                    print(f"  {name}: {desc}")
                else:
                    print(f"  {name}")
        else:
            print("  (无)")
    
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    """执行 show-config 命令"""
    import yaml
    from .config import ConfigManager
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}", file=sys.stderr)
        return 1
    
    try:
        manager = ConfigManager()
        configs = manager.load(str(config_path))
        
        for config in configs:
            print(f"\n{'='*50}")
            print(f"Node: {config.node_id}")
            print('='*50)
            # 转换为字典并打印
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
            print(yaml.dump(config_dict, default_flow_style=False, allow_unicode=True))
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """执行 version 命令"""
    print(f"OiaFed version {get_version()}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """执行 info 命令"""
    import platform
    
    print("OiaFed - One Framework for All Federation")
    print("=" * 50)
    print(f"Version:    {get_version()}")
    print(f"Python:     {platform.python_version()}")
    print(f"Platform:   {platform.platform()}")
    print()
    print("Supported Scenarios:")
    print("  - HFL: Horizontal Federated Learning")
    print("  - VFL: Vertical Federated Learning")
    print("  - FCL: Federated Continual Learning")
    print("  - PFL: Personalized Federated Learning")
    print("  - FU:  Federated Unlearning")
    print()
    print("Documentation: https://docs.oiafed.cn")
    print("Repository:    https://github.com/oiafed/oiafed")
    
    return 0


def _init_from_paper(args: argparse.Namespace, output_dir: Path) -> int:
    """从论文生成配置"""
    import yaml
    
    try:
        from .papers import get_registry
    except ImportError:
        from oiafed.papers import get_registry
    
    registry = get_registry()
    paper = registry.get(args.paper)
    
    if not paper:
        print(f"错误: 未找到论文 '{args.paper}'", file=sys.stderr)
        print(f"使用 'oiafed papers list' 查看所有可用论文")
        return 1
    
    # 生成配置文件
    num_clients = args.num_clients
    configs = registry.generate_node_configs(
        args.paper,
        num_clients=num_clients,
        output_dir=str(output_dir),
    )
    
    # 生成覆盖文件模板
    defaults = registry.get_defaults(args.paper)
    
    override_content = f"""# {paper.name}
# 用户覆盖配置 - 修改后使用 --override 参数运行
# 
# 运行方式:
#   oiafed papers run {args.paper} --override {output_dir}/override.yaml
#   或
#   oiafed run --config {output_dir}/

# 引用论文（可选，用于配置文件联动）
paper: {args.paper}

# 覆盖参数（取消注释并修改需要的参数）
# learner:
#   learning_rate: {defaults.get('learner', {}).get('learning_rate', 0.01)}
#   batch_size: {defaults.get('learner', {}).get('batch_size', 32)}

# trainer:
#   num_rounds: {defaults.get('trainer', {}).get('num_rounds', 100)}
#   local_epochs: {defaults.get('trainer', {}).get('local_epochs', 5)}

# experiment:
#   num_clients: {num_clients}
"""
    
    override_path = output_dir / "override.yaml"
    override_path.write_text(override_content)
    
    print(f"\n✓ 从论文 '{paper.name}' 生成配置")
    print(f"  目录: {output_dir}/")
    print(f"")
    print(f"  生成的文件:")
    for config in configs:
        print(f"    - {config['node_id']}.yaml")
    print(f"    - override.yaml (覆盖模板)")
    print(f"")
    print(f"  组件配置:")
    for comp_type, comp_name in paper.components.items():
        print(f"    - {comp_type}: {comp_name}")
    print(f"")
    print(f"  运行实验:")
    print(f"    oiafed run --config {output_dir}/")
    print(f"    或")
    print(f"    oiafed papers run {args.paper} --override {output_dir}/override.yaml")
    
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """执行 init 命令 - 生成实验配置模板"""
    import os
    
    output_dir = Path(args.output) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果指定了 --paper，使用论文注册表生成配置
    if args.paper:
        return _init_from_paper(args, output_dir)
    
    # 数据集配置
    dataset_configs = {
        "mnist": {"num_classes": 10, "model": "mnist_cnn"},
        "fmnist": {"num_classes": 10, "model": "mnist_cnn"},
        "cifar10": {"num_classes": 10, "model": "cifar10_cnn"},
        "cifar100": {"num_classes": 100, "model": "resnet18"},
    }
    
    ds_config = dataset_configs.get(args.dataset, dataset_configs["cifar10"])
    
    # 场景特定配置
    scenario_configs = {
        "hfl": {
            "trainer_type": "default",
            "learner_type": "default",
        },
        "pfl": {
            "trainer_type": "default",
            "learner_type": "moon",
            "learner_args": "temperature: 0.5\n        mu: 1.0",
        },
        "fcl": {
            "trainer_type": "continual",
            "learner_type": "cl.fot",
            "learner_args": "num_tasks: 5\n        classes_per_task: 2\n        orthogonal_weight: 1.0",
        },
        "vfl": {
            "trainer_type": "default",
            "learner_type": "vfl.splitnn",
            "learner_args": "split_layer: 2",
        },
        "fu": {
            "trainer_type": "default",
            "learner_type": "default",
            "aggregator_type": "faderaser",
            "aggregator_args": "history_dir: ./fed_history\n    record_history: true",
        },
    }
    
    sc_config = scenario_configs.get(args.scenario, scenario_configs["hfl"])
    
    # 生成 trainer 配置
    trainer_config = f'''# OiaFed 实验配置
# 生成命令: oiafed init {args.name} --algorithm {args.algorithm} --dataset {args.dataset} --scenario {args.scenario}

exp_name: {args.name}
node_id: trainer
role: trainer

# ==================== 训练配置 ====================
trainer:
  type: {sc_config.get("trainer_type", "default")}
  args:
    max_rounds: 100
    local_epochs: 5
    client_fraction: 1.0
    eval_every: 10

# ==================== 聚合配置 ====================
aggregator:
  type: {sc_config.get("aggregator_type", args.algorithm)}
  args:
    {sc_config.get("aggregator_args", "weighted: true")}

# ==================== 学习器配置 ====================
learner:
  type: {sc_config.get("learner_type", "default")}
  args:
    batch_size: 64
    learning_rate: 0.01
    {sc_config.get("learner_args", "")}

# ==================== 模型配置 ====================
model:
  type: {ds_config["model"]}
  args:
    num_classes: {ds_config["num_classes"]}

# ==================== 数据配置 ====================
datasets:
  - type: {args.dataset}
    split: train
    args:
      data_dir: ./data
      download: true
    partition:
      strategy: dirichlet
      num_partitions: {args.num_clients}
      config:
        alpha: 0.5
        seed: 42

  - type: {args.dataset}
    split: test
    args:
      data_dir: ./data

# ==================== 追踪配置 ====================
tracker:
  backends:
    - type: mlflow
      tracking_uri: ./mlruns
      experiment_name: {args.name}

# ==================== 运行命令 ====================
# oiafed run --config {output_dir}/trainer.yaml
'''

    # 写入 trainer 配置
    trainer_path = output_dir / "trainer.yaml"
    trainer_path.write_text(trainer_config)
    
    # 生成 learner 配置
    for i in range(args.num_clients):
        learner_config = f'''# Learner {i} 配置
# 继承自 trainer.yaml

exp_name: {args.name}
node_id: learner_{i}
role: learner

learner:
  type: {sc_config.get("learner_type", "default")}
  args:
    batch_size: 64
    learning_rate: 0.01
    {sc_config.get("learner_args", "")}

model:
  type: {ds_config["model"]}
  args:
    num_classes: {ds_config["num_classes"]}
'''
        learner_path = output_dir / f"learner_{i}.yaml"
        learner_path.write_text(learner_config)
    
    # 生成 base.yaml (用于 generate 命令)
    base_config = f'''# 基础配置 - 用于 oiafed generate 命令
exp_name: {args.name}

trainer:
  type: {sc_config.get("trainer_type", "default")}
  args:
    max_rounds: 100
    local_epochs: 5

aggregator:
  type: {sc_config.get("aggregator_type", args.algorithm)}

learner:
  type: {sc_config.get("learner_type", "default")}
  args:
    batch_size: 64
    learning_rate: 0.01

model:
  type: {ds_config["model"]}
  args:
    num_classes: {ds_config["num_classes"]}

dataset:
  type: {args.dataset}
  partition:
    strategy: dirichlet
    alpha: 0.5
'''
    base_path = output_dir / "base.yaml"
    base_path.write_text(base_config)
    
    print(f"✓ 实验配置已生成: {output_dir}/")
    print(f"  - trainer.yaml")
    for i in range(args.num_clients):
        print(f"  - learner_{i}.yaml")
    print(f"  - base.yaml")
    print()
    print(f"运行实验:")
    print(f"  oiafed run --config {output_dir}/trainer.yaml")
    print()
    print(f"或使用整个目录:")
    print(f"  oiafed run --config {output_dir}/")
    
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """执行 generate 命令 - 从基础配置生成多客户端配置"""
    import yaml
    
    base_path = Path(args.base)
    if not base_path.exists():
        print(f"错误: 基础配置文件不存在: {base_path}", file=sys.stderr)
        return 1
    
    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = base_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取基础配置（支持环境变量替换）
    base_config = _load_yaml_with_env(base_path)
    
    exp_name = base_config.get('exp_name', 'experiment')
    
    # 生成 trainer 配置
    trainer_config = {
        **base_config,
        'node_id': 'trainer',
        'role': 'trainer',
    }
    
    # 添加数据划分配置
    if 'datasets' not in trainer_config:
        dataset_type = base_config.get('dataset', {}).get('type', 'cifar10')
        partition_strategy = args.partition
        partition_config = {'seed': 42}
        
        if partition_strategy == 'dirichlet':
            partition_config['alpha'] = args.alpha
        
        trainer_config['datasets'] = [
            {
                'type': dataset_type,
                'split': 'train',
                'args': {'data_dir': './data', 'download': True},
                'partition': {
                    'strategy': partition_strategy,
                    'num_partitions': args.num_clients,
                    'config': partition_config,
                }
            },
            {
                'type': dataset_type,
                'split': 'test',
                'args': {'data_dir': './data'},
            }
        ]
    
    trainer_path = output_dir / "trainer.yaml"
    with open(trainer_path, 'w', encoding='utf-8') as f:
        yaml.dump(trainer_config, f, default_flow_style=False, allow_unicode=True)
    
    # 生成 learner 配置
    learner_files = []
    for i in range(args.num_clients):
        learner_config = {
            'exp_name': exp_name,
            'node_id': f'learner_{i}',
            'role': 'learner',
        }
        
        # 复制 learner 和 model 配置
        if 'learner' in base_config:
            learner_config['learner'] = base_config['learner']
        if 'model' in base_config:
            learner_config['model'] = base_config['model']
        
        learner_path = output_dir / f"learner_{i}.yaml"
        with open(learner_path, 'w', encoding='utf-8') as f:
            yaml.dump(learner_config, f, default_flow_style=False, allow_unicode=True)
        learner_files.append(f"learner_{i}.yaml")
    
    print(f"✓ 配置文件已生成: {output_dir}/")
    print(f"  - trainer.yaml")
    for lf in learner_files:
        print(f"  - {lf}")
    print()
    print(f"数据划分: {args.partition}" + (f" (alpha={args.alpha})" if args.partition == "dirichlet" else ""))
    print(f"客户端数: {args.num_clients}")
    print()
    print(f"运行实验:")
    print(f"  oiafed run --config {output_dir}/")
    
    return 0


def cmd_papers(args: argparse.Namespace) -> int:
    """执行 papers 命令"""
    try:
        from .papers import get_registry
    except ImportError:
        from oiafed.papers import get_registry
    
    registry = get_registry()
    
    # 子命令分发
    if args.papers_command == "list":
        return cmd_papers_list(args, registry)
    elif args.papers_command == "show":
        return cmd_papers_show(args, registry)
    elif args.papers_command == "init":
        return cmd_papers_init(args, registry)
    else:
        # 没有子命令，显示帮助
        print(registry.format_paper_list())
        print()
        print("运行论文实验请使用:")
        print("  oiafed run --paper <paper_id> -n <num_clients>")
        print()
        print("示例:")
        print("  oiafed run --paper fedavg -n 10")
        print("  oiafed run --paper fedavg -n 10 --config base.yaml")
        print("  oiafed run --paper fedavg -n 10 --rounds 50 --lr 0.01")
        return 0


def cmd_papers_list(args: argparse.Namespace, registry) -> int:
    """列出论文"""
    if args.category:
        papers = registry.list_by_category(args.category)
        if not papers:
            print(f"类别 {args.category} 没有论文")
            return 0
        
        category_name = registry.CATEGORY_NAMES.get(args.category, args.category)
        print(f"\n[{args.category}] {category_name}")
        print("-" * 40)
        for paper_id in papers:
            paper = registry.get(paper_id)
            print(f"  {paper_id:<15} {paper.name}")
        print()
    else:
        print(registry.format_paper_list())
    
    return 0


def cmd_papers_show(args: argparse.Namespace, registry) -> int:
    """显示论文详情"""
    paper = registry.get(args.paper_id)
    if not paper:
        print(f"错误: 未找到论文 '{args.paper_id}'", file=sys.stderr)
        print(f"使用 'oiafed papers list' 查看所有可用论文")
        return 1
    
    print(registry.format_paper_info(args.paper_id, show_params=args.params))
    return 0


def cmd_papers_init(args: argparse.Namespace, registry) -> int:
    """生成论文配置模板"""
    import yaml
    
    paper = registry.get(args.paper_id)
    if not paper:
        print(f"错误: 未找到论文 '{args.paper_id}'", file=sys.stderr)
        return 1
    
    output_dir = Path(args.output) / args.paper_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成配置文件
    num_clients = args.num_clients
    configs = registry.generate_node_configs(
        args.paper_id,
        num_clients=num_clients,
        output_dir=str(output_dir),
    )
    
    # 生成覆盖文件模板
    override_template = {
        "# 用户覆盖配置": "修改此文件后使用 --override 参数运行",
        "learner": {
            "learning_rate": paper.get_default("learner.learning_rate", 0.01),
            "batch_size": paper.get_default("learner.batch_size", 32),
        },
        "trainer": {
            "num_rounds": paper.get_default("trainer.num_rounds", 20),
        },
    }
    
    override_path = output_dir / "override.yaml"
    with open(override_path, "w", encoding="utf-8") as f:
        f.write(f"# {paper.name}\n")
        f.write(f"# 用户覆盖配置 - 修改后使用 --override 参数运行\n\n")
        yaml.dump(override_template, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✓ 配置已生成: {output_dir}/")
    print(f"")
    print(f"  生成的文件:")
    for config in configs:
        print(f"    - {config['node_id']}.yaml")
    print(f"    - override.yaml (覆盖模板)")
    print(f"")
    print(f"  运行实验:")
    print(f"    oiafed run --config {output_dir}/")
    print(f"")
    print(f"  或使用 papers run:")
    print(f"    oiafed papers run {args.paper_id} --override {override_path}")
    
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 主入口"""
    parser = get_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # 命令分发
    commands = {
        "run": cmd_run,
        "validate": cmd_validate,
        "list": cmd_list,
        "show-config": cmd_show_config,
        "version": cmd_version,
        "info": cmd_info,
        "init": cmd_init,
        "generate": cmd_generate,
        "papers": cmd_papers,
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())