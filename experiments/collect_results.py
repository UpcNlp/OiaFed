"""
实验结果收集工具
experiments/collect_results.py

功能：
- 收集一个实验的所有节点结果
- 汇总Server和Client的指标
- 生成实验摘要报告

使用方式：
    python experiments/collect_results.py experiment_name
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List


def collect_experiment_results(exp_name: str, results_dir: str = "experiments/results") -> Dict[str, Any]:
    """
    收集一个实验的所有节点结果

    Args:
        exp_name: 实验名称
        results_dir: 结果根目录

    Returns:
        包含所有节点结果的字典
    """
    exp_dir = Path(results_dir) / exp_name

    if not exp_dir.exists():
        print(f"Error: Experiment '{exp_name}' not found in {results_dir}")
        return {}

    results = {
        'experiment_name': exp_name,
        'server': {},
        'clients': {}
    }

    # 遍历所有节点目录
    for node_dir in exp_dir.iterdir():
        if not node_dir.is_dir():
            continue

        # 解析节点角色和ID
        # 例如: "server_server_1" or "client_client_0"
        node_name = node_dir.name
        parts = node_name.split('_', 1)

        if len(parts) < 2:
            continue

        role = parts[0]  # "server" or "client"
        node_id = parts[1]  # "server_1" or "client_0"

        # Sacred 的运行目录通常是 "1", "2" 等
        run_dirs = sorted([d for d in node_dir.iterdir() if d.is_dir() and d.name.isdigit()])

        if not run_dirs:
            continue

        # 读取最新的运行结果（通常是编号最大的）
        run_dir = run_dirs[-1]

        node_result = load_run_result(run_dir)

        if role == 'server':
            results['server'][node_id] = node_result
        else:
            results['clients'][node_id] = node_result

    return results


def load_run_result(run_dir: Path) -> Dict[str, Any]:
    """
    加载单次运行的结果

    Args:
        run_dir: 运行目录（如 experiments/results/exp_name/server_1/1/）

    Returns:
        运行结果字典
    """
    result = {}

    # 读取 run.json（运行元信息）
    run_json = run_dir / "run.json"
    if run_json.exists():
        with open(run_json) as f:
            result['run_info'] = json.load(f)

    # 读取 config.json（配置）
    config_json = run_dir / "config.json"
    if config_json.exists():
        with open(config_json) as f:
            result['config'] = json.load(f)

    # 读取 info.json（用户记录的信息）
    info_json = run_dir / "info.json"
    if info_json.exists():
        with open(info_json) as f:
            result['info'] = json.load(f)

    # 读取 metrics.json（时序指标）
    metrics_json = run_dir / "metrics.json"
    if metrics_json.exists():
        with open(metrics_json) as f:
            result['metrics'] = json.load(f)

    # 读取 cout.txt（标准输出）
    cout_txt = run_dir / "cout.txt"
    if cout_txt.exists():
        with open(cout_txt) as f:
            result['output'] = f.read()

    return result


def print_experiment_summary(results: Dict[str, Any]):
    """打印实验摘要"""
    print("\n" + "=" * 80)
    print(f"Experiment Summary: {results.get('experiment_name', 'Unknown')}")
    print("=" * 80)

    # 服务端结果
    print("\n[Server Results]")
    if not results['server']:
        print("  No server results found")
    else:
        for node_id, data in results['server'].items():
            print(f"\n  Node: {node_id}")
            print_node_summary(data)

    # 客户端结果
    print("\n[Client Results]")
    if not results['clients']:
        print("  No client results found")
    else:
        for node_id, data in results['clients'].items():
            print(f"\n  Node: {node_id}")
            print_node_summary(data)

    print("\n" + "=" * 80)


def print_node_summary(data: Dict[str, Any]):
    """打印单个节点的摘要"""
    # 运行状态
    run_info = data.get('run_info', {})
    status = run_info.get('status', 'UNKNOWN')
    print(f"    Status: {status}")

    # 用户记录的信息
    info = data.get('info', {})
    if info:
        print(f"    Recorded Info:")
        for key, value in info.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.4f}")
            else:
                print(f"      {key}: {value}")

    # 时序指标摘要
    metrics = data.get('metrics', {})
    if metrics:
        print(f"    Metrics: {len(metrics)} series recorded")
        # 显示部分指标
        for key in list(metrics.keys())[:3]:
            metric_data = metrics[key]
            if 'values' in metric_data and metric_data['values']:
                last_value = metric_data['values'][-1]
                print(f"      {key}: {last_value:.4f} (last)")


def export_results_to_csv(results: Dict[str, Any], output_path: str):
    """导出结果到CSV文件"""
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(['Role', 'Node ID', 'Status', 'Final Accuracy', 'Final Loss',
                        'Completed Rounds', 'Total Time'])

        # 写入服务端数据
        for node_id, data in results['server'].items():
            info = data.get('info', {})
            run_info = data.get('run_info', {})

            writer.writerow([
                'server',
                node_id,
                run_info.get('status', ''),
                info.get('final_accuracy', ''),
                info.get('final_loss', ''),
                info.get('completed_rounds', ''),
                info.get('total_time', '')
            ])

        # 写入客户端数据
        for node_id, data in results['clients'].items():
            info = data.get('info', {})
            run_info = data.get('run_info', {})

            writer.writerow([
                'client',
                node_id,
                run_info.get('status', ''),
                '',  # 客户端通常没有全局accuracy
                '',
                '',
                ''
            ])

    print(f"\n✓ Results exported to: {output_path}")


def main():
    """主函数：解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect and summarize experiment results"
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Results root directory"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export results to CSV file"
    )

    args = parser.parse_args()

    # 收集结果
    results = collect_experiment_results(args.experiment, args.results_dir)

    if not results or (not results['server'] and not results['clients']):
        print(f"No results found for experiment: {args.experiment}")
        sys.exit(1)

    # 打印摘要
    print_experiment_summary(results)

    # 导出到CSV（如果指定）
    if args.export:
        export_results_to_csv(results, args.export)


if __name__ == "__main__":
    main()
