#!/usr/bin/env python3
"""
实验运行和结果汇总脚本
Usage: python experiment_runner.py [--run | --analyze | --both]
"""

import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
import argparse
import sys
from test import *

class ExperimentRunner:
    def __init__(self, results_dir="results", summary_file="experiment_summary.json"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.summary_file = self.results_dir / summary_file
        self.script_name = "test.py"
        
    def run_experiments(self, configs):
        """运行多个实验配置"""
        print("="*80)
        print("RUNNING EXPERIMENTS")
        print("="*80)
        
        results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Running: {config['name']}")
            print("-"*80)
            
            result = self._run_single_experiment(config)
            results.append(result)
            
            print(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}")
            print("-"*80)
        
        # 保存结果
        self._save_results(results)
        
        return results
    
    def _run_single_experiment(self, config):
        """运行单个实验"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.results_dir / f"{config['name']}_{timestamp}.log"
        
        # 构建命令
        cmd = [sys.executable, self.script_name]
        for key, value in config['args'].items():
            cmd.append(f"--{key}")
            cmd.append(str(value))
        cmd.append(f"--log-file={log_file}")
        
        result = {
            'name': config['name'],
            'timestamp': timestamp,
            'config': config['args'],
            'log_file': str(log_file),
            'success': False,
            'metrics': {}
        }
        
        try:
            print(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=False)
            result['success'] = True
            
            # 解析日志提取指标
            result['metrics'] = self._parse_log_file(log_file)
            
        except subprocess.CalledProcessError as e:
            result['success'] = False
            result['error'] = str(e)
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def _parse_log_file(self, log_file):
        """解析日志文件提取关键指标"""
        metrics = {
            'task_accuracies': [],
            'final_accuracies': [],
            'forgetting': [],
            'avg_forgetting': None,
            'final_avg_accuracy': None,
            'reconstruction_losses': {}
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取每个任务后的准确率
            # 格式: "Task 0 accuracy: 98.96%"
            task_acc_pattern = r"Task (\d+) accuracy: ([\d.]+)%"
            matches = re.findall(task_acc_pattern, content)
            
            current_task_accs = []
            for task_id, acc in matches:
                current_task_accs.append({
                    'task_id': int(task_id),
                    'accuracy': float(acc)
                })
            
            # 提取最终平均准确率
            # 格式: "Final Average Accuracy: 70.49%"
            final_avg_match = re.search(r"Final Average Accuracy: ([\d.]+)%", content)
            if final_avg_match:
                metrics['final_avg_accuracy'] = float(final_avg_match.group(1))
            
            # 提取遗忘率
            # 格式: "Task 0: Initial 98.96% → Final 46.88% (Forgetting: 52.08%)"
            forgetting_pattern = r"Task (\d+).*Forgetting: ([\d.]+)%"
            forgetting_matches = re.findall(forgetting_pattern, content)
            for task_id, forget in forgetting_matches:
                metrics['forgetting'].append({
                    'task_id': int(task_id),
                    'forgetting': float(forget)
                })
            
            # 提取平均遗忘率
            avg_forget_match = re.search(r"Average Forgetting: ([\d.]+)%", content)
            if avg_forget_match:
                metrics['avg_forgetting'] = float(avg_forget_match.group(1))
            
            # 提取重建损失
            # 格式: "Best reconstruction loss for class 0: 0.111355"
            recon_pattern = r"Best reconstruction loss for class (\d+): ([\d.]+)"
            recon_matches = re.findall(recon_pattern, content)
            for class_id, loss in recon_matches:
                metrics['reconstruction_losses'][int(class_id)] = float(loss)
            
            # 提取最终每个任务的准确率
            # 在"Evaluating after Task X:"之后的最后一组
            eval_sections = re.findall(
                r"Evaluating after Task (\d+):.*?(?=Evaluating after Task|\Z)", 
                content, 
                re.DOTALL
            )
            if eval_sections:
                last_section = eval_sections[-1]
                final_task_pattern = r"Task (\d+).*?: ([\d.]+)%"
                final_matches = re.findall(final_task_pattern, last_section)
                metrics['final_accuracies'] = [
                    {'task_id': int(tid), 'accuracy': float(acc)} 
                    for tid, acc in final_matches
                ]
        
        except Exception as e:
            print(f"Warning: Failed to parse log file {log_file}: {e}")
        
        return metrics
    
    def _save_results(self, results):
        """保存结果到JSON文件"""
        # 加载现有结果
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = []
        
        # 添加新结果
        all_results.extend(results)
        
        # 保存
        with open(self.summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: {self.summary_file}")
    
    def analyze_results(self):
        """分析并汇总所有实验结果"""
        if not self.summary_file.exists():
            print("No results file found. Run experiments first.")
            return
        
        with open(self.summary_file, 'r') as f:
            results = json.load(f)
        
        if not results:
            print("No results to analyze.")
            return
        
        print("="*80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['success'])}")
        print(f"Failed: {sum(1 for r in results if not r['success'])}")
        print("="*80)
        
        # 按配置分组
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("\nNo successful experiments to analyze.")
            return
        
        # 详细结果表格
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        print(f"{'Experiment':<30} {'Final Acc':<12} {'Avg Forget':<12} {'Avg Recon Loss':<15}")
        print("-"*80)
        
        for result in successful_results:
            name = result['name'][:28]
            metrics = result['metrics']
            
            final_acc = metrics.get('final_avg_accuracy', 'N/A')
            final_acc_str = f"{final_acc:.2f}%" if isinstance(final_acc, float) else final_acc
            
            avg_forget = metrics.get('avg_forgetting', 'N/A')
            avg_forget_str = f"{avg_forget:.2f}%" if isinstance(avg_forget, float) else avg_forget
            
            recon_losses = metrics.get('reconstruction_losses', {})
            if recon_losses:
                avg_recon = sum(recon_losses.values()) / len(recon_losses)
                recon_str = f"{avg_recon:.6f}"
            else:
                recon_str = "N/A"
            
            print(f"{name:<30} {final_acc_str:<12} {avg_forget_str:<12} {recon_str:<15}")
        
        # 最佳结果
        print("\n" + "="*80)
        print("BEST RESULTS")
        print("="*80)
        
        # 最高准确率
        best_acc_result = max(
            successful_results, 
            key=lambda x: x['metrics'].get('final_avg_accuracy', 0)
        )
        print(f"\nBest Final Accuracy: {best_acc_result['name']}")
        print(f"  Accuracy: {best_acc_result['metrics']['final_avg_accuracy']:.2f}%")
        print(f"  Config: {best_acc_result['config']}")
        
        # 最低遗忘率
        results_with_forget = [
            r for r in successful_results 
            if r['metrics'].get('avg_forgetting') is not None
        ]
        if results_with_forget:
            best_forget_result = min(
                results_with_forget,
                key=lambda x: x['metrics']['avg_forgetting']
            )
            print(f"\nLowest Forgetting: {best_forget_result['name']}")
            print(f"  Avg Forgetting: {best_forget_result['metrics']['avg_forgetting']:.2f}%")
            print(f"  Config: {best_forget_result['config']}")
        
        # 最佳重建质量
        results_with_recon = [
            r for r in successful_results 
            if r['metrics'].get('reconstruction_losses')
        ]
        if results_with_recon:
            best_recon_result = min(
                results_with_recon,
                key=lambda x: sum(x['metrics']['reconstruction_losses'].values()) / 
                             len(x['metrics']['reconstruction_losses'])
            )
            recon_losses = best_recon_result['metrics']['reconstruction_losses']
            avg_recon = sum(recon_losses.values()) / len(recon_losses)
            print(f"\nBest Reconstruction Quality: {best_recon_result['name']}")
            print(f"  Avg Recon Loss: {avg_recon:.6f}")
            print(f"  Config: {best_recon_result['config']}")
        
        # 生成markdown格式的报告
        self._generate_markdown_report(successful_results)
        
    def _generate_markdown_report(self, results):
        """生成markdown格式的报告"""
        report_file = self.results_dir / "experiment_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Continual Learning Experiment Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"Total Experiments: {len(results)}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| Experiment | Final Accuracy | Avg Forgetting | Avg Recon Loss | Config |\n")
            f.write("|------------|----------------|----------------|----------------|--------|\n")
            
            for result in results:
                name = result['name']
                metrics = result['metrics']
                
                final_acc = metrics.get('final_avg_accuracy', 'N/A')
                final_acc_str = f"{final_acc:.2f}%" if isinstance(final_acc, float) else final_acc
                
                avg_forget = metrics.get('avg_forgetting', 'N/A')
                avg_forget_str = f"{avg_forget:.2f}%" if isinstance(avg_forget, float) else avg_forget
                
                recon_losses = metrics.get('reconstruction_losses', {})
                if recon_losses:
                    avg_recon = sum(recon_losses.values()) / len(recon_losses)
                    recon_str = f"{avg_recon:.6f}"
                else:
                    recon_str = "N/A"
                
                config_str = json.dumps(result['config'])[:50] + "..."
                
                f.write(f"| {name} | {final_acc_str} | {avg_forget_str} | {recon_str} | {config_str} |\n")
            
            f.write("\n## Per-Task Forgetting Analysis\n\n")
            for result in results:
                if result['metrics'].get('forgetting'):
                    f.write(f"\n### {result['name']}\n\n")
                    for forget_info in result['metrics']['forgetting']:
                        f.write(f"- Task {forget_info['task_id']}: {forget_info['forgetting']:.2f}%\n")
        
        print(f"\nMarkdown report saved to: {report_file}")


# ============================================================
# 实验配置定义
# ============================================================

def get_baseline_config():
    """基线配置（你当前的配置）"""
    return {
        'name': 'baseline',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 3,
            'batch-size': 128,
            'stage1-epochs': 5,
            'stage2-epochs': 50,
            'stage3-epochs': 20,
            'replay-samples': 300,
            'hidden-dim': 512,
            'seed': 42
        }
    }

def get_improved_configs():
    """改进配置"""
    configs = []
    
    # 配置1: 提高Stage 2训练轮数
    configs.append({
        'name': 'improved_stage2_40',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 3,
            'batch-size': 128,
            'stage1-epochs': 10,
            'stage2-epochs': 40,
            'stage3-epochs': 20,
            'replay-samples': 100,
            'hidden-dim': 512,
            'seed': 42
        }
    })
    
    # 配置2: 增加模型容量
    configs.append({
        'name': 'improved_hidden_1024',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 3,
            'batch-size': 128,
            'stage1-epochs': 10,
            'stage2-epochs': 20,
            'stage3-epochs': 20,
            'replay-samples': 100,
            'hidden-dim': 1024,
            'seed': 42
        }
    })
    
    # 配置3: 增加回放样本
    configs.append({
        'name': 'improved_replay_200',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 3,
            'batch-size': 128,
            'stage1-epochs': 10,
            'stage2-epochs': 20,
            'stage3-epochs': 20,
            'replay-samples': 200,
            'hidden-dim': 512,
            'seed': 42
        }
    })
    
    # 配置4: 组合改进
    configs.append({
        'name': 'improved_combined',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 3,
            'batch-size': 128,
            'stage1-epochs': 10,
            'stage2-epochs': 40,
            'stage3-epochs': 30,
            'replay-samples': 200,
            'hidden-dim': 1024,
            'stage2-lr': 0.0005,
            'seed': 42
        }
    })
    
    # 配置5: 激进改进
    configs.append({
        'name': 'improved_aggressive',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 3,
            'batch-size': 128,
            'stage1-epochs': 15,
            'stage2-epochs': 60,
            'stage3-epochs': 40,
            'replay-samples': 300,
            'hidden-dim': 1536,
            'stage1-lr': 0.0005,
            'stage2-lr': 0.0003,
            'stage3-lr': 0.0005,
            'seed': 42
        }
    })
    
    return configs

def get_ablation_configs():
    """消融实验配置"""
    configs = []
    
    # 只改Stage 2
    configs.append({
        'name': 'ablation_stage2_only',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 3,
            'stage1-epochs': 10,
            'stage2-epochs': 40,
            'stage3-epochs': 20,
            'replay-samples': 100,
            'hidden-dim': 512,
            'seed': 42
        }
    })
    
    # 只改hidden dim
    configs.append({
        'name': 'ablation_hidden_only',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 3,
            'stage1-epochs': 10,
            'stage2-epochs': 20,
            'stage3-epochs': 20,
            'replay-samples': 100,
            'hidden-dim': 1024,
            'seed': 42
        }
    })
    
    # 只改replay
    configs.append({
        'name': 'ablation_replay_only',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 3,
            'stage1-epochs': 10,
            'stage2-epochs': 20,
            'stage3-epochs': 20,
            'replay-samples': 200,
            'hidden-dim': 512,
            'seed': 42
        }
    })
    
    return configs

def get_quick_test_configs():
    """快速测试配置（2个任务，用于快速验证）"""
    configs = []
    
    configs.append({
        'name': 'quick_baseline',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 2,
            'stage1-epochs': 10,
            'stage2-epochs': 20,
            'stage3-epochs': 20,
            'replay-samples': 100,
            'hidden-dim': 512,
            'seed': 42
        }
    })
    
    configs.append({
        'name': 'quick_improved',
        'args': {
            'dataset': 'mnist',
            'num-tasks': 2,
            'stage1-epochs': 10,
            'stage2-epochs': 40,
            'stage3-epochs': 30,
            'replay-samples': 200,
            'hidden-dim': 1024,
            'seed': 42
        }
    })
    
    return configs


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run continual learning experiments and analyze results"
    )
    
    parser.add_argument(
        '--mode',
        choices=['run', 'analyze', 'both'],
        default='both',
        help='Mode: run experiments, analyze results, or both'
    )
    
    parser.add_argument(
        '--config-set',
        choices=['quick', 'baseline', 'improved', 'ablation', 'all'],
        default='improved',
        help='Which set of configurations to run'
    )
    
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    # 选择配置
    configs = []
    if args.config_set == 'quick':
        configs = get_quick_test_configs()
    elif args.config_set == 'baseline':
        configs = [get_baseline_config()]
    elif args.config_set == 'improved':
        configs = [get_baseline_config()] + get_improved_configs()
    elif args.config_set == 'ablation':
        configs = [get_baseline_config()] + get_ablation_configs()
    elif args.config_set == 'all':
        configs = (
            [get_baseline_config()] + 
            get_improved_configs() + 
            get_ablation_configs()
        )
    
    # 运行实验
    if args.mode in ['run', 'both']:
        print(f"\nRunning {len(configs)} experiments...")
        print("This may take a while. Press Ctrl+C to cancel.\n")
        
        try:
            runner.run_experiments(configs)
        except KeyboardInterrupt:
            print("\n\nExperiments interrupted by user.")
            print("Partial results may have been saved.")
    
    # 分析结果
    if args.mode in ['analyze', 'both']:
        print("\n")
        runner.analyze_results()


if __name__ == "__main__":
    main()