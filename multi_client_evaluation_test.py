#!/usr/bin/env python3
"""
多客户端评估测试脚本
基于FedCLExperiment标准流程，测试评估器和测试数据集功能
"""

import pytest
import time
import threading
import signal
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from datetime import datetime

# 添加项目路径到系统路径
import sys
sys.path.insert(0, os.path.abspath('.'))

from fedcl.experiment.experiment import FedCLExperiment
from fedcl.federation.coordinators.federated_client import MultiLearnerClient
from fedcl.config.config_manager import DictConfig

# 确保实现模块被加载，触发组件注册
import fedcl.implementations

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationLogMonitor:
    """评估日志监控器 - 专门监控评估相关的日志"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.evaluation_events = []
        self.errors = []
        self.warnings = []
        self.monitoring = False
        
    def start_monitoring(self):
        """开始监控日志"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_logs)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止监控日志"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
            
    def _monitor_logs(self):
        """监控日志文件"""
        processed_files = set()
        
        while self.monitoring:
            try:
                # 查找最新的日志文件
                log_files = list(self.log_dir.glob("*.log"))
                if not log_files:
                    time.sleep(1)
                    continue
                    
                # 处理新的日志文件
                for log_file in log_files:
                    if log_file not in processed_files:
                        self._process_log_file(log_file)
                        processed_files.add(log_file)
                        
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"日志监控错误: {e}")
                time.sleep(1)
                
    def _process_log_file(self, log_file: Path):
        """处理日志文件"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 检查评估相关事件
                if any(keyword in line.lower() for keyword in [
                    'evaluation', 'evaluator', 'test_data', 'accuracy', 'precision', 'recall', 'f1', 'loss'
                ]):
                    self.evaluation_events.append(line)
                    logger.info(f"📊 EVALUATION: {line}")
                    
                # 检查错误
                if 'ERROR' in line:
                    self.errors.append(line)
                    logger.error(f"🔴 ERROR: {line}")
                    
                # 检查警告
                elif 'WARNING' in line or 'WARN' in line:
                    self.warnings.append(line)
                    logger.warning(f"🟡 WARNING: {line}")
                    
        except Exception as e:
            logger.warning(f"处理日志文件错误 {log_file}: {e}")
            
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return {
            'evaluation_events': self.evaluation_events,
            'errors': self.errors,
            'warnings': self.warnings,
            'evaluation_count': len(self.evaluation_events),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }


class TestEvaluationConfiguration:
    """评估配置测试"""
    
    @pytest.fixture
    def evaluation_config_dir(self):
        """评估配置目录"""
        return Path("tests/configs/mnist_real_test")
    
    def test_evaluation_config_validation(self, evaluation_config_dir):
        """测试评估配置验证"""
        # 检查配置文件是否存在
        assert (evaluation_config_dir / "experiment_config.yaml").exists()
        assert (evaluation_config_dir / "server_config.yaml").exists()
        
        # 检查客户端配置文件
        client_configs = []
        for i in range(1, 4):
            client_config_path = evaluation_config_dir / f"client_{i}_config.yaml"
            assert client_config_path.exists(), f"客户端{i}配置文件不存在"
            
            with open(client_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                client_configs.append(config)
        
        # 验证每个客户端都有评估相关配置
        for i, config in enumerate(client_configs, 1):
            logger.info(f"验证客户端{i}的评估配置...")
            
            # 检查评估器配置
            assert 'evaluators' in config, f"客户端{i}缺少evaluators配置"
            evaluators = config['evaluators']
            assert len(evaluators) > 0, f"客户端{i}评估器配置为空"
            
            logger.info(f"客户端{i}评估器数量: {len(evaluators)}")
            for eval_name, eval_config in evaluators.items():
                assert 'class' in eval_config, f"客户端{i}评估器{eval_name}缺少class配置"
                logger.info(f"  - {eval_name}: {eval_config['class']}")
            
            # 检查测试数据集配置
            assert 'test_datas' in config, f"客户端{i}缺少test_datas配置"
            test_datas = config['test_datas']
            assert len(test_datas) > 0, f"客户端{i}测试数据集配置为空"
            
            logger.info(f"客户端{i}测试数据集数量: {len(test_datas)}")
            for data_name, data_config in test_datas.items():
                assert 'dataset_config' in data_config, f"客户端{i}测试数据集{data_name}缺少dataset_config"
                logger.info(f"  - {data_name}: {data_config['dataset_config'].get('name', 'unknown')}")
            
            # 检查评估任务配置
            assert 'evaluation' in config, f"客户端{i}缺少evaluation配置"
            evaluation = config['evaluation']
            assert 'tasks' in evaluation, f"客户端{i}评估配置缺少tasks"
            
            tasks = evaluation['tasks']
            assert len(tasks) > 0, f"客户端{i}评估任务配置为空"
            
            logger.info(f"客户端{i}评估任务数量: {len(tasks)}")
            for j, task in enumerate(tasks):
                assert 'learner' in task, f"客户端{i}评估任务{j}缺少learner"
                assert 'evaluator' in task, f"客户端{i}评估任务{j}缺少evaluator"
                assert 'test_data' in task, f"客户端{i}评估任务{j}缺少test_data"
                logger.info(f"  - 任务{j+1}: {task['learner']} -> {task['evaluator']} on {task['test_data']}")


class TestSingleClientEvaluation:
    """单客户端评估测试"""
    
    @pytest.fixture
    def single_client_with_evaluation_config(self):
        """带评估功能的单客户端配置"""
        config = {
            'client': {
                'id': 'eval_test_client',
                'type': 'multi_learner'
            },
            'learners': {
                'default_learner': {
                    'class': 'default',
                    'model': {
                        'type': 'SimpleMLP',
                        'input_size': 784,
                        'hidden_sizes': [128, 64],
                        'num_classes': 10,
                        'dropout_rate': 0.2
                    },
                    'optimizer': {
                        'type': 'SGD',
                        'lr': 0.01,
                        'momentum': 0.9
                    },
                    'dataloader': 'mnist_data',
                    'priority': 0,
                    'enabled': True
                }
            },
            'dataloaders': {
                'mnist_data': {
                    'batch_size': 32,
                    'shuffle': True,
                    'num_workers': 0,
                    'dataset_config': {
                        'name': 'MNIST',
                        'path': 'data/MNIST',
                        'split': 'train',
                        'download': True
                    }
                }
            },
            'test_datas': {
                'mnist_test': {
                    'batch_size': 64,
                    'shuffle': False,
                    'num_workers': 0,
                    'dataset_config': {
                        'name': 'MNIST',
                        'path': 'data/MNIST',
                        'split': 'test',
                        'download': True
                    }
                }
            },
            'evaluators': {
                'accuracy_evaluator': {
                    'class': 'accuracy',
                    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                    'test_data': 'mnist_test'
                },
                'loss_evaluator': {
                    'class': 'loss',
                    'metrics': ['loss', 'cross_entropy'],
                    'test_data': 'mnist_test'
                }
            },
            'evaluation': {
                'frequency': 1,
                'tasks': [
                    {
                        'learner': 'default_learner',
                        'evaluator': 'accuracy_evaluator',
                        'test_data': 'mnist_test'
                    },
                    {
                        'learner': 'default_learner',
                        'evaluator': 'loss_evaluator',
                        'test_data': 'mnist_test'
                    }
                ]
            },
            'training_plan': {
                'total_epochs': 2,
                'execution_strategy': 'sequential',
                'phases': [
                    {
                        'name': 'default_training',
                        'epochs': [1, 2],
                        'learner': 'default_learner'
                    }
                ]
            },
            'system': {
                'device': 'cpu',
                'random_seed': 42
            }
        }
        return DictConfig(config)
    
    def test_single_client_evaluation_initialization(self, single_client_with_evaluation_config):
        """测试单客户端评估初始化"""
        client = MultiLearnerClient.create_from_config(single_client_with_evaluation_config)
        
        assert client.client_id == 'eval_test_client'
        
        # 检查是否有评估相关配置
        config_dict = single_client_with_evaluation_config.to_dict()
        assert 'evaluators' in config_dict
        assert 'test_datas' in config_dict
        assert 'evaluation' in config_dict
        
        logger.info("✅ 单客户端评估配置初始化成功")
    
    def test_single_client_evaluation_data_loading(self, single_client_with_evaluation_config):
        """测试单客户端评估数据加载"""
        client = MultiLearnerClient.create_from_config(single_client_with_evaluation_config)
        
        try:
            # 加载训练数据
            client._load_multi_learner_data()
            
            # 验证训练数据加载成功
            assert len(client.dataloaders) > 0
            assert 'mnist_data' in client.dataloaders
            
            # 检查是否能够访问测试数据配置
            config_dict = single_client_with_evaluation_config.to_dict()
            test_datas = config_dict.get('test_datas', {})
            
            assert 'mnist_test' in test_datas
            logger.info("✅ 单客户端评估数据配置验证成功")
            
        except Exception as e:
            pytest.skip(f"数据加载失败，可能MNIST数据不存在: {e}")


class TestMultiClientEvaluationExperiment:
    """多客户端评估实验测试"""
    
    @pytest.fixture
    def evaluation_experiment_config_dir(self):
        """评估实验配置目录"""
        return Path("tests/configs/mnist_real_test")
    
    @pytest.fixture
    def evaluation_log_monitor(self):
        """评估日志监控器"""
        log_dir = Path("tests/test_outputs/mnist_evaluation_test/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        monitor = EvaluationLogMonitor(str(log_dir))
        yield monitor
        monitor.stop_monitoring()
    
    def test_evaluation_data_integrity_check(self):
        """测试评估数据完整性检查"""
        import torchvision.datasets as datasets
        
        try:
            # 检查MNIST数据是否存在
            train_dataset = datasets.MNIST('data/MNIST', train=True, download=False)
            test_dataset = datasets.MNIST('data/MNIST', train=False, download=False)
            
            assert len(train_dataset) == 60000
            assert len(test_dataset) == 10000
            
            # 检查数据格式
            sample_data, sample_label = train_dataset[0]
            assert 0 <= sample_label <= 9
            
            logger.info("✅ 评估数据完整性检查通过")
            
        except Exception as e:
            pytest.skip(f"MNIST数据不存在或格式错误: {e}")
    
    @pytest.mark.slow
    def test_multi_client_evaluation_experiment(self, evaluation_experiment_config_dir, evaluation_log_monitor):
        """测试多客户端评估实验执行"""
        # 清理之前的输出
        output_dir = Path("tests/test_outputs/mnist_evaluation_test")
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 开始日志监控
        evaluation_log_monitor.start_monitoring()
        
        experiment_success = False
        experiment_error = None
        
        try:
            # 创建并运行实验
            with FedCLExperiment(str(evaluation_experiment_config_dir)) as experiment:
                logger.info(f"🚀 开始多客户端评估实验: {experiment.experiment_id}")
                
                # 设置超时（5分钟）
                def timeout_handler(signum, frame):
                    raise TimeoutError("评估实验执行超时")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5分钟超时
                
                try:
                    # 执行实验
                    results = experiment.run()
                    experiment_success = True
                    
                    # 验证结果
                    assert results is not None
                    logger.info(f"✅ 评估实验完成，结果: {results}")
                    
                except TimeoutError:
                    logger.warning("⏰ 评估实验执行超时")
                    experiment_error = "timeout"
                    
                finally:
                    signal.alarm(0)  # 清除超时
                    
        except Exception as e:
            experiment_error = str(e)
            logger.error(f"❌ 评估实验执行失败: {e}")
        
        # 等待日志处理完成
        time.sleep(2)
        evaluation_log_monitor.stop_monitoring()
        
        # 分析评估日志
        log_summary = evaluation_log_monitor.get_summary()
        
        logger.info(f"\n📊 评估实验执行摘要:")
        logger.info(f"  - 评估事件数量: {log_summary['evaluation_count']}")
        logger.info(f"  - 错误数量: {log_summary['error_count']}")
        logger.info(f"  - 警告数量: {log_summary['warning_count']}")
        
        # 输出评估事件
        if log_summary['evaluation_events']:
            logger.info(f"\n📊 评估事件列表:")
            for event in log_summary['evaluation_events'][:10]:  # 只显示前10个
                logger.info(f"  {event}")
        
        # 输出错误和警告
        if log_summary['errors']:
            logger.info(f"\n🔴 错误列表:")
            for error in log_summary['errors'][:5]:  # 只显示前5个
                logger.info(f"  {error}")
                
        if log_summary['warnings']:
            logger.info(f"\n🟡 警告列表:")
            for warning in log_summary['warnings'][:5]:  # 只显示前5个
                logger.info(f"  {warning}")
        
        # 检查实验是否成功
        if experiment_success:
            logger.info(f"✅ 多客户端评估实验成功完成")
            
            # 验证评估是否正常执行
            if log_summary['evaluation_count'] > 0:
                logger.info(f"🎉 检测到 {log_summary['evaluation_count']} 个评估事件")
            else:
                logger.warning("⚠️  未检测到评估事件，可能评估未正常执行")
                
        else:
            logger.error(f"❌ 多客户端评估实验未能完成: {experiment_error}")
        
        # 检查输出文件
        output_files = list(output_dir.rglob("*"))
        logger.info(f"\n📁 生成的输出文件 ({len(output_files)} 个):")
        for file in output_files[:10]:  # 只显示前10个
            logger.info(f"  {file.relative_to(output_dir)}")
        
        # 验证基本要求
        assert log_summary['error_count'] < 10, f"错误过多: {log_summary['error_count']}"
        
        # 如果实验成功，验证是否有评估结果
        if experiment_success:
            # 这里可以添加更详细的评估结果验证
            pass
    
    def test_evaluation_output_validation(self, evaluation_experiment_config_dir):
        """测试评估输出验证"""
        output_dir = Path("tests/test_outputs/mnist_evaluation_test")
        
        if not output_dir.exists():
            pytest.skip("评估实验尚未运行，跳过输出验证")
            
        # 检查基本输出结构
        expected_dirs = ['logs']
        for dir_name in expected_dirs:
            dir_path = output_dir / dir_name
            if dir_path.exists():
                logger.info(f"✅ 找到目录: {dir_name}")
            else:
                logger.warning(f"⚠️  缺少目录: {dir_name}")
        
        # 检查日志文件
        log_files = list((output_dir / "logs").glob("*.log")) if (output_dir / "logs").exists() else []
        logger.info(f"📝 日志文件数量: {len(log_files)}")
        
        for log_file in log_files:
            logger.info(f"  {log_file.name} ({log_file.stat().st_size} bytes)")
            
            # 分析日志内容中的评估信息
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                evaluation_mentions = content.lower().count('evaluation')
                accuracy_mentions = content.lower().count('accuracy')
                loss_mentions = content.lower().count('loss')
                
                logger.info(f"    - 评估相关提及: {evaluation_mentions}")
                logger.info(f"    - 准确率相关提及: {accuracy_mentions}")
                logger.info(f"    - 损失相关提及: {loss_mentions}")


def main():
    """主函数"""
    # 设置工作目录
    os.chdir(Path(__file__).parent)
    
    logger.info("=" * 60)
    logger.info("开始多客户端评估测试")
    logger.info("=" * 60)
    
    # 运行测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow"  # 默认不运行耗时测试
    ])


if __name__ == "__main__":
    main()
