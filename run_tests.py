# run_tests.py
"""
联邦学习框架测试运行脚本

提供便捷的测试运行入口：
- 单元测试
- 集成测试
- 端到端测试
- 完整测试套件
"""

import os
import sys
import argparse
import subprocess
import time
import yaml
from pathlib import Path


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "tests" / "config"  # 使用tests/config目录
        self.tests_dir = self.project_root / "tests"
        self.output_dir = self.project_root / "test_outputs"
        
        # 确保输出目录存在
        self.output_dir.mkdir(exist_ok=True)
        
    def setup_environment(self):
        """设置测试环境"""
        print("Setting up test environment...")
        
        # 确保必要的目录存在
        self.config_dir.mkdir(exist_ok=True)
        self.tests_dir.mkdir(exist_ok=True)
        
        # 创建配置文件（如果不存在）
        self._create_config_files()
        
        print("Test environment setup complete")
    
    def _create_config_files(self):
        """创建配置文件"""
        # 服务端配置
        server_config_path = self.config_dir / "server_config.yaml"
        if not server_config_path.exists():
            server_config = {
                'server': {'id': 'mnist_test_server'},
                'communication': {
                    'host': 'localhost',
                    'port': 8080,
                    'max_workers': 10,
                    'timeout': 60.0
                },
                'aggregator': {
                    'class': 'fedavg',
                    'type': 'FedAvgAggregator'
                },
                'federation': {
                    'min_clients': 2,
                    'client_selection_strategy': 'random',
                    'aggregation_strategy': 'fedavg'
                },
                'round_config': {
                    'timeout': 120.0,
                    'min_client_updates': 2,
                    'retry_attempts': 2
                },
                'task_sequence': [
                    {
                        'task_id': 'mnist_classification',
                        'task_type': 'classification',
                        'dataset': {
                            'name': 'MNIST',
                            'root': './data',
                            'download': True,
                            'train': True
                        },
                        'model': {
                            'type': 'MLP',
                            'input_size': 784,
                            'hidden_sizes': [512, 256, 128],
                            'num_classes': 10,
                            'dropout_rate': 0.2,
                            'activation': 'relu',
                            'use_batch_norm': True
                        },
                        'learning': {'epochs': 5},
                        'sequence_position': 0,
                        'dependencies': [],
                        'metadata': {'description': 'MNIST handwritten digit classification'}
                    }
                ]
            }
            
            with open(server_config_path, 'w') as f:
                yaml.dump(server_config, f, default_flow_style=False)
        
        # 客户端配置
        client_config_path = self.config_dir / "client_config.yaml"
        if not client_config_path.exists():
            client_config = {
                'client': {'id': 'mnist_test_client'},
                'communication': {
                    'host': 'localhost',
                    'port': 8080,
                    'max_workers': 5,
                    'timeout': 60.0
                },
                'learners': {
                    'mnist_learner': {
                        'class': 'l2p',
                        'model': {
                            'type': 'MLP',
                            'input_size': 784,
                            'hidden_sizes': [256, 128],
                            'num_classes': 10,
                            'dropout_rate': 0.2,
                            'activation': 'relu',
                            'use_batch_norm': True
                        },
                        'optimizer': {'type': 'Adam', 'lr': 0.001},
                        'dataloader': 'mnist_train_data',
                        'scheduler': 'cosine_scheduler',
                        'priority': 10,
                        'enabled': True
                    }
                },
                'dataloaders': {
                    'mnist_train_data': {
                        'batch_size': 32,
                        'shuffle': True,
                        'num_workers': 0,
                        'dataset_config': {
                            'name': 'MNIST',
                            'root': './data',
                            'download': True,
                            'train': True
                        }
                    }
                },
                # 添加测试数据集配置
                'test_datas': {
                    'test_data': {
                        'type': 'StandardDataLoader',
                        'batch_size': 64,
                        'shuffle': False,
                        'num_workers': 0,
                        'dataset_config': {
                            'name': 'MNIST',
                            'root': './data',
                            'download': True,
                            'train': False  # 使用测试集
                        }
                    },
                    'validation_data': {
                        'type': 'StandardDataLoader',
                        'batch_size': 64,
                        'shuffle': False,
                        'num_workers': 0,
                        'dataset_config': {
                            'name': 'MNIST',
                            'root': './data',
                            'download': True,
                            'train': False,
                            'subset_ratio': 0.5  # 使用测试集的一半作为验证集
                        }
                    }
                },
                # 添加评估器配置
                'evaluators': {
                    'accuracy_evaluator': {
                        'type': 'accuracy_evaluator',
                        'metrics': ['accuracy', 'top1_accuracy']
                    },
                    'loss_evaluator': {
                        'type': 'loss_evaluator',
                        'metrics': ['cross_entropy_loss', 'avg_loss']
                    },
                    'comprehensive_evaluator': {
                        'type': 'comprehensive_evaluator',
                        'metrics': ['accuracy', 'precision', 'recall', 'f1_score']
                    }
                },
                # 添加评估配置
                'evaluation': {
                    'enabled': True,
                    'frequency': 'after_phase',
                    'learners': {
                        'mnist_learner': {
                            'evaluator_mapping': {
                                'accuracy_evaluator': ['test_data', 'validation_data'],
                                'loss_evaluator': ['test_data'],
                                'comprehensive_evaluator': ['validation_data']
                            }
                        }
                    },
                    'default': {
                        'evaluator': 'accuracy_evaluator',
                        'test_dataset': 'test_data'
                    }
                },
                'training_plan': {
                    'total_epochs': 5,
                    'execution_strategy': 'sequential',
                    'phases': [
                        {
                            'name': 'mnist_training',
                            'epochs': [1, 2, 3, 4, 5],
                            'learner': 'mnist_learner',
                            'scheduler': 'cosine_scheduler'
                        }
                    ]
                }
            }
            
            with open(client_config_path, 'w') as f:
                yaml.dump(client_config, f, default_flow_style=False)
        
        # 端到端测试配置
        e2e_config_path = self.config_dir / "e2e_test_config.yaml"
        if not e2e_config_path.exists():
            e2e_config = {
                'federation_test': {
                    'name': 'MNIST E2E Federation Test',
                    'description': 'End-to-end MNIST federation test',
                    'federation': {
                        'num_rounds': 3,
                        'min_clients': 2,
                        'timeout_per_round': 120,
                        'convergence_threshold': 0.85
                    },
                    'expected_results': {
                        'min_accuracy': 0.8,
                        'max_training_time_per_round': 60,
                        'successful_rounds_ratio': 0.9
                    }
                }
            }
            
            with open(e2e_config_path, 'w') as f:
                yaml.dump(e2e_config, f, default_flow_style=False)
    
    def run_unit_tests(self, verbose=True):
        """运行单元测试"""
        print("=== Running Unit Tests ===")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_federation_framework.py"),
            "-k", "TestImprovedFederatedServer or TestMultiLearnerFederatedClient or TestLightweightFederationEngine",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Unit Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
    def run_experiment_e2e_tests(self, verbose=True):
        """运行实验管理器端到端测试"""
        print("=== Running Experiment Manager E2E Tests ===")
        
        output_file = self.output_dir / f"experiment_e2e_results_{int(time.time())}.json"
        
        cmd = [
            sys.executable,
            str(self.tests_dir / "experiment_e2e_test.py"),
            "--output-dir", str(self.output_dir / "experiment_e2e"),
        ]
        
        if verbose:
            cmd.extend(["--log-level", "INFO"])
        else:
            cmd.extend(["--log-level", "ERROR"])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Experiment Manager E2E Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
    def run_component_tests(self, verbose=True):
        """运行完整组件测试"""
        print("=== Running Comprehensive Component Tests ===")
        
        cmd = [
            sys.executable,
            str(self.tests_dir / "comprehensive_component_test.py")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Component Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    
    def run_integration_tests(self, verbose=True):
        """运行集成测试"""
        print("=== Running Integration Tests ===")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_federation_framework.py"),
            "-k", "TestEndToEndFederation",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Integration Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    
    def run_e2e_tests(self, config_path=None, verbose=True):
        """运行端到端测试"""
        print("=== Running End-to-End Tests ===")
        
        if config_path is None:
            config_path = self.config_dir / "e2e_test_config.yaml"
        
        output_file = self.output_dir / f"e2e_test_results_{int(time.time())}.json"
        
        cmd = [
            sys.executable,
            str(self.tests_dir / "e2e_federation_test.py"),
            "--config", str(config_path),
            "--output", str(output_file)
        ]
        
        if verbose:
            cmd.extend(["--log-level", "INFO"])
        else:
            cmd.extend(["--log-level", "ERROR"])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("E2E Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        if output_file.exists():
            print(f"Detailed results saved to: {output_file}")
        
        return result.returncode == 0
    
    def run_experiment_tests(self, verbose=True):
        """运行实验管理器测试"""
        print("=== Running Experiment Manager Tests ===")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_federation_framework.py"),
            "-k", "TestFedCLExperiment or TestFedCLExperimentUtilities",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Experiment Manager Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    
    def run_performance_tests(self, verbose=True):
        """运行性能测试"""
        print("=== Running Performance Tests ===")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_federation_framework.py"),
            "-k", "TestRobustnessAndReliability",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Performance Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    
    def run_all_tests(self, verbose=True):
        """运行所有测试"""
        print("=== Running Complete Test Suite ===")
        
        results = {}
        
        # 设置环境
        self.setup_environment()
        
        # 运行各类测试
        print("\n" + "="*50)
        results['component_tests'] = self.run_component_tests(verbose)
        
        print("\n" + "="*50)
        results['unit_tests'] = self.run_unit_tests(verbose)
        
        print("\n" + "="*50)
        results['integration_tests'] = self.run_integration_tests(verbose)
        
        print("\n" + "="*50)
        results['experiment_tests'] = self.run_experiment_tests(verbose)
        
        print("\n" + "="*50)
        results['experiment_e2e_tests'] = self.run_experiment_e2e_tests(verbose)
        
        print("\n" + "="*50)
        results['e2e_tests'] = self.run_e2e_tests(verbose=verbose)
        
        print("\n" + "="*50)
        results['performance_tests'] = self.run_performance_tests(verbose)
        
        # 汇总结果
        print("\n" + "="*50)
        print("=== Test Suite Summary ===")
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        for test_type, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{test_type}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} test suites passed")
        
        overall_success = all(results.values())
        print(f"Overall Result: {'SUCCESS' if overall_success else 'FAILURE'}")
        
        return overall_success
    
    def run_quick_test(self):
        """运行快速测试（仅核心功能）"""
        print("=== Running Quick Test ===")
        
        self.setup_environment()
        
        # 只运行核心单元测试
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / "test_federation_framework.py"),
            "-k", "test_server_initialization or test_client_initialization",
            "-v", "--tb=short"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("Quick Test Results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    
    def create_test_report(self):
        """创建测试报告"""
        print("=== Generating Test Report ===")
        
        report_file = self.output_dir / f"test_report_{int(time.time())}.html"
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir),
            "--html", str(report_file),
            "--self-contained-html"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if report_file.exists():
            print(f"Test report generated: {report_file}")
        else:
            print("Failed to generate test report")
            if result.stderr:
                print("Errors:")
                print(result.stderr)
        
        return report_file.exists()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Federation Framework Test Runner')
    parser.add_argument('test_type', 
                       choices=['component', 'unit', 'integration', 'experiment', 'experiment-e2e', 'e2e', 'performance', 'all', 'quick', 'report'],
                       help='Type of test to run')
    parser.add_argument('--config', type=str, help='Configuration file path for e2e tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    verbose = args.verbose and not args.quiet
    
    if args.test_type == 'component':
        success = runner.run_component_tests(verbose)
    elif args.test_type == 'unit':
        success = runner.run_unit_tests(verbose)
    elif args.test_type == 'integration':
        success = runner.run_integration_tests(verbose)
    elif args.test_type == 'experiment':
        success = runner.run_experiment_tests(verbose)
    elif args.test_type == 'experiment-e2e':
        success = runner.run_experiment_e2e_tests(verbose)
    elif args.test_type == 'e2e':
        success = runner.run_e2e_tests(args.config, verbose)
    elif args.test_type == 'performance':
        success = runner.run_performance_tests(verbose)
    elif args.test_type == 'all':
        success = runner.run_all_tests(verbose)
    elif args.test_type == 'quick':
        success = runner.run_quick_test()
    elif args.test_type == 'report':
        success = runner.create_test_report()
    else:
        print(f"Unknown test type: {args.test_type}")
        sys.exit(1)
    
    if success:
        print("\n✅ Tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()