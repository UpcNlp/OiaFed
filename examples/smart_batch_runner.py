"""
智能批量实验调度器 - GPU显存感知 + SQLite断点续跑
examples/smart_batch_runner.py

个人需求工具，不属于开源项目核心代码

功能：
1. 动态GPU显存调度：运行一个实验后，查看显存使用情况，如果剩余显存够用就启动下一个
2. SQLite断点续跑：每个实验重复3次，自动跳过已完成的实验
3. 独立日志文件：每个实验的日志输出到独立文件（配置+日期命名）
"""

import asyncio
import logging
import sys
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

# 导入个人工具模块
from examples.experiment_tracker import ExperimentTracker
from examples.gpu_monitor import GPUMonitor
# from examples.logging_config_loader import LoggingConfigLoader  # Optional module


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str  # 实验名称
    dataset: str  # 数据集名称
    algorithm: str  # 算法名称
    noniid_type: str  # Non-IID类型
    config: Dict[str, Any]  # 完整配置


class SmartBatchRunner:
    """
    智能批量实验调度器

    特性：
    - GPU显存感知调度
    - SQLite断点续跑（支持3次重复执行）
    - 独立日志文件（每个实验+运行次数）
    """

    def __init__(
        self,
        config_base_dir: str,
        experiments: List[ExperimentConfig],
        max_repetitions: int = 3,
        db_path: str = "experiments/experiment_tracker.db",
        log_dir: str = "logs/smart_batch",
        safety_margin_mb: int = 1000,
        enable_gpu_scheduling: bool = True,
        max_concurrent_experiments: int = 5,
        dataset_concurrent_limits: Dict[str, int] = None
    ):
        """
        Args:
            config_base_dir: 配置文件基础目录
            experiments: 实验配置列表
            max_repetitions: 每个实验的最大重复次数
            db_path: SQLite数据库路径
            log_dir: 日志文件目录
            safety_margin_mb: GPU显存安全边际（MB）
            enable_gpu_scheduling: 是否启用GPU显存调度
            max_concurrent_experiments: 最大并发实验数（默认5，作为fallback）
            dataset_concurrent_limits: 每个数据集的并发限制字典 {'MNIST': 15, 'CINIC10': 3, ...}
        """
        self.config_base_dir = config_base_dir
        self.experiments = experiments
        self.max_repetitions = max_repetitions
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.safety_margin_mb = safety_margin_mb
        self.enable_gpu_scheduling = enable_gpu_scheduling
        self.max_concurrent_experiments = max_concurrent_experiments

        # 数据集特定的并发限制
        self.dataset_concurrent_limits = dataset_concurrent_limits or {}

        # 初始化追踪器和监控器
        self.tracker = ExperimentTracker(db_path)
        self.gpu_monitor = GPUMonitor()

        # 统计信息
        self.total_experiments = 0
        self.completed_experiments = 0
        self.failed_experiments = 0
        self.skipped_experiments = 0

    def get_concurrent_limit_for_dataset(self, dataset: str) -> int:
        """获取特定数据集的并发限制"""
        return self.dataset_concurrent_limits.get(dataset, self.max_concurrent_experiments)

    def _create_log_file_path(self, exp_config: ExperimentConfig, run_number: int) -> str:
        """
        创建日志文件路径（配置+日期命名）

        格式: logs/smart_batch/MNIST_IID_FedAvg_run1_20231114_093000.log
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{exp_config.dataset}_{exp_config.noniid_type}_{exp_config.algorithm}_run{run_number}_{timestamp}.log"
        return str(self.log_dir / filename)

    def _setup_experiment_logger(self, log_file: str) -> logging.Logger:
        """
        为实验设置独立的日志记录器

        Returns:
            Logger: 配置好的logger实例
        """
        # 创建唯一的logger名称
        logger_name = f"experiment_{Path(log_file).stem}"
        logger = logging.getLogger(logger_name)

        # 如果logger已经有handler,先清除
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(logging.INFO)

        # 文件handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.propagate = False  # 不传播到父logger

        return logger

    async def run_single_experiment(
        self,
        exp_config: ExperimentConfig,
        run_number: int,
        config_hash: str
    ) -> Dict[str, Any]:
        """
        运行单个实验的一次重复

        Args:
            exp_config: 实验配置
            run_number: 运行编号（1-based）
            config_hash: 配置哈希值

        Returns:
            Dict: 实验结果
        """
        # 创建日志文件
        log_file = self._create_log_file_path(exp_config, run_number)
        exp_logger = self._setup_experiment_logger(log_file)

        exp_logger.info(f"="*80)
        exp_logger.info(f"实验: {exp_config.name}")
        exp_logger.info(f"运行次数: {run_number}/{self.max_repetitions}")
        exp_logger.info(f"数据集: {exp_config.dataset}")
        exp_logger.info(f"算法: {exp_config.algorithm}")
        exp_logger.info(f"Non-IID: {exp_config.noniid_type}")
        exp_logger.info(f"="*80)

        # 标记实验开始
        self.tracker.start_run(config_hash, run_number, log_file)

        # 记录开始前的GPU显存（用于估算实验显存需求）
        gpu_before = None
        if self.enable_gpu_scheduling:
            gpu_info_before = self.gpu_monitor.get_gpu_with_most_free_memory()
            if gpu_info_before:
                gpu_before = gpu_info_before.used_memory_mb
                exp_logger.info(f"开始前GPU显存使用: {gpu_before}MB")

        start_time = datetime.now()
        result = {'status': 'failed', 'error': 'Unknown error'}

        try:
            # 运行实验
            exp_logger.info("开始运行实验...")

            # 调用BatchExperimentRunner运行实验
            from fedcl.experiment import BatchExperimentRunner

            runner = BatchExperimentRunner(
                self.config_base_dir,
                [exp_config.config]
            )

            results = await runner.run_all(parallel=False)
            result = results[0] if results else result

            exp_logger.info(f"实验完成: accuracy={result.get('accuracy', 'N/A')}, "
                           f"loss={result.get('loss', 'N/A')}")

        except Exception as e:
            error_msg = str(e)
            exp_logger.error(f"实验失败: {error_msg}")
            result = {
                'status': 'failed',
                'error': error_msg
            }

        finally:
            # 计算实验耗时
            duration = (datetime.now() - start_time).total_seconds()
            result['duration'] = duration

            # 记录实验结束后的GPU显存
            if self.enable_gpu_scheduling and gpu_before is not None:
                gpu_info_after = self.gpu_monitor.get_gpu_with_most_free_memory()
                if gpu_info_after:
                    gpu_after = gpu_info_after.used_memory_mb
                    memory_used = max(0, gpu_after - gpu_before)
                    exp_logger.info(f"实验显存使用估算: {memory_used}MB")

                    # 记录到GPU监控器的历史记录
                    self.gpu_monitor.record_experiment_memory(memory_used)

            # 更新实验状态到数据库
            self.tracker.complete_run(
                config_hash,
                run_number,
                result,
                status=result['status']
            )

            # 关闭logger
            for handler in exp_logger.handlers:
                handler.close()
            logging.getLogger(f"experiment_{Path(log_file).stem}").handlers.clear()

        return result

    async def run_all_experiments(
        self,
        mode: str = "sequential"
    ) -> List[Dict[str, Any]]:
        """
        运行所有实验

        Args:
            mode: 执行模式，可选：
                - "sequential": 顺序执行（默认，最稳定）
                - "concurrent": 并发执行（asyncio，适合IO密集型）
                - "multiprocess": 多进程并发（真实进程隔离，推荐用于长时间批量实验）

        Returns:
            List[Dict]: 所有实验的结果
        """
        # 根据模式选择执行方式
        if mode == "sequential":
            return await self._run_all_experiments_sequential()
        elif mode == "concurrent":
            return await self.run_all_experiments_concurrent()
        elif mode == "multiprocess":
            # 多进程模式不需要async，直接调用
            return self.run_all_experiments_multiprocess()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sequential', 'concurrent', or 'multiprocess'")

    async def _run_all_experiments_sequential(self) -> List[Dict[str, Any]]:
        """
        顺序运行所有实验（原始实现）

        Returns:
            List[Dict]: 所有实验的结果
        """
        all_results = []
        self.total_experiments = len(self.experiments) * self.max_repetitions

        print("="*80)
        print("智能批量实验调度器")
        print("="*80)
        print(f"总实验配置数: {len(self.experiments)}")
        print(f"每个配置重复次数: {self.max_repetitions}")
        print(f"总实验运行数: {self.total_experiments}")
        print(f"GPU显存调度: {'启用' if self.enable_gpu_scheduling else '禁用'}")
        print(f"日志目录: {self.log_dir}")
        print(f"数据库: {self.tracker.db_path}")
        print("="*80)
        print()

        # 显示当前GPU状态
        if self.enable_gpu_scheduling:
            self.gpu_monitor.print_gpu_status()

        for exp_idx, exp_config in enumerate(self.experiments, 1):
            print(f"\n[{exp_idx}/{len(self.experiments)}] 处理实验: {exp_config.name}")

            # 注册实验到数据库
            config_hash = self.tracker.register_experiment(
                exp_config.config,
                dataset=exp_config.dataset,
                algorithm=exp_config.algorithm,
                noniid_type=exp_config.noniid_type
            )

            # 检查是否已完成所有重复执行
            if self.tracker.is_experiment_completed(config_hash, self.max_repetitions):
                print(f"  ✓ 已完成所有 {self.max_repetitions} 次运行，跳过")
                self.skipped_experiments += self.max_repetitions
                continue

            # 执行未完成的runs
            for run_num in range(1, self.max_repetitions + 1):
                # 检查该run是否已成功完成
                next_run = self.tracker.get_next_run_number(config_hash, self.max_repetitions)
                if next_run is None or run_num < next_run:
                    print(f"  ✓ Run {run_num}/{self.max_repetitions} 已完成，跳过")
                    self.skipped_experiments += 1
                    continue

                print(f"\n  → Run {run_num}/{self.max_repetitions}:")

                # GPU显存检查（如果启用）
                if self.enable_gpu_scheduling:
                    if not self.gpu_monitor.can_launch_experiment(safety_margin_mb=self.safety_margin_mb):
                        print(f"    显存不足，等待中...")
                        self.gpu_monitor.wait_for_available_memory(safety_margin_mb=self.safety_margin_mb)

                    max_free = self.gpu_monitor.get_max_free_memory()
                    estimated = self.gpu_monitor.estimate_experiment_memory()
                    print(f"    GPU显存: {max_free}MB 可用 / {estimated}MB 估算需求")

                # 运行实验
                result = await self.run_single_experiment(exp_config, run_num, config_hash)

                if result['status'] == 'success':
                    print(f"    ✓ 成功: accuracy={result.get('accuracy', 'N/A'):.4f}, "
                          f"duration={result.get('duration', 0):.1f}s")
                    self.completed_experiments += 1
                else:
                    print(f"    ✗ 失败: {result.get('error', 'Unknown error')}")
                    self.failed_experiments += 1

                all_results.append({
                    'exp_name': exp_config.name,
                    'run_number': run_num,
                    **result
                })

        # 打印最终统计
        print("\n" + "="*80)
        print("实验批量运行完成")
        print("="*80)
        print(f"总实验数: {self.total_experiments}")
        print(f"完成: {self.completed_experiments}")
        print(f"失败: {self.failed_experiments}")
        print(f"跳过（已完成）: {self.skipped_experiments}")
        print("="*80)

        # 打印实验追踪摘要
        self.tracker.print_summary()

        return all_results

    async def run_all_experiments_concurrent(self) -> List[Dict[str, Any]]:
        """
        并发运行所有实验（基于GPU显存的动态调度）

        Returns:
            List[Dict]: 所有实验的结果
        """
        all_results = []
        self.total_experiments = len(self.experiments) * self.max_repetitions

        print("="*80)
        print("智能批量实验调度器 - 并发模式")
        print("="*80)
        print(f"总实验配置数: {len(self.experiments)}")
        print(f"每个配置重复次数: {self.max_repetitions}")
        print(f"总实验运行数: {self.total_experiments}")
        print(f"GPU显存调度: {'启用' if self.enable_gpu_scheduling else '禁用'}")
        print(f"最大并发数: {self.max_concurrent_experiments}")
        print(f"日志目录: {self.log_dir}")
        print(f"数据库: {self.tracker.db_path}")
        print("="*80)
        print()

        # 显示当前GPU状态
        if self.enable_gpu_scheduling:
            self.gpu_monitor.print_gpu_status()

        # 准备所有待运行的任务
        pending_tasks = []  # List of (exp_config, run_number, config_hash)

        for exp_idx, exp_config in enumerate(self.experiments, 1):
            # 注册实验到数据库
            config_hash = self.tracker.register_experiment(
                exp_config.config,
                dataset=exp_config.dataset,
                algorithm=exp_config.algorithm,
                noniid_type=exp_config.noniid_type
            )

            # 检查是否已完成所有重复执行
            if self.tracker.is_experiment_completed(config_hash, self.max_repetitions):
                print(f"[{exp_idx}/{len(self.experiments)}] {exp_config.name}: ✓ 已完成所有 {self.max_repetitions} 次运行，跳过")
                self.skipped_experiments += self.max_repetitions
                continue

            # 收集未完成的runs
            for run_num in range(1, self.max_repetitions + 1):
                next_run = self.tracker.get_next_run_number(config_hash, self.max_repetitions)
                if next_run is None or run_num < next_run:
                    self.skipped_experiments += 1
                    continue

                pending_tasks.append((exp_config, run_num, config_hash))

        print(f"\n待运行任务数: {len(pending_tasks)}")
        print(f"已跳过任务数: {self.skipped_experiments}")
        print("="*80)
        print()

        # 并发任务池
        running_tasks = {}  # {asyncio.Task: (exp_config, run_number, config_hash, start_time)}

        # 主调度循环
        while pending_tasks or running_tasks:
            # 1. 检查并处理已完成的任务
            if running_tasks:
                done_tasks = [task for task in running_tasks.keys() if task.done()]

                for task in done_tasks:
                    exp_config, run_num, config_hash, start_time = running_tasks.pop(task)

                    try:
                        result = await task

                        if result['status'] == 'success':
                            print(f"✓ [{exp_config.name} Run{run_num}] 完成: "
                                  f"accuracy={result.get('accuracy', 'N/A'):.4f}, "
                                  f"duration={result.get('duration', 0):.1f}s")
                            self.completed_experiments += 1
                        else:
                            print(f"✗ [{exp_config.name} Run{run_num}] 失败: "
                                  f"{result.get('error', 'Unknown error')[:60]}")
                            self.failed_experiments += 1

                        all_results.append({
                            'exp_name': exp_config.name,
                            'run_number': run_num,
                            **result
                        })

                    except Exception as e:
                        print(f"✗ [{exp_config.name} Run{run_num}] 异常: {str(e)[:60]}")
                        self.failed_experiments += 1
                        all_results.append({
                            'exp_name': exp_config.name,
                            'run_number': run_num,
                            'status': 'failed',
                            'error': str(e)
                        })

            # 2. 尝试启动新任务（如果条件允许）
            while pending_tasks:
                # 获取当前任务队列中第一个任务的数据集
                next_exp_config, next_run_num, next_config_hash = pending_tasks[0]
                current_dataset = next_exp_config.dataset

                # 获取该数据集的并发限制
                dataset_concurrent_limit = self.get_concurrent_limit_for_dataset(current_dataset)

                # 检查是否已达到并发限制
                if len(running_tasks) >= dataset_concurrent_limit:
                    break

                # GPU显存检查
                can_launch = True
                if self.enable_gpu_scheduling:
                    can_launch = self.gpu_monitor.can_launch_experiment(
                        safety_margin_mb=self.safety_margin_mb
                    )

                if not can_launch:
                    print(f"  [调度器] 显存不足，等待中... (当前运行: {len(running_tasks)})")
                    break

                # 启动新任务
                exp_config, run_num, config_hash = pending_tasks.pop(0)

                print(f"→ [{exp_config.name} Run{run_num}] 启动 (当前并发: {len(running_tasks)+1}/{dataset_concurrent_limit})")

                if self.enable_gpu_scheduling:
                    max_free = self.gpu_monitor.get_max_free_memory()
                    estimated = self.gpu_monitor.estimate_experiment_memory()
                    print(f"  GPU显存: {max_free}MB 可用 / {estimated}MB 估算需求")

                # 创建异步任务
                task = asyncio.create_task(
                    self.run_single_experiment(exp_config, run_num, config_hash)
                )
                running_tasks[task] = (exp_config, run_num, config_hash, datetime.now())

            # 3. 显示进度
            total_processed = self.completed_experiments + self.failed_experiments + self.skipped_experiments
            print(f"\n[进度] 总计: {total_processed}/{self.total_experiments} | "
                  f"运行中: {len(running_tasks)} | "
                  f"等待: {len(pending_tasks)} | "
                  f"完成: {self.completed_experiments} | "
                  f"失败: {self.failed_experiments} | "
                  f"跳过: {self.skipped_experiments}")

            # 4. 短暂等待后继续调度
            if running_tasks or pending_tasks:
                await asyncio.sleep(10)  # 每10秒检查一次

        # 打印最终统计
        print("\n" + "="*80)
        print("实验批量运行完成（并发模式）")
        print("="*80)
        print(f"总实验数: {self.total_experiments}")
        print(f"完成: {self.completed_experiments}")
        print(f"失败: {self.failed_experiments}")
        print(f"跳过（已完成）: {self.skipped_experiments}")
        print("="*80)

        # 打印实验追踪摘要
        self.tracker.print_summary()

        return all_results

    def run_all_experiments_multiprocess(self) -> List[Dict[str, Any]]:
        """
        多进程并发运行所有实验（真实进程隔离）

        与asyncio版本的区别：
        - 真实进程隔离，避免内存泄漏和状态污染
        - 每个实验在独立进程中运行，相互不影响
        - 更适合长时间批量实验

        Returns:
            List[Dict]: 所有实验的结果
        """
        all_results = []
        self.total_experiments = len(self.experiments) * self.max_repetitions

        print("="*80)
        print("智能批量实验调度器 - 多进程模式")
        print("="*80)
        print(f"总实验配置数: {len(self.experiments)}")
        print(f"每个配置重复次数: {self.max_repetitions}")
        print(f"总实验运行数: {self.total_experiments}")
        print(f"GPU显存调度: {'启用' if self.enable_gpu_scheduling else '禁用'}")
        print(f"最大并发数: {self.max_concurrent_experiments}")
        print(f"日志目录: {self.log_dir}")
        print(f"数据库: {self.tracker.db_path}")
        print("="*80)
        print()

        # 显示当前GPU状态
        if self.enable_gpu_scheduling:
            self.gpu_monitor.print_gpu_status()

        # 准备所有待运行的任务
        pending_tasks = []  # List of (exp_config, run_number, config_hash)

        for exp_idx, exp_config in enumerate(self.experiments, 1):
            # 注册实验到数据库
            config_hash = self.tracker.register_experiment(
                exp_config.config,
                dataset=exp_config.dataset,
                algorithm=exp_config.algorithm,
                noniid_type=exp_config.noniid_type
            )

            # 检查是否已完成所有重复执行
            if self.tracker.is_experiment_completed(config_hash, self.max_repetitions):
                print(f"[{exp_idx}/{len(self.experiments)}] {exp_config.name}: ✓ 已完成所有 {self.max_repetitions} 次运行，跳过")
                self.skipped_experiments += self.max_repetitions
                continue

            # 收集未完成的runs
            for run_num in range(1, self.max_repetitions + 1):
                next_run = self.tracker.get_next_run_number(config_hash, self.max_repetitions)
                if next_run is None or run_num < next_run:
                    self.skipped_experiments += 1
                    continue

                pending_tasks.append((exp_config, run_num, config_hash))

        print(f"\n待运行任务数: {len(pending_tasks)}")
        print(f"已跳过任务数: {self.skipped_experiments}")
        print("="*80)
        print()

        # 多进程任务池
        running_processes = {}  # {Process: (exp_config, run_number, config_hash, start_time, result_queue)}

        # 主调度循环
        while pending_tasks or running_processes:
            # 1. 检查并处理已完成的进程
            if running_processes:
                finished_processes = []

                for process, (exp_config, run_num, config_hash, start_time, result_queue) in list(running_processes.items()):
                    # 检查进程是否结束
                    if not process.is_alive():
                        finished_processes.append(process)

                        try:
                            # 从队列获取结果（非阻塞，带超时）
                            result = result_queue.get(timeout=1.0)

                            if result['status'] == 'success':
                                print(f"✓ [{exp_config.name} Run{run_num}] 完成: "
                                      f"accuracy={result.get('accuracy', 'N/A'):.4f}, "
                                      f"duration={result.get('duration', 0):.1f}s")
                                self.completed_experiments += 1
                            else:
                                print(f"✗ [{exp_config.name} Run{run_num}] 失败: "
                                      f"{result.get('error', 'Unknown error')[:60]}")
                                self.failed_experiments += 1

                            all_results.append({
                                'exp_name': exp_config.name,
                                'run_number': run_num,
                                **result
                            })

                        except Exception as e:
                            print(f"✗ [{exp_config.name} Run{run_num}] 进程异常: {str(e)[:60]}")
                            self.failed_experiments += 1
                            all_results.append({
                                'exp_name': exp_config.name,
                                'run_number': run_num,
                                'status': 'failed',
                                'error': f"Process error: {str(e)}"
                            })
                        finally:
                            # 清理进程
                            process.join(timeout=5)
                            if process.is_alive():
                                process.terminate()
                                process.join(timeout=5)

                            del running_processes[process]

            # 2. 尝试启动新进程（如果条件允许）
            while pending_tasks:
                # 获取当前任务队列中第一个任务的数据集
                next_exp_config, next_run_num, next_config_hash = pending_tasks[0]
                current_dataset = next_exp_config.dataset

                # 获取该数据集的并发限制
                dataset_concurrent_limit = self.get_concurrent_limit_for_dataset(current_dataset)

                # 检查是否已达到并发限制
                if len(running_processes) >= dataset_concurrent_limit:
                    break

                # GPU显存检查
                can_launch = True
                if self.enable_gpu_scheduling:
                    can_launch = self.gpu_monitor.can_launch_experiment(
                        safety_margin_mb=self.safety_margin_mb
                    )

                if not can_launch:
                    print(f"  [调度器] 显存不足，等待中... (当前运行: {len(running_processes)})")
                    break

                # 启动新进程
                exp_config, run_num, config_hash = pending_tasks.pop(0)

                print(f"→ [{exp_config.name} Run{run_num}] 启动进程 (当前并发: {len(running_processes)+1}/{dataset_concurrent_limit})")

                if self.enable_gpu_scheduling:
                    max_free = self.gpu_monitor.get_max_free_memory()
                    estimated = self.gpu_monitor.estimate_experiment_memory()
                    print(f"  GPU显存: {max_free}MB 可用 / {estimated}MB 估算需求")

                # 创建结果队列
                result_queue = mp.Queue()

                # 创建进程
                process = mp.Process(
                    target=_run_experiment_in_process,
                    args=(
                        exp_config,
                        run_num,
                        config_hash,
                        self.config_base_dir,
                        str(self.log_dir),  # Convert Path to string
                        self.tracker.db_path,
                        result_queue
                    )
                )
                process.start()

                running_processes[process] = (exp_config, run_num, config_hash, datetime.now(), result_queue)

            # 3. 显示进度
            total_processed = self.completed_experiments + self.failed_experiments + self.skipped_experiments
            print(f"\n[进度] 总计: {total_processed}/{self.total_experiments} | "
                  f"运行中: {len(running_processes)} | "
                  f"等待: {len(pending_tasks)} | "
                  f"完成: {self.completed_experiments} | "
                  f"失败: {self.failed_experiments} | "
                  f"跳过: {self.skipped_experiments}")

            # 4. 短暂等待后继续调度
            if running_processes or pending_tasks:
                time.sleep(10)  # 每10秒检查一次

        # 打印最终统计
        print("\n" + "="*80)
        print("实验批量运行完成（多进程模式）")
        print("="*80)
        print(f"总实验数: {self.total_experiments}")
        print(f"完成: {self.completed_experiments}")
        print(f"失败: {self.failed_experiments}")
        print(f"跳过（已完成）: {self.skipped_experiments}")
        print("="*80)

        # 打印实验追踪摘要
        self.tracker.print_summary()

        return all_results


def _run_experiment_in_process(
    exp_config: ExperimentConfig,
    run_number: int,
    config_hash: str,
    config_base_dir: str,
    log_dir: str,  # Changed from Path to str for multiprocessing
    db_path: str,
    result_queue: mp.Queue
):
    """
    在独立进程中运行单个实验

    这个函数作为multiprocessing.Process的target运行
    必须是顶层函数（不能是类方法）

    Args:
        exp_config: 实验配置
        run_number: 运行编号
        config_hash: 配置哈希值
        config_base_dir: 配置文件基础目录
        log_dir: 日志目录（字符串路径）
        db_path: 数据库路径
        result_queue: 用于返回结果的队列
    """
    import traceback
    from pathlib import Path

    try:
        # 在进程内部创建新的追踪器实例
        tracker = ExperimentTracker(db_path)

        # 创建日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{exp_config.dataset}_{exp_config.noniid_type}_{exp_config.algorithm}_run{run_number}_{timestamp}.log"
        log_file = str(Path(log_dir) / filename)

        # 标记实验开始
        tracker.start_run(config_hash, run_number, log_file)

        start_time = datetime.now()
        result = {'status': 'failed', 'error': 'Unknown error'}

        try:
            # 运行实验 (使用asyncio.run来运行async函数)
            from fedcl.experiment import BatchExperimentRunner

            runner = BatchExperimentRunner(
                config_base_dir,
                [exp_config.config]
            )

            # 在进程内运行异步函数
            import asyncio
            results = asyncio.run(runner.run_all(parallel=False))
            result = results[0] if results else result

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            result = {
                'status': 'failed',
                'error': error_msg,
                'traceback': error_trace
            }

        finally:
            # 计算实验耗时
            duration = (datetime.now() - start_time).total_seconds()
            result['duration'] = duration

            # 更新实验状态到数据库
            tracker.complete_run(
                config_hash,
                run_number,
                result,
                status=result['status']
            )

            # 将结果放入队列
            result_queue.put(result)

    except Exception as e:
        # 顶层异常处理，确保结果总是返回
        error_msg = f"Process top-level error: {str(e)}"
        error_trace = traceback.format_exc()
        result_queue.put({
            'status': 'failed',
            'error': error_msg,
            'traceback': error_trace,
            'duration': 0
        })


# 使用示例
if __name__ == "__main__":
    async def test_smart_batch_runner():
        """测试智能批量调度器"""

        # 创建测试实验配置
        test_experiments = [
            ExperimentConfig(
                name="MNIST_IID_FedAvg",
                dataset="MNIST",
                algorithm="FedAvg",
                noniid_type="IID",
                config={
                    "dataset": "MNIST",
                    "algorithm": "FedAvg",
                    "noniid_type": "iid"
                }
            ),
            ExperimentConfig(
                name="MNIST_IID_FedProx",
                dataset="MNIST",
                algorithm="FedProx",
                noniid_type="IID",
                config={
                    "dataset": "MNIST",
                    "algorithm": "FedProx",
                    "noniid_type": "iid"
                }
            ),
        ]

        # 创建智能调度器
        runner = SmartBatchRunner(
            config_base_dir="configs/distributed/experiments/table3/",
            experiments=test_experiments,
            max_repetitions=3,
            enable_gpu_scheduling=True
        )

        # 运行所有实验
        results = await runner.run_all_experiments()

        print(f"\n运行完成，共 {len(results)} 个实验结果")

    # 运行测试
    asyncio.run(test_smart_batch_runner())
