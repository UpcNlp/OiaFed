"""
实验追踪器 - 使用SQLite管理实验状态和断点续跑
examples/experiment_tracker.py

个人需求工具，不属于开源项目核心代码
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager


class ExperimentTracker:
    """
    实验状态追踪器

    功能：
    - 记录实验配置和执行状态
    - 支持重复执行（3次）
    - 断点续跑（跳过已完成的实验）
    """

    def __init__(self, db_path: str = "experiments/experiment_tracker.db"):
        """
        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """初始化数据库schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 实验配置表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    config_hash TEXT PRIMARY KEY,
                    exp_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    dataset TEXT,
                    algorithm TEXT,
                    noniid_type TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 实验执行记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_hash TEXT NOT NULL,
                    run_number INTEGER NOT NULL,
                    status TEXT NOT NULL,  -- 'pending', 'running', 'success', 'failed'
                    accuracy REAL,
                    loss REAL,
                    rounds INTEGER,
                    duration_sec REAL,
                    error_msg TEXT,
                    log_file TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY (config_hash) REFERENCES experiments(config_hash),
                    UNIQUE(config_hash, run_number)
                )
            """)

            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_status
                ON experiment_runs(config_hash, status)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        计算实验配置的哈希值（用于唯一标识）

        注意：忽略name字段，只根据实际配置参数计算hash
        """
        # 创建配置副本并移除name（因为name可能包含时间戳）
        config_for_hash = config.copy()
        config_for_hash.pop('name', None)

        # 将配置转换为稳定的JSON字符串
        config_str = json.dumps(config_for_hash, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def register_experiment(self, config: Dict[str, Any],
                           dataset: str = None,
                           algorithm: str = None,
                           noniid_type: str = None) -> str:
        """
        注册实验配置

        Args:
            config: 实验配置
            dataset: 数据集名称（可选，用于查询）
            algorithm: 算法名称（可选，用于查询）
            noniid_type: Non-IID类型（可选，用于查询）

        Returns:
            config_hash: 配置哈希值
        """
        config_hash = self._compute_config_hash(config)
        exp_name = config.get('name', f'exp_{config_hash}')
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 插入或更新实验配置
            cursor.execute("""
                INSERT OR REPLACE INTO experiments
                (config_hash, exp_name, config_json, dataset, algorithm, noniid_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (config_hash, exp_name, json.dumps(config), dataset, algorithm, noniid_type, now, now))

            conn.commit()

        return config_hash

    def get_next_run_number(self, config_hash: str, max_runs: int = 3) -> Optional[int]:
        """
        获取下一个可执行的run_number

        Args:
            config_hash: 配置哈希值
            max_runs: 最大重复次数（默认3次）

        Returns:
            run_number: 下一个可执行的run编号（1-based），如果已完成所有runs则返回None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 查询已成功完成的runs
            cursor.execute("""
                SELECT run_number FROM experiment_runs
                WHERE config_hash = ? AND status = 'success'
                ORDER BY run_number
            """, (config_hash,))

            completed_runs = set(row['run_number'] for row in cursor.fetchall())

            # 找到第一个未完成的run
            for run_num in range(1, max_runs + 1):
                if run_num not in completed_runs:
                    return run_num

            return None  # 所有runs都已完成

    def is_experiment_completed(self, config_hash: str, max_runs: int = 3) -> bool:
        """
        检查实验是否已完成所有重复执行

        Args:
            config_hash: 配置哈希值
            max_runs: 最大重复次数

        Returns:
            True if completed, False otherwise
        """
        return self.get_next_run_number(config_hash, max_runs) is None

    def start_run(self, config_hash: str, run_number: int, log_file: str = None) -> int:
        """
        标记实验开始执行

        Returns:
            run_id: 执行记录ID
        """
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 插入或更新执行记录
            cursor.execute("""
                INSERT OR REPLACE INTO experiment_runs
                (config_hash, run_number, status, log_file, started_at)
                VALUES (?, ?, 'running', ?, ?)
            """, (config_hash, run_number, log_file, now))

            run_id = cursor.lastrowid
            conn.commit()

        return run_id

    def complete_run(self, config_hash: str, run_number: int,
                    result: Dict[str, Any], status: str = 'success'):
        """
        标记实验执行完成

        Args:
            config_hash: 配置哈希值
            run_number: 执行编号
            result: 实验结果
            status: 'success' or 'failed'
        """
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE experiment_runs
                SET status = ?,
                    accuracy = ?,
                    loss = ?,
                    rounds = ?,
                    duration_sec = ?,
                    error_msg = ?,
                    completed_at = ?
                WHERE config_hash = ? AND run_number = ?
            """, (
                status,
                result.get('accuracy'),
                result.get('loss'),
                result.get('rounds'),
                result.get('duration'),
                result.get('error'),
                now,
                config_hash,
                run_number
            ))

            conn.commit()

    def get_experiment_summary(self, config_hash: str) -> Dict[str, Any]:
        """
        获取实验的摘要统计

        Returns:
            {
                'total_runs': int,
                'successful_runs': int,
                'failed_runs': int,
                'avg_accuracy': float,
                'avg_loss': float
            }
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(CASE WHEN status = 'success' THEN accuracy END) as avg_acc,
                    AVG(CASE WHEN status = 'success' THEN loss END) as avg_loss
                FROM experiment_runs
                WHERE config_hash = ?
            """, (config_hash,))

            row = cursor.fetchone()

            return {
                'total_runs': row['total'],
                'successful_runs': row['successful'] or 0,
                'failed_runs': row['failed'] or 0,
                'avg_accuracy': row['avg_acc'],
                'avg_loss': row['avg_loss']
            }

    def get_all_experiments_status(self) -> List[Dict[str, Any]]:
        """
        获取所有实验的状态概览

        Returns:
            List of experiment status dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    e.config_hash,
                    e.exp_name,
                    e.dataset,
                    e.algorithm,
                    e.noniid_type,
                    COUNT(r.id) as total_runs,
                    SUM(CASE WHEN r.status = 'success' THEN 1 ELSE 0 END) as completed_runs
                FROM experiments e
                LEFT JOIN experiment_runs r ON e.config_hash = r.config_hash
                GROUP BY e.config_hash
                ORDER BY e.updated_at DESC
            """)

            return [dict(row) for row in cursor.fetchall()]

    def print_summary(self):
        """打印实验追踪摘要"""
        status_list = self.get_all_experiments_status()

        print("\n" + "="*80)
        print("实验追踪摘要")
        print("="*80)
        print(f"数据库路径: {self.db_path}")
        print(f"总实验数: {len(status_list)}")
        print()

        if status_list:
            print(f"{'实验名称':<40s} {'数据集':<10s} {'算法':<10s} {'完成次数':<10s}")
            print("-"*80)
            for exp in status_list[:20]:  # 只显示前20个
                name = exp['exp_name'][:38]
                dataset = exp['dataset'] or 'N/A'
                algo = exp['algorithm'] or 'N/A'
                runs = f"{exp['completed_runs'] or 0}/{exp['total_runs'] or 0}"
                print(f"{name:<40s} {dataset:<10s} {algo:<10s} {runs:<10s}")

            if len(status_list) > 20:
                print(f"... 还有 {len(status_list) - 20} 个实验")

        print("="*80)
