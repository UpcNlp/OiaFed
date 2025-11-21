#!/usr/bin/env python3
"""
查看Table 3实验结果的工具脚本
用法: python view_table3_results.py
"""

import sqlite3
import pandas as pd
import sys

def main():
    db_path = "experiments/table3_tracker.db"

    try:
        conn = sqlite3.connect(db_path)

        print("="*80)
        print("Table 3 实验结果数据库查看器")
        print("="*80)

        # 1. 统计信息
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total_experiments = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM experiment_runs WHERE status='success'")
        successful_runs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT noniid_type) FROM experiments")
        num_noniid_types = cursor.fetchone()[0]

        print(f"\n数据库统计:")
        print(f"  总实验配置数: {total_experiments}")
        print(f"  成功运行次数: {successful_runs}")
        print(f"  Non-IID类型数: {num_noniid_types}")

        # 2. Non-IID类型分布
        print(f"\nNon-IID类型分布:")
        cursor.execute("""
            SELECT noniid_type, COUNT(*) as count
            FROM experiments
            GROUP BY noniid_type
            ORDER BY noniid_type
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]} 个实验配置")

        # 3. 数据集分布
        print(f"\n数据集分布:")
        cursor.execute("""
            SELECT dataset, COUNT(DISTINCT algorithm) as num_algorithms,
                   COUNT(DISTINCT noniid_type) as num_noniid_types
            FROM experiments
            GROUP BY dataset
            ORDER BY dataset
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]} 个算法, {row[2]} 种Non-IID类型")

        # 4. 读取所有结果
        df = pd.read_sql_query("""
            SELECT
                e.dataset,
                e.noniid_type,
                e.algorithm,
                ROUND(AVG(r.accuracy), 4) as avg_accuracy,
                ROUND(AVG(r.loss), 4) as avg_loss,
                ROUND(AVG(r.rounds), 1) as avg_rounds,
                COUNT(r.id) as num_repetitions
            FROM experiments e
            JOIN experiment_runs r ON e.config_hash = r.config_hash
            WHERE r.status = 'success'
            GROUP BY e.dataset, e.noniid_type, e.algorithm
            ORDER BY e.dataset, e.noniid_type, e.algorithm
        """, conn)

        print(f"\n实验结果预览 (前20行):")
        print(df.head(20).to_string(index=False))

        print(f"\n\n总结果数: {len(df)} 行")
        print(f"CSV文件已保存到: experiments/table3_results.csv")

        conn.close()

        print("\n" + "="*80)
        print("数据库可以正常访问！使用以下命令查看:")
        print("  sqlite3 experiments/table3_tracker.db")
        print("  或使用DB Browser for SQLite等图形化工具")
        print("="*80)

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
