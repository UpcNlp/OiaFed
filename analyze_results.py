#!/usr/bin/env python3
"""
实验结果分析脚本
读取CSV文件并生成统计报告
"""

import pandas as pd
import sys
from pathlib import Path


def analyze_results(csv_path: str):
    """分析实验结果CSV文件"""

    if not Path(csv_path).exists():
        print(f"✗ 文件不存在: {csv_path}")
        return

    # 读取CSV
    df = pd.read_csv(csv_path)

    print("=" * 100)
    print(f"实验结果分析报告")
    print(f"CSV文件: {csv_path}")
    print("=" * 100)

    # 1. 基本统计
    print("\n" + "=" * 100)
    print("1. 基本统计")
    print("=" * 100)

    total_experiments = len(df)
    successful = len(df[df['Status'] == 'success'])
    failed = len(df[df['Status'] == 'failed'])

    print(f"\n总实验数: {total_experiments}")
    print(f"  成功: {successful} ({successful/total_experiments*100:.1f}%)")
    print(f"  失败: {failed} ({failed/total_experiments*100:.1f}%)")

    if successful > 0:
        success_df = df[df['Status'] == 'success']
        print(f"\n准确率统计 (成功实验):")
        print(f"  平均: {success_df['Final_Accuracy'].mean():.4f}")
        print(f"  最高: {success_df['Final_Accuracy'].max():.4f}")
        print(f"  最低: {success_df['Final_Accuracy'].min():.4f}")
        print(f"  标准差: {success_df['Final_Accuracy'].std():.4f}")

        print(f"\n运行时间统计 (秒):")
        print(f"  平均: {success_df['Duration_sec'].mean():.2f}")
        print(f"  最长: {success_df['Duration_sec'].max():.2f}")
        print(f"  最短: {success_df['Duration_sec'].min():.2f}")

        print(f"\n完成轮数统计:")
        print(f"  平均: {success_df['Completed_Rounds'].mean():.1f}")
        print(f"  最多: {success_df['Completed_Rounds'].max()}")
        print(f"  最少: {success_df['Completed_Rounds'].min()}")

    # 2. 按数据集分组统计
    print("\n" + "=" * 100)
    print("2. 按数据集分组 - Top准确率")
    print("=" * 100)

    if successful > 0:
        dataset_stats = success_df.groupby('Dataset').agg({
            'Final_Accuracy': ['mean', 'max', 'min', 'count']
        }).round(4)
        print("\n" + dataset_stats.to_string())

        print("\n每个数据集的最佳实验:")
        for dataset in success_df['Dataset'].unique():
            dataset_df = success_df[success_df['Dataset'] == dataset]
            best = dataset_df.loc[dataset_df['Final_Accuracy'].idxmax()]
            print(f"\n  {dataset}:")
            print(f"    实验: {best['Experiment_Name']}")
            print(f"    算法: {best['Algorithm']}")
            print(f"    划分: {best['NonIID_Type']}")
            print(f"    准确率: {best['Final_Accuracy']:.4f}")
            print(f"    轮数: {int(best['Completed_Rounds'])}/{int(best['Max_Rounds'])}")

    # 3. 按算法分组统计
    print("\n" + "=" * 100)
    print("3. 按算法分组 - 性能对比")
    print("=" * 100)

    if successful > 0:
        algo_stats = success_df.groupby('Algorithm').agg({
            'Final_Accuracy': ['mean', 'max', 'min', 'count']
        }).round(4)
        print("\n" + algo_stats.to_string())

        # 算法排名
        algo_ranking = success_df.groupby('Algorithm')['Final_Accuracy'].mean().sort_values(ascending=False)
        print("\n算法平均准确率排名:")
        for i, (algo, acc) in enumerate(algo_ranking.items(), 1):
            count = len(success_df[success_df['Algorithm'] == algo])
            print(f"  {i}. {algo:15s}: {acc:.4f} (基于{count}个实验)")

    # 4. 按Non-IID类型分组统计
    print("\n" + "=" * 100)
    print("4. 按Non-IID类型分组 - 难度分析")
    print("=" * 100)

    if successful > 0:
        noniid_stats = success_df.groupby('NonIID_Type').agg({
            'Final_Accuracy': ['mean', 'max', 'min', 'count']
        }).round(4)
        print("\n" + noniid_stats.to_string())

        # Non-IID难度排名 (准确率越低越难)
        noniid_ranking = success_df.groupby('NonIID_Type')['Final_Accuracy'].mean().sort_values(ascending=True)
        print("\nNon-IID难度排名 (从难到易):")
        for i, (noniid, acc) in enumerate(noniid_ranking.items(), 1):
            count = len(success_df[success_df['NonIID_Type'] == noniid])
            print(f"  {i}. {noniid:20s}: {acc:.4f} (基于{count}个实验)")

    # 5. Early Stopping效果分析
    print("\n" + "=" * 100)
    print("5. Early Stopping效果分析")
    print("=" * 100)

    if successful > 0 and 'Early_Stopping' in success_df.columns:
        early_stopped = success_df[success_df['Completed_Rounds'] < success_df['Max_Rounds']]
        completed_all = success_df[success_df['Completed_Rounds'] == success_df['Max_Rounds']]

        print(f"\nEarly Stopping触发统计:")
        print(f"  提前停止: {len(early_stopped)} ({len(early_stopped)/len(success_df)*100:.1f}%)")
        print(f"  完成全部: {len(completed_all)} ({len(completed_all)/len(success_df)*100:.1f}%)")

        if len(early_stopped) > 0:
            avg_saved_rounds = (early_stopped['Max_Rounds'] - early_stopped['Completed_Rounds']).mean()
            print(f"\n平均节省轮数: {avg_saved_rounds:.1f}")

            total_possible = success_df['Max_Rounds'].sum()
            total_actual = success_df['Completed_Rounds'].sum()
            time_saved_pct = (total_possible - total_actual) / total_possible * 100
            print(f"总轮数节省: {time_saved_pct:.1f}% ({total_actual}/{total_possible})")

    # 6. Top 10 最佳实验
    print("\n" + "=" * 100)
    print("6. Top 10 最佳实验")
    print("=" * 100)

    if successful > 0:
        top10 = success_df.nlargest(10, 'Final_Accuracy')
        print("\n")
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            print(f"{i:2d}. {row['Final_Accuracy']:.4f} - {row['Experiment_Name']}")
            print(f"     Dataset: {row['Dataset']}, Model: {row['Model']}, Algorithm: {row['Algorithm']}")
            print(f"     Partition: {row['Partition_Detail']}, Rounds: {int(row['Completed_Rounds'])}/{int(row['Max_Rounds'])}")

    # 7. 失败实验分析
    if failed > 0:
        print("\n" + "=" * 100)
        print("7. 失败实验分析")
        print("=" * 100)

        failed_df = df[df['Status'] == 'failed']
        print(f"\n失败实验列表 (共{failed}个):")
        for i, (idx, row) in enumerate(failed_df.iterrows(), 1):
            print(f"\n  {i}. {row['Experiment_Name']}")
            print(f"     错误: {row['Error'][:100]}...")

    # 8. 生成Markdown报告
    print("\n" + "=" * 100)
    print("8. 导出Markdown报告")
    print("=" * 100)

    md_path = csv_path.replace('.csv', '_report.md')
    generate_markdown_report(df, success_df if successful > 0 else None, md_path)
    print(f"\n✓ Markdown报告已保存到: {md_path}")

    print("\n" + "=" * 100)
    print("分析完成!")
    print("=" * 100)


def generate_markdown_report(df, success_df, output_path):
    """生成Markdown格式的报告"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 联邦学习实验结果报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 基本统计
        f.write("## 1. 基本统计\n\n")
        f.write(f"- 总实验数: {len(df)}\n")
        f.write(f"- 成功: {len(df[df['Status'] == 'success'])}\n")
        f.write(f"- 失败: {len(df[df['Status'] == 'failed'])}\n\n")

        if success_df is not None and len(success_df) > 0:
            # 按数据集统计表格
            f.write("## 2. 按数据集分组统计\n\n")
            dataset_stats = success_df.groupby('Dataset').agg({
                'Final_Accuracy': ['mean', 'max', 'min', 'count']
            }).round(4)
            f.write(dataset_stats.to_markdown() + "\n\n")

            # 按算法统计表格
            f.write("## 3. 按算法分组统计\n\n")
            algo_stats = success_df.groupby('Algorithm').agg({
                'Final_Accuracy': ['mean', 'max', 'min', 'count']
            }).round(4)
            f.write(algo_stats.to_markdown() + "\n\n")

            # Top 10表格
            f.write("## 4. Top 10 最佳实验\n\n")
            top10 = success_df.nlargest(10, 'Final_Accuracy')[
                ['Experiment_Name', 'Dataset', 'Algorithm', 'NonIID_Type', 'Final_Accuracy', 'Completed_Rounds']
            ]
            f.write(top10.to_markdown(index=False) + "\n\n")


def main():
    """主函数"""

    # 默认CSV路径
    default_csv = "experiments/table3_results.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_csv

    print(f"\n尝试分析: {csv_path}\n")

    try:
        analyze_results(csv_path)
    except Exception as e:
        print(f"\n✗ 分析出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
