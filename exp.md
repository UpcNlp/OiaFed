 系统资源情况

  | 资源 | 配置                         | 10并发评估                    |
  |------|------------------------------|-------------------------------|
  | CPU  | 32核                         | ✅ 完全足够                   |
  | 内存 | 125GB total, 87GB available  | ✅ 完全足够                   |
  | GPU  | 2个 × 32GB (各有28-29GB可用) | ✅ 足够（建议每GPU跑5个实验） |
  | 端口 | 50051-60051 范围，仅5个在用  | ✅ 完全足够                   |
  | 磁盘 | 1.9TB 可用                   | ✅ 完全足够                   |

  结论：你的系统完全可以支持10个并发实验！ 🎉

  推荐的运行方案

  最优配置（充分利用2个GPU）

  # 方案A：直接10并发（让系统自动分配GPU）
  nohup python scripts/run_table3_experiments.py \
      --config-dir configs/table3_experiments \
      --max-parallel 10 \
      > experiments_10_parallel.log 2>&1 &

  # 记录进程ID
  echo $! > experiment_pid.txt

  或者手动分配GPU（更精确控制）

  # 终端1：GPU0上运行5个并发
  CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_table3_experiments.py \
      --config-dir configs/table3_experiments \
      --dataset mnist,fmnist,emnist \
      --max-parallel 5 \
      > gpu0_experiments.log 2>&1 &

  # 终端2：GPU1上运行5个并发
  CUDA_VISIBLE_DEVICES=1 nohup python scripts/run_table3_experiments.py \
      --config-dir configs/table3_experiments \
      --dataset cifar10,cifar100,cinic10,svhn \
      --max-parallel 5 \
      > gpu1_experiments.log 2>&1 &

  开始运行

  # 1. 清理之前可能的实验
  # (如果有端口占用)
  pkill -f run_experiment.py

  # 2. 检查端口是否释放
  netstat -tuln | grep 5005

  # 3. 开始运行
  nohup python scripts/run_table3_experiments.py \
      --config-dir configs/table3_experiments \
      --max-parallel 10 \
      > experiments_10_parallel.log 2>&1 &

  # 4. 保存进程ID
  echo $! > experiment_pid.txt

  # 5. 监控进度
  watch -n 10 ./scripts/monitor_experiments.sh

  # 6. 查看实时日志
  tail -f experiments_10_parallel.log

  预期性能

  根据你的配置：
  - 每个实验预计时间：视数据集和算法而定，大约5-30分钟
  - 10个并发：理论上可以将总时间缩短到原来的1/10
  - 总耗时估算：
    - 288个实验顺序运行：约24-144小时
    - 288个实验10并发：约2.4-14.4小时

  监控命令

  # 查看运行中的实验数量
  ps aux | grep run_experiment.py | grep -v grep | wc -l

  # 查看GPU使用情况
  watch -n 5 nvidia-smi

  # 查看端口占用
  netstat -tuln | grep python | grep -E ":(5[0-9]{4}|60000)" | wc -l

  # 查看实验进度
  python -c "
  import json
  try:
      with open('configs/table3_experiments/experiment_results.json') as f:
          data = json.load(f)
          print(f\"Progress: {data['summary']['success']}/{data['summary']['total']} completed\")
          print(f\"Success: {data['summary']['success']}, Failed: {data['summary']['failed']}\")
  except:
      print('No results yet')
  "

  停止实验（如果需要）

  # 方法1：使用保存的进程ID
  kill $(cat experiment_pid.txt)

  # 方法2：杀掉所有相关进程
  pkill -f run_table3_experiments.py
  pkill -f run_experiment.py

  # 方法3：优雅停止（等待当前实验完成）
  kill -TERM $(cat experiment_pid.txt)