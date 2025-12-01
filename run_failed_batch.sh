#!/bin/bash
# 运行output.log中失败的实验

# 失败最多的数据集：CINIC10 (189个), FedISIC2019 (189个)

echo "开始运行失败的实验..."
echo "只运行CINIC10和FedISIC2019的失败实验（共378个）"

# CINIC10的所有配置
PYTHONPATH=. python examples/reproduce_table3_experiments.py --mode single --dataset CINIC10 --use-smart-runner --multiprocess --max-concurrent 3 --quiet 2>&1 | tee /tmp/cinic10_rerun.log &

# FedISIC2019的所有配置  
PYTHONPATH=. python examples/reproduce_table3_experiments.py --mode single --dataset FedISIC2019 --use-smart-runner --multiprocess --max-concurrent 3 --quiet 2>&1 | tee /tmp/isic2019_full_rerun.log &

wait

echo "完成！"
