#!/bin/bash
# 使用简单模式运行失败实验（不用multiprocess）

echo "开始运行CINIC10和FedISIC2019实验（简单模式）..."

# CINIC10 - 顺序运行
PYTHONPATH=. python examples/reproduce_table3_experiments.py --mode single --dataset CINIC10 --quiet 2>&1 | tee /tmp/cinic10_simple.log

echo "CINIC10完成，开始FedISIC2019..."

# FedISIC2019 - 顺序运行
PYTHONPATH=. python examples/reproduce_table3_experiments.py --mode single --dataset FedISIC2019 --quiet 2>&1 | tee /tmp/isic2019_simple.log

echo "全部完成！"
