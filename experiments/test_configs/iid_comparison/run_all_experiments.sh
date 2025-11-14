#!/bin/bash
# 批量运行IID对比实验
# experiments/test_configs/iid_comparison/run_all_experiments.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始运行IID对比实验（3组）"
echo "=========================================="
echo ""

# 创建配置目录的临时副本（因为memory模式需要合并server和client配置）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# 实验1：FedAvg
echo "[实验 1/3] MNIST + FedAvg + IID"
echo "----------------------------------------"

# 创建临时配置目录
mkdir -p experiments/test_configs/iid_comparison/exp1_configs
cp experiments/test_configs/iid_comparison/exp1_fedavg_server.yaml experiments/test_configs/iid_comparison/exp1_configs/server.yaml
cp experiments/test_configs/iid_comparison/exp1_fedavg_client.yaml experiments/test_configs/iid_comparison/exp1_configs/client_0.yaml
cp experiments/test_configs/iid_comparison/exp1_fedavg_client.yaml experiments/test_configs/iid_comparison/exp1_configs/client_1.yaml
cp experiments/test_configs/iid_comparison/exp1_fedavg_client.yaml experiments/test_configs/iid_comparison/exp1_configs/client_2.yaml

# 修改client配置中的node_id
sed -i 's/node_id: "exp1_client"/node_id: "exp1_client_0"/' experiments/test_configs/iid_comparison/exp1_configs/client_0.yaml
sed -i 's/node_id: "exp1_client"/node_id: "exp1_client_1"/' experiments/test_configs/iid_comparison/exp1_configs/client_1.yaml
sed -i 's/node_id: "exp1_client"/node_id: "exp1_client_2"/' experiments/test_configs/iid_comparison/exp1_configs/client_2.yaml

python examples/run_with_sacred.py \
    --config experiments/test_configs/iid_comparison/exp1_configs \
    --exp_name iid_mnist_fedavg

echo ""
echo "✓ 实验1完成"
echo ""

# 实验2：FedProx
echo "[实验 2/3] MNIST + FedProx + IID"
echo "----------------------------------------"

mkdir -p experiments/test_configs/iid_comparison/exp2_configs
cp experiments/test_configs/iid_comparison/exp2_fedprox_server.yaml experiments/test_configs/iid_comparison/exp2_configs/server.yaml
cp experiments/test_configs/iid_comparison/exp2_fedprox_client.yaml experiments/test_configs/iid_comparison/exp2_configs/client_0.yaml
cp experiments/test_configs/iid_comparison/exp2_fedprox_client.yaml experiments/test_configs/iid_comparison/exp2_configs/client_1.yaml
cp experiments/test_configs/iid_comparison/exp2_fedprox_client.yaml experiments/test_configs/iid_comparison/exp2_configs/client_2.yaml

sed -i 's/node_id: "exp2_client"/node_id: "exp2_client_0"/' experiments/test_configs/iid_comparison/exp2_configs/client_0.yaml
sed -i 's/node_id: "exp2_client"/node_id: "exp2_client_1"/' experiments/test_configs/iid_comparison/exp2_configs/client_1.yaml
sed -i 's/node_id: "exp2_client"/node_id: "exp2_client_2"/' experiments/test_configs/iid_comparison/exp2_configs/client_2.yaml

python examples/run_with_sacred.py \
    --config experiments/test_configs/iid_comparison/exp2_configs \
    --exp_name iid_mnist_fedprox

echo ""
echo "✓ 实验2完成"
echo ""

# 实验3：SCAFFOLD
echo "[实验 3/3] MNIST + SCAFFOLD + IID"
echo "----------------------------------------"

mkdir -p experiments/test_configs/iid_comparison/exp3_configs
cp experiments/test_configs/iid_comparison/exp3_scaffold_server.yaml experiments/test_configs/iid_comparison/exp3_configs/server.yaml
cp experiments/test_configs/iid_comparison/exp3_scaffold_client.yaml experiments/test_configs/iid_comparison/exp3_configs/client_0.yaml
cp experiments/test_configs/iid_comparison/exp3_scaffold_client.yaml experiments/test_configs/iid_comparison/exp3_configs/client_1.yaml
cp experiments/test_configs/iid_comparison/exp3_scaffold_client.yaml experiments/test_configs/iid_comparison/exp3_configs/client_2.yaml

sed -i 's/node_id: "exp3_client"/node_id: "exp3_client_0"/' experiments/test_configs/iid_comparison/exp3_configs/client_0.yaml
sed -i 's/node_id: "exp3_client"/node_id: "exp3_client_1"/' experiments/test_configs/iid_comparison/exp3_configs/client_1.yaml
sed -i 's/node_id: "exp3_client"/node_id: "exp3_client_2"/' experiments/test_configs/iid_comparison/exp3_configs/client_2.yaml

python examples/run_with_sacred.py \
    --config experiments/test_configs/iid_comparison/exp3_configs \
    --exp_name iid_mnist_scaffold

echo ""
echo "✓ 实验3完成"
echo ""

# 汇总结果
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo ""
echo "查看实验结果："
echo "  python experiments/collect_results.py iid_mnist_fedavg"
echo "  python experiments/collect_results.py iid_mnist_fedprox"
echo "  python experiments/collect_results.py iid_mnist_scaffold"
echo ""
echo "结果目录："
echo "  experiments/results/iid_mnist_fedavg/"
echo "  experiments/results/iid_mnist_fedprox/"
echo "  experiments/results/iid_mnist_scaffold/"
