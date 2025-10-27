#!/bin/bash
# Network 模式测试 - 客户端快速启动脚本
# 使用方法：./start_client.sh <client_number> [server_ip]

set -e

# 默认值
SERVER_IP="${2:-192.168.31.68}"
SERVER_PORT=8000
CLIENT_NUM="${1:-1}"

# 检查参数
if [ -z "$CLIENT_NUM" ]; then
    echo "用法: $0 <client_number> [server_ip]"
    echo "示例: $0 1 192.168.31.68"
    exit 1
fi

# 验证客户端编号
if [ "$CLIENT_NUM" != "1" ] && [ "$CLIENT_NUM" != "2" ]; then
    echo "错误: client_number 必须是 1 或 2"
    exit 1
fi

CONFIG_FILE="configs/network_test/client${CLIENT_NUM}.yaml"

echo "============================================================"
echo "Network 模式测试 - 客户端 ${CLIENT_NUM}"
echo "============================================================"
echo "配置文件: ${CONFIG_FILE}"
echo "服务端地址: ${SERVER_IP}:${SERVER_PORT}"
echo "============================================================"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查网络连通性
echo "检查与服务端的连接..."
if ping -c 1 -W 2 "$SERVER_IP" > /dev/null 2>&1; then
    echo "✅ 网络连接正常"
else
    echo "⚠️  警告: 无法 ping 通服务端 ${SERVER_IP}"
    echo "继续启动客户端..."
fi

# 启动客户端
echo ""
echo "启动客户端..."
python examples/network_client_standalone.py \
    --config "$CONFIG_FILE" \
    --server-ip "$SERVER_IP" \
    --server-port "$SERVER_PORT"
