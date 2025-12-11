#!/usr/bin/env python3
"""
测试新的日志系统
验证目录结构：runtime/, training/, configs/
"""
from fedcl.utils.auto_logger import setup_auto_logging, get_logger, save_config_snapshot

# 初始化日志系统
auto_logger = setup_auto_logging(
    experiment_name="test_new_structure",
    config={'console_enabled': True}
)

print(f"日志根目录: {auto_logger.run_dir}")
print(f"Runtime 目录: {auto_logger.runtime_dir}")
print(f"Training 目录: {auto_logger.training_dir}")
print(f"Configs 目录: {auto_logger.configs_dir}")

# 测试 Server 日志
server_runtime_logger = get_logger("runtime", "server")
server_training_logger = get_logger("training", "server")

server_runtime_logger.info("Server started")
server_training_logger.info("Round 1: accuracy=0.85, loss=0.32")

# 测试 Client 日志
client0_runtime_logger = get_logger("runtime", "client_0")
client0_training_logger = get_logger("training", "client_0")

client0_runtime_logger.info("Client 0 connected")
client0_training_logger.info("Local training: accuracy=0.82, loss=0.35")

# 保存配置快照
save_config_snapshot("server", {"role": "server", "lr": 0.01})
save_config_snapshot("client_0", {"role": "client", "lr": 0.01, "client_id": 0})

print("\n查看生成的文件:")
import os
for root, dirs, files in os.walk(auto_logger.run_dir):
    level = root.replace(str(auto_logger.run_dir), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')

print("\n✓ 测试完成！")
