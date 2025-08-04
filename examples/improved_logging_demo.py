#!/usr/bin/env python3
"""
改进日志系统使用示例

这个示例展示了如何在 FedCL 项目中使用改进的日志系统
"""

from fedcl.utils.improved_logging_manager import (
    initialize_improved_logging, 
    get_component_logger, 
    log_training_info, 
    log_system_debug
)


def main():
    """主函数 - 演示改进的日志系统使用"""
    
    # 1. 初始化改进的日志系统
    print("初始化改进的日志系统...")
    log_manager = initialize_improved_logging(
        log_base_dir="./logs",
        experiment_name="mnist_联邦学习实验",
        enable_console=True,
        global_log_level="INFO"
    )
    
    # 2. 获取不同组件的日志器
    server_logger = get_component_logger("server", "主服务器")
    client1_logger = get_component_logger("client", "客户端_001")
    client2_logger = get_component_logger("client", "客户端_002")
    
    # 3. 服务器日志
    server_logger.info("联邦学习服务器启动完成")
    server_logger.debug("加载服务器配置文件")
    
    # 4. 客户端日志（写入独立文件）
    client1_logger.info("客户端001准备就绪，开始训练")
    client1_logger.info("本地训练第1个epoch完成")
    
    client2_logger.info("客户端002准备就绪，开始训练")
    client2_logger.info("本地训练第1个epoch完成")
    
    # 5. 使用便利函数记录训练和系统日志
    log_training_info("开始第1轮联邦训练")
    log_training_info("收集客户端模型更新")
    log_training_info("执行FedAvg聚合")
    log_training_info("第1轮训练完成")
    
    log_system_debug("检查系统资源使用情况")
    log_system_debug("清理临时文件")
    
    # 6. 显示日志文件信息
    print("\n生成的日志文件:")
    log_files = log_manager.get_log_files_info()
    for key, path in log_files.items():
        print(f"  {key}: {path}")
    
    print("\n✅ 日志系统示例完成！")
    print("📁 检查 logs/ 目录下的日志文件以查看效果")


if __name__ == "__main__":
    main()
