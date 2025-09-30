#!/usr/bin/env python3
"""
DDDR (Diffusion-Driven Data Replay) 联邦学习入口文件

基于FedCL框架实现DDDR方法的完整训练流程。
参考DDDR-master/main.py的设计，适配FedCL框架。
"""

import os
import argparse
import time
import torch
import numpy as np
import json
from omegaconf import OmegaConf
from loguru import logger
# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入FedCL框架组件
from fedcl import registry
from fedcl.methods.learners.dddr import DDDRLearner
from fedcl.methods.trainers.dddr_federation_trainer import DDDRFederationTrainer
print("✓ DDDR模块导入成功")
DDDR_AVAILABLE = True

# 尝试导入transforms
try:
    from fedcl.models.transforms import get_train_transform, SUPPORTED_DATASETS
    print("✓ Transforms模块导入成功")
except Exception as e:
    print(f"⚠️ Transforms模块导入失败: {e}")
    print("将跳过数据变换设置")


def args_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DDDR联邦持续学习基准测试')
    
    # 通用设置
    parser.add_argument('--exp_name', type=str, default='', help='实验名称')
    parser.add_argument('--save_dir', type=str, default="outputs", help='保存目录')
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    parser.add_argument('--g_sigma', type=float, default=0, help='生成器更新差分隐私sigma')
    parser.add_argument('--classifier_dp', type=float, default=0, help='分类器差分隐私')
    parser.add_argument('--dataset', type=str, default="cifar100", help='数据集名称')
    parser.add_argument('--tasks', type=int, default=5, help='任务数量')
    parser.add_argument('--method', type=str, default="dddr", help='学习方法')
    parser.add_argument('--net', type=str, default="resnet18", help='网络架构')
    parser.add_argument('--com_round', type=int, default=100, help='通信轮数')
    parser.add_argument('--num_users', type=int, default=5, help='客户端数量')
    parser.add_argument('--local_bs', type=int, default=128, help='本地批处理大小')
    parser.add_argument('--local_ep', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--beta', type=float, default=0.5, help='标签倾斜程度控制')
    parser.add_argument('--frac', type=float, default=1.0, help='选择客户端比例')
    
    # 目标设置
    parser.add_argument('--nums', type=int, default=8000, help='合成数据数量')
    
    # DDDR设置
    parser.add_argument('--w_kd', type=float, default=10., help='知识蒸馏损失权重')
    parser.add_argument('--w_ce_pre', type=float, default=0.5, help='合成数据交叉熵损失权重')
    parser.add_argument('--w_scl', type=float, default=1., help='监督对比学习损失权重')
    parser.add_argument('--com_round_gen', type=int, default=10, help='生成器通信轮数')
    parser.add_argument('--g_local_train_steps', type=int, default=50, help='生成器本地训练步数')
    parser.add_argument('--config', type=str, default="config/ldm_dddr.yaml", help='扩散模型配置')
    parser.add_argument('--ldm_ckpt', type=str, default="PM/ldm/text2img-large/model.ckpt", help='LDM检查点路径')
    parser.add_argument('--no_scale_lr', action='store_true', help='不缩放学习率')
    parser.add_argument('--g_local_bs', type=int, default=12, help='生成器本地批处理大小')
    parser.add_argument('--n_iter', type=int, default=5, help='生成合成数据迭代次数')
    parser.add_argument('--syn_image_path', type=str, default=None, help='合成数据路径')
    parser.add_argument('--pre_size', type=int, default=200, help='每类历史合成数据大小')
    parser.add_argument('--cur_size', type=int, default=50, help='每类当前合成数据大小')
    parser.add_argument('--save_cls_embeds', action='store_true', help='保存类别嵌入')
    
    # FedCL框架设置
    parser.add_argument('--mode', type=str, default="pseudo", help='执行模式: local, pseudo, federation')
    parser.add_argument('--device', type=str, default="cuda", help='设备')
    parser.add_argument('--log_level', type=str, default="INFO", help='日志级别')
    
    args = parser.parse_args()
    return args


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dddr_config(args):
    """创建DDDR配置"""
    # 计算类别信息
    num_class = 200 if args.dataset == "tiny_imagenet" else 100
    init_cls = int(num_class / args.tasks)
    increment = init_cls
    
    config = {
        # 基础配置（完全对齐ours.py的args超参数）
        "num_clients": args.num_users,
        "num_tasks": args.tasks,
        "classes_per_task": increment,
        "total_classes": num_class,
        "com_round": args.com_round,  # 对应ours.py的com_round
        "com_round_gen": args.com_round_gen,  # 对应ours.py的com_round_gen
        "local_ep": args.local_ep,  # 对应ours.py的local_ep
        "frac": args.frac,  # 对应ours.py的frac
        "num_users": args.num_users,  # 对应ours.py的num_users
        "batch_size": args.local_bs,
        "seed": args.seed,
        
        # 网络配置
        "base_model": args.net,
        "feature_dim": 512,
        "proj_dim": 256,
        
        # DDDR特定配置（完全对齐ours.py）
        "w_kd": args.w_kd,  # 对应ours.py的w_kd
        "w_ce_pre": args.w_ce_pre,  # 对应ours.py的w_ce_pre
        "w_scl": args.w_scl,  # 对应ours.py的w_scl
        "pre_size": args.pre_size,  # 对应ours.py的pre_size
        "cur_size": args.cur_size,  # 对应ours.py的cur_size
        "n_iter": args.n_iter,  # 对应ours.py的n_iter
        "g_local_train_steps": args.g_local_train_steps,  # 对应ours.py的g_local_train_steps
        "g_sigma": args.g_sigma,  # 对应ours.py的g_sigma
        "classifer_dp": args.classifier_dp,  # 对应ours.py的classifer_dp（注意拼写）
        "save_cls_embeds": args.save_cls_embeds,  # 对应ours.py的save_cls_embeds
        
        # 扩散模型配置
        "ldm_config": args.config,
        "ldm_ckpt": args.ldm_ckpt,
        "syn_image_path": args.syn_image_path,
        
        # 数据集配置
        "dataset": args.dataset,
        "beta": args.beta,
        "nums": args.nums,
        
        # FedCL框架配置
        "mode": args.mode,
        "device": args.device,
        "log_level": args.log_level,
        
        # 优化器配置
        "optimizer": {
            "type": "sgd",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 5e-4
        },
        
        # 保存配置
        "save_dir": args.save_dir,
        "exp_name": args.exp_name
    }
    
    return config


def check_dependencies(args):
    """检查依赖"""
    print("检查DDDR依赖...")
    
    if not DDDR_AVAILABLE:
        print("⚠️ DDDR模块不可用，跳过依赖检查")
        return True
    
    # 检查CLIP依赖
    try:
        import clip
        print("✓ CLIP模块已安装")
    except ImportError:
        print("✗ CLIP模块未安装，请运行: pip install git+https://github.com/openai/CLIP.git")
        print("将继续运行，但生成器功能可能受限")
    
    # 检查预训练模型（如果提供了路径）
    config_path = args.config
    ckpt_path = args.ldm_ckpt
    
    if config_path != "config/ldm_dddr.yaml" and not os.path.exists(config_path):
        print(f"⚠️ LDM配置文件不存在: {config_path}")
        print("将使用默认设置")
    
    if ckpt_path != "PM/ldm/text2img-large/model.ckpt" and not os.path.exists(ckpt_path):
        print(f"⚠️ LDM检查点不存在: {ckpt_path}")
        print("将跳过图像生成")
    
    print("✓ 依赖检查完成")
    return True


def create_data_manager(config):
    """创建数据管理器 - 参考ours.py的init_dataloader方法"""
    print("创建数据管理器...")
    
    # 这里应该实现数据划分逻辑
    # 参考ours.py的init_dataloader方法，使用Dirichlet分布进行非IID数据划分
    
    dataset_name = config.get("dataset", "cifar100")
    num_clients = config.get("num_clients", 10)
    beta = config.get("beta", 0.5)  # Dirichlet分布的alpha参数
    seed = config.get("seed", 2024)
    
    print(f"数据集: {dataset_name}")
    print(f"客户端数: {num_clients}")
    print(f"数据倾斜程度: {beta}")
    
    # 注意：实际的数据划分应该在FedCL框架的partition.py中实现
    # 这里只是示例，实际使用时用户需要提供自己的数据管理器
    
    class DummyDataManager:
        def __init__(self, dataset_name, num_clients, beta, seed):
            self.dataset_name = dataset_name
            self.num_clients = num_clients
            self.beta = beta
            self.seed = seed
            
        def get_client_data(self, client_id):
            """获取客户端数据（示例实现）"""
            # 这里应该返回客户端的数据加载器
            # 实际实现需要根据partition文件加载对应的数据
            return None
            
        def get_test_data(self):
            """获取测试数据（示例实现）"""
            # 这里应该返回测试数据加载器
            return None
    
    return DummyDataManager(dataset_name, num_clients, beta, seed)

def main():
    """主函数"""
    print("=" * 60)
    print("DDDR (Diffusion-Driven Data Replay) 联邦持续学习")
    print("基于FedCL框架实现")
    print("=" * 60)
    
    # 解析参数
    args = args_parser()
    
    # 设置实验名称
    if args.exp_name == "":
        args.exp_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    args.exp_name = f"beta_{args.beta}_tasks_{args.tasks}_seed_{args.seed}_sigma_{args.g_sigma}_{args.exp_name}"
    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, args.exp_name)
    
    print(f"实验名称: {args.exp_name}")
    print(f"保存目录: {args.save_dir}")
    
    # 检查依赖
    if not check_dependencies(args):
        print("依赖检查失败，退出")
        return
    
    # 设置随机种子
    setup_seed(args.seed)
    print(f"随机种子设置: {args.seed}")
    
    # 创建配置
    config = create_dddr_config(args)
    print("配置创建完成")
    

    print("使用FedCL框架执行DDDR联邦训练...")
    try:
        # 使用FedCL的标准入口函数
        from fedcl import train
        
        # 执行训练
        result = train(
            learner="dddr",
            trainer="dddr", 
            dataset="cifar100",
            num_clients=config["num_users"],
            num_rounds=config["com_round"],
            local_epochs=config["local_ep"],
            batch_size=config["batch_size"],
            device=config["device"],
            # LDM配置
            ldm_config="config/ldm_dddr.yaml",
            ldm_ckpt="PM/ldm/text2img-large/model.ckpt",
            # DDDR特定配置
            dddr_config={
                "kd_weight": 10.0,
                "syn_weight": 0.5,
                "contrastive_weight": 1.0,
                "com_round_gen": config["com_round_gen"],
                "g_local_train_steps": config["g_local_train_steps"],
                "history_size": config["pre_size"],
                "current_size": config["cur_size"]
            }
        )
        print("✓ DDDR联邦训练完成")
        print(result)
        
    except Exception as e:
        logger.exception(e)
    
    


if __name__ == '__main__':
    main()
