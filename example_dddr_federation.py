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
from omegaconf import OmegaConf

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入FedCL框架组件
from fedcl import registry
from fedcl.methods.learners.dddr import DDDRLearner
from fedcl.methods.trainers.dddr_federation_trainer import DDDRFederationTrainer
from fedcl.models.transforms import get_train_transform, SUPPORTED_DATASETS


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
    parser.add_argument('--config', type=str, default="ldm/ldm_dddr.yaml", help='扩散模型配置')
    parser.add_argument('--ldm_ckpt', type=str, default="models/ldm/text2img-large/model.ckpt", help='LDM检查点路径')
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
        # 基础配置
        "num_clients": args.num_users,
        "num_tasks": args.tasks,
        "classes_per_task": increment,
        "total_classes": num_class,
        "com_rounds": args.com_round,
        "local_epochs": args.local_ep,
        "batch_size": args.local_bs,
        "frac": args.frac,
        "seed": args.seed,
        
        # 网络配置
        "base_model": args.net,
        "feature_dim": 512,
        "proj_dim": 256,
        
        # DDDR特定配置
        "w_kd": args.w_kd,
        "w_ce_pre": args.w_ce_pre,
        "w_scl": args.w_scl,
        "pre_size": args.pre_size,
        "cur_size": args.cur_size,
        "n_iter": args.n_iter,
        "com_rounds_gen": args.com_round_gen,
        "g_local_train_steps": args.g_local_train_steps,
        "g_local_bs": args.g_local_bs,
        "g_sigma": args.g_sigma,
        
        # 扩散模型配置
        "ldm_config": args.config,
        "ldm_ckpt": args.ldm_ckpt,
        "syn_image_path": args.syn_image_path,
        "save_cls_embeds": args.save_cls_embeds,
        
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


def check_dependencies():
    """检查依赖"""
    print("检查DDDR依赖...")
    
    # 检查CLIP依赖
    try:
        import clip
        print("✓ CLIP模块已安装")
    except ImportError:
        print("✗ CLIP模块未安装，请运行: pip install git+https://github.com/openai/CLIP.git")
        return False
    
    # 检查预训练模型
    config_path = "ldm/ldm_dddr.yaml"
    ckpt_path = "models/ldm/text2img-large/model.ckpt"
    
    if not os.path.exists(config_path):
        print(f"✗ LDM配置文件不存在: {config_path}")
        print("请从DDDR项目复制配置文件")
        return False
    
    if not os.path.exists(ckpt_path):
        print(f"✗ LDM检查点不存在: {ckpt_path}")
        print("请下载预训练模型")
        return False
    
    print("✓ 所有依赖检查通过")
    return True


def create_data_manager(config):
    """创建数据管理器"""
    # 这里应该根据实际的数据管理器实现
    # 暂时返回None，实际使用时需要实现
    print("注意: 需要实现数据管理器")
    return None


async def main():
    """主函数"""
    print("=" * 60)
    print("DDDR (Diffusion-Driven Data Replay) 联邦学习")
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
    if not check_dependencies():
        print("依赖检查失败，退出")
        return
    
    # 设置随机种子
    setup_seed(args.seed)
    print(f"随机种子设置: {args.seed}")
    
    
    # 创建配置
    config = create_dddr_config(args)
    print("配置创建完成")
    
    # 创建数据管理器
    data_manager = create_data_manager(config)
    
    # 创建联邦训练器
    print("创建DDDR联邦训练器...")
    trainer = DDDRFederationTrainer(config)
    
    # 设置数据管理器
    if data_manager is not None:
        trainer.set_data_manager(data_manager)
    
    # 检查合成数据路径
    if args.syn_image_path is not None:
        print(f"使用预生成合成数据: {args.syn_image_path}")
        config['syn_image_path'] = args.syn_image_path
    else:
        print("将动态生成合成数据")
    
    # 显示训练信息
    print("\n训练配置:")
    print(f"  数据集: {args.dataset}")
    print(f"  任务数: {args.tasks}")
    print(f"  客户端数: {args.num_users}")
    print(f"  通信轮数: {args.com_round}")
    print(f"  本地轮数: {args.local_ep}")
    print(f"  批处理大小: {args.local_bs}")
    print(f"  执行模式: {args.mode}")
    
    print("\nDDDR配置:")
    print(f"  知识蒸馏权重: {args.w_kd}")
    print(f"  合成数据权重: {args.w_ce_pre}")
    print(f"  对比学习权重: {args.w_scl}")
    print(f"  生成器通信轮数: {args.com_round_gen}")
    print(f"  生成器训练步数: {args.g_local_train_steps}")
    print(f"  历史数据大小: {args.pre_size}")
    print(f"  当前数据大小: {args.cur_size}")
    
    # 开始训练
    print("\n开始DDDR联邦训练...")
    start_time = time.time()
    
    try:
        # 执行训练
        result = await trainer.train()
        
        # 显示结果
        training_time = time.time() - start_time
        print(f"\n训练完成!")
        print(f"总训练时间: {training_time:.2f}秒")
        print(f"最终准确率: {result['final_accuracy']:.4f}")
        print(f"总任务数: {result['total_tasks']}")
        print(f"总轮数: {result['total_rounds']}")
        
        # 保存结果
        result_file = os.path.join(args.save_dir, "training_result.json")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        import json
        with open(result_file, 'w') as f:
            json.dump({
                "config": config,
                "result": result,
                "training_time": training_time
            }, f, indent=2)
        
        print(f"结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("DDDR联邦学习完成")
    print("=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
