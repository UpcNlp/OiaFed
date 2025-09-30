#!/usr/bin/env python3
"""
测试DDDR类反演和图像生成过程
"""

import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append('/home/nlp/ct/projects/MOE-FedCL')

def test_diffusion_model_init():
    """测试扩散模型初始化"""
    print("=" * 50)
    print("测试扩散模型初始化")
    print("=" * 50)
    
    # 检查配置文件
    config_path = "config/ldm_dddr.yaml"
    ckpt_path = "PM/ldm/text2img-large/model.ckpt"
    
    print(f"配置文件路径: {config_path}")
    print(f"检查点路径: {ckpt_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 检查点文件不存在: {ckpt_path}")
        return False
    
    print("✅ 配置文件存在")
    print("✅ 检查点文件存在")
    
    try:
        # 加载配置
        config = OmegaConf.load(config_path)
        print("✅ 配置加载成功")
        
        # 测试LDM模块导入
        try:
            from fedcl.models.ldm import LatentDiffusion
            print("✅ LatentDiffusion模块导入成功")
        except Exception as e:
            print(f"❌ LatentDiffusion模块导入失败: {e}")
            return False
        
        # 尝试创建模型
        config.model.params.ckpt_path = ckpt_path
        config['model']["params"]['personalization_config']["params"]['num_classes'] = 20
        
        print("尝试创建LatentDiffusion模型...")
        generator = LatentDiffusion(**config['model']["params"])
        print("✅ LatentDiffusion模型创建成功")
        
        # 尝试加载检查点
        print("尝试加载检查点...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # 过滤掉不匹配的键
        model_state_dict = generator.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                skipped_keys.append(key)
        
        print(f"✅ 过滤后加载 {len(filtered_state_dict)} 个匹配的键")
        print(f"⚠️  跳过 {len(skipped_keys)} 个不匹配的键")
        
        # 加载匹配的权重
        missing_keys, unexpected_keys = generator.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  缺失的键: {len(missing_keys)}")
            for key in missing_keys[:3]:  # 只显示前3个
                print(f"   缺失: {key}")
        
        if unexpected_keys:
            print(f"⚠️  意外的键: {len(unexpected_keys)}")
            for key in unexpected_keys[:3]:  # 只显示前3个
                print(f"   意外: {key}")
        
        print("✅ 检查点加载成功")
        
        # 检查嵌入管理器
        if hasattr(generator, 'embedding_manager'):
            print("✅ 嵌入管理器存在")
            print(f"   嵌入管理器状态: {list(generator.embedding_manager.state_dict().keys())}")
        else:
            print("❌ 嵌入管理器不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 扩散模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_inversion():
    """测试类反演过程"""
    print("\n" + "=" * 50)
    print("测试类反演过程")
    print("=" * 50)
    
    try:
        from fedcl.methods.trainers.dddr_federation_trainer import DDDRFederationTrainer
        
        # 创建配置
        config = {
            "num_tasks": 1,
            "classes_per_task": 20,
            "total_classes": 100,
            "num_clients": 2,
            "com_rounds": 5,
            "local_epochs": 2,
            "batch_size": 32,
            "ldm_config": "config/ldm_dddr.yaml",
            "ldm_ckpt": "PM/ldm/text2img-large/model.ckpt",
            "pre_size": 200,
            "cur_size": 50,
            "n_iter": 2,
            "com_rounds_gen": 2,
            "g_local_train_steps": 5,
            "w_kd": 10.0,
            "w_ce_pre": 0.5,
            "w_scl": 1.0,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "frac": 1.0,
            "g_sigma": 0.0
        }
        
        print("创建DDDR联邦训练器...")
        trainer = DDDRFederationTrainer(config)
        
        # 检查扩散模型是否初始化
        if trainer._generator is not None:
            print("✅ 扩散模型初始化成功")
            
            # 测试类反演
            task_id = 0
            class_ids = [0, 1, 2, 3, 4]  # 前5个类别
            
            print(f"开始测试类反演，任务 {task_id}，类别 {class_ids}")
            
            # 创建模拟数据管理器
            trainer.data_manager = None
            
            # 创建学习器
            trainer.learners = []
            for i in range(config["num_clients"]):
                from fedcl.methods.learners.dddr import DDDRLearner
                learner_config = trainer._create_learner_config(f"client_{i}")
                learner = DDDRLearner(f"client_{i}", learner_config)
                trainer.learners.append(learner)
            
            # 执行类反演
            inv_text_embeds = trainer._federated_class_inversion()
            
            if inv_text_embeds is not None:
                print("✅ 类反演成功")
                print(f"   嵌入字典键: {list(inv_text_embeds.keys())}")
                return True
            else:
                print("❌ 类反演失败")
                return False
        else:
            print("❌ 扩散模型初始化失败")
            return False
            
    except Exception as e:
        print(f"❌ 类反演测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_generation():
    """测试图像生成过程"""
    print("\n" + "=" * 50)
    print("测试图像生成过程")
    print("=" * 50)
    
    try:
        from fedcl.methods.trainers.dddr_federation_trainer import DDDRFederationTrainer
        
        # 创建配置
        config = {
            "num_tasks": 1,
            "classes_per_task": 20,
            "total_classes": 100,
            "num_clients": 2,
            "com_rounds": 5,
            "local_epochs": 2,
            "batch_size": 32,
            "ldm_config": "config/ldm_dddr.yaml",
            "ldm_ckpt": "PM/ldm/text2img-large/model.ckpt",
            "pre_size": 200,
            "cur_size": 50,
            "n_iter": 1,  # 减少迭代次数用于测试
            "com_rounds_gen": 1,
            "g_local_train_steps": 2,
            "w_kd": 10.0,
            "w_ce_pre": 0.5,
            "w_scl": 1.0,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "frac": 1.0,
            "g_sigma": 0.0
        }
        
        print("创建DDDR联邦训练器...")
        trainer = DDDRFederationTrainer(config)
        
        if trainer._generator is not None:
            print("✅ 扩散模型初始化成功")
            
            # 测试图像生成
            task_id = 0
            class_ids = [0, 1]  # 只测试2个类别
            
            print(f"开始测试图像生成，任务 {task_id}，类别 {class_ids}")
            
            # 创建模拟嵌入
            inv_text_embeds = {}
            for class_id in class_ids:
                inv_text_embeds[f"*_{class_id}"] = torch.randn(1280)  # 模拟嵌入
            
            # 执行图像生成
            trainer._synthesis_images(inv_text_embeds)
            
            # 检查生成的文件
            outdir = os.path.join("syn_imgs", f"task_{task_id}")
            if os.path.exists(outdir):
                print(f"✅ 图像生成目录创建成功: {outdir}")
                
                # 检查每个类别的图像
                for class_id in class_ids:
                    class_dir = os.path.join(outdir, str(class_id))
                    if os.path.exists(class_dir):
                        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
                        print(f"   类别 {class_id}: {len(image_files)} 张图像")
                    else:
                        print(f"   ❌ 类别 {class_id} 目录不存在")
                
                return True
            else:
                print(f"❌ 图像生成目录不存在: {outdir}")
                return False
        else:
            print("❌ 扩散模型初始化失败")
            return False
            
    except Exception as e:
        print(f"❌ 图像生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("DDDR类反演和图像生成测试")
    print("=" * 60)
    
    # 测试1: 扩散模型初始化
    model_ok = test_diffusion_model_init()
    
    if model_ok:
        # 测试2: 类反演
        inversion_ok = test_class_inversion()
        
        # 测试3: 图像生成
        generation_ok = test_image_generation()
        
        print("\n" + "=" * 60)
        print("测试结果总结:")
        print(f"扩散模型初始化: {'✅ 成功' if model_ok else '❌ 失败'}")
        print(f"类反演过程: {'✅ 成功' if inversion_ok else '❌ 失败'}")
        print(f"图像生成过程: {'✅ 成功' if generation_ok else '❌ 失败'}")
        
        if model_ok and inversion_ok and generation_ok:
            print("\n🎉 所有测试通过！DDDR的类反演和图像生成功能正常工作。")
        else:
            print("\n⚠️  部分测试失败，请检查相关配置和依赖。")
    else:
        print("\n❌ 扩散模型初始化失败，无法进行后续测试。")

if __name__ == "__main__":
    main()
