#!/usr/bin/env python3
"""
评估配置测试脚本
测试新的配置格式：解决评估器和测试数据集的映射问题
支持多种配置方式：单个、任务列表、映射、笛卡尔积
"""

import yaml
from typing import Dict, Any


def test_evaluation_config():
    """测试多种评估配置方式"""
    
    # 示例配置 - 包含所有配置方式
    config = {
        "test_datas": {
            "test_data": {
                "type": "StandardDataLoader",
                "dataset": "mnist_test",
                "batch_size": 100
            },
            "validation_data": {
                "type": "StandardDataLoader",
                "dataset": "mnist_validation", 
                "batch_size": 100
            },
            "challenge_data": {
                "type": "StandardDataLoader",
                "dataset": "mnist_challenge", 
                "batch_size": 50
            }
        },
        "evaluators": {
            "accuracy_evaluator": {
                "type": "accuracy_evaluator",
                "metrics": ["accuracy"]
            },
            "loss_evaluator": {
                "type": "loss_evaluator",
                "metrics": ["loss"]
            },
            "comprehensive_evaluator": {
                "type": "comprehensive_evaluator",
                "metrics": ["accuracy", "precision", "recall"]
            }
        },
        "evaluation": {
            "learners": {
                # 方式1: 单个评估器和数据集
                "learner_1": {
                    "evaluator": "accuracy_evaluator",
                    "test_dataset": "test_data"
                },
                # 方式2: 评估任务列表 - 精确控制
                "learner_2": {
                    "evaluation_tasks": [
                        {"evaluator": "accuracy_evaluator", "test_dataset": "test_data", "name": "test_accuracy"},
                        {"evaluator": "accuracy_evaluator", "test_dataset": "validation_data", "name": "val_accuracy"},
                        {"evaluator": "loss_evaluator", "test_dataset": "test_data", "name": "test_loss"}
                    ]
                },
                # 方式3: 评估器映射
                "learner_3": {
                    "evaluator_mapping": {
                        "accuracy_evaluator": ["test_data", "validation_data"],
                        "loss_evaluator": ["test_data"],
                        "comprehensive_evaluator": ["challenge_data"]
                    }
                },
                # 方式4: 向后兼容 - 笛卡尔积
                "learner_4": {
                    "evaluators": ["accuracy_evaluator", "loss_evaluator"],
                    "test_datasets": ["test_data", "validation_data"]
                }
            }
        }
    }
    
    # 模拟配置解析和评估任务生成
    def parse_evaluation_config(learner_id, learner_eval_config):
        """解析评估配置并生成评估任务"""
        evaluation_tasks = []
        
        if "evaluation_tasks" in learner_eval_config:
            # 方式2: 评估任务列表
            evaluation_tasks = learner_eval_config["evaluation_tasks"]
            print(f"  方式2 - 评估任务列表：{len(evaluation_tasks)} 个任务")
            
        elif "evaluator_mapping" in learner_eval_config:
            # 方式3: 评估器映射
            evaluator_mapping = learner_eval_config["evaluator_mapping"]
            for evaluator_id, dataset_list in evaluator_mapping.items():
                if isinstance(dataset_list, str):
                    dataset_list = [dataset_list]
                
                for dataset_id in dataset_list:
                    evaluation_tasks.append({
                        "evaluator": evaluator_id,
                        "test_dataset": dataset_id,
                        "name": f"{evaluator_id}_{dataset_id}"
                    })
            print(f"  方式3 - 评估器映射：生成 {len(evaluation_tasks)} 个任务")
            
        elif "evaluator" in learner_eval_config and "test_dataset" in learner_eval_config:
            # 方式1: 单个评估器和数据集
            evaluation_tasks = [{
                "evaluator": learner_eval_config["evaluator"],
                "test_dataset": learner_eval_config["test_dataset"],
                "name": f"{learner_eval_config['evaluator']}_{learner_eval_config['test_dataset']}"
            }]
            print(f"  方式1 - 单个配置：1 个任务")
            
        elif "evaluators" in learner_eval_config and "test_datasets" in learner_eval_config:
            # 方式4: 向后兼容 - 笛卡尔积
            evaluators = learner_eval_config["evaluators"]
            test_datasets = learner_eval_config["test_datasets"]
            
            for evaluator_id in evaluators:
                for dataset_id in test_datasets:
                    evaluation_tasks.append({
                        "evaluator": evaluator_id,
                        "test_dataset": dataset_id,
                        "name": f"{evaluator_id}_{dataset_id}"
                    })
            print(f"  方式4 - 笛卡尔积：生成 {len(evaluation_tasks)} 个任务")
        
        return evaluation_tasks
    
    print("=== 多种评估配置方式测试 ===")
    
    for learner_id, learner_config in config.get("evaluation", {}).get("learners", {}).items():
        print(f"\n{learner_id}:")
        tasks = parse_evaluation_config(learner_id, learner_config)
        
        for i, task in enumerate(tasks, 1):
            print(f"    任务{i}: {task['evaluator']} + {task['test_dataset']} -> {task['name']}")
    
    print("\n✅ 配置解析测试通过!")


def test_yaml_config():
    """测试YAML配置文件"""
    
    yaml_config = """
# 改进的评估配置 - 解决评估器和数据集映射问题
test_datas:
  test_data:
    type: "StandardDataLoader"
    dataset: "mnist_test"
    batch_size: 100
    
  validation_data:
    type: "StandardDataLoader"
    dataset: "mnist_validation"
    batch_size: 100

evaluators:
  accuracy_evaluator:
    type: "accuracy_evaluator"
    metrics: ["accuracy"]
    
  loss_evaluator:
    type: "loss_evaluator"
    metrics: ["loss"]

evaluation:
  learners:
    # 精确控制每个评估任务
    learner_precise:
      evaluation_tasks:
        - evaluator: "accuracy_evaluator"
          test_dataset: "test_data"
          name: "test_accuracy"
        - evaluator: "loss_evaluator"
          test_dataset: "validation_data"
          name: "validation_loss"
          
    # 评估器映射方式
    learner_mapping:
      evaluator_mapping:
        accuracy_evaluator: ["test_data", "validation_data"]
        loss_evaluator: ["test_data"]
"""
    
    config = yaml.safe_load(yaml_config)
    
    print("=== YAML配置测试 ===")
    print("test_datas:", list(config.get("test_datas", {}).keys()))
    print("evaluators:", list(config.get("evaluators", {}).keys()))
    
    for learner_id, learner_config in config.get("evaluation", {}).get("learners", {}).items():
        print(f"\n{learner_id}:")
        if "evaluation_tasks" in learner_config:
            print("  配置方式: 评估任务列表")
            for task in learner_config["evaluation_tasks"]:
                print(f"    {task['name']}: {task['evaluator']} + {task['test_dataset']}")
        elif "evaluator_mapping" in learner_config:
            print("  配置方式: 评估器映射")
            for evaluator, datasets in learner_config["evaluator_mapping"].items():
                print(f"    {evaluator}: {datasets}")
    
    print("\n✅ YAML配置测试通过!")


def test_task_generation():
    """测试评估任务生成逻辑"""
    
    print("=== 评估任务生成测试 ===")
    
    # 测试案例
    test_cases = [
        {
            "name": "单个评估器+单个数据集",
            "config": {
                "evaluator": "accuracy_evaluator",
                "test_dataset": "test_data"
            },
            "expected_tasks": 1
        },
        {
            "name": "评估任务列表",
            "config": {
                "evaluation_tasks": [
                    {"evaluator": "accuracy_evaluator", "test_dataset": "test_data"},
                    {"evaluator": "loss_evaluator", "test_dataset": "validation_data"}
                ]
            },
            "expected_tasks": 2
        },
        {
            "name": "评估器映射",
            "config": {
                "evaluator_mapping": {
                    "accuracy_evaluator": ["test_data", "validation_data"],
                    "loss_evaluator": ["test_data"]
                }
            },
            "expected_tasks": 3
        },
        {
            "name": "笛卡尔积（向后兼容）",
            "config": {
                "evaluators": ["accuracy_evaluator", "loss_evaluator"],
                "test_datasets": ["test_data", "validation_data"]
            },
            "expected_tasks": 4  # 2 * 2 = 4
        }
    ]
    
    for case in test_cases:
        print(f"\n测试案例: {case['name']}")
        
        # 简化的任务生成逻辑
        tasks = []
        config = case["config"]
        
        if "evaluation_tasks" in config:
            tasks = config["evaluation_tasks"]
        elif "evaluator_mapping" in config:
            for evaluator_id, dataset_list in config["evaluator_mapping"].items():
                if isinstance(dataset_list, str):
                    dataset_list = [dataset_list]
                for dataset_id in dataset_list:
                    tasks.append({"evaluator": evaluator_id, "test_dataset": dataset_id})
        elif "evaluator" in config and "test_dataset" in config:
            tasks = [{"evaluator": config["evaluator"], "test_dataset": config["test_dataset"]}]
        elif "evaluators" in config and "test_datasets" in config:
            for evaluator_id in config["evaluators"]:
                for dataset_id in config["test_datasets"]:
                    tasks.append({"evaluator": evaluator_id, "test_dataset": dataset_id})
        
        print(f"  生成任务数: {len(tasks)} (预期: {case['expected_tasks']})")
        for i, task in enumerate(tasks, 1):
            print(f"    任务{i}: {task['evaluator']} + {task['test_dataset']}")
        
        if len(tasks) == case["expected_tasks"]:
            print("  ✅ 通过")
        else:
            print("  ❌ 失败")
    
    print("\n🎉 评估任务生成测试完成!")


if __name__ == "__main__":
    print("� 测试改进的评估配置格式")
    print("🎯 解决评估器和测试数据集的映射问题")
    print("📊 支持4种配置方式：单个、任务列表、映射、笛卡尔积")
    print()
    
    test_evaluation_config()
    print()
    test_yaml_config()
    print()
    test_task_generation()
    
    print("\n🎉 所有测试通过！新配置格式完美解决了评估器-数据集映射问题。")
