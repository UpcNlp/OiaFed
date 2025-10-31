"""
验证Builder与配置系统集成
examples/validate_builder_integration.py

展示Builder如何与现有的CommunicationConfig和TrainingConfig配置系统集成
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fedcl.api import ComponentBuilder, get_builder
from fedcl.api.registry import registry
from loguru import logger

print("=" * 80)
print("🔍 验证Builder与配置系统集成")
print("=" * 80)
print()

# ============================================================================
# 1. 验证组件注册表
# ============================================================================
print("📋 步骤1: 检查已注册的组件")
print("-" * 80)

all_components = registry.list_all_components()

print(f"✅ 已注册的训练器 (Trainers): {all_components.get('trainers', [])}")
print(f"✅ 已注册的学习器 (Learners): {all_components.get('learners', [])}")
print(f"✅ 已注册的聚合器 (Aggregators): {all_components.get('aggregators', [])}")
print(f"✅ 已注册的数据集 (Datasets): {all_components.get('datasets', [])}")
print(f"✅ 已注册的模型 (Models): {all_components.get('models', [])}")
print()

# ============================================================================
# 2. 验证Builder创建组件（模拟配置驱动）
# ============================================================================
print("🔧 步骤2: 验证Builder从配置字典创建组件")
print("-" * 80)

# 模拟从YAML配置文件加载的字典结构
# 这个结构与TrainingConfig中的字典字段对应
simulated_config = {
    # 聚合器配置（字典格式，与TrainingConfig.aggregator对应）
    "aggregator": {
        "name": "FedAvg",      # 必需：用于从注册表查找
        "weighted": True        # 可选：传递给聚合器__init__
    }
}

print("📝 配置示例（与TrainingConfig兼容的字典格式）:")
print(f"  aggregator:")
print(f"    name: '{simulated_config['aggregator']['name']}'")
print(f"    weighted: {simulated_config['aggregator']['weighted']}")
print()

# 使用Builder从配置创建组件
builder = ComponentBuilder()

try:
    # 方式1: 使用build_from_config（推荐）
    print("✨ 方式1: 使用 build_from_config() 创建组件")
    components = builder.build_from_config(simulated_config)

    if "aggregator" in components:
        aggregator = components["aggregator"]
        print(f"  ✅ 成功创建聚合器: {aggregator.__class__.__name__}")
        print(f"  ✅ 聚合器类型: {type(aggregator)}")
    print()

    # 方式2: 直接使用build_aggregator（适合单个组件）
    print("✨ 方式2: 使用 build_aggregator() 创建单个组件")
    aggregator2 = builder.build_aggregator(
        simulated_config["aggregator"]["name"],
        **{k: v for k, v in simulated_config["aggregator"].items() if k != "name"}
    )
    print(f"  ✅ 成功创建聚合器: {aggregator2.__class__.__name__}")
    print()

    print("✅ Builder与配置系统集成验证成功！")
    print()

except Exception as e:
    print(f"❌ 创建组件失败: {e}")
    import traceback
    traceback.print_exc()
    print()

# ============================================================================
# 3. 验证配置文件加载（如果存在）
# ============================================================================
print("📄 步骤3: 验证配置文件加载")
print("-" * 80)

config_path = Path(__file__).parent.parent / "configs" / "experiments" / "fedavg_mnist.yaml"

if config_path.exists():
    try:
        # 加载配置文件
        config_dict = builder.load_config(config_path)
        print(f"✅ 成功加载配置文件: {config_path.name}")

        # 显示配置结构
        print(f"\n📋 配置文件包含的组件:")
        for key in ['trainer', 'aggregator', 'learner', 'dataset', 'model']:
            if key in config_dict:
                component_config = config_dict[key]
                component_name = component_config.get('name', 'N/A')
                print(f"  • {key}: {component_name}")
        print()

        # 验证可以从配置创建组件（不实际创建，只验证结构）
        print("🔍 验证配置结构:")
        required_components = ['trainer', 'aggregator', 'learner']
        for component_type in required_components:
            if component_type in config_dict:
                component_cfg = config_dict[component_type]
                if 'name' in component_cfg:
                    print(f"  ✅ {component_type} 配置正确（包含 'name' 字段）")
                else:
                    print(f"  ⚠️  {component_type} 配置缺少 'name' 字段")
        print()

    except Exception as e:
        print(f"⚠️  加载配置文件时出错: {e}")
        print()
else:
    print(f"⚠️  配置文件不存在: {config_path}")
    print("   这是正常的，如果您还没有创建实验配置文件")
    print()

# ============================================================================
# 4. 使用总结
# ============================================================================
print("=" * 80)
print("📚 使用总结")
print("=" * 80)
print("""
✅ Builder模式已成功集成到现有配置系统

🔑 关键要点:

1. **配置格式（与TrainingConfig兼容）**
   training:
     aggregator:
       name: "FedAvg"      # 必需：从注册表查找组件类
       weighted: true       # 可选：传递给组件__init__

2. **使用Builder创建组件**
   # 方式1: 批量创建
   components = builder.build_from_config(config_dict)

   # 方式2: 单独创建
   aggregator = builder.build_aggregator("FedAvg", weighted=True)

3. **灵活性**
   • 不同组件需要不同参数 → 使用字典配置，除name外的字段完全自定义
   • 用户自定义组件 → 使用装饰器注册，然后在配置中引用name

4. **三种使用方式**
   • 配置文件驱动 → 批量实验，团队协作
   • 代码+Builder → 快速原型，灵活控制
   • 完全自定义 → 研究创新，完全控制

📖 参考文档:
   • 架构设计: docs/architecture_design.md
   • 配置指南: docs/configuration_guide.md
   • Builder API: fedcl/api/builder.py
""")

print("=" * 80)
print("✅ 验证完成！")
print("=" * 80)
