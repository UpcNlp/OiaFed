# client_index 自动分配方案

## 问题

`client_index` 必须是 **0 到 num_clients-1** 的连续整数，不能是随机数。

### 原因

```python
# 数据划分器的返回格式（固定键）
partition_result = {
    0: [样本1, 样本2, ...],   # 第0个客户端的数据
    1: [样本10, 样本11, ...],  # 第1个客户端的数据
    2: [样本20, 样本21, ...],  # 第2个客户端的数据
    ...
    9: [样本90, 样本91, ...]   # 第9个客户端的数据
}

# 使用 client_index 获取数据
my_data = partition_result[client_index]  # 必须是 0-9 之间的整数
```

如果使用随机数（如 `client_index: 123456789`），会导致 `KeyError`，因为字典中没有这个键。

---

## 解决方案1: 服务器端自动分配（推荐）

### 方案说明

1. 客户端使用**任意 node_id**（可以是UUID、名称等）
2. **不指定** `client_index`
3. 服务器在客户端注册时，按照**注册顺序**自动分配索引
4. 服务器将分配的索引发送回客户端
5. 客户端使用服务器分配的索引进行数据划分

### 配置示例

#### 客户端配置（无需指定 client_index）
```yaml
# hospital_A.yaml
node_id: "hospital_beijing_f3a9b2"  # 任意唯一标识

training:
  learner:
    params:
      # 不指定 client_index，等待服务器分配
      batch_size: 64
      local_epochs: 10
  dataset:
    partition:
      num_clients: 10
      # 客户端会在注册时从服务器获得 assigned_index
```

### 实现流程

```
1. 客户端 hospital_beijing_f3a9b2 连接服务器
2. 服务器: "你是第3个注册的客户端，分配 client_index=2"
3. 客户端: 收到 assigned_index=2，使用它划分数据
4. 客户端: 从 partition_result[2] 获取自己的训练数据
```

---

## 解决方案2: 配置文件映射表（当前推荐）

### 方案说明

创建一个映射配置文件，将任意 node_id 映射到 client_index。

### 实现方式

#### 主配置文件
```yaml
# configs/experiments/table3/client_mapping.yaml
client_index_mapping:
  # node_id → client_index
  "hospital_beijing_abc123": 0
  "hospital_shanghai_def456": 1
  "hospital_guangzhou_ghi789": 2
  "mobile_device_001": 3
  "mobile_device_002": 4
  "edge_node_alpha": 5
  "edge_node_beta": 6
  "iot_sensor_gamma": 7
  "cloud_worker_01": 8
  "cloud_worker_02": 9
```

#### 客户端配置引用映射
```yaml
# hospital_beijing.yaml
node_id: "hospital_beijing_abc123"

training:
  learner:
    params:
      # 从映射表自动获取
      client_index: "${client_index_mapping[hospital_beijing_abc123]}"
      batch_size: 64
  dataset:
    partition:
      num_clients: 10
```

---

## 解决方案3: 批量自动生成（最简单）

### 使用脚本自动生成配置

```python
# scripts/generate_client_configs.py
import yaml
import uuid

def generate_configs(num_clients=10, prefix="client"):
    """自动生成客户端配置"""

    for i in range(num_clients):
        # 生成唯一ID（可以是任意格式）
        unique_id = str(uuid.uuid4())[:8]
        node_id = f"{prefix}_{unique_id}"

        config = {
            'extends': '../../base/client_base.yaml',
            'node_id': node_id,  # 任意名称
            'training': {
                'learner': {
                    'params': {
                        'client_index': i,  # 自动分配索引
                        'batch_size': 64,
                        'local_epochs': 10
                    }
                },
                'dataset': {
                    'partition': {
                        'num_clients': num_clients,
                        'seed': 42
                    }
                }
            }
        }

        filename = f"configs/table3/client_{i}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(config, f)

        print(f"✓ 生成 {filename}: node_id={node_id}, client_index={i}")

if __name__ == '__main__':
    generate_configs(num_clients=10, prefix="hospital")
```

### 运行效果

```bash
$ python scripts/generate_client_configs.py

✓ 生成 client_0.yaml: node_id=hospital_a3f9b2c1, client_index=0
✓ 生成 client_1.yaml: node_id=hospital_d4e8f7a2, client_index=1
✓ 生成 client_2.yaml: node_id=hospital_b5c9a3d8, client_index=2
...
✓ 生成 client_9.yaml: node_id=hospital_e7f2b4c9, client_index=9
```

---

## 核心要点总结

### ✅ 可以做的
- ✅ `node_id` 可以是任意字符串（UUID、名称、编号等）
- ✅ 使用脚本自动生成配置（自动分配 client_index）
- ✅ 使用映射表管理 node_id → client_index 关系

### ❌ 不能做的
- ❌ `client_index` 不能是随机数或任意整数
- ❌ `client_index` 必须是 **0 到 num_clients-1** 的连续整数
- ❌ `client_index` 不能重复

### 为什么 client_index 必须连续？

```python
# Dirichlet 划分示例（label skew）
def partition(dataset, num_clients=10):
    # 预先生成 num_clients 份数据
    for client_idx in range(num_clients):  # 0, 1, 2, ..., 9
        client_data = sample_by_dirichlet(...)
        result[client_idx] = client_data

    return result  # {0: [...], 1: [...], ..., 9: [...]}

# 如果你的 client_index=123456，这里会 KeyError
my_data = result[123456]  # ❌ KeyError: 123456
```

---

## 建议方案（根据场景选择）

| 场景 | 推荐方案 | 优点 |
|------|----------|------|
| **开发测试** | 方案3 - 脚本生成 | 最简单快速 |
| **固定部署** | 方案2 - 映射表 | 便于管理和修改 |
| **动态环境** | 方案1 - 服务器分配 | 灵活，支持动态加入 |

---

## 实现方案1的步骤（未来改进）

如果要实现服务器自动分配，需要修改以下组件：

1. **服务器端**：`fedcl/federation/server.py`
   - 维护一个注册表：`{node_id: assigned_index}`
   - 在客户端注册时分配索引

2. **客户端端**：`fedcl/learner/base_learner.py`
   - 修改 `_extract_client_index` 方法
   - 如果配置中没有 `client_index`，向服务器请求分配

3. **注册协议**：
   ```python
   # 客户端注册请求
   {
       "action": "register",
       "node_id": "hospital_beijing_f3a9b2",
       "request_index_assignment": True
   }

   # 服务器响应
   {
       "status": "success",
       "assigned_index": 2,
       "total_clients": 10
   }
   ```

这个改进可以作为未来的功能增强！
