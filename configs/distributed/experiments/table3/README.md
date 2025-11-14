# TABLE III 实验配置说明

## 配置文件列表

- `server.yaml` - 服务器配置
- `client_0.yaml` ~ `client_9.yaml` - 10个客户端配置

## 自定义命名支持

现在支持使用**任意名称**作为 `node_id`，不再强制要求 `client_X` 或 `server_X` 格式！

### 客户端自定义命名示例

```yaml
# 可以使用任意名称作为 node_id
node_id: "hospital_beijing"  # ✓ 医院名称
node_id: "mobile_device_001"  # ✓ 移动设备ID
node_id: "edge_node_shanghai" # ✓ 边缘节点名称
node_id: "client_0"           # ✓ 传统格式也可以

# 关键：必须明确指定 client_index（0 到 num_clients-1）
training:
  learner:
    params:
      client_index: 0  # 必须指定！范围: 0 到 num_clients-1
      batch_size: 64
      local_epochs: 10
  dataset:
    partition:
      num_clients: 10  # 必须匹配实际客户端总数
```

### 服务器自定义命名示例

```yaml
# 服务器也可以使用任意名称
node_id: "central_coordinator"  # ✓ 中心协调器
node_id: "aggregation_server"   # ✓ 聚合服务器
node_id: "main_hub"            # ✓ 主节点
node_id: "server_1"            # ✓ 传统格式

training:
  max_rounds: 50
  min_clients: 10  # 应该匹配实际客户端数量
```

## 关键配置要点

### 1. 客户端配置

- ✅ **node_id**: 任意唯一字符串
- ✅ **client_index**: 必须明确指定（0 到 num_clients-1）
- ✅ **partition.num_clients**: 所有客户端必须一致
- ⚠️ 如果不指定 `client_index`，系统会尝试从 `node_id` 中解析数字

### 2. 服务器配置

- ✅ **node_id**: 任意唯一字符串
- ✅ **min_clients**: 应匹配实际客户端数量
- ✅ **max_rounds**: 训练轮数

### 3. 配置一致性检查

| 配置项 | 位置 | 说明 |
|--------|------|------|
| `num_clients` | 每个客户端的 `dataset.partition.num_clients` | 必须都是10 |
| `min_clients` | 服务器的 `training.min_clients` | 应该是10 |
| `client_index` | 每个客户端的 `learner.params.client_index` | 0~9，不重复 |

## 完整示例：医院联邦学习场景

### 服务器配置 (central_hospital.yaml)
```yaml
extends: "../../base/server_base.yaml"
node_id: "central_hospital"

training:
  max_rounds: 50
  min_clients: 3
  trainer:
    name: "Generic"
```

### 客户端配置 (hospital_A.yaml)
```yaml
extends: "../../base/client_base.yaml"
node_id: "hospital_beijing"

training:
  learner:
    params:
      client_index: 0  # 北京医院 - 索引0
      batch_size: 64
  dataset:
    partition:
      num_clients: 3
```

### 客户端配置 (hospital_B.yaml)
```yaml
extends: "../../base/client_base.yaml"
node_id: "hospital_shanghai"

training:
  learner:
    params:
      client_index: 1  # 上海医院 - 索引1
      batch_size: 64
  dataset:
    partition:
      num_clients: 3
```

### 客户端配置 (hospital_C.yaml)
```yaml
extends: "../../base/client_base.yaml"
node_id: "hospital_guangzhou"

training:
  learner:
    params:
      client_index: 2  # 广州医院 - 索引2
      batch_size: 64
  dataset:
    partition:
      num_clients: 3
```

## 故障排查

### 问题1: "索引X超出范围[0, Y)"

**原因**: `partition.num_clients` 配置不一致

**解决**: 确保所有客户端配置中的 `dataset.partition.num_clients` 都是相同的值

### 问题2: "无法从client_id中提取客户端索引"

**原因**: 使用了自定义 `node_id`，但没有指定 `client_index`

**解决**: 在配置中添加 `training.learner.params.client_index`

### 问题3: 客户端重复索引

**原因**: 多个客户端使用了相同的 `client_index`

**解决**: 确保每个客户端的 `client_index` 都是唯一的（0 到 num_clients-1）
