# Network 模式跨机器测试指南

## 📋 测试环境

| 角色 | IP 地址 | 操作系统 | 端口 |
|------|---------|----------|------|
| 服务端 | 192.168.31.68 (笔记本) | Windows | 8000, 9501 |
| 客户端1 | 192.168.31.75 (服务器1) | Linux | 8001, 9502 |
| 客户端2 | 192.168.31.166 (服务器2) | Linux | 8002, 9503 |

## 🚀 快速开始

### 第一步：准备服务端（笔记本 - 192.168.31.68）

#### 1.1 检查防火墙

**Windows 防火墙：**
```powershell
# 打开 PowerShell（管理员权限）

# 允许 8000 端口（HTTP）
netsh advfirewall firewall add rule name="FedCL Server HTTP" dir=in action=allow protocol=TCP localport=8000

# 允许 9501 端口（WebSocket）
netsh advfirewall firewall add rule name="FedCL Server WebSocket" dir=in action=allow protocol=TCP localport=9501
```

或者通过 Windows Defender 防火墙界面：
1. 搜索"Windows Defender 防火墙"
2. 点击"高级设置"
3. 点击"入站规则" -> "新建规则"
4. 选择"端口" -> "TCP" -> 输入"8000,9501"
5. 选择"允许连接"

#### 1.2 验证网络连通性

在笔记本上，确保可以被其他机器访问：
```powershell
# 查看本机 IP
ipconfig

# 应该看到：192.168.31.68
```

#### 1.3 启动服务端

```bash
# 进入项目目录
cd D:\PyCharm\MOE-FedCL

# 激活 conda 环境
conda activate your_env_name

# 运行服务端脚本（使用独立服务端脚本）
python examples/network_server_standalone.py
```

你应该看到：
```
============================================================
Network 模式测试 - 服务端
============================================================
服务端地址: 192.168.31.68:8000
等待客户端连接...
============================================================
配置模式: network
监听地址: 0.0.0.0:8000

✅ 服务端已启动: network_server_main
等待客户端注册...
```

---

### 第二步：准备客户端（Linux 服务器）

#### 2.1 同步代码到服务器

在**两台 Linux 服务器**上分别执行：

```bash
# 方法1：使用 git（推荐）
cd ~
git clone <your-repo-url> MOE-FedCL
cd MOE-FedCL

# 方法2：使用 scp 从笔记本传输
# 在笔记本上执行：
# scp -r D:\PyCharm\MOE-FedCL username@192.168.31.75:~/
# scp -r D:\PyCharm\MOE-FedCL username@192.168.31.166:~/
```

#### 2.2 安装依赖

在**两台服务器**上分别执行：

```bash
cd ~/MOE-FedCL

# 激活 conda 环境
conda activate your_env_name

# 验证依赖
pip list | grep -E "aiohttp|asyncio|pyyaml"
```

#### 2.3 检查防火墙（Linux）

在**两台服务器**上分别执行：

```bash
# 检查防火墙状态
sudo firewall-cmd --state
# 或
sudo ufw status

# 如果使用 firewalld：
sudo firewall-cmd --permanent --add-port=8001/tcp  # 服务器1
sudo firewall-cmd --permanent --add-port=9502/tcp
sudo firewall-cmd --reload

# 服务器2 使用不同端口：
# sudo firewall-cmd --permanent --add-port=8002/tcp
# sudo firewall-cmd --permanent --add-port=9503/tcp
# sudo firewall-cmd --reload

# 如果使用 ufw：
sudo ufw allow 8001/tcp  # 服务器1
sudo ufw allow 9502/tcp

# 服务器2：
# sudo ufw allow 8002/tcp
# sudo ufw allow 9503/tcp
```

#### 2.4 测试网络连通性

在**两台服务器**上分别测试与服务端的连接：

```bash
# 测试 HTTP 端口
curl http://192.168.31.68:8000/api/v1/health
# 或
telnet 192.168.31.68 8000

# 测试 WebSocket 端口
telnet 192.168.31.68 9501
```

如果连接失败，检查：
1. 服务端防火墙是否开放
2. 网络路由是否正常
3. 服务端程序是否已启动

---

### 第三步：启动客户端

#### 3.1 在服务器1 (192.168.31.75) 上启动客户端1

```bash
cd ~/MOE-FedCL

# 激活环境
conda activate your_env_name

# 启动客户端1（使用独立客户端脚本）
python examples/network_client_standalone.py \
    --config configs/network_test/client1.yaml \
    --server-ip 192.168.31.68 \
    --server-port 8000
```

你应该看到：
```
============================================================
Network 模式测试 - 客户端
============================================================
配置文件: configs/network_test/client1.yaml
服务端地址: 192.168.31.68:8000
============================================================
客户端ID: client_server1
监听端口: 8001

启动客户端...
✅ 客户端已启动: client_server1

等待服务端训练指令...
(按 Ctrl+C 停止)
```

#### 3.2 在服务器2 (192.168.31.166) 上启动客户端2

```bash
cd ~/MOE-FedCL

# 激活环境
conda activate your_env_name

# 启动客户端2（使用独立客户端脚本）
python examples/network_client_standalone.py \
    --config configs/network_test/client2.yaml \
    --server-ip 192.168.31.68 \
    --server-port 8000
```

你应该看到类似输出，但 client_id 是 `client_server2`，端口是 `8002`。

---

### 第四步：观察训练过程

#### 4.1 服务端输出

当两个客户端都连接后，服务端会自动开始训练，你会看到：

```
============================================================
[Trainer] 第 1 轮：聚合 2 个客户端模型
============================================================
  客户端 1: weights=1.1000
  客户端 2: weights=1.1000
  聚合结果: weights=1.1000

============================================================
[Trainer] 第 2 轮：聚合 2 个客户端模型
============================================================
  客户端 1: weights=1.2000
  客户端 2: weights=1.2000
  聚合结果: weights=1.2000

...

============================================================
训练完成!
============================================================
完成轮数: 3
最终准确率: 0.6500
最终损失: 0.1667
总耗时: 18.52秒
============================================================
```

#### 4.2 客户端输出

每个客户端会显示训练过程：

```
[client_server1] 执行本地训练 - 第 1 轮
[client_server1] 训练完成: loss=0.5000, acc=0.5000

[client_server1] 执行本地训练 - 第 2 轮
[client_server1] 训练完成: loss=0.2500, acc=0.5500

[client_server1] 执行本地训练 - 第 3 轮
[client_server1] 训练完成: loss=0.1667, acc=0.6000
```

---

## 🔧 故障排查

### 问题1：客户端无法连接到服务端

**症状：** 客户端报错 `Connection refused` 或超时

**解决方案：**

1. **检查服务端是否启动**
   ```bash
   # 在笔记本上检查
   netstat -an | findstr 8000
   ```

2. **检查防火墙**
   ```powershell
   # Windows 查看防火墙规则
   netsh advfirewall firewall show rule name=all | findstr 8000
   ```

3. **测试网络连通性**
   ```bash
   # 在服务器上测试
   ping 192.168.31.68
   telnet 192.168.31.68 8000
   ```

4. **检查服务端监听地址**
   - 确保 `server.yaml` 中 `host: "0.0.0.0"`（而不是 `127.0.0.1`）

### 问题2：客户端连接成功但无法通信

**症状：** 客户端注册失败或心跳超时

**解决方案：**

1. **检查客户端防火墙**
   ```bash
   # Linux 服务器上
   sudo firewall-cmd --list-ports
   # 应该看到 8001/tcp 或 8002/tcp
   ```

2. **检查客户端监听地址**
   - 确保客户端配置文件中 `host: "0.0.0.0"`

3. **查看详细日志**
   - 在启动脚本中添加 `--verbose` 标志

### 问题3：训练过程中断

**症状：** 训练开始后突然停止

**解决方案：**

1. **检查网络稳定性**
   ```bash
   # 持续 ping 测试
   ping -t 192.168.31.68  # Windows
   ping 192.168.31.68     # Linux
   ```

2. **增加超时时间**
   - 在配置文件中增加 `timeout` 和 `rpc_timeout` 值

3. **查看错误日志**
   - 检查服务端和客户端的错误输出

---

## 📊 验证测试成功

成功的测试应该满足：

✅ 服务端成功启动并监听 8000 端口
✅ 两个客户端都成功连接到服务端
✅ 客户端成功注册并开始心跳
✅ 完成3轮联邦学习训练
✅ 显示最终的训练结果

---

## 🎯 下一步

测试成功后，你可以：

1. **修改训练轮数**
   ```python
   # 在 network_server_standalone.py 中修改
   max_rounds = 10  # 改为10轮（默认是3轮）
   ```

2. **添加更多客户端**
   - 创建 `client3.yaml`, `client4.yaml`
   - 在更多服务器上启动客户端

3. **使用真实数据**
   - 替换 `SimpleLearner` 和 `SimpleTrainer`
   - 加载真实数据集（MNIST, CIFAR等）

4. **启用 SSL/TLS**
   - 在配置文件中设置 `ssl_enabled: true`
   - 配置证书路径

---

## 📝 配置文件说明

### 服务端配置 (`server.yaml`)

```yaml
transport:
  host: "0.0.0.0"  # ⚠️ 必须是 0.0.0.0 才能接受外部连接
  port: 8000        # HTTP 端口
  websocket_port: 9501  # WebSocket 端口
```

### 客户端配置 (`client1.yaml`, `client2.yaml`)

```yaml
client_id: client_server1  # ⚠️ 每个客户端必须有唯一 ID

transport:
  host: "0.0.0.0"  # ⚠️ 必须是 0.0.0.0 才能接受服务端回调
  port: 8001        # ⚠️ 每个客户端必须使用不同端口
```

---

## 🔐 安全注意事项

1. **生产环境建议：**
   - 启用 SSL/TLS 加密
   - 使用认证机制
   - 配置访问控制列表（ACL）

2. **防火墙配置：**
   - 只开放必要的端口
   - 限制来源 IP 范围
   - 使用 VPN 或专用网络

3. **数据隐私：**
   - 不要在日志中输出敏感数据
   - 使用差分隐私保护
   - 定期审计访问日志

---

## ❓ 常见问题

**Q: 为什么要使用 0.0.0.0 而不是 127.0.0.1？**
A: `127.0.0.1` 只监听本地回环接口，外部机器无法连接。`0.0.0.0` 监听所有网络接口，可以接受来自任何 IP 的连接。

**Q: 可以在公网环境测试吗？**
A: 可以，但需要：
   1. 配置公网 IP 或域名
   2. 配置端口转发（如果在 NAT 后）
   3. 强烈建议启用 SSL/TLS
   4. 配置认证机制

**Q: 客户端数量可以动态变化吗？**
A: 当前版本需要在启动时指定 `num_clients`。未来版本可以支持动态加入/退出。

---

## 📞 获取帮助

如果遇到问题：
1. 查看详细错误日志
2. 检查防火墙和网络配置
3. 参考故障排查部分
4. 提交 Issue 到项目仓库

祝测试顺利！🎉
