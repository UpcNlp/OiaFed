# OiaFed 文档

欢迎使用 OiaFed 文档！**OiaFed**（One Framework for All Federation）是一个统一的联邦学习框架，支持所有联邦场景。

---

## 📚 文档导航

### [00 - 快速开始](00-getting-started/)

新用户从这里开始：

| 文档 | 描述 |
|------|------|
| [安装指南](00-getting-started/installation.md) | 环境配置与安装 |
| [快速入门](00-getting-started/quickstart.md) | 5 分钟运行第一个实验 |
| [核心概念](00-getting-started/concepts.md) | 理解框架基础 |

### [01 - 用户指南](01-guides/)

深入了解 OiaFed 的使用：

| 文档 | 描述 |
|------|------|
| [配置系统](01-guides/configuration.md) | 完整的配置系统说明 |
| [运行模式](01-guides/running-modes.md) | Serial / Parallel / Distributed |
| [内置算法](01-guides/algorithms.md) | 20+ 算法详解与选择指南 |
| [数据划分](01-guides/data-partitioning.md) | IID 和 Non-IID 策略 |
| [分布式部署](01-guides/distributed-setup.md) | 多机部署指南 |
| [自定义算法](01-guides/custom-algorithm.md) | 开发自己的算法 |

### [02 - API 参考](02-api-reference/)

完整的 API 文档：

| 文档 | 描述 |
|------|------|
| [核心 API](02-api-reference/core-api.md) | Trainer, Learner, Aggregator |
| [通信 API](02-api-reference/comm-api.md) | Node, Proxy, Transport |
| [算法 API](02-api-reference/methods-api.md) | 内置算法实现 |
| [基础设施 API](02-api-reference/infra-api.md) | Tracker, Callback, Config |

### [03 - 架构设计](03-architecture/)

理解 OiaFed 的内部设计：

| 文档 | 描述 |
|------|------|
| [架构总览](03-architecture/overview.md) | 整体系统架构 |
| [通信层设计](03-architecture/communication.md) | 节点通信详解 |
| [Callback 机制](03-architecture/callback-system.md) | 生命周期钩子 |
| [注册系统](03-architecture/registry-system.md) | 组件注册机制 |

### [04 - 开发指南](04-development/)

贡献和扩展 OiaFed：

| 文档 | 描述 |
|------|------|
| [代码规范](04-development/coding-style.md) | 代码风格指南 |
| [测试指南](04-development/testing.md) | 编写和运行测试 |
| [插件开发](04-development/plugin-development.md) | 开发可复用插件 |

### [05 - 论文复现](05-papers/)

学术研究支持：

| 文档 | 描述 |
|------|------|
| [已复现论文](05-papers/reproduced-papers.md) | 论文与算法对应表 |
| [复现指南](05-papers/reproduction-guide.md) | 如何复现实验 |

---

## 🔗 快速链接

- **GitHub**: [https://github.com/oiafed/oiafed](https://github.com/oiafed/oiafed)
- **官网**: [https://oiafed.cn](https://oiafed.cn)
- **PyPI**: [https://pypi.org/project/oiafed](https://pypi.org/project/oiafed)

---

## 🎯 支持的联邦场景

| 场景 | 缩写 | 状态 |
|------|------|------|
| 横向联邦学习 | HFL | ✅ |
| 纵向联邦学习 | VFL | ✅ |
| 联邦持续学习 | FCL | ✅ |
| 个性化联邦学习 | PFL | ✅ |
| 联邦遗忘 | FU | ✅ |

---

## 🤖 AI 助手

如果你是 AI 助手（Claude/GPT），请参阅 [AI Guide](ai-guide.md) 获取结构化的框架信息。

---

*文档版本: 0.1.0 | 最后更新: 2025-12-26*
