当前项目是我的联邦持续学习开源项目，完整的使用例子在这个文件：examples/complete_mnist_demo.py
项目的设计文档可以在这里查看：docs/MOE-FedCL联邦通信系统架构设计.md
现在我需要复现一些经典的论文。复习完之后的代码需要依据功能进行拆分，放到fedcl/methods中对应的文件夹下。
现在请你先了解我的项目。

目前项目还有一个需求，我需要进行多次批量实验，我需要一个依据配置文件启动实验的脚本。给出你的设计思路
 # 完整数据集 (~6.7GB，包含train/valid/test)
  wget https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz

  # 下载后解压
  tar -xzf CINIC-10.tar.gz -C ./data/
  mv ./data/CINIC-10 ./data/cinic10
## 十三、待确认的问题

### 问题1: 目录结构
当前设计：
```
experiments/{experiment_name}/run_{timestamp}/
├── server/
└── clients/
```

是否满足需求？是否需要调整？默认是logs/xxx 文件夹，还有就是支持用户自己配置路径

### 问题2: Tracker 本地缓存
当前设计：依赖各 Tracker 自己的缓存机制
- MLflow: 自动缓存
- WandB: 自动缓存
- TensorBoard: 本地文件

是否需要额外的统一缓存层？ 不需要

### 问题3: 进度条策略
当前设计：
- 模式1（串行）：显示 Server 进度条
- 模式2（并行）：显示 Server 进度条
- 模式3（分布式）：Server 显示，Clients 静默

是否合理？分布式模式下， 服务端和客户端是否都需要显示进度条。然

### 问题4: Loguru 日志收集
当前设计：训练结束后，将 runtime.log 上传到 Tracker 作为 artifacts

是否需要实时上传？还是结束后批量上传？ 结束后批量上传，注意，需要考虑运行报错时，不要直接结束整个系统，而是先将处理日志、上传等工作后，然后再结束整个系统。

### 问题5: 配置文件位置
当前设计：logging 配置在 server.yaml 的顶层

是否需要支持单独的 logging.yaml 文件？
现在的configloader应该是支持yaml文件的拼接，请你检查

---