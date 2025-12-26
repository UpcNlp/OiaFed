# 发布到 PyPI 指南

## 准备工作

### 1. 设置 PyPI 账号

1. 在 [PyPI](https://pypi.org/) 注册账号
2. 在 [Test PyPI](https://test.pypi.org/) 注册账号（可选，用于测试）
3. 创建 API Token：
   - PyPI: https://pypi.org/manage/account/token/
   - Test PyPI: https://test.pypi.org/manage/account/token/

### 2. 配置 GitHub Secrets

在仓库的 `Settings > Secrets and variables > Actions` 中添加：

| Secret 名称 | 说明 |
|------------|------|
| `PYPI_API_TOKEN` | PyPI API token（可选，使用 Trusted Publishing 时不需要）|

### 3. 配置 Trusted Publishing（推荐）

使用 [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) 更安全：

1. 登录 PyPI，进入项目设置
2. 添加 "GitHub Actions" 作为 trusted publisher
3. 填写：
   - Owner: `oiafed`
   - Repository: `oiafed`
   - Workflow name: `publish.yml`
   - Environment: `pypi`

## 发布流程

### 自动发布（推荐）

1. **更新版本号**
   
   修改 `src/__init__.py` 中的版本号：
   ```python
   __version__ = "0.2.0"
   ```

2. **提交更改**
   ```bash
   git add .
   git commit -m "Bump version to 0.2.0"
   git push origin main
   ```

3. **创建并推送标签**
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. **GitHub Actions 自动执行**
   - 构建包
   - 运行测试
   - 发布到 PyPI
   - 创建 GitHub Release

### 手动发布

1. **安装工具**
   ```bash
   pip install build twine
   ```

2. **构建**
   ```bash
   python -m build
   ```

3. **检查**
   ```bash
   twine check dist/*
   ```

4. **上传到 Test PyPI（测试）**
   ```bash
   twine upload --repository testpypi dist/*
   ```

5. **上传到 PyPI**
   ```bash
   twine upload dist/*
   ```

## 版本号规范

使用 [语义化版本](https://semver.org/lang/zh-CN/)：

- `MAJOR.MINOR.PATCH`
- 例如：`0.1.0`, `1.0.0`, `1.2.3`

### 预发布版本

- Alpha: `0.1.0a1`, `0.1.0a2`
- Beta: `0.1.0b1`, `0.1.0b2`
- Release Candidate: `0.1.0rc1`, `0.1.0rc2`

## GitHub Actions 工作流

### CI (`.github/workflows/ci.yml`)

触发条件：
- 推送到 main/master/develop 分支
- Pull Request

执行：
- 代码检查 (black, isort, flake8)
- 多版本测试 (Python 3.10, 3.11, 3.12)
- 构建测试

### Publish (`.github/workflows/publish.yml`)

触发条件：
- 推送 `v*` 标签
- 手动触发

执行：
- 构建包
- 运行测试
- 发布到 Test PyPI（可选）
- 发布到 PyPI
- 创建 GitHub Release

## 检查清单

发布前确认：

- [ ] 版本号已更新
- [ ] CHANGELOG.md 已更新
- [ ] 所有测试通过
- [ ] README.md 是最新的
- [ ] 文档已同步更新

## 故障排除

### 常见问题

1. **包名已被占用**
   - 在 PyPI 搜索确认包名可用
   - 考虑使用其他名称

2. **上传失败**
   - 检查 API Token 是否正确
   - 确认网络连接正常

3. **版本冲突**
   - 不能重复上传相同版本
   - 必须使用新版本号

### 删除发布

- PyPI 不支持删除已发布的版本
- 只能通过发布新版本来修复问题
- 可以 "yank" 版本（标记为不推荐）