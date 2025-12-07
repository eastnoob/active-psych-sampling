# Active Psych Sampling - Git 管理指南

## 📋 快速导航

- [项目结构](#项目结构)
- [子模块管理](#子模块管理)
- [常见工作流](#常见工作流)
- [故障排除](#故障排除)

---

## 项目结构

### 仓库配置

```
active-psych-sampling (主仓库)
├── GitHub: https://github.com/eastnoob/active-psych-sampling
├── 分支: main（默认）、develop（可选）、feature/*
│
└── extensions/dynamic_eur_acquisition (子模块)
    └── GitHub: https://github.com/eastnoob/aepsych-eur-acqf
        ├── 分支: feature/hybrid-perturbation（当前）
        ├── feature/sps-convergence-metric
        └── master
```

### 目录权属

| 路径 | 管理方式 | 说明 |
|------|---------|------|
| `extensions/dynamic_eur_acquisition/` | **子模块** | 指向独立仓库，保持完整历史 |
| `extensions/custom_*/` | 主仓库 | 由主项目管理 |
| `extensions/config_builder/` | 主仓库 | 配置工具 |
| `extensions/docs/` | 主仓库 | 扩展文档 |
| `docs/` | 主仓库 | 项目文档 |
| `pixi.toml`, `pixi.lock` | 主仓库 | 环境配置 |

---

## 子模块管理

### 什么是子模块？

子模块是 Git 的功能，允许在一个仓库中嵌入另一个独立的仓库。主仓库只记录子模块指向的特定 commit。

```
主仓库记录:
  "extensions/dynamic_eur_acquisition" → commit abc123def456

子模块独立维护:
  - 完整的版本历史
  - 自己的分支（feature/hybrid-perturbation, master 等）
  - 自己的远程仓库
```

### 初次克隆项目

#### 方式 1: 递归克隆（推荐）
```bash
git clone --recurse-submodules https://github.com/eastnoob/active-psych-sampling.git
cd active-psych-sampling
pixi install
```

#### 方式 2: 分步克隆
```bash
git clone https://github.com/eastnoob/active-psych-sampling.git
cd active-psych-sampling

# 初始化子模块
git submodule init

# 获取子模块代码
git submodule update

pixi install
```

### 更新子模块到最新版本

```bash
# 查看子模块当前状态
git submodule status

# 更新到子模块远程的最新 commit
git submodule update --remote extensions/dynamic_eur_acquisition

# 检查更新
git status  # 显示 "extensions/dynamic_eur_acquisition" 已修改

# 提交更新到主仓库
git add extensions/dynamic_eur_acquisition
git commit -m "Update dynamic_eur_acquisition to latest version"
git push origin main
```

---

## 常见工作流

### 场景 1: 修改主项目的代码（不涉及子模块）

```bash
# 创建特性分支
git checkout -b feature/my-feature

# 修改文件
echo "new feature" >> extensions/custom_factory/new_file.py

# 提交
git add extensions/custom_factory/
git commit -m "feat: Add new feature to custom_factory"

# 推送
git push origin feature/my-feature

# 创建 Pull Request（GitHub 网页端）
```

### 场景 2: 修改子模块代码

#### 第一步: 在子模块中开发

```bash
# 进入子模块目录
cd extensions/dynamic_eur_acquisition

# 查看当前分支
git branch -a
# * feature/hybrid-perturbation
#   feature/sps-convergence-metric
#   master
#   remotes/origin/HEAD -> origin/feature/hybrid-perturbation
#   remotes/origin/feature/hybrid-perturbation
#   ...

# 确保在正确的分支上（已是 feature/hybrid-perturbation）
git checkout feature/hybrid-perturbation

# 修改代码
echo "improvement" >> eur_anova_pair.py

# 提交到子模块仓库
git add eur_anova_pair.py
git commit -m "refactor: Optimize EUR ANOVA calculation"

# 推送到子模块远程仓库
git push origin feature/hybrid-perturbation
```

#### 第二步: 在主项目中记录更新

```bash
# 返回主项目根目录
cd ../..  # 从 extensions/dynamic_eur_acquisition 返回到项目根

# 暂存子模块的新 commit 指针
git add extensions/dynamic_eur_acquisition

# 提交到主仓库
git commit -m "Update dynamic_eur_acquisition: Optimize EUR ANOVA calculation"

# 推送主仓库
git push origin feature/my-feature
```

**重点**: 必须执行两个 `git push`：
1. 在子模块中 push（到子仓库）
2. 在主项目中 push（记录子模块指针）

### 场景 3: 在新分支中工作

```bash
# 创建并切换到特性分支
git checkout -b feature/add-new-extension main

# 同时需要更新子模块？
git submodule update --remote

# 修改主项目文件
git add extensions/custom_new_module/
git commit -m "feat: Add new custom module"

# 修改子模块
cd extensions/dynamic_eur_acquisition
git checkout feature/hybrid-perturbation
# ... 修改代码 ...
git commit -m "..."
git push origin feature/hybrid-perturbation

# 返回主项目
cd ../..
git add extensions/dynamic_eur_acquisition
git commit -m "Update submodule to latest"

# 推送
git push origin feature/add-new-extension
```

### 场景 4: 查看子模块有哪些新提交未合并

```bash
# 查看子模块状态（+/- 表示领先/落后）
git submodule status
# +abc123def456 extensions/dynamic_eur_acquisition (describes new commits)
#   def789abc012 extensions/dynamic_eur_acquisition (current version)

# 进入子模块查看详细差异
cd extensions/dynamic_eur_acquisition
git log --oneline origin/feature/hybrid-perturbation ^HEAD

# 返回主项目
cd ../..
```

---

## Git 命令速查表

### 子模块操作

```bash
# 初始化子模块
git submodule init

# 获取子模块代码
git submodule update

# 一步到位（递归克隆）
git clone --recurse-submodules <repo-url>

# 更新所有子模块到远程最新
git submodule update --remote

# 更新特定子模块
git submodule update --remote extensions/dynamic_eur_acquisition

# 进入子模块工作
cd extensions/dynamic_eur_acquisition
git pull origin feature/hybrid-perturbation

# 查看子模块状态
git submodule status

# 强制同步子模块（当出现问题时）
git submodule sync --recursive
git submodule update --init --recursive
```

### 分支管理

```bash
# 创建并切换分支
git checkout -b feature/my-feature

# 列出所有分支
git branch -a

# 删除本地分支
git branch -d feature/old-feature

# 删除远程分支
git push origin --delete feature/old-feature

# 重命名分支
git branch -m old-name new-name
```

### 提交和推送

```bash
# 查看状态
git status

# 查看差异
git diff                    # 与最新 commit 的差异
git diff --staged          # 暂存区的差异

# 暂存文件
git add .                  # 暂存所有修改
git add <file>             # 暂存特定文件

# 提交
git commit -m "feat: Your message"

# 修改最后一个提交
git commit --amend

# 推送
git push origin feature/my-feature

# 强制推送（谨慎！仅在必要时）
git push --force-with-lease origin feature/my-feature
```

### 日志和历史

```bash
# 查看提交历史
git log --oneline -10         # 最近 10 条
git log --graph --all         # 显示分支图
git log --author=eastnoob     # 特定作者

# 查看特定文件的历史
git log -- extensions/custom_factory/

# 查看某次提交的详细内容
git show abc123def456
```

---

## 故障排除

### 问题 1: 克隆后子模块为空

**症状**: `extensions/dynamic_eur_acquisition` 目录存在但为空

**解决方案**:
```bash
git submodule init
git submodule update
# 或者
git submodule update --init --recursive
```

### 问题 2: 子模块出现 "detached HEAD" 状态

**症状**:
```
detached HEAD at abc123def456
```

**原因**: 子模块指向特定 commit（非分支末端），这是正常的

**如果需要继续开发**:
```bash
cd extensions/dynamic_eur_acquisition
git checkout feature/hybrid-perturbation
git pull origin feature/hybrid-perturbation
```

### 问题 3: 子模块更新后，主项目显示修改但未提交

**症状**:
```bash
git status
# modified:   extensions/dynamic_eur_acquisition (new commits)
```

**解决方案**:
```bash
# 如果你确实想更新子模块版本
git add extensions/dynamic_eur_acquisition
git commit -m "Update submodule to latest"
git push origin main

# 如果你想回滚到之前的版本
git checkout extensions/dynamic_eur_acquisition
```

### 问题 4: 在子模块中修改后无法推送

**症状**:
```bash
cd extensions/dynamic_eur_acquisition
git push origin feature/hybrid-perturbation
# error: permission denied
```

**解决方案**:
- 确保有 https 或 SSH 权限
- 检查 SSH 密钥配置
- 或使用 GitHub token 进行 https 认证

```bash
# 使用 SSH（推荐）
git remote set-url origin git@github.com:eastnoob/aepsych-eur-acqf.git

# 使用 HTTPS + token
git remote set-url origin https://github.com/eastnoob/aepsych-eur-acqf.git
# Git 会提示输入 token（使用 GitHub PAT）
```

### 问题 5: 主项目和子模块都有修改，不知道如何提交

**场景**:
```
主项目: extensions/custom_factory/ 有修改
子模块: extensions/dynamic_eur_acquisition/ 有修改
```

**解决方案** (分别处理):

```bash
# 1. 先处理子模块
cd extensions/dynamic_eur_acquisition
git add .
git commit -m "submodule: ..."
git push origin feature/hybrid-perturbation
cd ../..

# 2. 更新子模块指针
git add extensions/dynamic_eur_acquisition

# 3. 处理主项目
git add extensions/custom_factory/
git commit -m "feat: ... 

Also update dynamic_eur_acquisition submodule"
git push origin main
```

---

## 最佳实践

### ✅ DO (应该做)

- ✅ 修改子模块代码后，**先 push 子模块，再更新主项目**
- ✅ 使用 `--recurse-submodules` 克隆和更新
- ✅ 定期运行 `git submodule status` 检查版本
- ✅ 在 commit message 中清楚地说明修改的模块
- ✅ 使用 feature 分支进行开发
- ✅ 定期拉取最新版本: `git pull --recurse-submodules`

### ❌ DON'T (不应该做)

- ❌ 在主项目中直接编辑子模块文件后只 push 主项目
- ❌ 在子模块中 commit 后忘记 push
- ❌ 使用 `git push --force` 在共享分支上
- ❌ 在子模块中创建与主项目无关的分支
- ❌ 忽视 "detached HEAD" 警告而继续工作

---

## 提交信息规范

### 格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type（类型）
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档修改
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建、依赖等

### Scope（范围）
- `core`: 核心功能
- `submodule`: 子模块相关
- `custom_factory`: 工厂模块
- `config`: 配置相关

### 示例

```bash
# 主项目
git commit -m "feat(submodule): Update EUR ANOVA calculation

- Improved numerical stability
- Added caching for repeated calculations
- Also update dynamic_eur_acquisition reference"

# 子模块
cd extensions/dynamic_eur_acquisition
git commit -m "refactor(core): Optimize EUR main effect computation

Previously used gradient-based calculation, now using direct parameter changes.
Reduces computation time by ~30%."
```

---

## 有用的别名

添加到 `.git/config` 或全局配置 `~/.gitconfig`:

```bash
git config --global alias.subupdate 'submodule update --remote --recursive'
git config --global alias.subinit 'submodule update --init --recursive'
git config --global alias.substatus 'submodule status'
git config --global alias.sublog 'submodule foreach git log --oneline -5'
```

使用:
```bash
git subupdate              # 更新所有子模块
git substatus              # 查看所有子模块状态
git sublog                 # 查看所有子模块最近 5 条提交
```

---

## 联系和支持

- **主仓库**: https://github.com/eastnoob/active-psych-sampling
- **子仓库**: https://github.com/eastnoob/aepsych-eur-acqf
- **问题反馈**: GitHub Issues

---

**最后更新**: 2025-12-07

