# Active Psych Sampling

**主动学习在心理学实验中的自适应采样框架**

## 项目结构

```
active-psych-sampling/
├── .gitmodules                          # Git submodule 配置
├── pixi.toml & pixi.lock               # Pixi 环境管理
├── .gitignore                          # Git 忽略规则
│
├── extensions/
│   ├── dynamic_eur_acquisition/        # 📌 [submodule] 动态欧拉采集函数
│   ├── custom_factory/                 # 自定义基础高斯过程工厂
│   ├── custom_generators/              # 自定义生成器
│   ├── custom_likelihood/              # 自定义似然函数
│   ├── custom_mean/                    # 自定义均值函数
│   ├── config_builder/                 # 配置构建工具
│   ├── docs/                           # 扩展文档
│   ├── handoff/                        # 交接文档
│   └── test/                           # 整合测试
│
├── docs/                               # 项目文档
├── data/                               # 实验数据
├── tests/                              # 测试
├── tools/                              # 工具脚本
├── logs/                               # 日志输出
│
└── README.md                           # 本文件
```

## 关键特性

- ✅ **主动学习采样**：使用欧拉ANOVA采集函数实现高效的心理学实验设计
- ✅ **多变量支持**：处理分类、整数和连续变量的混合
- ✅ **动态权重**：自适应加权策略优化采样效率
- ✅ **序数响应**：支持序数数据的专门建模

## 环境管理

### 使用 Pixi

```bash
# 安装依赖（自动创建虚拟环境）
pixi install

# 激活环境
pixi shell

# 运行脚本
python your_script.py
```

### Pixi 配置文件

- `pixi.toml`：项目依赖和配置（提交到 Git）
- `pixi.lock`：锁定的依赖版本（提交到 Git）
- `.pixi/envs/`：实际虚拟环境（Git 忽略）

## Git 子模块管理

### `dynamic_eur_acquisition` 子项目

这个目录被管理为 **Git submodule**，指向独立仓库：
- 独立仓库：https://github.com/eastnoob/aepsych-eur-acqf.git
- 保持完整的版本历史和分支结构

#### 更新子模块

```bash
# 获取子模块最新代码
git submodule update --remote extensions/dynamic_eur_acquisition

# 在子模块中开发
cd extensions/dynamic_eur_acquisition
git checkout feature/hybrid-perturbation
# ... 修改代码 ...
git add .
git commit -m "Your commit message"
git push origin feature/hybrid-perturbation

# 在主项目中记录子模块更新
cd ../..
git add extensions/dynamic_eur_acquisition
git commit -m "Update dynamic_eur_acquisition to latest"
git push origin main
```

#### 克隆包含子模块的项目

```bash
# 方式1：克隆时自动初始化 submodules
git clone --recurse-submodules https://github.com/eastnoob/active-psych-sampling.git

# 方式2：先克隆后初始化
git clone https://github.com/eastnoob/active-psych-sampling.git
cd active-psych-sampling
git submodule init
git submodule update
```

## 快速开始

```python
from aepsych.server import AEPsychServer

# 加载配置
with open('extensions/dynamic_eur_acquisition/configs/QUICKSTART.ini') as f:
    config_str = f.read()

# 创建服务器
server = AEPsychServer()
server.configure(config_str=config_str)

# 运行实验
for trial in range(25):
    next_x = server.ask()
    outcome = get_response(next_x)  # 你的实验代码
    server.tell(config_str, outcome)
```

## 项目管理

### 添加新的 submodule

如果某个 `extensions/` 下的模块变得足够独立，可以转换为 submodule：

```bash
# 1. 为模块创建独立 GitHub 仓库
# 2. 在项目根目录执行
git submodule add <repo-url> extensions/<module-name>
git add .gitmodules extensions/<module-name>
git commit -m "Add <module-name> as submodule"
git push origin main
```

### 目录说明

| 目录 | 说明 | Git 管理 |
|------|------|---------|
| `extensions/dynamic_eur_acquisition` | 动态欧拉采集函数核心模块 | submodule |
| `extensions/custom_*` | 自定义扩展（工厂、生成器等） | main repo |
| `extensions/config_builder` | 自动配置生成工具 | main repo |
| `extensions/docs` & `handoff` | 扩展文档和交接资料 | main repo |
| `docs/` | 项目总体文档 | main repo |
| `data/` | 实验数据样本 | main repo |
| `.pixi/` | Pixi 虚拟环境 | 忽略 |
| `logs/` | 运行日志 | 忽略 |

## 贡献指南

### 工作流

1. **创建 feature 分支**（主项目）
2. **修改代码**（可能涉及多个模块）
3. **提交变更**：
   - 子模块：先 push 到子项目仓库
   - 主项目：更新 submodule 指针后 push
4. **创建 Pull Request**

### 提交规范

```bash
# 主项目
git commit -m "feat: Add new acquisition function"
git commit -m "docs: Update EUR documentation"
git commit -m "fix: Resolve ordinal encoding bug"

# 子项目（在 extensions/dynamic_eur_acquisition 目录）
git commit -m "refactor: Optimize EUR ANOVA calculation"
```

## 许可证

遵循 AEPsych 的许可证政策。

## 相关资源

- **AEPsych 官方**：https://github.com/facebookresearch/aepsych
- **子项目文档**：[extensions/dynamic_eur_acquisition/README.md](extensions/dynamic_eur_acquisition/README.md)
- **配置指南**：[extensions/dynamic_eur_acquisition/archive/docs/AEPSYCH_CONFIG_GUIDE.md](extensions/dynamic_eur_acquisition/archive/docs/AEPSYCH_CONFIG_GUIDE.md)

## 联系方式

- 作者：eastnoob
- GitHub：https://github.com/eastnoob
