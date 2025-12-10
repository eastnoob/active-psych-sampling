# CustomPoolBasedGenerator

池基础采集函数生成器，支持三种去重数据库模式和被试隔离。

## 功能

- **采集函数评估**：用采集函数评估所有候选点，选择最优点
- **跨运行去重**：自动记录采样历史，防止重复采样
- **被试隔离**：每个被试使用独立数据库，数据完全隔离
- **三种数据库模式**：
  - **Mode 1**（持久化）：手动指定路径，数据跨实验累积
  - **Mode 2**（临时）：自动内存数据库，仅当前运行有效
  - **Mode 3**（自动命名）：自动生成路径，支持 (subject_id, run_id) 元组

## 快速开始

### Mode 1: 持久化模式（生产环境推荐）

```python
from custom_pool_based_generator import CustomPoolBasedGenerator
from botorch.acquisition import qExpectedImprovement

gen = CustomPoolBasedGenerator(
    lb=torch.tensor([-1., -1., -1.]),
    ub=torch.tensor([1., 1., 1.]),
    pool_points=pool,
    acqf=qExpectedImprovement,
    dedup_database_path="./data/subject_A/dedup.db"  # 完整路径
)

# 自动加载历史点、排除重复、记录新点
points = gen.gen(num_points=1, model=fitted_model)
```

### Mode 2: 临时模式（快速测试）

```python
gen = CustomPoolBasedGenerator(
    lb=torch.tensor([-1., -1., -1.]),
    ub=torch.tensor([1., 1., 1.]),
    pool_points=pool,
    acqf=qExpectedImprovement,
    dedup_database_path=None  # 自动内存数据库
)
```

### Mode 3: 自动命名模式（推荐用于被试实验）

```python
# 简单格式：(subject_id, run_id) → ./data/subject_A_run001_dedup.db
gen = CustomPoolBasedGenerator(
    lb=torch.tensor([-1., -1., -1.]),
    ub=torch.tensor([1., 1., 1.]),
    pool_points=pool,
    acqf=qExpectedImprovement,
    dedup_database_path=("subject_A", "run001")
)

# 自定义目录：(subject_id, run_id, save_dir) → {save_dir}/subject_A_run001_dedup.db
gen = CustomPoolBasedGenerator(
    lb=torch.tensor([-1., -1., -1.]),
    ub=torch.tensor([1., 1., 1.]),
    pool_points=pool,
    acqf=qExpectedImprovement,
    dedup_database_path=("subject_A", "run001", "./custom_path")
)
```

## 配置

### INI 文件

```ini
[CustomPoolBasedGenerator]
pool_points = ./data/candidate_pool.pt
acqf = qExpectedImprovement
acqf_kwargs = {}
allow_resampling = False
shuffle = True
seed = 42
dedup_database_path = ("subject_A", "run001")  # Mode 3: 自动命名
; dedup_database_path = ./data/subject_A/dedup.db  # Mode 1: 持久化
; dedup_database_path = None  # Mode 2: 临时内存
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `pool_points` | Tensor | 候选点池 [n_points, dim] |
| `acqf` | AcqfClass | 采集函数类型 |
| `acqf_kwargs` | dict | 采集函数参数 |
| `allow_resampling` | bool | 是否允许重复采样（默认False） |
| `shuffle` | bool | 初始化时是否打乱池（默认True） |
| `seed` | int | 随机种子 |
| `dedup_database_path` | str/tuple/None | 去重数据库配置（可选）：<br>- str: 完整路径（Mode 1）<br>- tuple: (subject_id, run_id) 或 (subject_id, run_id, save_dir)（Mode 3）<br>- None: 临时内存数据库（Mode 2） |

## 被试隔离

每个被试使用独立数据库：

```
./data/
├── subject_A/dedup.db
├── subject_B/dedup.db
└── subject_C/dedup.db
```

- ✅ 完全隔离：A的采样不影响B和C
- ✅ 并行安全：支持多被试并行实验
- ✅ 增量累积：每次实验数据自动追加

## 工作流

```
初始化:
  ├─ 创建或连接数据库
  ├─ 初始化 param_data 表
  └─ 加载历史点到内存

每次 gen():
  ├─ 排除历史点和已使用点
  ├─ 用采集函数评估可用点
  ├─ 选择最优点
  └─ 自动记录到数据库
```

## 数据库

**标准 AEPsych `param_data` 表**：

```sql
CREATE TABLE param_data (
    param_name TEXT NOT NULL,
    param_value REAL NOT NULL,
    iteration_id INTEGER
)
```

查询历史采样：

```python
import sqlite3

conn = sqlite3.connect("./data/subject_A/dedup.db")
cursor = conn.execute("""
    SELECT iteration_id, GROUP_CONCAT(param_value, ',') as point
    FROM param_data
    GROUP BY iteration_id
    ORDER BY iteration_id
""")

for iter_id, point in cursor:
    print(f"迭代 {iter_id}: {point}")

conn.close()
```

## 最佳实践

✅ **推荐**：

- 指定 `dedup_database_path`（生产环境）
- 使用被试特定目录：`./data/subject_<ID>/dedup.db`
- 定期备份数据库文件

❌ **避免**：

- 多进程共享同一生成器实例
- 手工修改数据库
- 依赖临时库进行重要实验

## 内部方法

| 方法 | 作用 |
|------|------|
| `_initialize_dedup_database()` | 初始化数据库 |
| `_load_historical_points_from_dedup_db()` | 加载历史点 |
| `_record_points_to_dedup_db(points)` | 记录采样点 |
| `_get_available_indices()` | 排除历史点，返回可用索引 |
| `_close_dedup_database()` | 关闭数据库连接 |

## 架构

CustomPoolBasedGenerator 采用模块化设计，核心功能委托给专用管理器：

```
CustomPoolBasedGenerator (主控制器)
├── DedupDatabaseManager (models/dedup_manager.py)
│   ├── 初始化三种数据库模式
│   ├── 管理数据库连接生命周期
│   └── 加载/保存历史点
├── HistoryManager (models/history_manager.py)
│   ├── 追踪采样历史
│   └── 排除已采样点
├── AcquisitionManager (models/acquisition_manager.py)
│   ├── 缓存采集函数实例
│   └── 评估采集函数
└── pool_utils (models/pool_utils.py)
    ├── get_available_indices() - 计算可用索引
    ├── find_aepsych_server() - 发现服务器实例
    ├── get_sampling_history_from_server() - 从服务器获取历史
    └── match_points_to_pool_indices() - 点匹配
```

**优势**：

- 清晰的职责分离
- 易于单元测试
- 便于独立维护和扩展
- 代码行数减少 30%（982→576 行）

## 兼容性

- ✅ AEPsych v8f68733+
- ✅ Python 3.8+
- ✅ Windows/Linux/macOS
- ✅ Config-driven 配置
- ✅ 多阶段策略支持

## 文件

- `custom_pool_based_generator.py` — 生成器实现
- `custom_pool_based_generator.ini` — 配置模板
- `tests/` — 功能测试和演示
