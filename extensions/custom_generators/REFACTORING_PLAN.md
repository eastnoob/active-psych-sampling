# CustomPoolBasedGenerator 重构计划

## 当前状态

- **文件大小**: 42KB (982 行代码)
- **职责分散**: 包含去重、历史管理、采集函数等多个关注点

## 提议的模块化结构

```
extensions/custom_generators/
├── custom_pool_based_generator.py (核心类，重构后 ~200-250 行)
├── models/
│   ├── __init__.py
│   ├── dedup_manager.py         (去重数据库管理)
│   ├── history_manager.py       (采样历史管理)
│   ├── acquisition_manager.py   (采集函数管理)
│   └── pool_utils.py            (池相关工具)
├── README.md
└── tests/
    └── ...
```

## 模块分割方案

### 1. **dedup_manager.py** (~150 行)

**职责**: 去重数据库管理
**迁移方法**:

- `_initialize_dedup_database()`
- `_generate_db_path()`
- `_load_historical_points_from_dedup_db()`
- `_record_points_to_dedup_db()`
- `_close_dedup_database()`

**新类**:

```python
class DedupDatabaseManager:
    """管理去重数据库的全生命周期"""
    def __init__(self, dedup_database_path, dim)
    def initialize(self)
    def load_historical_points(self)
    def record_points(self, points)
    def close(self)
    def get_historical_points(self)
```

### 2. **history_manager.py** (~100 行)

**职责**: 采样历史追踪和管理
**迁移方法**:

- `_get_sampling_history_from_server()`
- `_exclude_historical_points_from_history()`
- `_match_points_to_pool_indices()`

**新类**:

```python
class HistoryManager:
    """追踪和管理采样历史"""
    def __init__(self, pool_points, dim)
    def get_server_history(self, server)
    def exclude_historical_points(self, indices)
    def match_points_to_indices(self, points)
    def get_used_indices(self)
```

### 3. **acquisition_manager.py** (~80 行)

**职责**: 采集函数管理和评估
**迁移方法**:

- `get_acqf_instance()`
- 采集函数相关的缓存和诊断逻辑

**新类**:

```python
class AcquisitionManager:
    """管理采集函数的创建、缓存和评估"""
    def __init__(self, acqf, acqf_kwargs)
    def get_instance(self, model)
    def evaluate(self, points, model)
    def invalidate_cache(self)
```

### 4. **pool_utils.py** (~60 行)

**职责**: 池管理工具函数
**包含内容**:

- `_get_available_indices()` - 获取可用池索引
- `_find_aepsych_server()` - 服务器查找逻辑
- `_match_points_to_pool_indices()` 辅助方法
- 常量和工具函数

**新函数**:

```python
def get_available_indices(used_indices, pool_size, ...)
def find_aepsych_server()
def validate_pool_points(points, bounds)
```

## 核心类精简方案

**重构后的 CustomPoolBasedGenerator** (~250 行):

```python
class CustomPoolBasedGenerator(AcqfGenerator):
    def __init__(self, ...):
        # 初始化各个管理器
        self.dedup_manager = DedupDatabaseManager(...)
        self.history_manager = HistoryManager(...)
        self.acqf_manager = AcquisitionManager(...)
    
    def gen(self, num_points, model=None):
        # 核心生成逻辑（简洁）
        available_indices = pool_utils.get_available_indices(...)
        acqf_values = self.acqf_manager.evaluate(...)
        selected_indices = torch.topk(acqf_values, num_points)
        selected_points = self.pool_points[selected_indices]
        self.dedup_manager.record_points(selected_points)
        return selected_points
    
    def get_config_options(self):
        # 配置接口保持不变
        ...
```

## 迁移策略

### 步骤 1: 创建模块框架

- [ ] 创建 `models/` 目录和 `__init__.py`
- [ ] 创建四个新模块文件（空框架）

### 步骤 2: 提取 dedup_manager

- [ ] 创建 `DedupDatabaseManager` 类
- [ ] 迁移所有去重相关方法
- [ ] 编写单元测试

### 步骤 3: 提取 history_manager

- [ ] 创建 `HistoryManager` 类
- [ ] 迁移历史追踪方法
- [ ] 编写单元测试

### 步骤 4: 提取 acquisition_manager

- [ ] 创建 `AcquisitionManager` 类
- [ ] 迁移采集函数管理
- [ ] 编写单元测试

### 步骤 5: 提取 pool_utils

- [ ] 创建工具函数模块
- [ ] 迁移辅助方法
- [ ] 编写单元测试

### 步骤 6: 重构核心类

- [ ] 更新 `CustomPoolBasedGenerator` 导入和使用新模块
- [ ] 简化 `gen()` 方法
- [ ] 保持 API 兼容性
- [ ] 运行完整测试

### 步骤 7: 清理和文档

- [ ] 更新 README.md
- [ ] 添加模块文档
- [ ] 更新配置示例

## 预期效果

| 指标 | 当前 | 重构后 | 改进 |
|------|------|--------|------|
| **单文件行数** | 982 | ~250 | -75% |
| **单文件大小** | 42KB | ~10KB | -76% |
| **职责分离** | 混合 | 清晰 | ✅ |
| **可测试性** | 困难 | 简单 | ✅ |
| **可维护性** | 困难 | 容易 | ✅ |
| **代码重用** | 低 | 高 | ✅ |

## 兼容性保证

- ✅ 公共 API 完全保持不变
- ✅ 配置接口不变
- ✅ 现有代码无需修改
- ✅ 所有测试通过

## 是否开始实施？

**推荐**：从 `dedup_manager` 开始，逐步迁移。这样可以：

1. 逐步验证每个模块
2. 及时发现问题
3. 降低风险
4. 保持测试覆盖

确认后我会按这个计划逐步执行。
