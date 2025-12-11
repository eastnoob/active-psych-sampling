# 有序参数扩展 - 快速参考

**文件**: `20251211_ordinal_monotonic_parameter_extension.md`  
**任务**: 在AEPsych + dynamic_eur_acquisition中添加ordinal参数类型  
**工作量**: 2-3天 (~300-400 LOC)  

---

## 核心概念

```
参数类型对比:

Categorical:   A,B,C,D        (无序，无偏好结构)  ❌ 无顺序
Integer:       1,2,3,4,5      (等差整数，但仅整数)  ⚠️  限制为整数
Ordinal:       1,2,3,4,5      (等差有序，任意数值)  ✅ 推荐用于Likert
Ordinal_mono:  0.1,0.5,2,5,10 (非等差但单调)      ✅ 用于幂律/指数
```

---

## 实现范围

### 📁 aepsych (主库修改)

| 文件 | 行数 | 修改内容 |
|------|------|--------|
| `transforms/ops/ordinal.py` | 150 | **新建** Transform类 |
| `transforms/ops/__init__.py` | 2 | 导入ordinal |
| `transforms/parameters.py` | 50 | get_config_options()中添加par_type处理 |
| `config.py` | 10 | par_type验证 |
| **小计** | **212** | |

### 📁 dynamic_eur_acquisition (扩展库修改)

| 文件 | 行数 | 修改内容 |
|------|------|--------|
| `modules/local_sampler.py` | 50 | 添加_perturb_ordinal() |
| `modules/config_parser.py` | 30 | parse_variable_types()中添加ordinal识别 |
| `eur_anova_pair.py` | 20 | _infer_variable_types_from_transforms()中添加ordinal推断 |
| `modules/diagnostics.py` | 30 | (可选)诊断报告增强 |
| **小计** | **130** | |

---

## Ordinal Transform核心API

```python
# 初始化
ordinal = Ordinal(
    indices=[0, 1],                           # 参数维度
    values={0: [1, 2, 3, 4, 5],             # 维度0的值列表
            1: [0.1, 0.5, 2.0, 5.0, 10.0]}, # 维度1的值列表
    is_uniform=True  # 等差(True) vs 非等差(False)
)

# 变换: 值 → rank
X_original = torch.tensor([[2.0], [5.0]])      # [1,2,3,4,5] 中的值
X_rank = ordinal.transform(X_original)         # [1.0, 4.0] (rank)

# 逆变换: rank → 值
X_back = ordinal.untransform(X_rank)           # [2.0, 5.0]

# bounds变换 (原始空间 → rank空间)
bounds = torch.tensor([[0.5, 9.5]])            # 原始值范围
bounds_rank = ordinal.transform_bounds(bounds) # [-0.5, 4.5]
```

---

## LocalSampler整合

```python
# 扰动逻辑伪代码
def sample(X_can_t, dims):
    for k in dims:
        vtype = self.variable_types.get(k)
        
        if vtype == "categorical":
            base = self._perturb_categorical(...)  # 离散采样 (rank空间)
        elif vtype == "ordinal":                   # ⭐ 新增
            base = self._perturb_ordinal(...)      # 值空间高斯 + 最近邻约束
        elif vtype == "integer":
            base = self._perturb_integer(...)      # 值空间高斯 + 舍入
        else:  # continuous
            base = self._perturb_continuous(...)   # 值空间高斯
    
    return base
```

**关键**: ordinal在**值空间**(物理参数实际值)内扰动，保留间距信息

```
例: 天花板高度 [2.0m, 2.5m, 3.5m]

中心值: 2.5m
噪声: σ = 0.1 × (3.5-2.0) = 0.15m
样本: 2.5 + N(0, 0.15) ≈ 2.38m → 最近邻 → 2.5m (或3.5m)

✅ 保留了0.5m vs 1.0m的间距关系
✅ ANOVA能正确看到参数效应
```

---

## 配置格式

### INI配置

```ini
[common]
parnames = [rating, intensity, dose]
lb = [0, 0, 0.0]
ub = [4, 4, 1.0]

[rating]
par_type = ordinal_arithmetic
values = [very_bad, bad, neutral, good, very_good]

[intensity]  
par_type = ordinal_monotonic
values = [0.1, 0.5, 2.0, 5.0, 10.0]

[dose]
par_type = continuous
lb = 0.0
ub = 1.0
```

### Python API

```python
from aepsych.config import Config

config_str = """
[common]
parnames = [x1, x2]
lb = [0, 0]
ub = [4, 3]

[x1]
par_type = ordinal
values = [1, 2, 3, 4, 5]

[x2]
par_type = ordinal_monotonic
values = [0.1, 1.0, 5.0, 10.0]
"""

config = Config(config_str=config_str)
# 自动创建Ordinal Transform，处理rank空间转换
```

---

## 与dynamic_eur_acquisition的交互

### 自动推断

```python
# eur_anova_pair.py 中
variable_types = self._infer_variable_types_from_transforms(transforms)
# 如果某维有 Ordinal transform，自动推断为 "ordinal"
```

### 混合扰动策略

```python
# local_sampler.py 中
if use_hybrid_perturbation and n_ranks <= exhaustive_level_threshold:
    # 穷举所有rank [0,1,2,...,n-1]
    # 适用: ordinal水平数≤3时覆盖所有
else:
    # 随机rank采样+舍入
    # 适用: 高维ordinal参数
```

---

## 测试清单

### 单元测试

- [ ] Ordinal.transform() / untransform()往返精确
- [ ] transform_bounds()正确生成rank界
- [ ] 字符串值列表支持 (e.g., ["a","b","c"])
- [ ] 浮点值列表支持 (e.g., [0.1, 0.5, 2.0])
- [ ] get_config_options()正确解析INI

### 集成测试

- [ ] ordinal参数通过ParameterTransforms完整流程
- [ ] LocalSampler._perturb_ordinal()输出合法rank
- [ ] EURAnovaPairAcqf正确推断ordinal类型
- [ ] 混合参数(ordinal+categorical+integer+continuous)协作
- [ ] 混合扰动策略(穷举vs随机)正确切换

### 性能测试

- [ ] ordinal扰动与categorical性能相当
- [ ] 无内存泄漏(长运行)

---

## 关键决策

| 决策 | 选项 | 原因 |
|------|------|------|
| **Transform空间** | Rank(0,1,2,...) | 统一Categorical, bounds简单 |
| **扰动空间** | Rank空间 | 高斯扰动自然, 舍入明确 |
| **is_uniform** | 配置指定 | 用户显式控制, 避免自动推断误差 |
| **向后兼容** | 完全兼容 | 无需修改existing配置 |

---

## 常见问题

**Q: 为什么不直接用Integer?**  
A: Integer仅支持整数值，ordinal支持任意数值(0.1, 0.5, ...)

**Q: Ordinal vs Categorical的区别?**  
A: Categorical无序(A,B,C无差别), Ordinal有序(1<2<3保有偏好结构)

**Q: rank空间的0.5是什么意思?**  
A: 两个rank之间的中点，便于均衡分布(如Categorical的±0.5)

**Q: 性能开销?**  
A: 最小，仅多做一次rank lookup表查询(O(1))

---

## 快速实现步骤

```
Day 1上午:
  1. transforms/ops/ordinal.py (150 LOC)
  2. 单元测试 + 集成 to aepsych

Day 1下午:
  1. config.py + parameters.py 修改 (60 LOC)
  2. 端到端测试

Day 2:
  1. local_sampler.py 修改 (50 LOC)
  2. config_parser.py 修改 (30 LOC)
  3. eur_anova_pair.py 修改 (20 LOC)
  4. 集成测试

Day 3:
  1. 性能测试
  2. 文档 + 配置示例
  3. 边界情况处理
```

---

**参考**: 详细计划见 `20251211_ordinal_monotonic_parameter_extension.md`
