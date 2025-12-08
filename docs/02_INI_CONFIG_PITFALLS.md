# 02 INI 配置常见陷阱速查

**用途**: 快速诊断 AEPsych INI 配置错误,避免 `ast.literal_eval()` 解析失败

---

## 核心原则

AEPsych 使用 `ast.literal_eval()` 解析 INI 值 → **只接受 Python 字面量 (literals)**

---

## 引号规则速查表

| 配置项 | ❌ 错误 | ✅ 正确 | 原因 |
|--------|---------|---------|------|
| **策略名/类型** | `strategy_names = ['init_strat']` | `strategy_names = [init_strat]` | 裸标识符,非字符串 |
| **结果类型** | `outcome_types = ['continuous']` | `outcome_types = [continuous]` | 裸标识符,非字符串 |
| **参数名列表** | `parnames = [x1_Height, x2_Width]` | `parnames = ['x1_Height', 'x2_Width']` | 参数名**需要**引号 |
| **类别值** | `choices = [Chaos, Rotated]` | `choices = ['Chaos', 'Rotated']` | 字符串值需要引号 |
| **离散参数键** | `discrete_params = {x1_Height: 3}` | `discrete_params = {'x1_Height': 3}` | 字典键需要引号 |
| **数值列表** | `lb = [0, 0, 0]` | `lb = [0, 0, 0]` | 数值不需引号 |
| **采样点** | `points = [[2.8, 'Strict']]` | `points = [[2.8, 2]]` | 类别参数用数值索引 |
| **变量类型** | `variable_types_list = ['categorical']` | `variable_types_list = categorical` | 逗号分隔裸标识符 |

---

## 常见错误快速定位

### 错误 1: `malformed node: Name(id='...')`
**症状**: `ValueError: malformed node or string: <ast.Name object...>`
**根因**: 裸标识符被加了引号 (strategy_names, outcome_types)
**修复**: 移除引号: `['init_strat']` → `[init_strat]`

### 错误 2: `could not convert string to float: "'continuous'"`
**症状**: 类型转换失败,值带外层引号
**根因**: `outcome_types = ['continuous']` 被解析为字符串 `"'continuous'"`
**修复**: 移除引号: `outcome_types = [continuous]`

### 错误 3: `No section: "'init_strat'"`
**症状**: ConfigParser 找不到 section
**根因**: Section 名被引号包裹
**修复**: `strategy_names = ['init_strat']` → `[init_strat]`

### 错误 4: `KeyError: 'points'` (ManualGenerator)
**症状**: ManualGenerator 缺少 points 参数
**根因**: `points = [[2.8, 'Strict']]` 包含字符串字面量
**修复**: 类别参数用数值索引: `[[2.8, 2]]` (Chaos=0, Rotated=1, Strict=2)

### 错误 5: `KeyError: 'acqf'` (CustomPoolBasedGenerator)
**症状**: Generator 初始化缺少 acqf 参数
**根因**: CustomPoolBasedGenerator 从**自己的 section** 读取 acqf (非 `[opt_strat]`)
**修复**: 添加独立 section:
```ini
[CustomPoolBasedGenerator]
acqf = EURAnovaMultiAcqf
pool_points = [[...]]  # 由 server_manager.py 动态注入
```

### 错误 6: `TypeError: missing required argument 'pool_points'`
**症状**: PoolBasedGenerator 子类缺少 pool_points
**根因**: `server_manager.py` 只注入到 `[PoolBasedGenerator]`,未注入 `[CustomPoolBasedGenerator]`
**修复**: 修改动态注入逻辑 (检测所有 `*PoolBasedGenerator` section)

### 错误 7: `ValueError: Dimension mismatch: continuous(0) + discrete(0) = 0`
**症状**: Factory 解析不到连续/离散参数
**根因**: `get_config_args()` 收到 `name=None`,导致 `config.get(None, ...)` 失败
**修复**: 重写 `get_config_options()` 方法:
```python
@classmethod
def get_config_options(cls, config, name=None, options=None):
    if name is None:
        name = cls.__name__
    return cls.get_config_args(config, name)
```

### 错误 8: BaseGP CSV not found
**症状**: `FileNotFoundError: basegp_scan_csv`
**根因**: 相对路径从脚本目录解析,而非项目根目录
**修复**: `server_manager.py` 动态转换为绝对路径:
```python
project_root = Path(__file__).resolve().parents[6]  # 向上 7 级
absolute_path = project_root / relative_path
```

### 错误 9: Missing ub/lb in [common]
**症状**: `KeyError: 'lb'` or `'ub'` in parameter initialization
**根因**: AEPsych 要求**全局边界** (在 `[common]`) 或**每参数边界**
**修复**: 添加全局边界:
```ini
[common]
lb = [0, 0, 0, 0, 0, 0]
ub = [2, 1, 2, 2, 1, 2]
```

---

## 完整配置模板

```ini
[common]
parnames = ['x1_Height', 'x2_Width', 'x3_Type']  # ✅ 参数名需引号
stimuli_per_trial = 1
outcome_types = [continuous]  # ✅ 裸标识符
strategy_names = [init_strat, opt_strat]  # ✅ 裸标识符
lb = [0, 0, 0]
ub = [2, 1, 2]

[x3_Type]
par_type = categorical
choices = ['Chaos', 'Rotated', 'Strict']  # ✅ 字符串值需引号
lb = 0
ub = 2

[init_strat]
generator = ManualGenerator

[ManualGenerator]
points = [[2.8, 8.0, 2], [2.8, 6.5, 0]]  # ✅ 类别用数值索引

[opt_strat]
generator = CustomPoolBasedGenerator

[CustomPoolBasedGenerator]
acqf = EURAnovaMultiAcqf  # ⚠️ 必须在 generator 自己的 section
pool_points = [[...]]  # 由 server_manager.py 动态注入

[CustomBaseGPResidualMixedFactory]
continuous_params = []  # ✅ 空列表表示无连续参数
discrete_params = {'x1_Height': 3, 'x2_Width': 2, 'x3_Type': 3}  # ✅ 键需引号
basegp_scan_csv = extensions/warmup_budget_check/.../design_space_scan.csv
mean_type = pure_residual
lengthscale_prior = lognormal
ls_loc = []  # 仅连续参数需要 (空列表表示无)
ls_scale = []

[EURAnovaMultiAcqf]
variable_types_list = categorical, categorical, categorical  # ✅ 逗号分隔裸标识符
```

---

## 调试技巧

1. **启用调试输出**: 在 Factory 的 `get_config_args()` 添加 print:
   ```python
   print(f"[DEBUG] continuous_params_str = {continuous_params_str}")
   print(f"[DEBUG] discrete_params_str = {discrete_params_str}")
   ```

2. **验证解析结果**: 在 Python REPL 测试:
   ```python
   import ast
   ast.literal_eval("[init_strat, opt_strat]")  # ❌ NameError
   ast.literal_eval("['init_strat', 'opt_strat']")  # ✅ ['init_strat', 'opt_strat']
   ```

3. **检查 ConfigParser 行为**: ConfigParser 自动去除外层引号
   ```python
   config.get("common", "outcome_types")  # 返回: "[continuous]"
   ast.literal_eval("[continuous]")  # ❌ NameError
   ```

4. **类别编码一致性**: 确保 ManualGenerator points 的数值索引与 BaseGP 编码一致
   ```python
   # 检查 base_gp_encodings.json:
   {"x3_Type": {"Chaos": 0, "Rotated": 1, "Strict": 2}}
   # INI 中使用对应数值:
   points = [[2.8, 2]]  # Strict = 2
   ```

---

## 相关文件

- [custom_basegp_residual_mixed_factory.py](../extensions/custom_factory/custom_basegp_residual_mixed_factory.py): Factory 配置解析
- [server_manager.py](../tests/is_EUR_work/00_plans/251206/scripts/modules/server_manager.py): 动态 INI 修改
- [eur_config_residual.ini](../tests/is_EUR_work/00_plans/251206/scripts/eur_config_residual.ini): EUR 残差实验配置

---

**最后更新**: 2025-12-08
