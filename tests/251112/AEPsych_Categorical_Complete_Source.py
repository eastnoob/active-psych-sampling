"""
AEPsych Categorical Transform - 完整源代码提取和注释分析

本文件包含：
1. 完整的 Categorical 类实现（从 .pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py 提取）
2. 关键方法的详细注释
3. 问题分析和建议
"""

# ============================================================================
# 文件: aepsych/transforms/ops/categorical.py
# ============================================================================

from typing import Any, Literal
import torch
from aepsych.config import Config
from aepsych.transforms.ops.base import StringParameterMixin, Transform
from botorch.models.transforms.input import subset_transform


class Categorical(Transform, StringParameterMixin):
    """
    分类参数转换类
    
    关键特性：
    - 不改变张量本身（transform/untransform 都只是四舍五入）
    - 实际的 categorical → indices 映射由 StringParameterMixin.indices_to_str() 处理
    - 支持混合类型数据（通过 NumPy object arrays）
    """
    
    # 这些属性确保与 BoTorch 的兼容性
    is_one_to_many = False
    transform_on_train = True
    transform_on_eval = True
    transform_on_fantasize = True
    reverse = False

    def __init__(
        self,
        indices: list[int],
        categories: dict[int, list[str]],
    ) -> None:
        """
        初始化分类转换
        
        参数：
        ------
        indices : list[int]
            分类参数的位置索引
            例如：[0, 2] 表示第0列和第2列是分类参数
            
        categories : dict[int, list[str]]
            分类值字典，格式：{index: [category_list]}
            例如：{0: ['2.8', '4.0', '8.5'], 2: ['A', 'B', 'C']}
            
            ⚠️ 注意：即使 choices 定义为数值（[2.8, 4.0, 8.5]），
               get_config_options() 会强制转换为字符串
        
        代码：
        -----
        super().__init__()
        self.indices = indices
        self.categories = categories
        self.string_map = self.categories
        
        其中：
        - self.string_map 用于 StringParameterMixin.indices_to_str() 方法
        - StringParameterMixin 提供 indices → str 的映射
        """
        super().__init__()
        self.indices = indices
        self.categories = categories
        self.string_map = self.categories

    @subset_transform
    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        前向转换（从实际值→indices）
        
        ⚠️ 重要：当前实现只做四舍五入，没有实际的索引映射！
        
        参数：
        ------
        X : torch.Tensor
            形状为 [batch_shape, n, d] 的输入张量
        
        返回：
        ------
        torch.Tensor
            原样返回，仅进行四舍五入
            
        装饰器 @subset_transform：
        - 自动限制操作范围到 self.indices 指定的列
        - 例如：如果 indices=[0,2]，只对第0和2列应用此转换
        
        代码实现：
        ---------
        return X.round()
        
        问题分析：
        --------
        这个方法没有做任何有意义的转换，只是四舍五入。
        实际的 indices → str 映射由外部的 indices_to_str() 处理。
        """
        return X.round()

    @subset_transform
    def _untransform(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        反向转换（从indices→实际值）
        
        ⚠️ 关键BUG位置：这里假设输入是 indices，但如果输入已经是实际值，
           就会出现双重转换的问题！
        
        参数：
        ------
        X : torch.Tensor
            形状为 [batch_shape, n, d] 的已转换（或未转换）张量
        
        返回：
        ------
        torch.Tensor
            四舍五入后的张量
        
        代码实现：
        ---------
        return X.round()
        
        🐛 Bug 场景：
        -----------
        1. Generator 返回实际值：[2.8, 8.0, ...]
        2. ParameterTransformedGenerator 调用 untransform()
        3. Categorical.untransform() 只做 X.round()
        4. 如果调用了 indices_to_str([2.8])，会尝试 categories[int(2.8)]
           → 出现索引越界或错误映射
        
        修复方案：
        ---------
        def _untransform(self, X: torch.Tensor) -> torch.Tensor:
            # 检查是否已经是实际值
            for idx in self.indices:
                if X[0, idx] in self.categories[idx]:
                    continue  # 已经是实际值，跳过
                else:
                    # 进行 indices → values 映射
                    X[0, idx] = self.categories[idx][int(X[0, idx])]
            return X.round()
        """
        return X.round()

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        r"""
        从配置文件中提取初始化选项
        
        参数：
        ------
        config : Config
            AEPsych 配置对象
        name : str, optional
            参数名称，例如 'x1_CeilingHeight'
        options : dict, optional
            用于覆盖的选项
        
        返回：
        ------
        dict[str, Any]
            初始化 Categorical 所需的选项，包括：
            - indices: [参数在 parnames 中的位置]
            - categories: {index: [value_list]}
        
        执行流程：
        ---------
        1. options = super().get_config_options(...)
           ↓ 从父类 Transform 继承
           ↓ 设置 indices = [parnames.index(name)]
           
        2. 验证 name 非空
        
        3. 如果 categories 不在 options 中：
           idx = options["indices"][0]
           cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
           options["categories"] = cat_dict
           
           ⚠️ 关键问题：element_type=str 强制所有 choices 转换为字符串！
        
        4. 删除 options 中的 "bounds" 键（分类参数无需连续边界）
        
        代码实现：
        ---------
        options = super().get_config_options(config=config, name=name, options=options)

        if name is None:
            raise ValueError(f"{name} must be set to initialize a transform.")

        if "categories" not in options:
            idx = options["indices"][0]  # 应该只有一个 index
            # ⚠️ BUG 在这里！
            cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
                                                         # ^^^^^^^^^^^^^^^^
                                                         # 强制转换为字符串！
            options["categories"] = cat_dict

        if "bounds" in options:
            del options["bounds"]  # 删除范围

        return options
        
        🐛 核心问题分析：
        ---------------
        
        配置示例：
        ---------
        [x1_CeilingHeight]
        par_type = categorical
        choices = [2.8, 4.0, 8.5]  # 数值列表
        
        当前处理过程（错误）：
        -------------------
        1. config.getlist(name, "choices", element_type=str)
        2. 将 "2.8", "4.0", "8.5" 解析为字符串
        3. 结果: {0: ['2.8', '4.0', '8.5']}
        4. indices_to_str([0]) 返回 '2.8'（字符串）
        5. Server 应该返回 2.8（浮点），但返回了 '2.8'（字符串）❌
        
        修复方案 1（推荐）：
        ------------------
        if "categories" not in options:
            idx = options["indices"][0]
            choices_raw = config.getlist(name, "choices")
            
            # 尝试转换为浮点，失败则保持为字符串
            try:
                choices = [float(c) for c in choices_raw]
            except (ValueError, TypeError):
                choices = choices_raw
            
            cat_dict = {idx: choices}
            options["categories"] = cat_dict
        
        修复方案 2：
        -----------
        在 get_config_options 的前面检查 par_type：
        
        par_type = config.get(name, "par_type")
        if par_type == "categorical":
            # 检查 choices 是否都是数值
            choices_str = config.get(name, "choices")
            try:
                choices = eval(choices_str)  # [2.8, 4.0, 8.5]
                if all(isinstance(c, (int, float)) for c in choices):
                    # 保持为数值
                    cat_dict = {idx: choices}
                else:
                    # 转换为字符串
                    cat_dict = {idx: [str(c) for c in choices]}
            except:
                # 字符串分类
                cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
        """
        options = super().get_config_options(config=config, name=name, options=options)

        if name is None:
            raise ValueError(f"{name} must be set to initialize a transform.")

        if "categories" not in options:
            idx = options["indices"][0]  # 应该只有一个 index
            cat_dict = {idx: config.getlist(name, "choices", element_type=str)}
            options["categories"] = cat_dict

        if "bounds" in options:
            del options["bounds"]  # 删除范围

        return options

    def transform_bounds(
        self, X: torch.Tensor, bound: Literal["lb", "ub"] | None = None, **kwargs
    ) -> torch.Tensor:
        r"""
        转换参数边界（将分类值映射到连续空间中的范围）
        
        参数：
        ------
        X : torch.Tensor
            形状为 [1, d] 或 [2, d] 的边界张量
            [1, d]: 单个边界
            [2, d]: [lb, ub] 堆叠的边界对
        
        bound : Literal["lb", "ub"], optional
            指定这是下界还是上界
            如果为 None，假设 X 是 [2, d] 格式的完整边界对
        
        **kwargs : dict
            其他参数，包括：
            - epsilon: 调整舍入偏移，确保每个离散值在参数空间中有相等的区间
              默认值：1e-6
        
        返回：
        ------
        torch.Tensor
            转换后的边界张量
        
        代码实现：
        ---------
        epsilon = kwargs.get("epsilon", 1e-6)
        return self._transform_bounds(X, bound=bound, epsilon=epsilon)
        """
        epsilon = kwargs.get("epsilon", 1e-6)
        return self._transform_bounds(X, bound=bound, epsilon=epsilon)

    def _transform_bounds(
        self,
        X: torch.Tensor,
        bound: Literal["lb", "ub"] | None = None,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        r"""
        实际的边界转换实现
        
        参数：
        ------
        X : torch.Tensor
            边界张量，形状为 [1, d] 或 [2, d]
        
        bound : Literal["lb", "ub"], optional
            边界类型
        
        epsilon : float
            舍入偏移修正，默认 1e-6
        
        返回：
        ------
        torch.Tensor
            转换后的边界
        
        原理分析：
        ---------
        
        假设有 3 个分类选项（indices = [0, 1, 2]）：
        
        原始配置：
        -------
        choices = [2.8, 4.0, 8.5]
        
        索引映射：
        -------
        Index 0 ↔ Value 2.8
        Index 1 ↔ Value 4.0
        Index 2 ↔ Value 8.5
        
        转换后的参数空间（连续化）：
        -----
        Index 0: [-0.5, 0.5)
        Index 1: [0.5, 1.5)
        Index 2: [1.5, 2.5)
        
        边界设置逻辑：
        ----------
        
        当 bound == "lb"（下界）：
        X[0, self.indices] -= 0.5
        
        示例：
        ----
        输入: X[0, 0] = 0（代表 index 0）
        输出: X[0, 0] = -0.5（转换到连续空间的下界）
        
        当 bound == "ub"（上界）：
        X[0, self.indices] += (0.5 - epsilon)
        
        示例：
        ----
        输入: X[0, 2] = 2（代表 index 2）
        输出: X[0, 2] = 2.5 - epsilon（转换到连续空间的上界）
        
        当 bound == None（完整边界对）：
        - 第一行（lb）：减去 0.5
        - 第二行（ub）：加上 (0.5 - epsilon)
        
        示例：
        ----
        输入: X = [[0, 0, 0],
                  [2, 2, 2]]（从 index 0 到 index 2）
        
        输出: X = [[-0.5, -0.5, -0.5],
                  [2.5-eps, 2.5-eps, 2.5-eps]]
        
        代码实现：
        ---------
        X = X.clone()

        if bound == "lb":
            X[0, self.indices] -= torch.tensor([0.5] * len(self.indices))
        elif bound == "ub":
            X[0, self.indices] += torch.tensor([0.5 - epsilon] * len(self.indices))
        else:  # 完整边界对
            X[0, self.indices] -= torch.tensor([0.5] * len(self.indices))
            X[1, self.indices] += torch.tensor([0.5 - epsilon] * len(self.indices))

        return X
        """
        X = X.clone()

        if bound == "lb":
            X[0, self.indices] -= torch.tensor([0.5] * len(self.indices))
        elif bound == "ub":
            X[0, self.indices] += torch.tensor([0.5 - epsilon] * len(self.indices))
        else:  # 完整边界对
            X[0, self.indices] -= torch.tensor([0.5] * len(self.indices))
            X[1, self.indices] += torch.tensor([0.5 - epsilon] * len(self.indices))

        return X


# ============================================================================
# 继承的 StringParameterMixin 方法（来自 base.py）
# ============================================================================

class StringParameterMixin:
    """
    将 indices 转换为字符串的 mixin 类
    """
    string_map: dict[int, list[str]] | None

    def indices_to_str(self, X: np.ndarray) -> np.ndarray:
        r"""
        将数值 indices 转换为字符串值
        
        参数：
        ------
        X : np.ndarray
            混合类型的 NumPy 数组，包含某些应转换为字符串的 indices
        
        返回：
        ------
        np.ndarray
            object 类型的数组，相关参数已转换为字符串
        
        代码实现：
        ---------
        obj_arr = X.astype("O")  # 转换为 object 类型

        if self.string_map is not None:
            for idx, cats in self.string_map.items():
                obj_arr[:, idx] = [cats[int(i)] for i in obj_arr[:, idx]]

        return obj_arr
        
        使用示例：
        ---------
        
        情景 1（字符串分类）：
        -------------------
        categories = {0: ['Chaos', 'Rotated', 'Strict']}
        X = np.array([[0, 1, 2]], dtype=object)
        
        result = indices_to_str(X)
        # result[0, 0] = 'Chaos'     （0 → categories[0][0]）
        # result[0, 1] = 'Rotated'   （1 → categories[0][1]）
        # result[0, 2] = 'Strict'    （2 → categories[0][2]）
        
        情景 2（数值分类，当前被错误地转换为字符串）：
        -------
        categories = {0: ['2.8', '4.0', '8.5']}  # 应该是 [2.8, 4.0, 8.5]
        X = np.array([[0, 1, 2]], dtype=object)
        
        result = indices_to_str(X)
        # result[0, 0] = '2.8'  （字符串，而非浮点） ❌
        # result[0, 1] = '4.0'  （字符串，而非浮点） ❌
        # result[0, 2] = '8.5'  （字符串，而非浮点） ❌
        
        🐛 核心 BUG：
        -----------
        当 categories 中的值本应是数值时，被错误地存储为字符串。
        这导致下游系统（如 Oracle）接收到错误的数据类型。
        """
        obj_arr = X.astype("O")

        if self.string_map is not None:
            for idx, cats in self.string_map.items():
                obj_arr[:, idx] = [cats[int(i)] for i in obj_arr[:, idx]]

        return obj_arr


# ============================================================================
# 总结：完整的数据流和问题点
# ============================================================================

"""
数据流分析（数值型分类参数示例）
==================================

配置：
-----
[x1_CeilingHeight]
par_type = categorical
choices = [2.8, 4.0, 8.5]  # 数值列表

步骤 1: 初始化阶段
------------------
Categorical.get_config_options(name='x1_CeilingHeight')
  ↓
config.getlist('x1_CeilingHeight', 'choices', element_type=str)
  ↓
categories = {0: ['2.8', '4.0', '8.5']}  # ❌ 被转换为字符串！
  ↓
self.string_map = {0: ['2.8', '4.0', '8.5']}

预期结果：
categories = {0: [2.8, 4.0, 8.5]}  # ✓ 保持为数值

步骤 2: Generator 阶段
---------------------
CustomPoolBasedGenerator.gen()
  ↓
返回实际值：[2.8, ...]  或 indices: [0, ...]

步骤 3: Transform 阶段（问题所在！）
------------------------------------
如果 Generator 返回实际值 [2.8, ...]：

ParameterTransformedGenerator.gen()
  ↓
x = base_generator.gen()  # 返回 [2.8, ...]
  ↓
self.transforms.untransform(x)  # ❌ 无条件调用！
  ↓
Categorical._untransform([2.8, ...])
  ↓
return X.round()  # 只做四舍五入，没有映射
  ↓
最终返回 [2.8, ...]

步骤 4: indices_to_str 阶段
---------------------------
如果后续调用了 indices_to_str([0, ...])：

indices_to_str([0, ...])
  ↓
obj_arr[:, 0] = [categories[0][int(0)] for i in obj_arr[:, 0]]
  ↓
obj_arr[:, 0] = ['2.8']  # 字符串！
  
问题：
-----
1. 应该返回数值 2.8，却返回字符串 '2.8'
2. 下游系统（Oracle）期望数值，接收到字符串

现有修复方案
=============

方案 A: 修复 get_config_options（根本解决）
-----------
修改 element_type 逻辑，自动检测并保留数值类型

方案 B: Generator Fallback（已集成）
-----------
在 CustomPoolBasedGenerator 中实现自动映射，检测到 indices 时自动转换

方案 C: 使 untransform 幂等（清晰解决）
-----------
修改 _untransform 使其检测输入类型，避免重复转换
"""
