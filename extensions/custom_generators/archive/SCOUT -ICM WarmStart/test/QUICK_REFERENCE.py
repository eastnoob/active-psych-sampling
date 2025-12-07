"""
SCOUT Multi-Batch, Multi-Subject Phase-1 Warmup System
Quick Reference Guide
"""

# ============================================================================
# 1. COORDINATOR API - 全局规划与状态管理
# ============================================================================

from study_coordinator import StudyCoordinator

# 初始化 - 一次性调用
coordinator = StudyCoordinator(
    design_df=design_df,  # pd.DataFrame with f1, f2, ..., fd columns
    n_subjects=6,  # 总被试数
    total_budget=300,  # 总trial数量
    n_batches=3,  # 批次数
    seed=42,  # 全局RNG种子
)
coordinator.fit_initial_plan()  # 生成全局Core-1候选、交互对、boundary库

# 跨进程状态 - 每批次首次加载/创建
run_state = coordinator.load_run_state(
    study_id="STUDY_001", runs_dir="runs"  # Path: runs/STUDY_001/run_state.json
)
# Returns: {
#   "study_id": "STUDY_001",
#   "current_batch": 1,            # 1-indexed
#   "base_seed": 42,
#   "core1_last_batch_ids": [],    # 来自上一批的Core-1点
#   "history": [],                 # 批历史记录
#   ...
# }

# 被试计划 - 每被试每批调用一次
subject_plan = coordinator.make_subject_plan(
    subject_id=0, batch_id=run_state["current_batch"], run_state=run_state
)
# Returns: {
#   "subject_id": 0,
#   "batch_id": 1,
#   "is_bridge": False,
#   "quotas": {"core1": 10, "main": 15, "inter": 6, "boundary": 3, "lhs": 3},
#   "constraints": {
#       "core1_pool_indices": [...],       # 全局候选
#       "core1_repeat_indices": [...],     # 来自prev batch (若是桥接者)
#       "interaction_pairs": [...],
#       "boundary_library": [...]
#   },
#   "seed": 42  # Per-subject seed = base_seed + subject_id
# }

# 批后更新 - 汇总所有被试后调用
all_trials_df = pd.concat([trials_subj1, trials_subj2, ...])
summaries = [summary_subj1, summary_subj2, ...]

run_state = coordinator.update_after_batch(
    run_state=run_state,
    batch_id=1,  # 刚完成的批次号
    all_trials_df=all_trials_df,  # 所有被试的汇总
    all_summaries=summaries,  # 各被试的summarize()输出
)
# Updates run_state: {
#   "current_batch": 2,            # 递进到下一批
#   "core1_last_batch_ids": [...], # 本批实际使用的Core-1点 (供下批重复)
#   "history": [..., {
#       "batch_id": 1,
#       "coverage": 0.95,
#       "gini": 0.089,
#       "core1_repeat_rate": 0.0,  # 首批无重复
#       ...
#   }]
# }

# 保存状态
coordinator.save_run_state(study_id="STUDY_001", run_state=run_state, runs_dir="runs")

# 验证全局约束
validation = coordinator.validate_global_constraints(all_trials_combined)
# Returns: {
#   "core1_repeat_ratio": 0.50,    # 跨批重复率
#   "coverage_rate": 0.92,
#   "gini_coefficient": 0.085,
#   "warnings": [...]
# }

# ============================================================================
# 2. WARMUP GENERATOR API - 被试级采样
# ============================================================================

from scout_warmup_generator import WarmupAEPsychGenerator

# 初始化 - 每被试创建一个
gen = WarmupAEPsychGenerator(
    design_df=design_df, n_subjects=6, total_budget=300, n_batches=3, seed=42
)

# 应用coordinator的计划
gen.apply_plan(subject_plan)
# 覆盖: quotas, constraints, seed
# 特别地:
#   - gen.core1_pool_indices = subject_plan["constraints"]["core1_pool_indices"]
#   - gen.core1_repeat_indices = subject_plan["constraints"]["core1_repeat_indices"]
#   - np.random.seed(subject_plan["seed"])

# 全局规划 - 一次性，计算因子类型、预算split等
gen.fit_planning()

# 生成trial表
trials_df = gen.generate_trials()
# DataFrame with columns:
#   subject_id, batch_id, is_bridge,
#   block_type (core1/core2/interaction/boundary/lhs),
#   is_core1, is_core2, is_individual, is_boundary, is_lhs,
#   is_core1_repeat (CRITICAL!),
#   interaction_pair_id,
#   design_row_id (用于跨批追踪),
#   f1, f2, ..., fd (因子值)

# 汇总统计
summary = gen.summarize()
# Returns: {
#   "metadata": {
#       "subject_id": 0,
#       "batch_id": 1,
#       "is_bridge": False,
#       "coverage_rate": 0.92,     # 用于Coordinator检查
#       "gini": 0.089,             # 用于Coordinator检查
#       "core1_repeat_rate": 0.0,  # 首批为0
#       "block_counts": {
#           "core1": 10,
#           "main": 15,
#           "inter": 6,
#           "boundary": 3,
#           "lhs": 3
#       },
#       ...
#   },
#   "spatial_coverage": { "coverage_rate": 0.92, "gini_coefficient": 0.089 },
#   ...
# }

# ============================================================================
# 3. 完整工作流示例（Pseudo-Code）
# ============================================================================

"""
# 初始化
coordinator.fit_initial_plan()

for batch_id in range(1, n_batches + 1):
    # 加载状态
    if batch_id == 1:
        run_state = coordinator.load_run_state(study_id, runs_dir)  # 创建新
    else:
        run_state = coordinator.load_run_state(study_id, runs_dir)  # 加载prev

    all_batch_trials = []
    all_batch_summaries = []
    
    # 为每被试采样
    for subject_id in range(n_subjects):
        plan = coordinator.make_subject_plan(subject_id, batch_id, run_state)
        
        gen = WarmupAEPsychGenerator(design_df, ...)
        gen.apply_plan(plan)
        gen.fit_planning()
        trials_df = gen.generate_trials()
        summary = gen.summarize()
        
        all_batch_trials.append(trials_df)
        all_batch_summaries.append(summary)
        
        # [执行实验，收集响应...]
    
    # 汇总本批
    batch_all_trials = pd.concat(all_batch_trials)
    
    # 更新状态
    run_state = coordinator.update_after_batch(
        run_state, batch_id, batch_all_trials, all_batch_summaries
    )
    
    # 保存状态 (支持跨进程恢复)
    coordinator.save_run_state(study_id, run_state, runs_dir)

# 最终验证
validation = coordinator.validate_global_constraints(all_trials_combined)
print(f"Core-1 repeat ratio: {validation['core1_repeat_ratio']:.1%}")
print(f"Coverage: {validation['coverage_rate']:.3f}")
"""

# ============================================================================
# 4. 核心概念
# ============================================================================

"""
Core-1 (全局骨架):
  - 所有被试都做的固定题
  - Batch 2+ 中，部分Core-1被重复 (cross-batch continuity)
  - is_core1_repeat = True 表示重复，False 表示新点

桥接被试 (Bridge Subjects):
  - 参与多个批次的被试
  - 使得Batch k 与 k+1 的数据可比
  - 从prev batch的Core-1中重复约50%的点

Core-1重复逻辑 (严格优先级):
  Step 1: 优先放置 core1_repeat_indices (is_core1_repeat=True)
  Step 2: 从 core1_pool_indices 填补剩余quota (is_core1_repeat=False)
  Step 3: 重复数量 <= 50% * core1_quota

Core-2 (主效应 + 交互):
  - Core2_main: D-optimal 选择，覆盖主效应
  - Core2_inter: 交互对的象限采样

个体点 (Individual):
  - Boundary: 极端点，探索不确定性高的区域
  - LHS: Latin hypercube sampling，保证覆盖

设计行ID (design_row_id):
  - 在design_df中的原始行索引
  - 用于跨batch追踪点，特别是重复点
  - Coordinator 用其更新 core1_last_batch_ids
"""

# ============================================================================
# 5. 常见操作
# ============================================================================

# 检查某被试是否桥接
is_bridge = subject_id in run_state.get("bridge_subjects", {}).get(str(batch_id), [])

# 查看Core-1重复点是否被应用
core1_trials = trials_df[trials_df["block_type"] == "core1"]
repeat_count = len(core1_trials[core1_trials["is_core1_repeat"] == True])
print(f"Core-1 repeats: {repeat_count} / {len(core1_trials)}")

# 从next batch的plan中获取prev batch的Core-1点
next_plan = coordinator.make_subject_plan(subject_id, batch_id + 1, run_state)
prev_core1_to_repeat = next_plan["constraints"]["core1_repeat_indices"]

# 验证coverage/gini达标 (决定是否调整策略)
coverage = summary["metadata"]["coverage_rate"]
gini = summary["metadata"]["gini"]
if coverage < 0.60 or gini > 0.60:
    print("Below target - next batch will increase LHS/boundary by 10%")
    # 这会在 update_after_batch 中设置 strategy_adjustment 标记

# ============================================================================
# 6. 故障排查
# ============================================================================

# 问题: run_state.json 不存在
# 解决: coordinator.load_run_state() 会自动创建初始状态

# 问题: Core-1 重复点在Batch 2中没有出现
# 检查:
#   1. plan["constraints"]["core1_repeat_indices"] 是否为空
#   2. gen._generate_core1_trials() 是否接收并应用了它
#   3. 输出trials_df中is_core1_repeat列是否有True值

# 问题: coverage < 0.6 或 gini > 0.6
# 分析:
#   1. 检查design_df是否充分
#   2. 检查quota分配（core1是否过小）
#   3. 考虑提高LHS/boundary占比 (strategy_adjustment)

# ============================================================================
# 7. 参数建议
# ============================================================================

PARAMS = {
    "n_subjects": 6,  # 4-10人
    "n_batches": 3,  # 2-4批
    "per_subject_budget": 50,  # 题数 = total_budget / n_subjects
    "core1_quota": 10,  # 全局骨架，所有被试共同
    "core2_main_quota": 15,  # 主效应覆盖
    "core2_inter_quota": 6,  # 交互采样
    "boundary_quota": 3,  # 极端点
    "lhs_quota": 3,  # 覆盖探索
    "bridge_subjects_per_batch": 2,  # 2-3人
    "core1_repeat_ratio": 0.5,  # 50%上限
    "coverage_target": 0.60,  # 最小覆盖度
    "gini_target": 0.60,  # 最大不均匀度
    "base_seed": 42,  # 可复现性
}
