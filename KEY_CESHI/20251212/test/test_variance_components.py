#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 variance components 评估功能

测试场景:
1. 简单null模型 (仅随机截距)
2. 包含预测变量的conditional模型
3. 不同被试观测数不均衡的情况
4. 估计准确性验证 (已知真实方差成分)
"""

import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
from modules.evaluation_effect_recovery import evaluate_variance_components


def test_null_model_balanced():
    """测试1: Null模型,均衡设计 (每个被试相同观测数)"""
    print("\n" + "="*70)
    print("测试1: Null Model - 均衡设计")
    print("="*70)
    
    # 模拟参数
    n_subjects = 5
    n_obs_per_subject = 10
    true_sigma2_between = 0.5  # 被试间差异
    true_sigma2_within = 0.3   # 被试内噪声
    
    # 生成数据
    np.random.seed(42)
    subject_ids = np.repeat(np.arange(n_subjects), n_obs_per_subject)
    subject_effects = np.random.randn(n_subjects) * np.sqrt(true_sigma2_between)
    
    # y = subject_effect + noise
    y = subject_effects[subject_ids] + np.random.randn(len(subject_ids)) * np.sqrt(true_sigma2_within)
    X = np.zeros((len(y), 0))  # 无预测变量
    
    # 真实ICC
    true_icc = true_sigma2_between / (true_sigma2_between + true_sigma2_within)
    
    # 评估
    result = evaluate_variance_components(
        X_train=X,
        y_train=y,
        subject_ids=subject_ids,
        true_variance_components={
            "sigma2_between": true_sigma2_between,
            "sigma2_within": true_sigma2_within,
        },
        include_predictors=False,
    )
    
    # 输出结果
    print(f"\n数据生成参数:")
    print(f"  被试数: {n_subjects}")
    print(f"  每被试观测数: {n_obs_per_subject}")
    print(f"  真实σ²_between: {true_sigma2_between:.4f}")
    print(f"  真实σ²_within: {true_sigma2_within:.4f}")
    print(f"  真实ICC: {true_icc:.4f}")
    
    print(f"\n估计结果:")
    print(f"  估计σ²_between: {result['sigma2_between']:.4f}")
    print(f"  估计σ²_within: {result['sigma2_within']:.4f}")
    print(f"  估计ICC: {result['icc']:.4f}")
    print(f"  模型类型: {result['model_type']}")
    
    if result['estimation_accuracy']:
        acc = result['estimation_accuracy']
        print(f"\n估计准确性:")
        print(f"  ICC误差: {acc['icc_error']:.4f} ({acc['icc_relative_error']*100:.1f}%)")
        print(f"  σ²_between误差: {acc['sigma2_between_error']:.4f}")
        print(f"  σ²_within误差: {acc['sigma2_within_error']:.4f}")
    
    # 验证
    assert result['n_subjects'] == n_subjects, "被试数量不匹配"
    assert result['n_observations'] == n_subjects * n_obs_per_subject, "观测数不匹配"
    assert abs(result['icc'] - true_icc) < 0.25, f"ICC估计偏差过大: {abs(result['icc'] - true_icc):.4f}"
    
    print("\n✅ 测试1通过!")
    return result


def test_conditional_model_with_predictors():
    """测试2: Conditional模型,包含预测变量"""
    print("\n" + "="*70)
    print("测试2: Conditional Model - 包含预测变量")
    print("="*70)
    
    # 模拟参数
    n_subjects = 8
    n_obs_per_subject = 6
    d = 3  # 预测变量维度
    true_sigma2_between = 0.4
    true_sigma2_within = 0.25
    
    # 生成数据
    np.random.seed(123)
    subject_ids = np.repeat(np.arange(n_subjects), n_obs_per_subject)
    n_total = len(subject_ids)
    
    # 预测变量
    X = np.random.randn(n_total, d)
    true_beta = np.array([0.5, -0.3, 0.2])  # 真实系数
    
    # 被试随机效应
    subject_effects = np.random.randn(n_subjects) * np.sqrt(true_sigma2_between)
    
    # y = X*beta + subject_effect + noise
    y = X @ true_beta + subject_effects[subject_ids] + np.random.randn(n_total) * np.sqrt(true_sigma2_within)
    
    true_icc = true_sigma2_between / (true_sigma2_between + true_sigma2_within)
    
    # 评估
    result = evaluate_variance_components(
        X_train=X,
        y_train=y,
        subject_ids=subject_ids,
        true_variance_components={
            "sigma2_between": true_sigma2_between,
            "sigma2_within": true_sigma2_within,
        },
        include_predictors=True,
    )
    
    # 输出结果
    print(f"\n数据生成参数:")
    print(f"  被试数: {n_subjects}")
    print(f"  每被试观测数: {n_obs_per_subject}")
    print(f"  预测变量维度: {d}")
    print(f"  真实β: {true_beta}")
    print(f"  真实σ²_between: {true_sigma2_between:.4f}")
    print(f"  真实σ²_within: {true_sigma2_within:.4f}")
    print(f"  真实Conditional ICC: {true_icc:.4f}")
    
    print(f"\n估计结果:")
    print(f"  估计σ²_between: {result['sigma2_between']:.4f}")
    print(f"  估计σ²_within: {result['sigma2_within']:.4f}")
    print(f"  估计ICC: {result['icc']:.4f}")
    print(f"  模型类型: {result['model_type']}")
    
    if result['estimation_accuracy']:
        acc = result['estimation_accuracy']
        print(f"\n估计准确性:")
        print(f"  ICC误差: {acc['icc_error']:.4f}")
        print(f"  σ²_between相对误差: {acc['icc_relative_error']*100:.1f}%")
    
    # 验证
    assert result['model_type'] == 'conditional_model', "模型类型应为conditional_model"
    assert abs(result['icc'] - true_icc) < 0.3, f"ICC估计偏差过大: {abs(result['icc'] - true_icc):.4f}"
    
    print("\n✅ 测试2通过!")
    return result


def test_unbalanced_design():
    """测试3: 不均衡设计 (不同被试观测数不同)"""
    print("\n" + "="*70)
    print("测试3: 不均衡设计")
    print("="*70)
    
    # 模拟参数 - 不同被试不同观测数
    obs_counts = [3, 5, 8, 4, 10, 2]  # 每个被试的观测数
    n_subjects = len(obs_counts)
    true_sigma2_between = 0.6
    true_sigma2_within = 0.2
    
    # 生成数据
    np.random.seed(999)
    subject_ids = np.concatenate([np.full(count, i) for i, count in enumerate(obs_counts)])
    n_total = len(subject_ids)
    
    X = np.random.randn(n_total, 4)
    subject_effects = np.random.randn(n_subjects) * np.sqrt(true_sigma2_between)
    true_beta = np.array([0.4, -0.2, 0.1, 0.3])
    
    y = X @ true_beta + subject_effects[subject_ids] + np.random.randn(n_total) * np.sqrt(true_sigma2_within)
    
    true_icc = true_sigma2_between / (true_sigma2_between + true_sigma2_within)
    
    # 评估
    result = evaluate_variance_components(
        X_train=X,
        y_train=y,
        subject_ids=subject_ids,
        true_variance_components={
            "sigma2_between": true_sigma2_between,
            "sigma2_within": true_sigma2_within,
        },
        include_predictors=True,
    )
    
    # 输出结果
    print(f"\n数据生成参数:")
    print(f"  被试数: {n_subjects}")
    print(f"  每被试观测数: {obs_counts}")
    print(f"  总观测数: {n_total}")
    print(f"  真实σ²_between: {true_sigma2_between:.4f}")
    print(f"  真实σ²_within: {true_sigma2_within:.4f}")
    print(f"  真实ICC: {true_icc:.4f}")
    
    print(f"\n估计结果:")
    print(f"  估计σ²_between: {result['sigma2_between']:.4f}")
    print(f"  估计σ²_within: {result['sigma2_within']:.4f}")
    print(f"  估计ICC: {result['icc']:.4f}")
    print(f"  被试观测数统计:")
    print(f"    - 平均: {result['mean_obs_per_subject']:.1f}")
    print(f"    - 最小: {result['min_obs_per_subject']}")
    print(f"    - 最大: {result['max_obs_per_subject']}")
    
    if result['estimation_accuracy']:
        acc = result['estimation_accuracy']
        print(f"\n估计准确性:")
        print(f"  ICC误差: {acc['icc_error']:.4f}")
        print(f"  σ²_between误差: {acc['sigma2_between_error']:.4f}")
        print(f"  σ²_within误差: {acc['sigma2_within_error']:.4f}")
    
    # 验证
    assert result['n_subjects'] == n_subjects, "被试数量不匹配"
    assert result['n_observations'] == n_total, "总观测数不匹配"
    assert result['mean_obs_per_subject'] == np.mean(obs_counts), "平均观测数计算错误"
    
    print("\n✅ 测试3通过!")
    return result


def test_small_sample_warning():
    """测试4: 小样本情况 (检测警告和降级处理)"""
    print("\n" + "="*70)
    print("测试4: 小样本测试 (可能触发降级估计)")
    print("="*70)
    
    # 极小样本: 3个被试,每人2次观测
    n_subjects = 3
    n_obs_per_subject = 2
    
    np.random.seed(777)
    subject_ids = np.repeat(np.arange(n_subjects), n_obs_per_subject)
    X = np.random.randn(len(subject_ids), 2)
    
    # 人工设置大的被试间差异
    subject_effects = np.array([1.0, -1.5, 0.8])
    y = subject_effects[subject_ids] + np.random.randn(len(subject_ids)) * 0.2
    
    # 评估
    result = evaluate_variance_components(
        X_train=X,
        y_train=y,
        subject_ids=subject_ids,
        include_predictors=True,
    )
    
    print(f"\n数据生成参数:")
    print(f"  被试数: {n_subjects} (⚠️ 小样本)")
    print(f"  每被试观测数: {n_obs_per_subject} (⚠️ 小样本)")
    print(f"  总观测数: {len(subject_ids)}")
    
    print(f"\n估计结果:")
    print(f"  估计σ²_between: {result['sigma2_between']:.4f}")
    print(f"  估计σ²_within: {result['sigma2_within']:.4f}")
    print(f"  估计ICC: {result['icc']:.4f}")
    print(f"  模型类型: {result['model_type']}")
    
    # 小样本下,只验证计算不崩溃
    assert not np.isnan(result['icc']), "ICC不应为NaN"
    assert result['icc'] >= 0 and result['icc'] <= 1, "ICC应在[0,1]范围内"
    
    print("\n✅ 测试4通过 (小样本情况处理正常)!")
    return result


def test_edge_cases():
    """测试5: 边界情况"""
    print("\n" + "="*70)
    print("测试5: 边界情况测试")
    print("="*70)
    
    # Case 1: ICC=0 (无被试间差异)
    print("\n[Case 1] ICC ≈ 0 (所有被试相同)")
    np.random.seed(111)
    subject_ids = np.repeat([0, 1, 2, 3], 5)
    X = np.random.randn(20, 2)
    # 所有被试效应为0
    y = np.random.randn(20) * 0.5  # 仅噪声
    
    result1 = evaluate_variance_components(X, y, subject_ids, include_predictors=False)
    print(f"  估计ICC: {result1['icc']:.4f} (期望接近0)")
    print(f"  σ²_between: {result1['sigma2_between']:.4f}")
    print(f"  σ²_within: {result1['sigma2_within']:.4f}")
    
    # Case 2: ICC≈1 (被试间差异极大)
    print("\n[Case 2] ICC ≈ 1 (被试间差异极大)")
    subject_effects = np.array([10, -10, 15, -15])
    y2 = subject_effects[subject_ids] + np.random.randn(20) * 0.01  # 极小噪声
    
    result2 = evaluate_variance_components(X, y2, subject_ids, include_predictors=False)
    print(f"  估计ICC: {result2['icc']:.4f} (期望接近1)")
    print(f"  σ²_between: {result2['sigma2_between']:.4f}")
    print(f"  σ²_within: {result2['sigma2_within']:.4f}")
    
    # 验证
    assert result1['icc'] < 0.3, "Case 1: ICC应接近0"
    assert result2['icc'] > 0.8, "Case 2: ICC应接近1"
    
    print("\n✅ 测试5通过!")
    return result1, result2


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Variance Components Evaluation - 完整测试套件")
    print("="*70)
    
    try:
        test_null_model_balanced()
        test_conditional_model_with_predictors()
        test_unbalanced_design()
        test_small_sample_warning()
        test_edge_cases()
        
        print("\n" + "="*70)
        print("🎉 所有测试通过!")
        print("="*70)
        print("\n功能验证:")
        print("  ✅ Null模型ICC估计")
        print("  ✅ Conditional模型(含预测变量)")
        print("  ✅ 不均衡设计处理")
        print("  ✅ 小样本降级处理")
        print("  ✅ 边界情况 (ICC=0, ICC=1)")
        print("  ✅ 估计准确性验证")
        print("\n可用于后续LMM分析评估!")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
