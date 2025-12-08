"""测试 ALL_CONFIG 是否正确应用到各个 STEP 配置"""
# -*- coding: utf-8 -*-

from quick_start import ALL_CONFIG, STEP1_CONFIG, STEP2_CONFIG, STEP3_CONFIG, STEP1_5_CONFIG

print("=" * 60)
print("ALL_CONFIG Application Test")
print("=" * 60)
print()

# 测试 Step 1 配置
print("[OK] Step 1 Config:")
print(f"  n_subjects: {STEP1_CONFIG['n_subjects']} (expected: 5)")
print(f"  trials_per_subject: {STEP1_CONFIG['trials_per_subject']} (expected: 30)")
print(f"  skip_interaction: {STEP1_CONFIG['skip_interaction']} (expected: False)")
assert STEP1_CONFIG['n_subjects'] == 5
assert STEP1_CONFIG['trials_per_subject'] == 30
assert STEP1_CONFIG['skip_interaction'] == False
print()

# 测试 Step 1.5 配置
print("[OK] Step 1.5 Config:")
print(f"  seed: {STEP1_5_CONFIG['seed']} (expected: 42)")
print(f"  likert_levels: {STEP1_5_CONFIG['likert_levels']} (expected: 5)")
print(f"  population_std: {STEP1_5_CONFIG['population_std']} (expected: 0.4)")
assert STEP1_5_CONFIG['seed'] == 42
assert STEP1_5_CONFIG['likert_levels'] == 5
assert STEP1_5_CONFIG['population_std'] == 0.4
print()

# 测试 Step 2 配置
print("[OK] Step 2 Config:")
print(f"  max_pairs: {STEP2_CONFIG['max_pairs']} (expected: 5)")
print(f"  phase2_n_subjects: {STEP2_CONFIG['phase2_n_subjects']} (expected: 20)")
print(f"  lambda_adjustment: {STEP2_CONFIG['lambda_adjustment']} (expected: 1.2)")
assert STEP2_CONFIG['max_pairs'] == 5
assert STEP2_CONFIG['phase2_n_subjects'] == 20
assert STEP2_CONFIG['lambda_adjustment'] == 1.2
print()

# 测试 Step 3 配置
print("[OK] Step 3 Config:")
print(f"  max_iters: {STEP3_CONFIG['max_iters']} (expected: 200)")
print(f"  learning_rate: {STEP3_CONFIG['learning_rate']} (expected: 0.05)")
print(f"  use_cuda: {STEP3_CONFIG['use_cuda']} (expected: False)")
assert STEP3_CONFIG['max_iters'] == 200
assert STEP3_CONFIG['learning_rate'] == 0.05
assert STEP3_CONFIG['use_cuda'] == False
print()

print("=" * 60)
print("[SUCCESS] ALL_CONFIG successfully applied to all STEP configs!")
print("=" * 60)
