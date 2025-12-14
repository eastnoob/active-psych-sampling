# 20251128 计划总结

## 📋 文档清单

### 1. **BaseGP_Prior_Integration_Full.md** (⭐ 推荐 - 完整详解)

**完整的 BaseGP 先验集成指南，包含数据获取路径**

核心内容:

- **第 0 节**: 数据和文件获取路径 (BaseGP 输出 + 代码位置)
- **第 1 节**: BaseGP 学习到的参数完整清单 + 核函数配置代码
- **第 2 节**: 如何从 BaseGP 提取参数并转换为 INI 配置 (含脚本)
- **第 3-4 节**: OrdinalGPModel / GPRegressionModel 配置 + 代码示例
- **第 5 节**: 核函数对比 (Matern vs RBF)
- **第 6 节**: 关键文件路径快速参考
- **第 7 节**: 执行步骤 (完整流程)
- **第 8 节**: 故障排查

**快速导航**:

- BaseGP 数据位置: 第 0 节
- Lengthscales 计算: 第 2.1 节
- Noise Variance: 第 4.1 节
- 核函数代码: 第 1.2 / 0.2 节
- 故障排查: 第 8 节

### 2. **BaseGP_Prior_Integration.md** (简明版 - 已保留)

精简版指南，核心配置方案

- BaseGP 参数值
- INI 配置模板
- 核函数说明

---

## 🎯 执行检查清单

```
Phase 1 完成: BaseGP 已训练
├── 输出位置: extensions/warmup_budget_check/phase1_analysis_output/202511271557/
├── 核函数: Matern(2.5) + ARD + Scale ✓
└── 参数学习完成 ✓

Phase 2 准备: 引入先验到 EUR 实验
├── OrdinalGPModel:
│   ├── ls_loc = [0.9124, 0.6325, 1.6092, -0.5785, 1.6942, 1.5350]
│   ├── ls_scale = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
│   └── fixed_kernel_amplitude = True
├── GPRegressionModel:
│   ├── 需要自定义 ConfigurableGaussianLikelihood
│   ├── noise_init = 0.814
│   └── outputscale_prior = gamma
└── 配置文件: eur_config_basegp_prior.ini

当前问题: SPS Lambda 保持常量 (0.7)
└── 解决方案: 见 current_work/sps_lambda_constant_analysis.md
```

---

## 📌 最关键的三个配置值

| 项目 | 值 | 用途 |
|------|-----|------|
| **Lengthscales** | `[1.9395, 1.4659, 3.8931, 0.4367, 4.2384, 3.6145]` | ARD 初始化 |
| **Noise Variance** | `0.814` | Regression 模型 |
| **Output Scale** | `0.167` | Regression outputscale |

---

## 🔗 相关文件

- **BaseGP 报告**: `extensions/warmup_budget_check/phase1_analysis_output/202511271557/base_gp/base_gp_report.md`
- **SPS 分析**: `current_work/sps_lambda_constant_analysis.md`
- **当前配置**: `eur_config_basegp_prior.ini`

---

**生成时间**: 2025-11-28 22:38
